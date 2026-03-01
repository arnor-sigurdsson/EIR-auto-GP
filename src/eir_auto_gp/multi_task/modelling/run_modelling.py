import contextlib
import os
import shutil
import subprocess
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import luigi
import pandas as pd
import yaml
from aislib.misc_utils import ensure_path_exists
from eir.setup.config_setup_modules.config_setup_utils import recursive_dict_inject

from eir_auto_gp.multi_task.modelling.configs import (
    AdversarialParams,
    AggregateConfig,
    ArchitectureParams,
    TabularSkipParams,
    get_aggregate_config,
    get_num_lcl_blocks,
)
from eir_auto_gp.multi_task.modelling.hyperparameters import get_gln_kernel_parameters
from eir_auto_gp.multi_task.modelling.injections import (
    _get_all_dynamic_injections,
    build_injection_params,
)
from eir_auto_gp.preprocess.converge import (
    ParseDataWrapper,
)
from eir_auto_gp.single_task.modelling.run_modelling import (
    lines_in_file,
)
from eir_auto_gp.utils.shared_modelling_utils import (
    get_bim_path,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def validate_architecture_config(modelling_config: dict[str, Any]) -> None:
    required_arch_flags = ["use_lcl_to_output_skips", "use_lcl_fusion_skips"]

    for flag in required_arch_flags:
        if flag not in modelling_config:
            raise ValueError(
                f"Architecture flag '{flag}' is missing from modelling_config. "
                f"This flag controls model architecture and must be set explicitly. "
                f"Available keys: {list(modelling_config.keys())}"
            )

    valid_lcl_output_values = [True, False, "fc_1_only"]
    if modelling_config["use_lcl_to_output_skips"] not in valid_lcl_output_values:
        raise ValueError(
            f"Invalid value for 'use_lcl_to_output_skips': "
            f"{modelling_config['use_lcl_to_output_skips']}. "
            f"Must be one of: {valid_lcl_output_values}"
        )

    if not isinstance(modelling_config["use_lcl_fusion_skips"], bool):
        raise ValueError(
            f"Invalid value for 'use_lcl_fusion_skips': "
            f"{modelling_config['use_lcl_fusion_skips']}. "
            f"Must be a boolean (True/False)."
        )

    logger.info(
        "Architecture configuration validated: "
        "use_lcl_to_output_skips=%s, use_lcl_fusion_skips=%s",
        modelling_config["use_lcl_to_output_skips"],
        modelling_config["use_lcl_fusion_skips"],
    )


class RunModellingWrapper(luigi.Task):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        task_object = (
            TestSingleRun if self.modelling_config["do_test"] else TrainSingleRun
        )
        for fold in _get_fold_iterator(folds=str(self.folds)):
            yield task_object(
                fold=fold,
                data_config=self.data_config,
                modelling_config=self.modelling_config,
            )

        cleanup_tmp_files(modelling_config=self.modelling_config)

    def output(self):
        return self.input()


def cleanup_tmp_files(modelling_config: dict[str, Any]) -> None:
    tmp_dir = Path(modelling_config["modelling_output_folder"]) / "tmp"
    if tmp_dir.exists():
        for file in tmp_dir.iterdir():
            file.unlink()
        tmp_dir.rmdir()


def _get_fold_iterator(folds: str) -> Iterable:
    if "-" in folds:
        start, end = folds.split("-")
        return range(int(start), int(end) + 1)
    elif "," in folds:
        return [int(i) for i in folds.split(",")]
    else:
        return range(int(folds))


def calculate_n_lcl_blocks(genotype_data_path: str) -> int:
    bim_path = get_bim_path(genotype_data_path=genotype_data_path)
    n_snps = lines_in_file(file_path=bim_path)

    kernel_width, first_kernel_expansion = get_gln_kernel_parameters(n_snps=n_snps)

    n_lcl_blocks = get_num_lcl_blocks(
        n_snps=n_snps,
        kernel_width=kernel_width,
        first_kernel_expansion=first_kernel_expansion,
        channel_exp_base=3,
        rb_do=0.10,
        stochastic_depth_p=0.00,
        cutoff=4096,
    )

    return n_lcl_blocks


def _filter_input_configs(configs: list[dict]) -> list[dict]:
    filtered = [c for c in configs if c["input_info"]["input_name"] != "eir_tabular"]
    if len(filtered) < len(configs):
        logger.info(f"Filtered {len(configs) - len(filtered)} tabular input config(s)")
    return filtered


def _filter_adversarial_from_global_config(config: dict) -> dict:
    if "adversarial_training" in config:
        filtered_config = config.copy()
        del filtered_config["adversarial_training"]
        return filtered_config
    return config


def _filter_tabular_from_fusion_config(config: dict) -> dict:
    if "tensor_broker_config" not in config:
        return config

    tb_config = config["tensor_broker_config"]
    if "message_configs" not in tb_config:
        return config

    original_messages = tb_config["message_configs"]
    filtered_messages = []

    for msg in original_messages:
        if "tabular" in msg.get("name", "").lower():
            logger.info(f"Filtering tabular TB message: {msg.get('name')}")
            continue

        if msg.get("from") == "tabular_output":
            logger.info(
                f"Filtering TB message with tabular origin: {msg.get('name')} "
                f"(from: {msg.get('from')})"
            )
            continue

        if "use_from_cache" in msg:
            original_cache = msg["use_from_cache"]
            filtered_cache = [c for c in original_cache if c != "tabular_output"]
            if len(filtered_cache) < len(original_cache):
                logger.info(
                    f"Removed 'tabular_output' from TB message: {msg.get('name')}"
                )
                msg["use_from_cache"] = filtered_cache

        filtered_messages.append(msg)

    config["tensor_broker_config"]["message_configs"] = filtered_messages
    return config


@contextmanager
def _filter_serialized_configs_for_genotype_only(train_run_folder: Path):
    configs_folder = train_run_folder / "serializations" / "configs_stripped"

    if not configs_folder.exists():
        logger.warning(
            f"Serialized configs folder not found: {configs_folder}."
            f" Skipping filtering."
        )
        yield
        return

    backup_folder = configs_folder.parent / "configs_stripped_backup"
    backup_folder.mkdir(exist_ok=True, parents=True)

    try:
        logger.info(
            f"Backing up serialized configs from {configs_folder} to {backup_folder}"
        )
        for config_file in configs_folder.iterdir():
            if config_file.suffix == ".yaml":
                shutil.copy2(config_file, backup_folder / config_file.name)

        logger.info("Filtering serialized configs for genotype-only prediction")
        for config_file in configs_folder.iterdir():
            if config_file.suffix != ".yaml":
                continue

            if config_file.stem == "input_configs":
                configs = yaml.safe_load(config_file.read_text())
                filtered_configs = _filter_input_configs(configs=configs)

                if len(filtered_configs) == len(configs):
                    logger.info("No tabular input found in training config, skipping")
                    continue

                config_file.write_text(yaml.dump(filtered_configs))

            elif config_file.stem == "fusion_config":
                config = yaml.safe_load(config_file.read_text())
                filtered_config = _filter_tabular_from_fusion_config(config=config)
                config_file.write_text(yaml.dump(filtered_config))

            elif config_file.stem == "global_config":
                config = yaml.safe_load(config_file.read_text())
                filtered_config = _filter_adversarial_from_global_config(config=config)
                if filtered_config != config:
                    logger.info(
                        "Removed adversarial_training from global_config for "
                        "genotype-only prediction"
                    )
                    config_file.write_text(yaml.dump(filtered_config))

        yield

    finally:
        logger.info(f"Restoring original serialized configs from {backup_folder}")
        for backup_file in backup_folder.iterdir():
            if backup_file.suffix == ".yaml":
                target_path = configs_folder / backup_file.name
                if target_path.exists():
                    target_path.unlink()
                shutil.copy2(backup_file, target_path)

        shutil.rmtree(backup_folder)
        logger.info("Restored original serialized configs")


class TestSingleRun(luigi.Task):
    fold = luigi.IntParameter()

    data_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        base = {
            "data": ParseDataWrapper(
                data_config=self.data_config,
                modelling_config=self.modelling_config,
            ),
            "train_run": TrainSingleRun(
                fold=self.fold,
                data_config=self.data_config,
                modelling_config=self.modelling_config,
            ),
        }

        return base

    def run(self):
        output_root = Path(self.output().path).parent
        ensure_path_exists(output_root, is_folder=True)

        validate_architecture_config(modelling_config=self.modelling_config)

        all_target_columns = (
            self.modelling_config["output_cat_columns"]
            + self.modelling_config["output_con_columns"]
        )

        n_lcl_blocks = calculate_n_lcl_blocks(
            genotype_data_path=self.data_config["genotype_data_path"]
        )

        has_tabular_columns = bool(
            self.modelling_config["input_cat_columns"]
            or self.modelling_config["input_con_columns"]
        )
        tabular_to_output_skips = has_tabular_columns and not self.modelling_config.get(
            "genotype_only_test", False
        )

        arch_params = ArchitectureParams.from_modelling_config(
            config=self.modelling_config,
        )
        tabular_params = TabularSkipParams(enabled=tabular_to_output_skips)
        adversarial_params = AdversarialParams(
            enabled=self.modelling_config.get("adversarial_enabled", True),
            lambda_=self.modelling_config.get("adversarial_lambda", 0.5),
        )

        base_aggregate_config = get_aggregate_config(
            arch_params=arch_params,
            target_columns=all_target_columns,
            output_cat_columns=self.modelling_config["output_cat_columns"],
            output_con_columns=self.modelling_config["output_con_columns"],
            n_lcl_blocks=n_lcl_blocks,
            tabular_params=tabular_params,
            adversarial_params=adversarial_params,
        )

        injection_params = build_injection_params(
            fold=int(self.fold),
            data_input_dict=self.input()["data"][0],
            task="test",
            data_config=self.data_config,
            modelling_config=self.modelling_config,
            output_configs=base_aggregate_config.output_config,
        )

        injections = _get_all_dynamic_injections(
            injection_params=injection_params,
            genotype_data_path=self.data_config["genotype_data_path"],
            expert_snp_groups_file=base_aggregate_config.expert_snp_groups_file,
        )

        with TemporaryDirectory() as temp_dir:
            temp_config_folder = Path(temp_dir)
            build_configs(
                injections=injections,
                aggregate_config_base=base_aggregate_config,
                output_folder=temp_config_folder,
            )

            train_run_folder = Path(self.input()["train_run"].path).parent

            filter_context = (
                _filter_serialized_configs_for_genotype_only(
                    train_run_folder=train_run_folder
                )
                if self.modelling_config.get("genotype_only_test", False)
                else contextlib.nullcontext()
            )

            with filter_context:
                base_predict_command = get_testing_string_from_config_folder(
                    config_folder=temp_config_folder,
                    train_run_folder=train_run_folder,
                    with_labels=True,
                )
                logger.info("Testing command: %s", base_predict_command)

                base_command_split = base_predict_command.split()
                process_info = subprocess.run(
                    base_command_split,
                    env=dict(os.environ, EIR_SEED=str(self.fold)),
                )

                if process_info.returncode != 0:
                    raise RuntimeError(
                        f"Testing failed with return code {process_info.returncode}"
                    )
                testing_complete_file = Path(self.output().path)
                testing_complete_file.touch()

    def output(self):
        run_folder = Path(self.input()["train_run"].path).parent
        output_path = run_folder / "test_set_predictions/test_complete.txt"
        return luigi.LocalTarget(str(output_path))


def get_testing_string_from_config_folder(
    config_folder: Path,
    train_run_folder: Path,
    with_labels: bool,
) -> str:
    base_string = "eirpredict"
    globals_string = " --global_configs "
    inputs_string = " --input_configs "
    fusion_string = " --fusion_configs "
    output_string = " --output_configs "

    for file in config_folder.iterdir():
        if file.suffix == ".yaml":
            if "global" in file.stem:
                globals_string += " " + f"{str(file)}" + " "
            elif "input" in file.stem:
                inputs_string += " " + f"{str(file)}" + " "
            elif "fusion" in file.stem:
                fusion_string += " " + f"{str(file)}" + " "
            elif "output" in file.stem:
                output_string += " " + f"{str(file)}" + " "

    final_string = (
        base_string + globals_string + inputs_string + fusion_string + output_string
    )

    saved_models = list((train_run_folder / "saved_models").iterdir())
    assert len(saved_models) == 1, "Expected only one saved model."

    final_string += f" --model_path {saved_models[0]}"
    final_string += " --no-strict"
    if with_labels:
        final_string += " --evaluate"

    test_output_folder = train_run_folder / "test_set_predictions"
    ensure_path_exists(path=test_output_folder, is_folder=True)

    final_string += f" --output_folder {test_output_folder}"

    return final_string


class TrainSingleRun(luigi.Task):
    fold = luigi.IntParameter()
    data_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        base = {
            "data": ParseDataWrapper(
                data_config=self.data_config,
                modelling_config=self.modelling_config,
            )
        }
        for i in range(self.fold):
            base[f"train_{i}"] = TrainSingleRun(
                fold=i,
                data_config=self.data_config,
                modelling_config=self.modelling_config,
            )
        return base

    def run(self):
        output_root = Path(self.output().path).parent
        ensure_path_exists(output_root, is_folder=True)

        validate_architecture_config(modelling_config=self.modelling_config)

        all_target_columns = (
            self.modelling_config["output_cat_columns"]
            + self.modelling_config["output_con_columns"]
        )

        n_lcl_blocks = calculate_n_lcl_blocks(
            genotype_data_path=self.data_config["genotype_data_path"]
        )

        has_tabular_columns = bool(
            self.modelling_config["input_cat_columns"]
            or self.modelling_config["input_con_columns"]
        )

        arch_params = ArchitectureParams.from_modelling_config(
            config=self.modelling_config,
        )
        tabular_params = TabularSkipParams(enabled=has_tabular_columns)
        adversarial_params = AdversarialParams(
            enabled=self.modelling_config.get("adversarial_enabled", True),
            lambda_=self.modelling_config.get("adversarial_lambda", 0.5),
        )

        base_aggregate_config = get_aggregate_config(
            arch_params=arch_params,
            target_columns=all_target_columns,
            output_cat_columns=self.modelling_config["output_cat_columns"],
            output_con_columns=self.modelling_config["output_con_columns"],
            n_lcl_blocks=n_lcl_blocks,
            tabular_params=tabular_params,
            adversarial_params=adversarial_params,
        )

        injection_params = build_injection_params(
            fold=int(self.fold),
            data_input_dict=self.input()["data"][0],
            task="train",
            data_config=self.data_config,
            modelling_config=self.modelling_config,
            output_configs=base_aggregate_config.output_config,
        )

        injections = _get_all_dynamic_injections(
            injection_params=injection_params,
            genotype_data_path=self.data_config["genotype_data_path"],
            expert_snp_groups_file=base_aggregate_config.expert_snp_groups_file,
        )

        with TemporaryDirectory() as temp_dir:
            temp_config_folder = Path(temp_dir)
            build_configs(
                injections=injections,
                aggregate_config_base=base_aggregate_config,
                output_folder=temp_config_folder,
            )

            base_train_command = get_training_string_from_config_folder(
                config_folder=temp_config_folder
            )
            logger.info("Training command: %s", base_train_command)

            base_command_split = base_train_command.split()
            process_info = subprocess.run(
                base_command_split,
                env=dict(os.environ, EIR_SEED=str(self.fold)),
            )

            if process_info.returncode != 0:
                raise RuntimeError(
                    f"Training failed with return code {process_info.returncode}"
                )
            training_complete_file = Path(self.output().path)
            training_complete_file.touch()

    def output(self):
        modelling_output_folder = Path(self.modelling_config["modelling_output_folder"])
        fold_output_folder = modelling_output_folder / f"fold_{self.fold}"
        file_name = fold_output_folder / "completed_train.txt"
        local_target = luigi.LocalTarget(path=str(file_name))
        return local_target


def build_tmp_label_file(
    label_file_path: str,
    tmp_dir: Path,
    output_cat_columns: list[str],
    output_con_columns: list[str],
    prefix_name: str,
) -> str:
    tmp_label_file_path = Path(tmp_dir) / f"{prefix_name}_label_file.csv"

    if tmp_label_file_path.exists():
        return str(tmp_label_file_path)

    df = pd.read_csv(label_file_path)

    if prefix_name == "test":
        train_file_path = Path(tmp_dir) / "train_label_file.csv"
        assert Path(train_file_path).exists(), train_file_path
        df_train = pd.read_csv(train_file_path, usecols=["ID"])
        ids_train = set(df_train["ID"].tolist())
        ids_test = set(df["ID"].tolist())
        assert len(ids_train.intersection(ids_test)) == 0

    ensure_path_exists(path=tmp_label_file_path.parent, is_folder=True)
    df.to_csv(tmp_label_file_path, index=False)

    return str(tmp_label_file_path)


def build_configs(
    aggregate_config_base: AggregateConfig,
    injections: dict[str, Any],
    output_folder: Path,
) -> None:
    for config_field in fields(aggregate_config_base):
        config_name = config_field.name
        config = getattr(aggregate_config_base, config_name)
        if config_name in injections:
            if config_name == "output_config":
                assert isinstance(config, list)

                for output_config in config:
                    output_config_name = output_config["output_info"]["output_name"]
                    cur_to_inject = injections[config_name][output_config_name]
                    output_config = recursive_dict_inject(
                        dict_=output_config,
                        dict_to_inject=cur_to_inject,
                    )

                    validate_complete_config(config_element=output_config)

                    out_path = output_folder / f"output_{output_config_name}.yaml"

                    with open(out_path, "w") as f:
                        yaml.dump(output_config, f)

            else:
                config = recursive_dict_inject(
                    dict_=config,
                    dict_to_inject=injections[config_name],
                )
                validate_complete_config(config_element=config)
                with open(output_folder / f"{config_name}.yaml", "w") as f:
                    yaml.dump(config, f)
        else:
            continue


def validate_complete_config(config_element: dict | list | str) -> None:
    match config_element:
        case dict(config_element):
            for _key, value in config_element.items():
                validate_complete_config(config_element=value)
        case list(config_element):
            for value in config_element:
                validate_complete_config(config_element=value)
        case str(config_element):
            assert config_element != "FILL"


def get_training_string_from_config_folder(config_folder: Path) -> str:
    base_string = "eirtrain"
    globals_string = " --global_configs "
    inputs_string = " --input_configs "
    fusion_string = " --fusion_configs "
    output_string = " --output_configs "

    for file in config_folder.iterdir():
        if file.suffix == ".yaml" and "_test" not in file.stem:
            if "global" in file.stem:
                globals_string += " " + f"{str(file)}" + " "
            elif "input" in file.stem:
                inputs_string += " " + f"{str(file)}" + " "
            elif "fusion" in file.stem:
                fusion_string += " " + f"{str(file)}" + " "
            elif "output" in file.stem:
                output_string += " " + f"{str(file)}" + " "

    final_string = (
        base_string + globals_string + inputs_string + fusion_string + output_string
    )

    return final_string
