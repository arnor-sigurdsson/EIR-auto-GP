import os
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Literal, Optional

import luigi
import pandas as pd
import polars as pl
import yaml
from aislib.misc_utils import ensure_path_exists
from eir.setup.config_setup_modules.config_setup_utils import recursive_dict_inject
from eir.setup.input_setup_modules.setup_omics import read_bim

from eir_auto_gp.multi_task.modelling.configs import (
    AggregateConfig,
    get_aggregate_config,
)
from eir_auto_gp.preprocess.converge import (
    ParseDataWrapper,
    get_batch_size,
    get_dynamic_valid_size,
)
from eir_auto_gp.single_task.modelling.run_modelling import (
    get_bim_path,
    get_dataloader_workers,
    get_device,
    get_memory_dataset,
    get_samples_per_epoch,
    lines_in_file,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


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


def cleanup_tmp_files(modelling_config: Dict[str, Any]) -> None:
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


class TestSingleRun(luigi.Task):
    fold = luigi.IntParameter()

    data_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        base = {
            "data": ParseDataWrapper(data_config=self.data_config),
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

        all_target_columns = (
            self.modelling_config["output_cat_columns"]
            + self.modelling_config["output_con_columns"]
        )

        base_aggregate_config = get_aggregate_config(
            output_head="linear",
            output_groups=self.modelling_config["output_groups"],
            model_size=self.modelling_config["model_size"],
            target_columns=all_target_columns,
            output_cat_columns=self.modelling_config["output_cat_columns"],
            output_con_columns=self.modelling_config["output_con_columns"],
            n_random_groups=self.modelling_config["n_random_output_groups"],
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
        )

        with TemporaryDirectory() as temp_dir:
            temp_config_folder = Path(temp_dir)
            build_configs(
                injections=injections,
                aggregate_config_base=base_aggregate_config,
                output_folder=temp_config_folder,
            )

            train_run_folder = Path(self.input()["train_run"].path).parent
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
        base = {"data": ParseDataWrapper(data_config=self.data_config)}
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

        all_target_columns = (
            self.modelling_config["output_cat_columns"]
            + self.modelling_config["output_con_columns"]
        )

        base_aggregate_config = get_aggregate_config(
            output_head="linear",
            output_groups=self.modelling_config["output_groups"],
            model_size=self.modelling_config["model_size"],
            target_columns=all_target_columns,
            output_cat_columns=self.modelling_config["output_cat_columns"],
            output_con_columns=self.modelling_config["output_con_columns"],
            n_random_groups=self.modelling_config["n_random_output_groups"],
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


@dataclass
class ModelInjectionParams:
    fold: int
    modelling_base_output_folder: str
    output_folder: str
    manual_valid_ids_file: Optional[str]
    genotype_input_source: str
    genotype_feature_selection: str
    label_file_path: str
    input_cat_columns: list[str]
    input_con_columns: list[str]
    output_cat_columns: list[str]
    output_con_columns: list[str]
    weighted_sampling_columns: list[str]
    modelling_data_format: str
    output_configs: list[dict[str, Any]]


def build_injection_params(
    fold: int,
    data_input_dict: Dict[str, luigi.LocalTarget],
    task: Literal["train", "test"],
    data_config: Dict[str, Any],
    modelling_config: Dict[str, Any],
    output_configs: list[Dict[str, Any]],
) -> ModelInjectionParams:
    weighted_sampling_columns = None

    base_output_folder = modelling_config["modelling_output_folder"]
    cur_run_output_folder = f"{base_output_folder}/fold_{fold}"

    label_file_path = data_input_dict[f"{task}_tabular"].path

    manual_valid_ids_file = get_manual_valid_ids_file(
        task=task,
        data_config=data_config,
        label_file_path=label_file_path,
        base_output_folder=base_output_folder,
    )

    params = ModelInjectionParams(
        fold=fold,
        output_folder=cur_run_output_folder,
        modelling_base_output_folder=base_output_folder,
        manual_valid_ids_file=str(manual_valid_ids_file),
        genotype_input_source=data_input_dict[f"{task}_genotype"].path,
        label_file_path=label_file_path,
        genotype_feature_selection=modelling_config["genotype_feature_selection"],
        input_cat_columns=modelling_config["input_cat_columns"],
        input_con_columns=modelling_config["input_con_columns"],
        output_cat_columns=modelling_config["output_cat_columns"],
        output_con_columns=modelling_config["output_con_columns"],
        weighted_sampling_columns=weighted_sampling_columns,
        modelling_data_format=data_config["modelling_data_format"],
        output_configs=output_configs,
    )

    return params


def get_weighted_sampling_columns(
    modelling_config: Dict[str, Any]
) -> Optional[list[str]]:
    return ["all"] if modelling_config["output_cat_columns"] else None


def get_manual_valid_ids_file(
    task: str,
    data_config: Dict[str, Any],
    label_file_path: str,
    base_output_folder: str,
) -> Optional[str]:
    if task != "train":
        return None

    valid_ids_file = Path(data_config["data_output_folder"], "ids/valid_ids.txt")
    if not valid_ids_file.exists():
        return None

    manual_valid_ids = pd.read_csv(valid_ids_file, header=None)[0].tolist()
    df_ids = pd.read_csv(label_file_path, usecols=["ID"]).dropna()
    intersected_ids = list(
        set(manual_valid_ids).intersection(set(df_ids["ID"].tolist()))
    )

    manual_valid_ids_file = Path(base_output_folder, "tmp", "valid_ids.txt")
    manual_valid_ids_file.parent.mkdir(parents=True, exist_ok=True)

    with open(manual_valid_ids_file, "w") as f:
        for id_ in intersected_ids:
            f.write(f"{id_}\n")

    return str(manual_valid_ids_file)


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
    injections: Dict[str, Any],
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
            for key, value in config_element.items():
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


def _get_global_injections(
    fold: int,
    output_folder: str,
    valid_size: int,
    batch_size: int,
    manual_valid_ids_file: Optional[str],
    n_snps: int,
    n_samples: int,
    iter_per_epoch: int,
    weighted_sampling_columns: list[str],
    modelling_data_format: str,
) -> Dict[str, Any]:
    mixing_candidates = [0.0]
    cur_mixing = mixing_candidates[fold % len(mixing_candidates)]

    device = get_device()

    if modelling_data_format == "auto":
        memory_dataset = get_memory_dataset(n_snps=n_snps, n_samples=n_samples)
    elif modelling_data_format == "disk":
        memory_dataset = False
    elif modelling_data_format == "memory":
        memory_dataset = True
    else:
        raise ValueError(f"Unknown data format: '{modelling_data_format}'.")

    n_workers = get_dataloader_workers(memory_dataset=memory_dataset, device=device)
    early_stopping_buffer = min(5000, iter_per_epoch * 5)
    early_stopping_buffer = max(early_stopping_buffer, 1000)
    sample_interval = min(1000, iter_per_epoch)
    lr = _get_learning_rate(n_snps=n_snps)

    injections = {
        "basic_experiment": {
            "output_folder": output_folder,
            "device": device,
            "batch_size": batch_size,
            "valid_size": valid_size,
            "manual_valid_ids_file": manual_valid_ids_file,
            "dataloader_workers": n_workers,
            "memory_dataset": memory_dataset,
        },
        "optimization": {
            "lr": lr,
        },
        "evaluation_checkpoint": {
            "sample_interval": sample_interval,
            "checkpoint_interval": sample_interval,
        },
        "training_control": {
            "mixing_alpha": cur_mixing,
            "early_stopping_buffer": early_stopping_buffer,
            "weighted_sampling_columns": weighted_sampling_columns,
        },
    }

    return injections


def _get_learning_rate(n_snps: int) -> float:
    if n_snps < 1_000:
        lr = 1e-03
    elif n_snps < 10_000:
        lr = 5e-04
    elif n_snps < 100_000:
        lr = 2e-04
    elif n_snps < 500_000:
        lr = 1e-04
    elif n_snps < 2_000_000:
        lr = 5e-05
    else:
        lr = 1e-05

    logger.info("Setting learning rate to %f due to %d SNPs.", lr, n_snps)

    return lr


def _get_genotype_injections(
    input_source: str,
    n_snps: int,
    subset_snp_path: Optional[Path],
) -> Dict[str, Any]:
    base_snp_path = (
        Path(input_source).parent.parent / "processed/parsed_files/data_final.bim"
    )
    assert base_snp_path.exists(), f"SNP file not found at {base_snp_path}"

    kernel_width, first_kernel_expansion = get_gln_kernel_parameters(n_snps=n_snps)

    injections = {
        "input_info": {
            "input_source": input_source,
        },
        "input_type_info": {
            "snp_file": str(base_snp_path),
        },
        "model_config": {
            "model_init_config": {
                "kernel_width": kernel_width,
                "first_kernel_expansion": first_kernel_expansion,
            }
        },
    }

    if subset_snp_path:
        injections["input_type_info"]["subset_snps_file"] = str(subset_snp_path)

    return injections


def get_gln_kernel_parameters(n_snps: int) -> tuple[int, int]:
    if n_snps < 1000:
        params = 16, -4
    elif n_snps < 10000:
        params = 16, -2
    elif n_snps < 100000:
        params = 16, 1
    elif n_snps < 500000:
        params = 16, 2
    elif n_snps < 2000000:
        params = 16, 4
    else:
        params = 16, 8

    logger.info(
        "Setting kernel width to %d and first kernel expansion to %d due to %d SNPs.",
        params[0],
        params[1],
        n_snps,
    )

    return params


def _get_tabular_injections(
    input_source: str, input_cat_columns: list[str], input_con_columns: list[str]
) -> Dict[str, Any]:
    injections = {
        "input_info": {
            "input_source": input_source,
        },
        "input_type_info": {
            "input_cat_columns": input_cat_columns,
            "input_con_columns": input_con_columns,
        },
    }
    return injections


def _get_output_injections(
    label_file_path: str,
    output_cat_columns: list[str],
    output_con_columns: list[str],
) -> dict[str, Any]:

    if output_cat_columns:
        df = pl.scan_csv(source=label_file_path).select(output_cat_columns).collect()

        all_binary = all(is_binary_column(df=df, col=col) for col in output_cat_columns)

        cat_loss = "BCEWithLogitsLoss" if all_binary else "CrossEntropyLoss"
        if all_binary:
            logger.info("Setting categorical loss to BCEWithLogitsLoss.")
    else:
        cat_loss = "CrossEntropyLoss"

    injections = {
        "output_info": {
            "output_source": label_file_path,
        },
        "output_type_info": {
            "target_cat_columns": output_cat_columns,
            "target_con_columns": output_con_columns,
            "cat_loss_name": cat_loss,
            "uncertainty_weighted_mt_loss": True,
        },
    }

    return injections


def is_binary_column(df: pl.DataFrame, col: str) -> bool:
    n_unique = df.select(pl.col(col)).filter(pl.col(col).is_not_null()).unique().height
    return n_unique <= 2


def _get_all_dynamic_injections(
    injection_params: ModelInjectionParams,
    genotype_data_path: str,
) -> Dict[str, Any]:
    mip = injection_params

    spe = get_samples_per_epoch(model_injection_params=mip)

    batch_size = get_batch_size(samples_per_epoch=spe.samples_per_epoch)

    valid_size = get_dynamic_valid_size(
        num_samples_per_epoch=spe.samples_per_epoch,
        minimum=batch_size,
    )
    iter_per_epoch = get_num_iter_per_epoch(
        num_samples_per_epoch=spe.samples_per_epoch,
        num_samples_total=spe.num_samples_total,
        batch_size=batch_size,
        valid_size=valid_size,
    )
    bim_path = get_bim_path(genotype_data_path=genotype_data_path)

    subset_folder = Path(mip.modelling_base_output_folder) / "snp_subset_files"

    subset_snp_path = None
    if mip.genotype_feature_selection == "random":
        n_snps, subset_snp_path = build_random_snp_subset_file(
            original_bim_path=Path(bim_path),
            output_folder=subset_folder,
            fold=mip.fold,
            fraction_per_chr=0.1,
        )
    else:
        assert not mip.genotype_feature_selection
        n_snps = lines_in_file(file_path=bim_path)

    injections = {
        "global_config": _get_global_injections(
            fold=mip.fold,
            output_folder=mip.output_folder,
            batch_size=batch_size,
            manual_valid_ids_file=mip.manual_valid_ids_file,
            valid_size=valid_size,
            iter_per_epoch=iter_per_epoch,
            n_snps=n_snps,
            n_samples=spe.num_samples_total,
            weighted_sampling_columns=mip.weighted_sampling_columns,
            modelling_data_format=mip.modelling_data_format,
        ),
        "input_genotype_config": _get_genotype_injections(
            input_source=mip.genotype_input_source,
            n_snps=n_snps,
            subset_snp_path=subset_snp_path,
        ),
        "fusion_config": {},
        "output_config": {},
    }

    for output_config in mip.output_configs:
        cat_cols = output_config["output_type_info"]["target_cat_columns"]
        con_cols = output_config["output_type_info"]["target_con_columns"]

        cur_config_name = output_config["output_info"]["output_name"]

        cur_injections = _get_output_injections(
            label_file_path=mip.label_file_path,
            output_cat_columns=cat_cols,
            output_con_columns=con_cols,
        )

        injections["output_config"][cur_config_name] = cur_injections

    if mip.input_cat_columns or mip.input_con_columns:
        injections["input_tabular_config"] = _get_tabular_injections(
            input_source=mip.label_file_path,
            input_cat_columns=list(mip.input_cat_columns),
            input_con_columns=list(mip.input_con_columns),
        )

    return injections


def get_num_iter_per_epoch(
    num_samples_per_epoch: int,
    num_samples_total: int,
    batch_size: int,
    valid_size: int,
) -> int:

    min_iter_per_epoch = 500
    if num_samples_total < 10_000:
        min_iter_per_epoch = 100

    iter_per_epoch = (num_samples_per_epoch - valid_size) // batch_size
    iter_per_epoch = max(min_iter_per_epoch, iter_per_epoch)

    logger.info(
        "Setting iter_per_epoch to %d with %d samples and %d valid samples.",
        iter_per_epoch,
        num_samples_per_epoch,
        valid_size,
    )

    return iter_per_epoch


def build_random_snp_subset_file(
    original_bim_path: Path,
    output_folder: Path,
    fold: int,
    fraction_per_chr: float = 0.1,
) -> tuple[int, Path]:
    df_bim = read_bim(bim_file_path=str(original_bim_path))

    grouped = df_bim.groupby("CHR_CODE")

    sampled_dfs = []
    for _, group in grouped:
        sample_size = int(len(group) * fraction_per_chr)
        sampled_dfs.append(
            group.sample(
                n=sample_size,
                random_state=fold,
                replace=False,
            )
        )

    df_sampled = pd.concat(sampled_dfs).sort_values(["CHR_CODE", "BP_COORD"])

    output_file = output_folder / f"random_subset_fold={fold}.txt"

    if output_file.exists():
        logger.info("%s already exists, using file.", output_file)
        n_snps = lines_in_file(file_path=output_file)
        return n_snps, output_file

    ensure_path_exists(path=output_file.parent, is_folder=True)

    df_sampled = df_sampled[["VAR_ID"]]

    df_sampled.to_csv(
        output_file,
        sep="\t",
        header=False,
        index=False,
    )

    logger.info(
        "Created random SNP subset file with %d SNPs: %s for fold %d",
        len(df_sampled),
        output_file,
        fold,
    )

    return len(df_sampled), output_file
