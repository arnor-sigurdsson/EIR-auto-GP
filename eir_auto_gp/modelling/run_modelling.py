import os
import subprocess
from dataclasses import fields, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Dict, Any, Optional, Iterable, Literal, Sequence

import luigi
import pandas as pd
import psutil
import torch
import yaml
from aislib.misc_utils import ensure_path_exists
from eir.setup.config import recursive_dict_replace

from eir_auto_gp.modelling.configs import get_aggregate_config, AggregateConfig
from eir_auto_gp.modelling.dl_feature_selection import get_genotype_subset_snps_file
from eir_auto_gp.modelling.gwas_feature_selection import run_gwas_feature_selection
from eir_auto_gp.preprocess.converge import ParseDataWrapper
from eir_auto_gp.utils.utils import get_logger
from eir_auto_gp.preprocess.converge import get_dynamic_valid_size, get_batch_size

logger = get_logger(name=__name__)


class RunModellingWrapper(luigi.Task):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        task_object = (
            TestSingleRun if self.modelling_config["do_test"] else TrainSingleRun
        )
        for fold in _get_fold_iterator(folds=str(self.folds)):
            yield task_object(
                fold=fold,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
            )

    def output(self):
        return self.input()


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
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        base = {
            "data": ParseDataWrapper(data_config=self.data_config),
            "train_run": TrainSingleRun(
                fold=self.fold,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
            ),
        }

        return base

    def run(self):
        output_root = Path(self.output().path).parent
        ensure_path_exists(output_root, is_folder=True)

        injection_params = build_injection_params(
            fold=int(self.fold),
            folder_with_runs=output_root.parent,
            genotype_data_path=self.data_config["genotype_data_path"],
            data_input_dict=self.input()["data"][0],
            task="test",
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
        )

        injections = _get_all_dynamic_injections(
            injection_params=injection_params,
            genotype_data_path=self.data_config["genotype_data_path"],
        )

        base_aggregate_config = get_aggregate_config()
        with TemporaryDirectory() as temp_dir:
            temp_config_folder = Path(temp_dir)
            build_configs(
                injections=injections,
                aggregate_config_base=base_aggregate_config,
                output_folder=temp_config_folder,
            )

            train_run_folder = Path(self.input()["train_run"].path).parent
            base_predict_command = get_testing_string_from_config_folder(
                config_folder=temp_config_folder, train_run_folder=train_run_folder
            )

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
    config_folder: Path, train_run_folder: Path
) -> str:
    base_string = "eirpredict"
    globals_string = " --global_configs "
    inputs_string = " --input_configs "
    output_string = " --output_configs "

    for file in config_folder.iterdir():
        if file.suffix == ".yaml":
            if "global" in file.stem:
                globals_string += " " + f"{str(file)}" + " "
            elif "input" in file.stem:
                inputs_string += " " + f"{str(file)}" + " "
            elif "output" in file.stem:
                output_string += " " + f"{str(file)}" + " "

    final_string = base_string + globals_string + inputs_string + output_string

    saved_models = list((train_run_folder / "saved_models").iterdir())
    assert len(saved_models) == 1, "Expected only one saved model."

    final_string += f" --model_path {saved_models[0]} --evaluate"

    test_output_folder = train_run_folder / "test_set_predictions"
    ensure_path_exists(path=test_output_folder, is_folder=True)

    final_string += f" --output_folder {test_output_folder}"

    return final_string


class TrainSingleRun(luigi.Task):
    fold = luigi.IntParameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()

    def requires(self):
        base = {"data": ParseDataWrapper(data_config=self.data_config)}
        for i in range(self.fold):
            base[f"train_{i}"] = TrainSingleRun(
                fold=i,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
            )
        return base

    def run(self):
        output_root = Path(self.output().path).parent
        ensure_path_exists(output_root, is_folder=True)

        injection_params = build_injection_params(
            fold=int(self.fold),
            folder_with_runs=output_root.parent,
            genotype_data_path=self.data_config["genotype_data_path"],
            data_input_dict=self.input()["data"][0],
            task="train",
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
        )

        injections = _get_all_dynamic_injections(
            injection_params=injection_params,
            genotype_data_path=self.data_config["genotype_data_path"],
        )

        base_aggregate_config = get_aggregate_config()
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
    output_folder: str
    manual_valid_ids_file: Optional[str]
    genotype_input_source: str
    genotype_subset_snps_file: Optional[str]
    label_file_path: str
    input_cat_columns: list[str]
    input_con_columns: list[str]
    output_cat_columns: list[str]
    output_con_columns: list[str]
    compute_attributions: bool
    weighted_sampling_columns: list[str]


def build_injection_params(
    fold: int,
    folder_with_runs: Path,
    genotype_data_path: str,
    data_input_dict: Dict[str, luigi.LocalTarget],
    task: Literal["train", "test"],
    data_config: Dict[str, Any],
    feature_selection_config: Dict[str, Any],
    modelling_config: Dict[str, Any],
) -> ModelInjectionParams:
    compute_attributions = False

    fs = feature_selection_config["feature_selection"]
    n_act_folds = feature_selection_config["n_dl_feature_selection_setup_folds"]
    if task == "train" and fs in ("dl", "gwas->dl") and fold < n_act_folds:
        compute_attributions = True

    weighted_sampling_columns = None
    if modelling_config["output_cat_columns"]:
        weighted_sampling_columns = ["all"]

    feature_selection_tasks = feature_selection_config["feature_selection"]

    gwas_manual_subset_file = None
    if feature_selection_tasks is not None:
        if "gwas" in feature_selection_tasks and task == "train":
            gwas_manual_subset_file = run_gwas_feature_selection(
                genotype_data_path=genotype_data_path,
                data_config=data_config,
                modelling_config=modelling_config,
                feature_selection_config=feature_selection_config,
            )
        elif "gwas" in feature_selection_tasks and task == "test":
            fs_output_folder = Path(
                feature_selection_config["feature_selection_output_folder"]
            )
            gwas_output_folder = fs_output_folder / "gwas_output"
            gwas_manual_subset_file = Path(gwas_output_folder, "snps_to_keep.txt")

    bim_file = _get_bim_path(genotype_data_path=genotype_data_path)
    snp_subset_file = get_genotype_subset_snps_file(
        fold=fold,
        folder_with_runs=folder_with_runs,
        feature_selection_approach=feature_selection_config["feature_selection"],
        feature_selection_output_folder=Path(
            feature_selection_config["feature_selection_output_folder"]
        ),
        bim_file=bim_file,
        n_dl_feature_selection_setup_folds=n_act_folds,
        manual_subset_from_gwas=gwas_manual_subset_file,
    )

    base_output_folder = modelling_config["modelling_output_folder"]
    cur_run_output_folder = f"{base_output_folder}/fold_{fold}"

    manual_valid_ids_file = None
    valid_ids_file = Path(data_config["data_output_folder"], "ids/valid_ids.txt")
    if task == "train" and valid_ids_file.exists():
        manual_valid_ids_file = str(valid_ids_file)

    params = ModelInjectionParams(
        fold=fold,
        output_folder=cur_run_output_folder,
        manual_valid_ids_file=manual_valid_ids_file,
        genotype_input_source=data_input_dict[f"{task}_genotype"].path,
        genotype_subset_snps_file=snp_subset_file,
        label_file_path=data_input_dict[f"{task}_tabular"].path,
        input_cat_columns=modelling_config["input_cat_columns"],
        input_con_columns=modelling_config["input_con_columns"],
        output_cat_columns=modelling_config["output_cat_columns"],
        output_con_columns=modelling_config["output_con_columns"],
        compute_attributions=compute_attributions,
        weighted_sampling_columns=weighted_sampling_columns,
    )

    return params


def build_configs(
    aggregate_config_base: AggregateConfig,
    injections: Dict[str, Any],
    output_folder: Path,
) -> None:
    for config_field in fields(aggregate_config_base):
        config_name = config_field.name
        config = getattr(aggregate_config_base, config_name)
        if config_name in injections:
            config = recursive_dict_replace(
                dict_=config, dict_to_inject=injections[config_name]
            )
        else:
            continue

        validate_complete_config(config_element=config)
        with open(output_folder / f"{config_name}.yaml", "w") as f:
            yaml.dump(config, f)


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
    output_string = " --output_configs "

    for file in config_folder.iterdir():
        if file.suffix == ".yaml" and "_test" not in file.stem:
            if "global" in file.stem:
                globals_string += " " + f"{str(file)}" + " "
            elif "input" in file.stem:
                inputs_string += " " + f"{str(file)}" + " "
            elif "output" in file.stem:
                output_string += " " + f"{str(file)}" + " "

    final_string = base_string + globals_string + inputs_string + output_string

    return final_string


def _get_global_injections(
    fold: int,
    output_folder: str,
    valid_size: int,
    batch_size: int,
    manual_valid_ids_file: Optional[str],
    n_snps: int,
    n_samples: int,
    compute_attributions: bool,
    iter_per_epoch: int,
    weighted_sampling_columns: list[str],
) -> Dict[str, Any]:
    mixing_candidates = [0.2]
    cur_mixing = mixing_candidates[fold % len(mixing_candidates)]

    device = _get_device()
    memory_dataset = _get_memory_dataset(n_snps=n_snps, n_samples=n_samples)
    n_workers = _get_dataloader_workers(memory_dataset=memory_dataset)
    early_stopping_buffer = min(5000, iter_per_epoch * 5)
    sample_interval = min(2000, iter_per_epoch)

    injections = {
        "output_folder": output_folder,
        "device": device,
        "batch_size": batch_size,
        "valid_size": valid_size,
        "manual_valid_ids_file": manual_valid_ids_file,
        "dataloader_workers": n_workers,
        "memory_dataset": memory_dataset,
        "mixing_alpha": cur_mixing,
        "sample_interval": sample_interval,
        "checkpoint_interval": sample_interval,
        "early_stopping_buffer": early_stopping_buffer,
        "compute_attributions": compute_attributions,
        "weighted_sampling_columns": weighted_sampling_columns,
    }

    return injections


def _get_memory_dataset(n_snps: int, n_samples: int) -> bool:
    available_memory = psutil.virtual_memory().available
    upper_bound = 0.6 * available_memory

    # 4 for one-hot encoding
    total_size = n_snps * n_samples * 4

    percent = total_size / available_memory
    if total_size < upper_bound:
        logger.info(
            "Estimated dataset size %.4f GB is %.4f%% of available memory %.4f GB, "
            "using memory dataset.",
            total_size / 1e9,
            percent * 100,
            available_memory / 1e9,
        )
        return True

    logger.info(
        "Estimated dataset size %.4f GB is %.4f%% of available memory %.4f GB, "
        "using disk dataset.",
        total_size / 1e9,
        percent * 100,
        available_memory / 1e9,
    )
    return False


def _get_device() -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.warning(
            "Using CPU as no CUDA device found, "
            "this might be much slower than using a CUDA device."
        )
    elif device == "cuda:0":
        logger.info("Using CUDA device 0 for modelling.")

    return device


def _get_dataloader_workers(memory_dataset: bool) -> int:
    if memory_dataset:
        return 0

    n_cores = os.cpu_count()
    n_workers = int(0.8 * n_cores / 2)

    if n_workers <= 2:
        n_workers = 0

    logger.info(
        "Using %d workers for data loading based on %d available cores.",
        n_workers,
        n_cores,
    )
    return n_workers


def _get_genotype_injections(
    input_source: str,
    genotype_use_snps_file: Optional[str],
) -> Dict[str, Any]:
    base_snp_path = (
        Path(input_source).parent.parent / "processed/parsed_files/data_final.bim"
    )
    assert base_snp_path.exists(), f"SNP file not found at {base_snp_path}"

    injections = {
        "input_info": {
            "input_source": input_source,
        },
        "input_type_info": {
            "snp_file": str(base_snp_path),
        },
    }

    if genotype_use_snps_file is not None:
        injections["input_type_info"]["subset_snps_file"] = str(genotype_use_snps_file)

    return injections


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
    label_file_path: str, output_cat_columns: list[str], output_con_columns: list[str]
) -> Dict[str, Any]:
    injections = {
        "output_info": {
            "output_source": label_file_path,
        },
        "output_type_info": {
            "target_cat_columns": output_cat_columns,
            "target_con_columns": output_con_columns,
        },
    }

    return injections


def _get_all_dynamic_injections(
    injection_params: ModelInjectionParams, genotype_data_path: str
) -> Dict[str, Any]:
    mip = injection_params

    samples_per_epoch = get_samples_per_epoch(model_injection_params=mip)

    batch_size = get_batch_size(samples_per_epoch=samples_per_epoch)

    valid_size = get_dynamic_valid_size(
        num_samples_per_epoch=samples_per_epoch, batch_size=batch_size
    )
    iter_per_epoch = get_num_iter_per_epoch(
        num_samples_per_epoch=samples_per_epoch,
        batch_size=batch_size,
        valid_size=valid_size,
    )
    bim_path = _get_bim_path(genotype_data_path=genotype_data_path)

    n_snps = _lines_in_file(file_path=bim_path)
    if mip.genotype_subset_snps_file is not None:
        n_snps = _lines_in_file(file_path=mip.genotype_subset_snps_file)

    n_samples = _lines_in_file(file_path=mip.label_file_path) - 1

    injections = {
        "global_config": _get_global_injections(
            fold=mip.fold,
            output_folder=mip.output_folder,
            batch_size=batch_size,
            manual_valid_ids_file=mip.manual_valid_ids_file,
            valid_size=valid_size,
            iter_per_epoch=iter_per_epoch,
            n_snps=n_snps,
            n_samples=n_samples,
            compute_attributions=mip.compute_attributions,
            weighted_sampling_columns=mip.weighted_sampling_columns,
        ),
        "input_genotype_config": _get_genotype_injections(
            input_source=mip.genotype_input_source,
            genotype_use_snps_file=mip.genotype_subset_snps_file,
        ),
        "output_config": _get_output_injections(
            label_file_path=mip.label_file_path,
            output_cat_columns=list(mip.output_cat_columns),
            output_con_columns=list(mip.output_con_columns),
        ),
    }

    if mip.input_cat_columns or mip.input_con_columns:
        injections["input_tabular_config"] = _get_tabular_injections(
            input_source=mip.label_file_path,
            input_cat_columns=list(mip.input_cat_columns),
            input_con_columns=list(mip.input_con_columns),
        )

    return injections


@lru_cache()
def _lines_in_file(file_path: str | Path) -> int:
    with open(file_path, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def _get_bim_path(genotype_data_path: str) -> str:
    bim_files = [i for i in Path(genotype_data_path).glob("*.bim")]
    assert len(bim_files) == 1, bim_files

    path = bim_files[0]
    assert path.exists(), f".bim file not found at {path}"
    return str(path)


def get_samples_per_epoch(model_injection_params: ModelInjectionParams) -> int:
    mip = model_injection_params

    if not mip.weighted_sampling_columns:
        num_samples = _lines_in_file(file_path=mip.label_file_path) - 1
        return num_samples

    logger.info(
        "Setting up weighted sampling for categorical output columns: %s.",
        mip.output_cat_columns,
    )
    label_counts = get_column_label_counts(
        label_file_path=mip.label_file_path, output_cat_columns=mip.output_cat_columns
    )

    mean_per_target = (min(i.values()) for i in label_counts.values())
    mean_all_outputs = int(mean(mean_per_target))

    return mean_all_outputs


def get_column_label_counts(
    label_file_path: str | Path, output_cat_columns: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    columns = ["ID"] + list(output_cat_columns)
    df = pd.read_csv(label_file_path, index_col=["ID"], usecols=columns)

    label_counts = {}

    for col in output_cat_columns:
        label_counts[col] = df[col].value_counts().to_dict()

    return label_counts


def get_num_iter_per_epoch(
    num_samples_per_epoch: int, batch_size: int, valid_size: int
) -> int:
    iter_per_epoch = (num_samples_per_epoch - valid_size) // batch_size
    iter_per_epoch = max(50, iter_per_epoch)
    return iter_per_epoch


if __name__ == "__main__":
    luigi.build([RunModellingWrapper()], local_scheduler=True)
