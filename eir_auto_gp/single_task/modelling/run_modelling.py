import os
import re
import subprocess
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, Literal, Optional, Sequence

import luigi
import pandas as pd
import polars as pl
import psutil
import torch
import yaml
from aislib.misc_utils import ensure_path_exists
from eir.setup.config_setup_modules.config_setup_utils import recursive_dict_inject

from eir_auto_gp.preprocess.converge import (
    ParseDataWrapper,
    get_batch_size,
    get_dynamic_valid_size,
)
from eir_auto_gp.single_task.modelling.configs import (
    AggregateConfig,
    get_aggregate_config,
)
from eir_auto_gp.single_task.modelling.feature_selection import (
    get_genotype_subset_snps_file,
)
from eir_auto_gp.single_task.modelling.gwas_feature_selection import (
    run_gwas_feature_selection,
)
from eir_auto_gp.utils.utils import get_logger

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
    config_folder: Path,
    train_run_folder: Path,
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
    compute_attributions = should_compute_attributions(
        task=task,
        feature_selection_config=feature_selection_config,
        fold=fold,
    )
    weighted_sampling_columns = get_weighted_sampling_columns(
        modelling_config=modelling_config
    )
    gwas_manual_subset_file = get_gwas_manual_subset_file(
        task=task,
        feature_selection_config=feature_selection_config,
        genotype_data_path=genotype_data_path,
        data_config=data_config,
        modelling_config=modelling_config,
    )

    bim_file = get_bim_path(genotype_data_path=genotype_data_path)
    n_act_folds = feature_selection_config["n_dl_feature_selection_setup_folds"]
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
        gwas_p_value_threshold=feature_selection_config["gwas_p_value_threshold"],
    )

    base_output_folder = modelling_config["modelling_output_folder"]
    cur_run_output_folder = f"{base_output_folder}/fold_{fold}"

    label_file_path = build_tmp_label_file(
        label_file_path=data_input_dict[f"{task}_tabular"].path,
        output_cat_columns=modelling_config["output_cat_columns"],
        tmp_dir=Path(base_output_folder, "tmp"),
        output_con_columns=modelling_config["output_con_columns"],
        prefix_name=task,
    )

    manual_valid_ids_file = get_manual_valid_ids_file(
        task=task,
        data_config=data_config,
        label_file_path=label_file_path,
        base_output_folder=base_output_folder,
    )

    params = ModelInjectionParams(
        fold=fold,
        output_folder=cur_run_output_folder,
        manual_valid_ids_file=str(manual_valid_ids_file),
        genotype_input_source=data_input_dict[f"{task}_genotype"].path,
        genotype_subset_snps_file=snp_subset_file,
        label_file_path=label_file_path,
        input_cat_columns=modelling_config["input_cat_columns"],
        input_con_columns=modelling_config["input_con_columns"],
        output_cat_columns=modelling_config["output_cat_columns"],
        output_con_columns=modelling_config["output_con_columns"],
        compute_attributions=compute_attributions,
        weighted_sampling_columns=weighted_sampling_columns,
    )

    return params


def should_compute_attributions(
    task: str, feature_selection_config: Dict[str, Any], fold: int
) -> bool:
    fs = feature_selection_config["feature_selection"]
    n_act_folds = feature_selection_config["n_dl_feature_selection_setup_folds"]
    return (
        task == "train" and fs in ("dl", "gwas->dl", "dl+gwas") and fold < n_act_folds
    )


def get_weighted_sampling_columns(
    modelling_config: Dict[str, Any]
) -> Optional[list[str]]:
    return ["all"] if modelling_config["output_cat_columns"] else None


def get_gwas_manual_subset_file(
    task: str,
    feature_selection_config: Dict[str, Any],
    genotype_data_path: str,
    data_config: Dict[str, Any],
    modelling_config: Dict[str, Any],
) -> Optional[Path]:
    feature_selection_tasks = feature_selection_config["feature_selection"]
    if feature_selection_tasks is None:
        return None

    if "gwas" in feature_selection_tasks:
        if task == "train":
            gwas_snps_to_keep_path = run_gwas_feature_selection(
                genotype_data_path=genotype_data_path,
                data_config=data_config,
                modelling_config=modelling_config,
                feature_selection_config=feature_selection_config,
            )
            return gwas_snps_to_keep_path
        elif task == "test":
            fs_output_folder = Path(
                feature_selection_config["feature_selection_output_folder"]
            )
            gwas_output_folder = fs_output_folder / "gwas_output"
            return Path(gwas_output_folder, "snps_to_keep.txt")

    return None


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

    if output_cat_columns:
        df = df.dropna(subset=output_cat_columns)
    if output_con_columns:
        df = df.dropna(subset=output_con_columns)

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
            config = recursive_dict_inject(
                dict_=config,
                dict_to_inject=injections[config_name],
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
    mixing_candidates = [0.0]
    cur_mixing = mixing_candidates[fold % len(mixing_candidates)]

    device = get_device()
    memory_dataset = get_memory_dataset(n_snps=n_snps, n_samples=n_samples)
    n_workers = get_dataloader_workers(memory_dataset=memory_dataset, device=device)
    early_stopping_buffer = min(5000, iter_per_epoch * 5)
    early_stopping_buffer = max(early_stopping_buffer, 1000)
    sample_interval = min(2000, iter_per_epoch)

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
        "evaluation_checkpoint": {
            "sample_interval": sample_interval,
            "checkpoint_interval": sample_interval,
        },
        "training_control": {
            "mixing_alpha": cur_mixing,
            "early_stopping_buffer": early_stopping_buffer,
            "weighted_sampling_columns": weighted_sampling_columns,
        },
        "attribution_analysis": {
            "compute_attributions": compute_attributions,
        },
    }

    return injections


def _maybe_get_slurm_job_memory() -> Optional[int]:
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        logger.info("Running in a SLURM environment. Using SLURM job memory.")
        try:
            output = subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "job",
                    job_id,
                ]
            ).decode("utf-8")
            match = re.search(r"mem=(\d+)([MG])", output)
            if match:
                mem_value, unit = match.groups()
                mem_value = int(mem_value)
                if unit == "G":
                    return int(mem_value * 1e9)
                elif unit == "M":
                    return int(mem_value * 1e6)
        except Exception as e:
            logger.error(
                f"Could not fetch SLURM job memory: {e}. Assuming non-SLURM job."
            )
    else:
        logger.info(
            "Not running in a SLURM environment or "
            "SLURM_JOB_ID not set. Using system's available memory."
        )

    return None


def get_memory_dataset(n_snps: int, n_samples: int) -> bool:
    slurm_memory = _maybe_get_slurm_job_memory()
    available_memory = (
        slurm_memory if slurm_memory is not None else psutil.virtual_memory().available
    )
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


def get_device() -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        logger.warning(
            "Using CPU as no CUDA device found, "
            "this might be much slower than using a CUDA device."
        )
    elif device == "cuda:0":
        logger.info("Using CUDA device 0 for modelling.")

    return device


def _maybe_get_slurm_job_cores() -> Optional[int]:
    job_id = os.getenv("SLURM_JOB_ID")
    if job_id:
        logger.info("Running in a SLURM environment. Using SLURM job core count.")
        try:
            output = subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "job",
                    job_id,
                ]
            ).decode("utf-8")
            match = re.search(r"NumCPUs=(\d+)", output)
            if match:
                return int(match.group(1))
        except Exception as e:
            logger.info(
                f"Could not fetch SLURM job core count: {e}. "
                f"Assuming non-SLURM environment."
            )
    else:
        logger.info(
            "Not running in a SLURM environment or SLURM_JOB_ID not set. "
            "Using system's CPU count."
        )

    return None


def get_dataloader_workers(memory_dataset: bool, device: str) -> int:
    if memory_dataset:
        logger.info(
            "Dataset is loaded into memory; "
            "using 0 workers to avoid unnecessary multiprocessing overhead."
        )
        return 0

    slurm_cores = _maybe_get_slurm_job_cores()
    n_cores = slurm_cores if slurm_cores is not None else os.cpu_count() or 1

    if device == "cpu":
        n_workers = int(0.8 * n_cores / 2)
    else:
        n_workers = int(0.8 * n_cores)

    if n_workers <= 2:
        logger.info(
            "Based on available cores, "
            "fewer than 2 workers were calculated; "
            "setting workers to 0 to avoid overhead."
        )
        n_workers = 0
    else:
        n_workers = min(12, n_workers)

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

    spe = get_samples_per_epoch(model_injection_params=mip)

    batch_size = get_batch_size(samples_per_epoch=spe.samples_per_epoch)

    valid_size = get_dynamic_valid_size(
        num_samples_per_epoch=spe.samples_per_epoch,
        minimum=batch_size,
    )
    iter_per_epoch = get_num_iter_per_epoch(
        num_samples_per_epoch=spe.samples_per_epoch,
        batch_size=batch_size,
        valid_size=valid_size,
    )
    bim_path = get_bim_path(genotype_data_path=genotype_data_path)

    n_snps = lines_in_file(file_path=bim_path)
    if mip.genotype_subset_snps_file is not None:
        n_snps = lines_in_file(file_path=mip.genotype_subset_snps_file)

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
def lines_in_file(file_path: str | Path) -> int:
    with open(file_path, "r") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


def get_bim_path(genotype_data_path: str) -> str:
    bim_files = [i for i in Path(genotype_data_path).glob("*.bim")]
    assert len(bim_files) == 1, bim_files

    path = bim_files[0]
    assert path.exists(), f".bim file not found at {path}"
    return str(path)


@dataclass()
class SampleEpochInfo:
    num_samples_total: int
    samples_per_epoch: int


def format_column_list(columns: Sequence[str], max_show: int = 10) -> str:
    if len(columns) <= max_show:
        return str(list(columns))

    return f"[{', '.join(repr(col) for col in columns[:max_show])}, ...]"


def get_valid_sample_count(
    label_file_path: str | Path,
    output_cat_columns: Sequence[str],
    output_con_columns: Sequence[str],
    *,
    id_column: str = "ID",
) -> int:
    output_columns = list(output_cat_columns) + list(output_con_columns)
    columns_to_read = [id_column] + output_columns

    formatted_columns = format_column_list(columns=columns_to_read)
    logger.info(
        "Reading %s with columns: %s",
        Path(label_file_path).name,
        formatted_columns,
    )

    df = pl.scan_csv(source=label_file_path).select(columns_to_read)
    is_nan_exprs = [pl.col(col).is_null() for col in output_columns]

    valid_samples = (
        df.filter(~pl.fold(True, lambda acc, x: acc & x, is_nan_exprs)).collect().height
    )

    logger.info(
        "Found %d valid samples in %s with output columns: %s",
        valid_samples,
        Path(label_file_path).name,
        format_column_list(columns=output_columns),
    )

    return valid_samples


def get_samples_per_epoch(
    model_injection_params: ModelInjectionParams,
) -> SampleEpochInfo:
    mip = model_injection_params

    num_samples = get_valid_sample_count(
        label_file_path=mip.label_file_path,
        output_cat_columns=mip.output_cat_columns,
        output_con_columns=mip.output_con_columns,
    )

    if not mip.weighted_sampling_columns:
        return SampleEpochInfo(
            num_samples_total=num_samples,
            samples_per_epoch=num_samples,
        )

    logger.info(
        "Setting up weighted sampling for categorical output columns: %s.",
        mip.output_cat_columns,
    )
    label_counts = get_column_label_counts(
        label_file_path=mip.label_file_path,
        output_cat_columns=mip.output_cat_columns,
    )

    mean_per_target = (min(i.values()) for i in label_counts.values())
    mean_all_outputs = int(mean(mean_per_target))

    return SampleEpochInfo(
        num_samples_total=num_samples,
        samples_per_epoch=mean_all_outputs,
    )


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
