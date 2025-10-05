import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Union

import pandas as pd
import polars as pl
import psutil
import torch

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)

if TYPE_CHECKING:
    from eir_auto_gp.multi_task.modelling.run_modelling import (
        MultiTaskModelInjectionParams,
    )
    from eir_auto_gp.single_task.modelling.run_modelling import (
        SingleTaskModelInjectionParams,
    )


def _maybe_get_slurm_job_memory() -> int | None:
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
    upper_bound = 0.5 * available_memory

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


def _maybe_get_slurm_job_cores() -> int | None:
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


def get_bim_path(genotype_data_path: str) -> str:
    bim_files = list(Path(genotype_data_path).glob("*.bim"))
    assert len(bim_files) == 1, bim_files

    path = bim_files[0]
    assert path.exists(), f".bim file not found at {path}"
    return str(path)


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


@dataclass()
class SampleEpochInfo:
    num_samples_total: int
    samples_per_epoch: int


def get_samples_per_epoch(
    model_injection_params: Union[
        "SingleTaskModelInjectionParams", "MultiTaskModelInjectionParams"
    ],
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
) -> dict[str, dict[str, int]]:
    columns = ["ID"] + list(output_cat_columns)
    df = pd.read_csv(label_file_path, index_col=["ID"], usecols=columns)

    label_counts = {}

    for col in output_cat_columns:
        label_counts[col] = df[col].value_counts().to_dict()

    return label_counts
