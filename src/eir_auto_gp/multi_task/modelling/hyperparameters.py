import math
from pathlib import Path

import pandas as pd
import torch
from aislib.misc_utils import ensure_path_exists
from eir.setup.input_setup_modules.setup_omics import read_bim

from eir_auto_gp.single_task.modelling.run_modelling import lines_in_file
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def _get_resolved_precision(mixed_precision: bool) -> str:
    if not mixed_precision:
        logger.info("Mixed precision disabled. Using 32-true precision.")
        return "32-true"

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("Hardware supports bf16-mixed on CUDA.")
        return "bf16-mixed"

    logger.info("bf16-mixed not supported or optimal on this hardware. Using 32-true.")
    return "32-true"


def _get_resolved_compile_model(compile_model: bool) -> bool:
    if not compile_model:
        logger.info("torch.compile disabled.")
        return False

    if torch.backends.mps.is_available():
        logger.info("Disabling torch.compile on MPS (experimental/unstable support).")
        return False

    if torch.cuda.is_available():
        logger.info("Enabling torch.compile for CUDA.")
        return True

    logger.info("torch.compile disabled on CPU (experimental for training).")
    return False


def _get_checkpoint_interval(
    iter_per_epoch: int,
    evaluations_per_epoch: int = 4,
    min_interval: int = 50,
    max_interval: int = 1000,
) -> int:
    if iter_per_epoch <= min_interval:
        return iter_per_epoch

    target_interval = iter_per_epoch / evaluations_per_epoch
    if target_interval <= min_interval:
        return min_interval

    power = 10 ** math.floor(math.log10(target_interval))
    rounding_base = power / 2
    nice_interval = round(target_interval / rounding_base) * rounding_base

    final_interval = max(min_interval, int(nice_interval))
    final_interval = min(final_interval, iter_per_epoch, max_interval)

    return final_interval


def _get_learning_rate(n_snps: int, batch_size: int) -> float:
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

    base_bs = 64
    max_lr = 1e-03
    min_lr = 1e-05

    lr = lr * (batch_size / base_bs)
    lr = max(min_lr, min(max_lr, lr))

    logger.info(
        "Setting learning rate to %f (n_snps=%d, batch_size=%d).",
        lr,
        n_snps,
        batch_size,
    )

    return lr


def get_gln_kernel_parameters(n_snps: int) -> tuple[int, int]:
    if n_snps < 1_000:
        params = 12, -4
    elif n_snps < 10_000:
        params = 12, -2
    elif n_snps < 100_000:
        params = 12, 1
    elif n_snps < 500_000:
        params = 12, 2
    elif n_snps < 2_000_000:
        params = 12, 4
    else:
        params = 12, 8

    logger.info(
        "Setting kernel width to %d and first kernel expansion to %d due to %d SNPs.",
        params[0],
        params[1],
        n_snps,
    )

    return params


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
