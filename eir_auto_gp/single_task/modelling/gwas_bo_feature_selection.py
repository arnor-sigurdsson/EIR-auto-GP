from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from aislib.misc_utils import ensure_path_exists
from skopt import Optimizer

from eir_auto_gp.single_task.modelling.feature_selection_utils import (
    gather_fractions_and_performances,
    read_gwas_df,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def run_gwas_bo_feature_selection(
    fold: int,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    gwas_output_folder: Optional[Path],
    gwas_p_value_threshold: Optional[float],
) -> Optional[Path]:
    fs_out_folder = feature_selection_output_folder
    subsets_out_folder = fs_out_folder / "snp_importance" / "snp_subsets"
    snp_subset_file = subsets_out_folder / f"chosen_snps_{fold}.txt"

    if snp_subset_file.exists():
        return snp_subset_file

    fractions_file = subsets_out_folder / f"chosen_snps_fraction_{fold}.txt"

    assert gwas_output_folder is not None
    df_gwas = read_gwas_df(gwas_output_folder=gwas_output_folder)
    df_gwas = df_gwas.rename(columns={"P": "GWAS P-VALUE"})
    df_gwas = df_gwas[["GWAS P-VALUE"]]

    top_n, fraction = get_gwas_bo_auto_top_n(
        df_gwas=df_gwas,
        folder_with_runs=folder_with_runs,
        feature_selection_output_folder=feature_selection_output_folder,
        fold=fold,
        gwas_p_value_threshold=gwas_p_value_threshold,
    )
    logger.info("Top %d SNPs selected.", top_n)

    df_top_n = get_gwas_top_n_snp_list_df(df_gwas=df_gwas, top_n_snps=top_n)
    df_top_n_snps_only = df_top_n[["SNP"]]
    ensure_path_exists(path=snp_subset_file)
    df_top_n_snps_only.to_csv(path_or_buf=snp_subset_file, index=False, header=False)
    fractions_file.write_text(str(fraction))

    return snp_subset_file


def get_gwas_bo_auto_top_n(
    df_gwas: pd.DataFrame,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    fold: int,
    gwas_p_value_threshold: Optional[float],
    min_n_snps: int = 16,
) -> Tuple[int, float]:
    max_fraction = _compute_max_fraction(
        df_gwas=df_gwas,
        gwas_p_value_threshold=gwas_p_value_threshold,
    )

    manual_fractions, manual_p_values = _get_manual_gwas_bo_fractions(
        df_gwas=df_gwas,
        min_snps_cutoff=1,
        max_fraction=max_fraction,
    )

    n_snps = len(df_gwas)

    if fold < len(manual_fractions):
        next_fraction = manual_fractions[fold]
        logger.info(
            "Next manual fraction for GWAS+BO: %f (p-value: %.2e)",
            next_fraction,
            manual_p_values[fold],
        )
    else:
        threshold_snps = min(min_n_snps, n_snps)
        min_fraction = threshold_snps / n_snps
        logger.debug("Setting minimum fraction to %.2e.", min_fraction)

        opt = Optimizer(
            dimensions=[(min_fraction, max_fraction, "log-uniform")],
            n_initial_points=len(manual_fractions),
        )
        df_history = gather_fractions_and_performances(
            folder_with_runs=folder_with_runs,
            feature_selection_output_folder=feature_selection_output_folder,
        )

        for t in df_history.itertuples():
            negated_performance = -t.best_val_performance
            opt.tell([t.fraction], negated_performance)

        next_fraction = opt.ask()[0]
        logger.info("Next computed fraction for GWAS+BO: %f", next_fraction)

    top_n = int(next_fraction * len(df_gwas))

    if top_n < min_n_snps:
        if n_snps >= min_n_snps:
            top_n = min_n_snps
            logger.info(
                "Computed top_n for GWAS+BO %d is too small (< %d). Setting to 16.",
                min_n_snps,
                top_n,
            )
        else:
            top_n = n_snps
            logger.info(
                "Dataset contains only %d SNPs, less than %d. Using all %d SNPs.",
                n_snps,
                min_n_snps,
                top_n,
            )

        next_fraction = top_n / n_snps

    return top_n, next_fraction


def _compute_max_fraction(
    df_gwas: pd.DataFrame, gwas_p_value_threshold: Optional[float]
) -> float:
    if gwas_p_value_threshold is None:
        return 1.0

    df_subset = df_gwas[df_gwas["GWAS P-VALUE"] < gwas_p_value_threshold].copy()
    n_snps = len(df_subset)
    fraction = n_snps / len(df_gwas)

    logger.info(
        "Computed max fraction of SNPs with p-value < %f: %f for GWAS+BO.",
        gwas_p_value_threshold,
        fraction,
    )

    assert fraction > 0.0
    return fraction


def _get_manual_gwas_bo_fractions(
    df_gwas: pd.DataFrame,
    min_snps_cutoff: int,
    max_fraction: float,
) -> tuple[list[float], list[float]]:
    p_values = []
    fractions = []
    for p in range(8, 2, -1):
        p_value = 10**-p
        df_subset = df_gwas[df_gwas["GWAS P-VALUE"] < p_value]
        n_snps = len(df_subset)

        if n_snps < min_snps_cutoff:
            logger.info(
                "Skipping p-value for GWAS+BO %f due to %d SNPs being too few (<%d).",
                p_value,
                n_snps,
                min_snps_cutoff,
            )
            continue

        fraction = n_snps / len(df_gwas)
        if fraction > max_fraction:
            logger.info(
                "Skipping p-value for GWAS+BO %f due to %d SNPs being too many (> %f).",
                p_value,
                n_snps,
                max_fraction,
            )
            continue

        fractions.append(fraction)
        p_values.append(p_value)

    return fractions, p_values


def get_gwas_top_n_snp_list_df(df_gwas: pd.DataFrame, top_n_snps: int) -> pd.DataFrame:
    df = df_gwas.sort_values(by="GWAS P-VALUE", ascending=True)
    df_top_n = df.iloc[:top_n_snps, :].copy()
    df_top_n.index.name = "SNP"
    df_top_n["SNP"] = df_top_n.index
    return df_top_n
