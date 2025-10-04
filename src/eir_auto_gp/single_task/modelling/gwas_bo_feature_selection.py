from pathlib import Path

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists
from skopt import Optimizer

from eir_auto_gp.single_task.modelling.feature_selection_utils import (
    gather_fractions_and_performances,
    read_gwas_df,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)

SMALLEST_P_VAL = 1e-300


def calculate_dynamic_bounds(
    df_gwas: pd.DataFrame,
    gwas_p_value_threshold: float | None,
    min_n_snps: int,
) -> tuple[float, float]:
    df_gwas_safe = df_gwas.copy()

    df_gwas_safe["GWAS P-VALUE"] = df_gwas_safe["GWAS P-VALUE"].replace(
        0.0, SMALLEST_P_VAL
    )
    df_gwas_safe["GWAS P-VALUE"] = np.maximum(
        df_gwas_safe["GWAS P-VALUE"], SMALLEST_P_VAL
    )

    if gwas_p_value_threshold:
        max_log_p = np.log10(gwas_p_value_threshold)
    else:
        max_log_p = np.log10(df_gwas_safe["GWAS P-VALUE"].quantile(0.5))

    n_snps = len(df_gwas_safe)
    if n_snps > min_n_snps:
        min_p_val_cutoff = df_gwas_safe.nsmallest(min_n_snps, "GWAS P-VALUE")[
            "GWAS P-VALUE"
        ].iloc[-1]
        min_log_p = np.log10(min_p_val_cutoff)
    else:
        min_log_p = np.log10(df_gwas_safe["GWAS P-VALUE"].min())

    min_log_p = max(min_log_p, -8.0)

    if min_log_p >= max_log_p:
        logger.warning(
            "Invalid search space bounds: min_log_p=%.2f >= max_log_p=%.2f. "
            "Using fallback range [-8.0, -3.0].",
            min_log_p,
            max_log_p,
        )
        return -8.0, -3.0

    logger.debug(
        "Dynamic bounds calculated: log10(p-value) in [%.2f, %.2f]",
        min_log_p,
        max_log_p,
    )
    return min_log_p, max_log_p


def fraction_to_log_p_value(fraction: float, df_gwas: pd.DataFrame) -> float:
    if fraction <= 0.0:
        return -8.0

    df_sorted = df_gwas.sort_values("GWAS P-VALUE")
    df_sorted = df_sorted.copy()
    df_sorted["GWAS P-VALUE"] = df_sorted["GWAS P-VALUE"].replace(
        0.0, np.finfo(float).tiny
    )

    n_snps = len(df_sorted)
    index = min(max(0, int(fraction * n_snps) - 1), n_snps - 1)

    if index < 0:
        return -8.0

    p_value = df_sorted.iloc[index]["GWAS P-VALUE"]
    return np.log10(p_value)


def run_gwas_bo_feature_selection(
    fold: int,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    gwas_output_folder: Path | None,
    gwas_p_value_threshold: float | None,
) -> Path | None:
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
    gwas_p_value_threshold: float | None,
    min_n_snps: int = 16,
) -> tuple[int, float]:
    n_snps = len(df_gwas)

    min_log_p, max_log_p = calculate_dynamic_bounds(
        df_gwas=df_gwas,
        gwas_p_value_threshold=gwas_p_value_threshold,
        min_n_snps=min_n_snps,
    )

    df_history = gather_fractions_and_performances(
        folder_with_runs=folder_with_runs,
        feature_selection_output_folder=feature_selection_output_folder,
    )

    num_priming_points = 4
    manual_priming_points = np.linspace(min_log_p, max_log_p, num_priming_points)

    if len(df_history) < num_priming_points:
        next_log_p_value = manual_priming_points[len(df_history)]
        logger.info(
            "Suggesting manual priming point %d/%d: log10(p-value)=%.2f",
            len(df_history) + 1,
            num_priming_points,
            next_log_p_value,
        )
    else:
        logger.info("Manual priming complete. Using BO for next suggestion.")
        n_initial_points = max(5, min(10, int((max_log_p - min_log_p) * 2)))

        opt = Optimizer(
            dimensions=[(min_log_p, max_log_p, "uniform")],
            n_initial_points=n_initial_points,
            initial_point_generator="sobol",
        )

        points_loaded = 0
        points_skipped = 0

        for _, row in df_history.iterrows():
            log_p_value = fraction_to_log_p_value(row["fraction"], df_gwas)
            if min_log_p <= log_p_value <= max_log_p:
                negated_performance = -row["best_val_performance"]
                opt.tell([log_p_value], negated_performance)
                points_loaded += 1
            else:
                points_skipped += 1

        if points_skipped > 0:
            logger.debug(
                "Loaded %d historical points, skipped %d out-of-bounds points",
                points_loaded,
                points_skipped,
            )

        next_log_p_value = opt.ask()[0]

    next_p_value_threshold = 10**next_log_p_value

    logger.info(
        "Next suggestion: log10(p-value)=%.2f (p-value=%.2e)",
        next_log_p_value,
        next_p_value_threshold,
    )

    top_n = len(df_gwas[df_gwas["GWAS P-VALUE"] < next_p_value_threshold])

    if top_n < min_n_snps:
        if n_snps >= min_n_snps:
            top_n = min_n_snps
            logger.info(
                "Computed top_n for GWAS+BO %d is too small (< %d). Setting to %d.",
                top_n,
                min_n_snps,
                min_n_snps,
            )
        else:
            top_n = n_snps
            logger.info(
                "Dataset contains only %d SNPs, less than %d. Using all %d SNPs.",
                n_snps,
                min_n_snps,
                top_n,
            )

    final_fraction = top_n / n_snps
    logger.info("Selected %d SNPs (fraction: %.4f)", top_n, final_fraction)
    return top_n, final_fraction


def get_gwas_top_n_snp_list_df(df_gwas: pd.DataFrame, top_n_snps: int) -> pd.DataFrame:
    df = df_gwas.sort_values(by="GWAS P-VALUE", ascending=True)
    df_top_n = df.iloc[:top_n_snps, :].copy()
    df_top_n.index.name = "SNP"
    df_top_n["SNP"] = df_top_n.index
    return df_top_n
