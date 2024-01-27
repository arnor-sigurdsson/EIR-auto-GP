from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from aislib.misc_utils import ensure_path_exists
from skopt import Optimizer

from eir_auto_gp.modelling.feature_selection_utils import (
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
) -> Optional[Path]:
    fs_out_folder = feature_selection_output_folder
    subsets_out_folder = fs_out_folder / "dl_importance" / "snp_subsets"
    snp_subset_file = subsets_out_folder / f"dl_snps_{fold}.txt"

    if snp_subset_file.exists():
        return snp_subset_file

    fractions_file = subsets_out_folder / f"dl_snps_fraction_{fold}.txt"

    assert gwas_output_folder is not None
    df_gwas = read_gwas_df(gwas_output_folder=gwas_output_folder)
    df_gwas = df_gwas.rename(columns={"P": "GWAS P-VALUE"})
    df_gwas = df_gwas[["GWAS P-VALUE"]]

    top_n, fraction = get_gwas_bo_auto_top_n(
        df_gwas=df_gwas,
        folder_with_runs=folder_with_runs,
        feature_selection_output_folder=feature_selection_output_folder,
        fold=fold,
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
) -> Tuple[int, float]:
    manual_fractions = _get_manual_gwas_bo_fractions(
        df_gwas=df_gwas,
        min_snps_cutoff=1,
    )

    if fold < len(manual_fractions):
        next_fraction = manual_fractions[fold]
    else:
        opt = Optimizer(dimensions=[(0.0, 1.0)])
        df_history = gather_fractions_and_performances(
            folder_with_runs=folder_with_runs,
            feature_selection_output_folder=feature_selection_output_folder,
        )

        for t in df_history.itertuples():
            negated_performance = -t.best_val_performance
            opt.tell([t.fraction], negated_performance)

        next_fraction = opt.ask()[0]
        logger.info("Next fraction: %f", next_fraction)

    top_n = int(next_fraction * len(df_gwas))
    if top_n < 16:
        top_n = 16
        next_fraction = top_n / len(df_gwas)

    return top_n, next_fraction


def _get_manual_gwas_bo_fractions(
    df_gwas: pd.DataFrame,
    min_snps_cutoff: int,
) -> list[float]:
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
        fractions.append(fraction)

    return fractions


def get_gwas_top_n_snp_list_df(df_gwas: pd.DataFrame, top_n_snps: int) -> pd.DataFrame:
    df = df_gwas.sort_values(by="GWAS P-VALUE", ascending=True)
    df_top_n = df.iloc[:top_n_snps, :]
    df_top_n.index.name = "SNP"
    df_top_n["SNP"] = df_top_n.index
    return df_top_n
