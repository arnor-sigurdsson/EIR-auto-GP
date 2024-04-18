from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger
from matplotlib import pyplot as plt

logger = get_logger(name=__name__)


def get_snp_allele_maps(
    df_acts: pd.DataFrame, rs_ids: Iterable[str]
) -> Dict[str, Dict[str, str]]:
    snp_allele_maps = {}
    for rs_id in rs_ids:
        snp_allele_maps[rs_id] = get_snp_allele_nucleotide_map(
            df_acts=df_acts, rs_id=rs_id
        )

    return snp_allele_maps


def get_snp_allele_nucleotide_map(df_acts: pd.DataFrame, rs_id: str) -> Dict[str, str]:
    nucleotide_map = {}

    ref = df_acts[df_acts["VAR_ID"] == rs_id]["REF"].item()
    alt = df_acts[df_acts["VAR_ID"] == rs_id]["ALT"].item()

    nucleotide_map["REF"] = ref * 2
    nucleotide_map["HET"] = f"{ref}{alt}"
    nucleotide_map["ALT"] = alt * 2

    return nucleotide_map


def plot_top_snps(
    df: pd.DataFrame,
    p_value_threshold: float,
    top_n: int,
    output_dir: Path,
) -> None:
    ensure_path_exists(path=output_dir, is_folder=True)

    df_allele_effects = _parse_df_allele_effects(df_allele_effects=df)
    top_snps = _get_snps_with_strongest_effects(
        df_allele_effects=df_allele_effects,
        top_n=top_n,
    )

    for snp in top_snps:
        df_snp = df.loc[snp]

        df_total_effect = _compute_total_effect(df=df_snp)

        if np.any(df_total_effect["p_value"] > p_value_threshold):
            logger.debug(
                "SNP '%s' has p-value > %f. Skipping allele plot.",
                snp,
                p_value_threshold,
            )
            continue

        if len(df_total_effect) < 3:
            logger.debug(
                "SNP '%s' has less than 3 genotypes. Skipping allele plot.",
                snp,
            )
            continue

        plt.figure()
        plt.errorbar(
            x=df_total_effect.index.get_level_values("Genotype"),
            y=df_total_effect["Coefficient"],
            yerr=[
                abs(df_total_effect["Coefficient"] - df_total_effect["0.025 CI"]),
                abs(df_total_effect["0.975 CI"] - df_total_effect["Coefficient"]),
            ],
            fmt="o",
            capsize=5,
        )

        plt.title(f"SNP: {snp}")
        plt.xlabel("Genotype")
        plt.ylabel("Total Effect Size")
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(output_dir / f"{snp}.pdf")
        plt.close()


def _parse_df_allele_effects(df_allele_effects: pd.DataFrame) -> pd.DataFrame:
    index_df = _get_index_df(df_allele_effects=df_allele_effects)

    df_allele_effects.reset_index(drop=True, inplace=True)

    cols = ["SNP", "Genotype", "Is_Intercept"]
    df_allele_effects[cols] = index_df[cols]
    df_allele_effects.set_index(keys=["SNP", "Genotype"], inplace=True)

    return df_allele_effects


def _get_index_df(df_allele_effects: pd.DataFrame) -> pd.DataFrame:
    index_df = df_allele_effects.index.to_frame(index=False)
    index_df["Is_Intercept"] = index_df["allele"].str.contains("Intercept")
    index_df["allele"] = index_df["allele"].str.replace(" (Intercept)", "")
    index_df[["SNP", "Genotype"]] = index_df["allele"].str.split(" ", expand=True)

    return index_df


def _get_snps_with_strongest_effects(
    df_allele_effects: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    df_no_intercepts = df_allele_effects[~df_allele_effects["Is_Intercept"]]
    snp_strength = df_no_intercepts.groupby(level=0)["Coefficient"].apply(
        lambda x: np.abs(x).sum()
    )

    top_snps = snp_strength.nlargest(top_n).index

    return top_snps


def _compute_total_effect(
    df: pd.DataFrame,
) -> pd.DataFrame:
    row_intercept = df[df["Is_Intercept"]]
    assert len(row_intercept) == 1
    intercept_value = row_intercept["Coefficient"].item()
    intercept_index = row_intercept.index

    for index in df.index.drop(intercept_index):
        df.loc[index, "Coefficient"] += intercept_value
        df.loc[index, "0.025 CI"] += intercept_value
        df.loc[index, "0.975 CI"] += intercept_value

    return df
