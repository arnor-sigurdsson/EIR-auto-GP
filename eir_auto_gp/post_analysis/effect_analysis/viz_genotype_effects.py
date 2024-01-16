from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists
from matplotlib import pyplot as plt


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
    df: pd.DataFrame, p_value_threshold: float, top_n: int, output_dir: Path
) -> None:
    ensure_path_exists(path=output_dir, is_folder=True)

    index_df = df.index.to_frame(index=False)
    index_df[["SNP", "Genotype"]] = index_df["allele"].str.split(" ", expand=True)
    df.reset_index(drop=True, inplace=True)
    df[["SNP", "Genotype"]] = index_df[["SNP", "Genotype"]]
    df.set_index(keys=["SNP", "Genotype"], inplace=True)

    snp_diff = df.groupby(level=0)["Coefficient"].apply(lambda x: np.abs(x - x.iloc[0]))

    snp_strength = snp_diff.groupby(level=0).sum()

    top_snps = snp_strength.nlargest(top_n).index

    for snp in top_snps:
        df_snp = df.loc[snp]

        if np.any(df_snp["p_value"] > p_value_threshold):
            continue

        if len(df_snp) < 3:
            continue

        plt.figure()
        plt.errorbar(
            x=df_snp.index.get_level_values("Genotype"),
            y=df_snp["Coefficient"],
            yerr=[
                abs(df_snp["Coefficient"] - df_snp["0.025 CI"]),
                abs(df_snp["0.975 CI"] - df_snp["Coefficient"]),
            ],
            fmt="o",
            capsize=5,
        )

        plt.title(f"SNP: {snp}")
        plt.xlabel("Genotype")
        plt.ylabel("Effect Size")
        plt.grid(True)

        plt.savefig(output_dir / f"{snp}.pdf")
