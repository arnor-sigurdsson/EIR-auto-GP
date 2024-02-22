from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.effect_analysis.interaction_effects import (
    get_snp_allele_maps,
    read_bim,
)

logger = get_logger(name=__name__)


def run_grouped_interaction_analysis(
    df_genotype: pd.DataFrame,
    df_target: pd.DataFrame,
    df_interaction_effects: pd.DataFrame,
    top_n_snps: int,
    bim_file: Path,
    output_folder: Path,
) -> None:
    df_bim = read_bim(bim_file_path=str(bim_file))
    snp_ids = df_genotype.columns
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=snp_ids)

    df_genotype_prepared = prepare_genotype_data(
        df=df_genotype,
        allele_maps=allele_maps,
    )

    assert len(df_target.columns) == 1
    target_name = df_target.columns[0]

    df_combined = pd.concat(objs=[df_target, df_genotype_prepared], axis=1)

    snps_to_check = _get_snp_pairs_to_check(
        df_interaction_effects=df_interaction_effects,
        top_n=top_n_snps,
    )

    if not snps_to_check:
        logger.info("No SNP pairs to check for grouped interaction analysis.")
        return

    all_results = []

    for snp1, snp2 in snps_to_check:
        df_results = fit_models_for_combinations(
            df=df_combined,
            snp1=snp1,
            snp2=snp2,
            target_name=target_name,
            allele_maps=allele_maps,
        )
        all_results.append(df_results)

        fig = plot_snp_coefficients_as_points(
            df_results=df_results,
            allele_maps=allele_maps,
            snp1_name=snp1,
            snp2_name=snp2,
        )

        output_path = output_folder / "figures" / f"{snp1}_{snp2}_interaction.pdf"
        ensure_path_exists(path=output_path)

        fig.savefig(output_path)

    df_all_results = pd.concat(all_results)

    df_all_results.to_csv(
        output_folder / "snp_interactions_as_groups.csv",
        index=False,
    )


def _get_snp_pairs_to_check(
    df_interaction_effects: pd.DataFrame,
    top_n: int,
) -> list[list[str, str]]:
    interaction_keys = df_interaction_effects["KEY"].unique()

    df_ie = df_interaction_effects
    df_interactions = df_ie[df_ie.index.isin(interaction_keys)]

    df_interactions_top = df_interactions.sort_values(
        "Coefficient",
        ascending=False,
    ).head(top_n)

    return df_interactions_top["KEY"].str.split(":", expand=True).to_numpy().tolist()


def prepare_genotype_data(
    df: pd.DataFrame, allele_maps: dict[str, dict[str, str]]
) -> pd.DataFrame:
    df_prepared = df.copy()

    for snp, mappings in allele_maps.items():
        if snp in df.columns:
            df_prepared[snp] = df[snp].map(
                {
                    0: mappings["REF"],
                    1: mappings["HET"],
                    2: mappings["ALT"],
                }
            )
    return df_prepared


def fit_models_for_combinations(
    df: pd.DataFrame,
    snp1: str,
    snp2: str,
    target_name: str,
    allele_maps: dict[str, dict[str, str]],
):
    results = []

    for genotype1 in df[snp1].unique():
        for genotype2 in df[snp2].unique():
            subset = df[(df[snp1] == genotype1) & (df[snp2] == genotype2)]

            snp1_expected_values = allele_maps[snp1].values()
            snp2_expected_values = allele_maps[snp2].values()

            subset = subset[
                subset[snp1].isin(snp1_expected_values)
                & subset[snp2].isin(snp2_expected_values)
            ].copy()

            if not subset.empty:
                model = smf.ols(f"{target_name} ~ 1", data=subset).fit()
                conf_int = model.conf_int().loc["Intercept"]
                p_value = model.pvalues["Intercept"]
                results.append(
                    {
                        f"{snp1}_genotype": genotype1,
                        f"{snp2}_genotype": genotype2,
                        "Coefficient": model.params["Intercept"],
                        "CI_lower": conf_int[0],
                        "CI_upper": conf_int[1],
                        "P_value": p_value,
                        "KEY": f"{snp1}:{snp2}",
                    }
                )

    return pd.DataFrame(results)


def plot_snp_coefficients_as_points(
    df_results: pd.DataFrame,
    allele_maps: dict[str, dict[str, str]],
    snp1_name: str,
    snp2_name: str,
) -> plt.Figure:
    sns.set_style("white")

    snp1_map = allele_maps[snp1_name]
    snp2_map = allele_maps[snp2_name]
    genotype_order_snp1 = [snp1_map["REF"], snp1_map["HET"], snp1_map["ALT"]]
    genotype_order_snp2 = [snp2_map["REF"], snp2_map["HET"], snp2_map["ALT"]]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["#E69F00", "#56B4E9", "#009E73"]
    color_map = dict(zip(genotype_order_snp2, colors))

    for i, genotype1 in enumerate(genotype_order_snp1):
        for j, genotype2 in enumerate(genotype_order_snp2):
            subset = df_results[
                (df_results[f"{snp1_name}_genotype"] == genotype1)
                & (df_results[f"{snp2_name}_genotype"] == genotype2)
            ]

            if not subset.empty:
                row = subset.iloc[0]
                y = row["Coefficient"]
                ci_lower = row["CI_lower"]
                ci_upper = row["CI_upper"]
                x = i + j * 0.1
                ax.errorbar(
                    x,
                    y,
                    yerr=[[y - ci_lower], [ci_upper - y]],
                    fmt="o",
                    markersize=8,
                    color=color_map[genotype2],
                    ecolor=color_map[genotype2],
                    elinewidth=2,
                    capsize=5,
                    capthick=2,
                    label=genotype2 if i == 0 else "",
                )

    ax.set_xticks(np.arange(len(genotype_order_snp1)) + 0.1)
    ax.set_xticklabels(genotype_order_snp1)

    ax.set_xlabel(snp1_name + " Genotype")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Effect Sizes for {snp1_name} and {snp2_name} Genotypes")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        title=f"{snp2_name} Genotype",
        loc="upper left",
        frameon=False,
    )

    plt.setp(legend.get_title(), fontsize="13")

    sns.despine()

    plt.tight_layout()

    return fig