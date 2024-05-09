from pathlib import Path
from typing import Optional

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
    snp_ids = [i for i in df_genotype.columns if not i.startswith("COVAR_")]
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=snp_ids)

    df_genotype_prepared = prepare_genotype_data(
        df=df_genotype,
        allele_maps=allele_maps,
    )

    assert len(df_target.columns) == 1
    target_name = df_target.columns[0]

    df_combined = pd.concat(objs=[df_target, df_genotype_prepared], axis=1)

    all_snp_pairs = _get_snp_pairs_to_check(
        df_interaction_effects=df_interaction_effects,
        top_n=None,
    )

    snps_to_plot = _get_snp_pairs_to_check(
        df_interaction_effects=df_interaction_effects,
        top_n=top_n_snps,
    )

    all_results = []

    for snp1, snp2 in all_snp_pairs:
        df_results = fit_models_for_combinations(
            df=df_combined,
            snp1=snp1,
            snp2=snp2,
            target_name=target_name,
            allele_maps=allele_maps,
        )

        if [snp1, snp2] in snps_to_plot:
            output_path = output_folder / "figures" / f"{snp1}_{snp2}_interaction.pdf"
            ensure_path_exists(path=output_path)
            fig = plot_snp_coefficients_as_points(
                df_results=df_results,
                allele_maps=allele_maps,
                snp1_name=snp1,
                snp2_name=snp2,
            )

            fig.savefig(output_path)

        df_results = df_results.rename(
            columns={
                f"{snp1}_genotype": "SNP1_genotype",
                f"{snp2}_genotype": "SNP2_genotype",
            }
        )

        all_results.append(df_results)

    if not all_results:
        logger.warning("No significant SNP pairs found for grouped analysis.")
        return

    df_all_results = pd.concat(all_results)

    df_all_results.to_csv(
        output_folder / "snp_interactions_as_groups.csv",
        index=False,
    )

    all_snps = set([snp for pair in all_snp_pairs for snp in pair])
    snp_frequencies = []

    for snp in all_snps:
        cur_freq = calculate_snp_genotype_frequencies(
            df=df_genotype_prepared,
            snp=snp,
            genotype_maps=allele_maps,
        )
        snp_frequencies.append(cur_freq)

    df_snp_frequencies = pd.concat(snp_frequencies)
    df_snp_frequencies.to_csv(
        output_folder / "snp_frequencies.csv",
        index=False,
    )


def _get_snp_pairs_to_check(
    df_interaction_effects: pd.DataFrame,
    top_n: Optional[int] = None,
    p_value_threshold: float | str = "auto",
) -> list[list[str]]:
    if p_value_threshold == "auto":
        num_tests = df_interaction_effects["KEY"].nunique()
        p_value_threshold = 0.05 / num_tests
    elif not isinstance(p_value_threshold, float):
        raise ValueError("p_value_threshold must be 'auto' or a float.")

    logger.info(
        "Using p-value threshold of %f for grouped analysis.", p_value_threshold
    )

    interaction_keys = df_interaction_effects["KEY"].unique()
    df_all = df_interaction_effects
    df_ie = df_all[df_all.index.isin(interaction_keys)].copy()

    df_ie_filtered = df_ie[df_ie["P>|t|"] <= p_value_threshold]
    df_ie_interaction_keys = df_ie_filtered["KEY"].unique()
    assert len(df_ie_interaction_keys) == len(df_ie_filtered)

    df_to_check = df_ie_filtered.copy()

    if top_n is not None:
        df_to_check = df_ie_filtered.sort_values(
            "Coefficient",
            ascending=False,
        ).head(top_n)

    logger.info("Checking %d SNP pairs for grouped analysis.", len(df_to_check))

    return df_to_check["KEY"].str.split("--:--", expand=True).to_numpy().tolist()


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
) -> pd.DataFrame:
    results = []
    allele_order = ["REF", "HET", "ALT"]
    total_samples = len(df)

    for allele1 in allele_order:
        genotype1 = allele_maps[snp1].get(allele1)
        if genotype1 is None:
            continue

        for allele2 in allele_order:
            genotype2 = allele_maps[snp2].get(allele2)
            if genotype2 is None:
                continue

            subset = df[(df[snp1] == genotype1) & (df[snp2] == genotype2)]

            if not subset.empty:
                model = smf.ols(f"{target_name} ~ 1", data=subset).fit()
                conf_int = model.conf_int().loc["Intercept"]
                p_value = model.pvalues["Intercept"]
                combination_freq = len(subset) / total_samples
                results.append(
                    {
                        f"{snp1}_genotype": genotype1,
                        f"{snp2}_genotype": genotype2,
                        "Value": model.params["Intercept"],
                        "CI_lower": conf_int[0],
                        "CI_upper": conf_int[1],
                        "P_value": p_value,
                        "KEY": f"{snp1}--:--{snp2}",
                        "n": len(subset),
                        "Combination_freq": combination_freq,
                    }
                )
            else:
                logger.warning(
                    "No data for SNP1: %s, SNP2: %s, genotype1: %s, genotype2: %s",
                    snp1,
                    snp2,
                    genotype1,
                    genotype2,
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
                y = row["Value"]
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
    ax.set_ylabel("Value")
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


def calculate_snp_genotype_frequencies(
    df: pd.DataFrame,
    snp: str,
    genotype_maps: dict[str, dict[str, str]],
) -> pd.DataFrame:
    results = []
    total_samples = len(df)

    for genotype_label, genotype in genotype_maps[snp].items():
        count = df[snp].value_counts(dropna=False).get(genotype, 0)
        frequency = count / total_samples if total_samples > 0 else 0
        results.append(
            {
                "SNP": snp,
                "Genotype": genotype,
                "Genotype_Label": genotype_label,
                "Count": count,
                "Frequency": frequency,
            }
        )

    all_genotypes = set(df[snp].unique())
    known_genotypes = set(genotype_maps[snp].values())
    additional_genotypes = all_genotypes - known_genotypes - {float("nan")}

    for genotype in additional_genotypes:
        count = df[snp].value_counts(dropna=False).get(genotype, 0)
        frequency = count / total_samples if total_samples > 0 else 0
        results.append(
            {
                "SNP": snp,
                "Genotype": genotype,
                "Genotype_Label": "Other/Unknown",
                "Count": count,
                "Frequency": frequency,
            }
        )

    missing_count = df[snp].isna().sum()
    if missing_count > 0:
        results.append(
            {
                "SNP": snp,
                "Genotype": "Missing",
                "Genotype_Label": "Missing",
                "Count": missing_count,
                "Frequency": missing_count / total_samples,
            }
        )

    return pd.DataFrame(results)
