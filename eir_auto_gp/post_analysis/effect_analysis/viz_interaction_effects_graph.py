import heapq
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from _operator import itemgetter
from aislib.misc_utils import ensure_path_exists, get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from matplotlib import pyplot as plt

logger = get_logger(name=__name__)


def generate_interaction_snp_graph_figure(
    df_interaction_effects: pd.DataFrame,
    bim_file_path: Path,
    plot_output_root: Path,
    df_target: pd.DataFrame,
    top_n_snps: Optional[int] = None,
    trait: Optional[str] = None,
) -> None:

    df_interaction_effects_filtered = _filter_interaction_df_for_p_value(
        df=df_interaction_effects,
        p_threshold="auto",
    )

    if len(df_interaction_effects_filtered) == 0:
        logger.info("No interactions found for SNP interaction graph figure, skipping.")
        return

    train_interaction_info = _extract_cluster_info_from_interaction_df(
        df_interactions=df_interaction_effects_filtered,
        bim_file_path=bim_file_path,
    )
    df_interaction_coefficients = build_interaction_coefficient_df(
        df_interaction_effects=df_interaction_effects_filtered,
        snps=train_interaction_info.snps,
    )

    df_interaction_coefficients = filter_df_interactions_for_top_n_snps(
        df_interactions=df_interaction_coefficients,
        top_n=top_n_snps,
    )

    summed_interactions = _get_snp_interactions_from_coefficients(
        df_cluster=df_interaction_coefficients,
        top_n=top_n_snps,
    )

    graph = build_graph_from_summed_interactions(
        summed_interactions=summed_interactions
    )

    snp_chr_map = _get_snp_chr_map(bim_file_path=bim_file_path)

    cur_fig = _get_interaction_graph_figure(
        graph=graph,
        trait=trait,
        node_color_map=snp_chr_map,
        manual_order=None,
        df_target=df_target,
    )

    cur_output_path = plot_output_root / "figures" / "snp_interactions.pdf"
    ensure_path_exists(path=cur_output_path)

    cur_fig.savefig(cur_output_path, bbox_inches="tight")


def _filter_interaction_df_for_p_value(
    df: pd.DataFrame, p_threshold: float | str = "auto"
) -> pd.DataFrame:
    interaction_df = df[df.index.str.contains("--:--")]

    if p_threshold == "auto":
        n_pairs_tested = df["KEY"].nunique()
        p_threshold = 0.05 / n_pairs_tested

        logger.info(
            "Setting p-value threshold to %f for SNP interaction graph figure "
            "based on %d pairs tested.",
            p_threshold,
            n_pairs_tested,
        )

    valid_interactions = interaction_df[interaction_df["P>|t|"] <= p_threshold]

    valid_keys = set(valid_interactions["KEY"])

    logger.info(
        "Keeping %d interactions out of %d for SNP interaction graph figure.",
        len(valid_keys),
        len(interaction_df),
    )

    df_filtered = df[df["KEY"].isin(valid_keys)]

    return df_filtered


def build_interaction_coefficient_df(
    df_interaction_effects: pd.DataFrame, snps: list[str]
) -> pd.DataFrame:
    array = np.zeros((len(snps), len(snps)))

    for row_idx, snp_1 in enumerate(snps):
        for column_idx, snp_2 in enumerate(snps):
            if snp_1 == snp_2:
                continue

            mask = (df_interaction_effects["KEY"].str.contains(snp_1)) & (
                df_interaction_effects["KEY"].str.contains(snp_2)
            )
            df_slice = df_interaction_effects[mask]

            if len(df_slice) == 0:
                continue

            cur_coefficient = df_slice.iloc[-1]["Coefficient"]

            array[row_idx, column_idx] = cur_coefficient

    df_interaction_effects = pd.DataFrame(array, columns=snps, index=snps)

    return df_interaction_effects


def _get_snp_interactions_from_coefficients(
    df_cluster: pd.DataFrame, top_n: Optional[int] = None
) -> dict:
    interactions = {}

    for snp_1 in df_cluster.index:
        for snp_2 in df_cluster.columns:
            cur_interaction = df_cluster.loc[snp_1, snp_2]

            if cur_interaction == 0.0:
                continue

            key = "---".join(sorted([snp_1, snp_2]))

            if key not in interactions:
                interactions[key] = []

            interactions[key].append(cur_interaction)

    summed_interactions = {
        key: sum(value) / len(value) for key, value in interactions.items()
    }

    if top_n and top_n < len(summed_interactions):
        summed_interactions = dict(
            heapq.nlargest(
                n=top_n,
                iterable=summed_interactions.items(),
                key=itemgetter(1),
            )
        )

    return summed_interactions


def build_graph_from_summed_interactions(
    summed_interactions: dict[str, float]
) -> nx.Graph:
    graph = nx.Graph(name="Interaction Effects")

    for key, value in summed_interactions.items():
        node_1_label, node_2_label = key.split("---")
        direction = "negative" if value < 0 else "positive"
        graph.add_weighted_edges_from(
            ebunch_to_add=[
                (
                    node_1_label,
                    node_2_label,
                    abs(value),
                )
            ],
            direction=direction,
        )

    return graph


@dataclass
class SNPInteractionClusterInfo:
    snps: list[str]
    main_effect_total: float
    interaction_effect_total: float
    percentage_interaction_effects_of_total: float


def _extract_cluster_info_from_interaction_df(
    df_interactions: pd.DataFrame, bim_file_path: Path
) -> SNPInteractionClusterInfo:
    """
    We add and trip the chromosomes when doing the order to maintain chr-pos order.
    Then we map these to rsIDs.
    """
    snps = set()

    snp_chr_map = _get_snp_chr_map(bim_file_path=bim_file_path)
    snp_pos_map = _get_snp_pos_map(bim_file_path=bim_file_path)

    main_effect_total = 0.0
    interaction_effect_total = 0.0

    for key, df_slice in df_interactions.groupby("KEY"):
        df_slice = df_slice.reset_index()
        snp_1, snp_2 = str(key).split("--:--")

        snp_1_chr = snp_chr_map[snp_1]
        snp_2_chr = snp_chr_map[snp_2]

        snp_1_pos = snp_pos_map[snp_1]
        snp_2_pos = snp_pos_map[snp_2]

        snp_1_w_chr_pos = f"chr{snp_1_chr}:{snp_1_pos}---{snp_1}"
        snp_2_w_chr_pos = f"chr{snp_2_chr}:{snp_2_pos}---{snp_2}"

        snps.update({snp_1_w_chr_pos, snp_2_w_chr_pos})

        main_effects = abs(df_slice.iloc[:-1]["Coefficient"]).sum()
        interaction_effect = abs(df_slice.iloc[:-1]["Coefficient"]).sum()

        main_effect_total += main_effects
        interaction_effect_total += interaction_effect

    snps = _natural_sort(tuple(snps))
    snps = tuple(i.split("---")[1] for i in snps)

    total_effects = main_effect_total + interaction_effect_total
    percentage_interaction = interaction_effect_total / total_effects

    cluster_info = SNPInteractionClusterInfo(
        snps=list(snps),
        main_effect_total=main_effect_total,
        interaction_effect_total=interaction_effect_total,
        percentage_interaction_effects_of_total=percentage_interaction,
    )

    return cluster_info


def _get_snp_pos_map(bim_file_path: Path) -> dict[str, str]:
    df = read_bim(bim_file_path=str(bim_file_path))
    chr_map = dict(zip(df["VAR_ID"], df["BP_COORD"]))
    return chr_map


def _get_snp_chr_map(bim_file_path: Path) -> dict[str, int]:
    df = read_bim(bim_file_path=str(bim_file_path))
    chr_map = dict(zip(df["VAR_ID"], df["CHR_CODE"]))

    return chr_map


def _natural_sort(sequence: Sequence) -> Sequence:
    def _convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def _alphanumeric_key(key: str) -> list:
        return [_convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(sequence, key=_alphanumeric_key)


def _get_interaction_graph_figure(
    graph: nx.Graph,
    node_color_map: dict[str, int],
    trait: str,
    df_target: pd.DataFrame,
    manual_order: Sequence[str] = None,
) -> plt.Figure:
    pos = nx.spring_layout(G=graph, k=0.15, iterations=20)

    if manual_order:
        new_pos = {}
        for idx, (prev_snp, position) in enumerate(pos.items()):
            cur_snp = manual_order[idx]
            new_pos[cur_snp] = position

        pos = new_pos

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap("tab20")
    node_color = [cmap(node_color_map[i]) for i in graph.nodes()]

    edge_colors = []
    for u, v in graph.edges():
        cur_edge = graph[u][v]
        cur_direction = cur_edge["direction"]

        if cur_direction == "positive":
            edge_colors.append("#f54248")
        elif cur_direction == "negative":
            edge_colors.append("#42b9f5")

    nx.draw_networkx_nodes(
        G=graph,
        pos=pos,
        node_color=node_color,
        node_size=300,
    )

    nx.draw_networkx_labels(
        G=graph,
        pos=pos,
        font_size=8,
    )

    weights = scale_weights(
        graph=graph,
        df_target=df_target,
        trait=trait,
    )
    nx.draw_networkx_edges(
        G=graph,
        pos=pos,
        width=weights,
        edge_color=edge_colors,
        alpha=0.8,
    )

    cur_colors = sorted(tuple(set([node_color_map[i] for i in graph.nodes])))
    for v in cur_colors:
        plt.scatter(x=[], y=[], c=[cmap(v)], label=f"{v}")

    ax.legend(
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.1),
        borderaxespad=0.0,
        title="Chromosome",
        frameon=False,
    )

    ax.set_title(trait)

    plt.axis("off")
    plt.tight_layout()

    return fig


def scale_weights(
    graph: nx.Graph,
    df_target: pd.DataFrame,
    trait: str,
    scaling_factor: float = 100.0,
    max_threshold: float = 18.0,
    min_threshold: float = 1.0,
) -> list[float]:
    target_min = df_target[trait].min()
    target_range = df_target[trait].max() - target_min

    weights = [
        ((abs(graph[u][v]["weight"]) - target_min) / target_range) * scaling_factor
        for u, v in graph.edges()
    ]

    max_weight = max(weights)
    if max_weight > max_threshold:
        weights = [weight * (max_threshold / max_weight) for weight in weights]

    weights = [max(weight, min_threshold) for weight in weights]

    assert all(
        min_threshold <= weight <= max_threshold for weight in weights
    ), "Weights out of bounds"

    return weights


def filter_df_interactions_for_top_n_snps(
    df_interactions: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    df_copy = df_interactions.copy()

    top_snps = abs(df_copy).sum().sort_values().tail(top_n).index
    df_copy = df_copy.loc[df_copy.index.isin(top_snps), top_snps]

    return df_copy
