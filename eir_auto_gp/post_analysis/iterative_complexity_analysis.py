from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.run_complexity_analysis import (
    ModelReadyObject,
    convert_split_data_to_model_ready_object,
    train_and_evaluate_linear,
)

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_post_analysis import PostAnalysisObject

logger = get_logger(name=__name__)


@dataclass
class StepInformationObjects:
    complexity_results: pd.DataFrame
    allele_effects: pd.DataFrame
    interaction_effects: pd.DataFrame


def _get_step_information_objects(
    post_analysis_object: "PostAnalysisObject",
) -> StepInformationObjects:
    output_root = post_analysis_object.data_paths.analysis_output_path

    df_complexity = pd.read_csv(output_root / "complexity/all_results.csv")

    df_allele_effects = pd.read_csv(
        output_root / "effect_analysis/allele_effects/allele_effects.csv"
    )

    df_interaction_effects = pd.read_csv(
        output_root / "effect_analysis/interaction_effects/interaction_effects.csv"
    )

    return StepInformationObjects(
        complexity_results=df_complexity,
        allele_effects=df_allele_effects,
        interaction_effects=df_interaction_effects,
    )


def run_iterative_complexity_analysis(
    post_analysis_object: "PostAnalysisObject",
) -> None:
    results = []

    pao = post_analysis_object
    sidp = _get_step_information_objects(post_analysis_object=pao)

    data_iter = get_step_training_iterator(
        post_analysis_object=pao,
        step_information_objects=sidp,
        max_one_hot=5,
        max_interaction_exe=5,
        max_interaction_gxe=5,
        max_interaction_gxg=5,
    )

    for idx, (mro, model_type) in enumerate(data_iter):
        train_eval_results = train_and_evaluate_linear(
            modelling_data=mro,
            target_type=pao.experiment_info.target_type,
        )
        df_performance = train_eval_results.performance
        df_performance["model_type"] = model_type
        df_performance["step"] = idx
        results.append(df_performance)

    df_results = pd.concat(results)

    output_root = pao.data_paths.analysis_output_path / "iterative_complexity"
    ensure_path_exists(path=output_root, is_folder=True)

    df_results.to_csv(output_root / "iterative_complexity_results.csv")

    plot_iterative_complexity_results(
        df_results=df_results,
        df_complexity_results=sidp.complexity_results,
        output_folder=output_root / "figures",
    )


def plot_iterative_complexity_results(
    df_results: pd.DataFrame,
    df_complexity_results: pd.DataFrame,
    output_folder: Path,
) -> None:
    ensure_path_exists(path=output_folder, is_folder=True)

    df_results, metric_columns = parse_metrics_changes(df=df_results)

    for metric in metric_columns:
        top_performance, line_label = get_top_performance_info(
            df_complexity_results=df_complexity_results,
            metric=metric,
        )

        fig = plot_metric_figure(
            df=df_results,
            metric=metric,
            top_performance=top_performance,
            line_label=line_label,
        )
        fig.savefig(output_folder / f"{metric}_change.pdf")
        plt.close(fig)


def parse_metrics_changes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    skip_cols = ["Unnamed: 0", "model_type", "step"]
    metric_columns = [col for col in df.columns if col not in skip_cols]
    for metric in metric_columns:
        df[f"{metric}_change"] = df[metric].diff().fillna(0)
    return df, metric_columns


def plot_metric_figure(
    df: pd.DataFrame, metric: str, top_performance: float, line_label: str
) -> plt.Figure:
    change_col = f"{metric}_change"
    y_size = 0.75 * len(df["model_type"].unique())
    plt.figure(figsize=(8, y_size))

    sns.set(style="whitegrid", palette="deep")

    ax = sns.barplot(
        data=df,
        x=metric,
        y="model_type",
        orient="h",
    )

    ax.set_title(
        f"{metric.upper()} Change by Model Type",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel(metric.upper(), fontsize=14)
    ax.set_ylabel("Model Type", fontsize=14)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    for p, change in zip(ax.patches, df[change_col]):
        change_sign = "+" if change >= 0 else "-"
        ax.annotate(
            f"{change_sign}{abs(change):.3f}",
            xy=(p.get_width(), p.get_y() + p.get_height() / 2),
            xytext=(0, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=12,
            color="black",
            weight="bold",
        )

    line_color = "black"
    plt.axvline(
        x=top_performance,
        color=line_color,
        linestyle="--",
        linewidth=1,
    )
    plt.text(
        1.01,
        0.5,
        f"--- {line_label}",
        transform=ax.transAxes,
        color=line_color,
        va="center",
        ha="left",
        fontsize=12,
        weight="bold",
        bbox=dict(
            facecolor="white",
            alpha=1.0,
            edgecolor="black",
            boxstyle="round,pad=0.5",
        ),
    )
    sns.despine(left=False, bottom=False, right=True, top=True)
    plt.tight_layout()

    return plt.gcf()


def get_top_performance_info(df_complexity_results: pd.DataFrame, metric: str):
    if metric == "rmse":
        top_row = df_complexity_results.loc[df_complexity_results[metric].idxmin()]
    else:
        top_row = df_complexity_results.loc[df_complexity_results[metric].idxmax()]

    label_components = [
        "XGB" if top_row["model_type"] == "xgboost" else "LIN",
        "GT" if top_row["include_genotype"] else "",
        "TAB" if top_row["include_tabular"] else "",
        "OH" if top_row["one_hot_encode"] else "",
    ]
    label = "-".join(filter(None, label_components))

    return top_row[metric], label


def get_step_training_iterator(
    post_analysis_object: "PostAnalysisObject",
    step_information_objects: StepInformationObjects,
    max_one_hot: int,
    max_interaction_exe: int,
    max_interaction_gxe: int,
    max_interaction_gxg: int,
) -> Generator:
    """
    1. Only tabular
    2. Only genotype
    3. Genotype + tabular
    4. Iteratively one-hot encode SNPs up to max_one_hot
    5. One-hot encode all SNPs
    6. Iteratively add Tabular x Tabular interaction terms up to max_interaction_exe
    7. Iteratively add SNP x Tabular interaction terms up to max_interaction_gxe
    8. Iteratively add SNP x SNP interaction terms up to max_interaction_gxg
    """

    ei = post_analysis_object.experiment_info
    any_tabular = len(ei.all_input_columns) > 0

    # 1
    tabular_feature_importance = None
    if any_tabular:
        logger.info("Running iterative complexity analysis: tabular only")
        mro_tabular = convert_split_data_to_model_ready_object(
            split_model_data=post_analysis_object.modelling_data,
            include_genotype=False,
            include_tabular=True,
            one_hot_encode=False,
        )

        yield mro_tabular, "Tabular"

        tabular_results = train_and_evaluate_linear(
            modelling_data=mro_tabular,
            target_type=ei.target_type,
        )
        tabular_feature_importance = tabular_results.feature_importance

    # 2
    logger.info("Running iterative complexity analysis: genotype only")
    mro_genotype = convert_split_data_to_model_ready_object(
        split_model_data=post_analysis_object.modelling_data,
        include_genotype=True,
        include_tabular=False,
        one_hot_encode=False,
    )

    yield mro_genotype, "Genotype"

    # 3
    if any_tabular:
        logger.info("Running iterative complexity analysis: genotype + tabular")
        mro_genotype_tabular = convert_split_data_to_model_ready_object(
            split_model_data=post_analysis_object.modelling_data,
            include_genotype=True,
            include_tabular=True,
            one_hot_encode=False,
        )

        yield mro_genotype_tabular, "Genotype + Tabular"

    # 4
    logger.info("Running iterative complexity analysis: one-hot encoding")
    one_hot_iterator = _get_one_hot_iterator(
        post_analysis_object=post_analysis_object,
        df_allele_effects=step_information_objects.allele_effects,
        max_one_hot=max_one_hot,
        any_tabular=any_tabular,
    )

    for mro, model_type in one_hot_iterator:
        yield mro, model_type

    # 5
    logger.info("Running iterative complexity analysis: all one-hot encoding")
    mro_all_oh = convert_split_data_to_model_ready_object(
        split_model_data=post_analysis_object.modelling_data,
        include_genotype=True,
        include_tabular=any_tabular,
        one_hot_encode=True,
    )

    yield mro_all_oh, "+ All OH"
    running_mro = mro_all_oh

    # 6
    if any_tabular:
        logger.info("Running iterative complexity analysis: tabular x tabular")
        txt_iterator, _, _ = get_txt_iterator(
            post_analysis_object=post_analysis_object,
            top_n=max_interaction_exe,
        )

        for mro_txt, model_type in txt_iterator:
            yield mro_txt, model_type
            running_mro = mro_txt

    # 7
    if any_tabular:
        logger.info("Running iterative complexity analysis: genotype x tabular")
        assert tabular_feature_importance is not None
        gxt_iterator, _, _ = get_gxt_iterator(
            running_mro=running_mro,
            post_analysis_object=post_analysis_object,
            top_n=max_interaction_gxe,
            step_information_objects=step_information_objects,
            tabular_feature_importance=tabular_feature_importance,
        )

        for mro_gxt, model_type in gxt_iterator:
            yield mro_gxt, model_type
            running_mro = mro_gxt

    # 8
    logger.info("Running iterative complexity analysis: genotype x genotype")
    gxg_iterator = get_gxg_iterator(
        running_mro=running_mro,
        post_analysis_object=post_analysis_object,
        interaction_effects_df=step_information_objects.interaction_effects,
        top_n=max_interaction_gxg,
    )

    for mro, model_type in gxg_iterator:
        yield mro, model_type


def _get_one_hot_iterator(
    post_analysis_object: "PostAnalysisObject",
    df_allele_effects: pd.DataFrame,
    max_one_hot: int,
    any_tabular: bool,
) -> Generator:
    cur_mro = convert_split_data_to_model_ready_object(
        split_model_data=post_analysis_object.modelling_data,
        include_genotype=True,
        include_tabular=any_tabular,
        one_hot_encode=False,
    )

    df_top_non_additive_snps = _find_non_additive_snps(
        df=df_allele_effects, n=max_one_hot
    )

    for snp in df_top_non_additive_snps["KEY"]:

        def one_hot_encode_snp(
            inputs: pd.DataFrame, targets: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            if snp in inputs.columns:
                inputs = pd.get_dummies(inputs, columns=[snp], prefix=snp + "_OH")
            return inputs, targets

        cur_mro = _merge_operate_and_split(
            model_ready_object=cur_mro, function=one_hot_encode_snp
        )

        cur_name = f"+ OH {snp}"

        yield cur_mro, cur_name


def _find_non_additive_snps(df: pd.DataFrame, n: int) -> pd.DataFrame:
    non_additive_effects = []

    genotype_order = {"REF": 0, "HET": 1, "ALT": 2}
    df["genotype_order"] = df["Label"].map(genotype_order)

    for key, group in df.sort_values(["KEY", "genotype_order"]).groupby("KEY"):
        if len(group) == 3:
            sorted_group = group.sort_values("genotype_order")
            ref_effect, het_effect, alt_effect = sorted_group["Coefficient"].values
            non_additivity_metric = abs(alt_effect - 2 * het_effect)
            non_additive_effects.append((key, non_additivity_metric))

    non_additive_df = (
        pd.DataFrame(non_additive_effects, columns=["KEY", "Non_Additive_Metric"])
        .sort_values(by="Non_Additive_Metric", ascending=False)
        .head(n)
    )

    return non_additive_df


def get_txt_iterator(
    post_analysis_object: "PostAnalysisObject",
    top_n: int,
) -> tuple[Generator, pd.DataFrame, ModelReadyObject]:
    pao = post_analysis_object

    target_type = pao.experiment_info.target_type
    tabular_columns = pao.experiment_info.all_input_columns

    mro_txt_search = convert_split_data_to_model_ready_object(
        split_model_data=pao.modelling_data,
        include_genotype=False,
        include_tabular=True,
        one_hot_encode=False,
    )

    txt_candidates = _find_txt_candidates(
        model_ready_object=mro_txt_search,
        tabular_columns=tabular_columns,
        target_type=target_type,
        top_n=top_n,
    )

    mro = convert_split_data_to_model_ready_object(
        split_model_data=pao.modelling_data,
        include_genotype=True,
        include_tabular=True,
        one_hot_encode=True,
    )

    logger.debug(
        "Top %d TxT candidates: %s", top_n, txt_candidates["interaction"].tolist()
    )

    def generator():
        nonlocal mro
        cur_mro = mro
        for idx, row in txt_candidates.iterrows():
            col1, col2 = row["interaction"].split("_x_")

            def add_interaction(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                col1_name = f"COVAR_{col1}"
                col2_name = f"COVAR_{col2}"
                interaction_term = f"{col1_name}_x_{col2_name}"
                inputs[interaction_term] = inputs[col1_name] * inputs[col2_name]
                return inputs, targets

            cur_mro = _merge_operate_and_split(
                model_ready_object=cur_mro,
                function=add_interaction,
            )

            yield cur_mro, f"+ TxT {col1}_x_{col2}"

    return generator(), txt_candidates, mro


def _find_txt_candidates(
    model_ready_object: ModelReadyObject,
    tabular_columns: list[str],
    target_type: str,
    top_n: int,
) -> pd.DataFrame:
    """
    TODO: Possibly make this just return the top_n feature important ones from
          the linear model only trained on tabular.
    """
    results = []
    mro = model_ready_object

    logger.debug(
        "Searching for top TxT candidates among %d tabular columns, "
        "testing %d combinations",
        len(tabular_columns),
        len(tabular_columns) * (len(tabular_columns) - 1) / 2,
    )

    for col1, col2 in combinations(iterable=tabular_columns, r=2):

        def add_interaction(
            inputs: pd.DataFrame,
            targets: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            col1_name = f"COVAR_{col1}"
            col2_name = f"COVAR_{col2}"
            interaction_term = f"{col1_name}_x_{col2_name}"
            inputs[interaction_term] = inputs[col1_name] * inputs[col2_name]
            return inputs, targets

        modified_mro = _merge_operate_and_split(
            model_ready_object=mro,
            function=add_interaction,
        )

        train_eval_results = train_and_evaluate_linear(
            modelling_data=modified_mro,
            target_type=target_type,
        )
        df_performance = train_eval_results.performance
        df_performance["interaction"] = f"{col1}_x_{col2}"

        results.append(df_performance)

    df_results = pd.concat(results)
    metric = "r2" if target_type == "regression" else "mcc"
    df_sorted = df_results.sort_values(by=metric, ascending=False)
    return df_sorted.head(top_n)


def get_gxt_iterator(
    running_mro: ModelReadyObject,
    post_analysis_object: "PostAnalysisObject",
    step_information_objects: StepInformationObjects,
    tabular_feature_importance: pd.DataFrame,
    top_n: int,
) -> tuple[Generator, pd.DataFrame, ModelReadyObject]:
    pao = post_analysis_object
    target_type = pao.experiment_info.target_type

    mro_search = convert_split_data_to_model_ready_object(
        split_model_data=pao.modelling_data,
        include_genotype=True,
        include_tabular=True,
        one_hot_encode=False,
    )
    genotype_columns = pao.modelling_data.train.df_genotype_input.columns.tolist()
    gxt_candidates = _find_gxt_candidates(
        model_ready_object=mro_search,
        target_type=target_type,
        genotype_columns=genotype_columns,
        top_n=top_n,
        df_allele_effects=step_information_objects.allele_effects,
        df_tabular_feature_importance=tabular_feature_importance,
    )

    logger.debug(
        "Top %d GxT candidates: %s", top_n, gxt_candidates["interaction"].tolist()
    )

    def generator():
        nonlocal running_mro

        for idx, row in gxt_candidates.iterrows():
            snp_column, tab_column = row["interaction"].split("_x_")

            def add_interaction(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                snp_column_additive = (
                    inputs[f"{snp_column}_0"] * 0
                    + inputs[f"{snp_column}_1"] * 1
                    + inputs[f"{snp_column}_2"] * 2
                )

                interaction_term = f"{snp_column}_x_{tab_column}"

                inputs[interaction_term] = snp_column_additive * inputs[tab_column]

                return inputs, targets

            running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_interaction,
            )

            yield running_mro, f"+ GxT {snp_column}_x_{tab_column}"

    return generator(), gxt_candidates, running_mro


def _find_gxt_candidates(
    model_ready_object: ModelReadyObject,
    target_type: str,
    top_n: int,
    genotype_columns: list[str],
    df_allele_effects: pd.DataFrame,
    df_tabular_feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Here we assume we are operating on additive encodings.

    TODO: Possibly we can skip the filtering there below, this is likely just an
          issue when manual doing partial runs with top_n_snps being different
          from the original run and the current iterative complexity run.
    """
    results = []
    mro = model_ready_object

    df_allele_effects_filtered = df_allele_effects[
        df_allele_effects["KEY"].isin(genotype_columns)
    ]

    top_snp_candidates = _find_top_snp_candidates(
        df_allele_effects=df_allele_effects_filtered,
        top_n=top_n,
    )

    top_tabular_columns = df_tabular_feature_importance.head(top_n)["Feature"].tolist()

    logger.debug(
        "Estimated n(TxG) combinations: %d for top %d SNPs: %s x top %d tabular: %s",
        len(top_snp_candidates) * len(top_tabular_columns),
        len(top_snp_candidates),
        top_snp_candidates,
        len(top_tabular_columns),
        top_tabular_columns,
    )

    for snp_column in top_snp_candidates:
        for tab_column in top_tabular_columns:

            def add_interaction(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                interaction_values = inputs[snp_column] * inputs[tab_column]

                interaction_name = f"{snp_column}_x_{tab_column}"
                inputs[interaction_name] = interaction_values
                return inputs, targets

            modified_mro = _merge_operate_and_split(
                model_ready_object=mro,
                function=add_interaction,
            )

            train_eval_results = train_and_evaluate_linear(
                modelling_data=modified_mro,
                target_type=target_type,
            )
            df_performance = train_eval_results.performance
            df_performance["interaction"] = f"{snp_column}_x_{tab_column}"

            results.append(df_performance)

    df_results = pd.concat(results)
    metric = "r2" if target_type == "regression" else "mcc"
    return df_results.sort_values(by=metric, ascending=False).head(top_n)


def _find_top_snp_candidates(df_allele_effects: pd.DataFrame, top_n: int) -> list[str]:
    snp_effects_total = {}

    for snp, group in df_allele_effects.groupby("KEY"):
        snp_effects_total[snp] = group["Coefficient"].abs().sum()

    snp_effects_total_df = pd.DataFrame(
        snp_effects_total.items(), columns=["KEY", "Total_Effect"]
    )

    top_snps_list = (
        snp_effects_total_df.sort_values(by="Total_Effect", ascending=False)
        .head(top_n)["KEY"]
        .tolist()
    )

    return top_snps_list


def get_gxg_iterator(
    running_mro: ModelReadyObject,
    post_analysis_object: "PostAnalysisObject",
    interaction_effects_df: pd.DataFrame,
    top_n: int,
) -> Generator:
    """
    TODO: Possibly we can skip the filtering there below, this is likely just an
          issue when manual doing partial runs with top_n_snps being different
          from the original run and the current iterative complexity run.
    """
    mro = running_mro
    pao = post_analysis_object

    genotype_columns = pao.modelling_data.train.df_genotype_input.columns.tolist()

    if len(interaction_effects_df) == 0:
        logger.warning(
            "No interaction effects found between SNPs in the genotype"
            " columns, skipping."
        )
        return

    interaction_effects_df["SNP1"] = (
        interaction_effects_df["KEY"].str.split("--:--").str[0]
    )
    interaction_effects_df["SNP2"] = (
        interaction_effects_df["KEY"].str.split("--:--").str[1]
    )

    interaction_effects_df_filtered = interaction_effects_df[
        (interaction_effects_df["SNP1"].isin(genotype_columns))
        & (interaction_effects_df["SNP2"].isin(genotype_columns))
    ].copy()

    if len(interaction_effects_df_filtered) == 0:
        logger.warning(
            "No interaction effects found between SNPs in the genotype"
            " columns, skipping."
        )
        return

    top_gxg_candidates = _find_top_gxg_candidates(
        interaction_effects_df=interaction_effects_df_filtered,
        top_n=top_n,
    )

    for idx, row in top_gxg_candidates.iterrows():
        snp_pair = row["KEY"]
        snp1, snp2 = snp_pair.split("--:--")

        def add_interaction(
            inputs: pd.DataFrame,
            targets: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            snp1_additive = (
                inputs[f"{snp1}_0"] * 0
                + inputs[f"{snp1}_1"] * 1
                + inputs[f"{snp1}_2"] * 2
            )
            snp2_additive = (
                inputs[f"{snp2}_0"] * 0
                + inputs[f"{snp2}_1"] * 1
                + inputs[f"{snp2}_2"] * 2
            )

            interaction_term = snp1_additive * snp2_additive
            interaction_name = f"{snp1}_x_{snp2}"

            inputs[interaction_name] = interaction_term

            return inputs, targets

        modified_mro = _merge_operate_and_split(
            model_ready_object=mro,
            function=add_interaction,
        )

        yield modified_mro, f"+ GxG {snp_pair}"


def _find_top_gxg_candidates(
    interaction_effects_df: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    interaction_rows = interaction_effects_df[
        interaction_effects_df["allele"] == interaction_effects_df["KEY"]
    ].copy()

    interaction_rows["Coefficient"] = interaction_rows["Coefficient"].abs()

    top_interactions = interaction_rows.sort_values(
        by="Coefficient", ascending=False
    ).head(top_n)

    return top_interactions[["KEY", "Coefficient"]]


def _merge_operate_and_split(
    model_ready_object: ModelReadyObject,
    function: Callable[
        [pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> ModelReadyObject:
    df_all_inputs = pd.concat(
        (
            model_ready_object.input_train,
            model_ready_object.input_val,
            model_ready_object.input_test,
        )
    )

    df_all_targets = pd.concat(
        (
            model_ready_object.target_train,
            model_ready_object.target_val,
            model_ready_object.target_test,
        )
    )

    df_all_inputs, df_all_targets = function(df_all_inputs, df_all_targets)

    input_train = df_all_inputs.loc[model_ready_object.input_train.index]
    target_train = df_all_targets.loc[model_ready_object.target_train.index]
    assert input_train.index.equals(model_ready_object.input_train.index)
    assert target_train.index.equals(model_ready_object.target_train.index)

    input_val = df_all_inputs.loc[model_ready_object.input_val.index]
    target_val = df_all_targets.loc[model_ready_object.target_val.index]
    assert input_val.index.equals(model_ready_object.input_val.index)
    assert target_val.index.equals(model_ready_object.target_val.index)

    input_test = df_all_inputs.loc[model_ready_object.input_test.index]
    target_test = df_all_targets.loc[model_ready_object.target_test.index]
    assert input_test.index.equals(model_ready_object.input_test.index)
    assert target_test.index.equals(model_ready_object.target_test.index)

    return ModelReadyObject(
        input_train=input_train,
        target_train=target_train,
        input_val=input_val,
        target_val=target_val,
        input_test=input_test,
        target_test=target_test,
        transformers=model_ready_object.transformers,
    )
