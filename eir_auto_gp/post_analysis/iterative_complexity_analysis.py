from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, Optional
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from aislib.misc_utils import ensure_path_exists, get_logger
from sklearn.preprocessing import StandardScaler

from eir_auto_gp.post_analysis.run_complexity_analysis import (
    ModelReadyObject,
    convert_split_data_to_model_ready_object,
    train_and_evaluate_linear,
)

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_post_analysis import PostAnalysisObject

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = get_logger(name=__name__)


@dataclass
class StepInformationObjects:
    complexity_results: pd.DataFrame
    allele_effects: pd.DataFrame
    interaction_effects: pd.DataFrame


def _get_step_information_objects(
    post_analysis_object: "PostAnalysisObject",
    eval_set: str,
) -> StepInformationObjects:
    assert eval_set in ["valid", "test"]
    output_root = post_analysis_object.data_paths.analysis_output_path

    df_complexity = pd.read_csv(output_root / f"complexity/{eval_set}/all_results.csv")

    df_allele_effects = pd.read_csv(
        output_root / "effect_analysis/allele_effects/allele_effects.csv"
    )
    df_allele_effects = filter_snp_rows(df=df_allele_effects)

    df_interaction_effects = pd.read_csv(
        output_root / "effect_analysis/interaction_effects/interaction_effects.csv"
    )

    if len(df_interaction_effects) != 0:
        df_interaction_effects = filter_snp_rows(df=df_interaction_effects)

    return StepInformationObjects(
        complexity_results=df_complexity,
        allele_effects=df_allele_effects,
        interaction_effects=df_interaction_effects,
    )


def filter_snp_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[~df["allele"].str.contains("COVAR")].copy()
    return df_filtered


def run_iterative_complexity_analysis(
    post_analysis_object: "PostAnalysisObject",
    n_iterative_complexity_candidates: int,
    eval_set: str,
) -> None:
    if eval_set not in ["test", "valid"]:
        raise ValueError(
            f"Unsupported eval_set: {eval_set}. Must be 'test' or 'valid'."
        )

    pao = post_analysis_object
    ei = pao.experiment_info
    sidp = _get_step_information_objects(
        post_analysis_object=pao,
        eval_set=eval_set,
    )

    outputs = ei.output_cat_columns + ei.output_con_columns
    if len(outputs) > 1:
        raise ValueError(
            "Multiple outputs detected, iterative complexity analysis "
            "only supports single output."
        )

    output_name = outputs[0]

    output_root = (
        pao.data_paths.analysis_output_path / "iterative_complexity" / eval_set
    )
    ensure_path_exists(path=output_root, is_folder=True)

    results = []

    n = n_iterative_complexity_candidates
    data_iter = get_step_training_iterator(
        post_analysis_object=pao,
        step_information_objects=sidp,
        max_one_hot=n,
        max_interaction_exe=n,
        max_interaction_gxe=n,
        max_interaction_gxg=n,
        eval_set=eval_set,
    )

    for idx, (mro, group, model_type) in enumerate(data_iter):
        train_eval_results = train_and_evaluate_linear(
            modelling_data=mro,
            target_type=ei.target_type,
            eval_set=eval_set,
            model_type="linear_model",
            cv_use_val_split=True,
            with_fallback=True,
        )
        df_performance = train_eval_results.performance
        df_performance["model_type"] = model_type
        df_performance["group"] = group
        df_performance["step"] = idx
        results.append(df_performance)

    df_results = pd.concat(results)
    df_results = df_results.reset_index(drop=True)

    df_results.to_csv(output_root / "iterative_complexity_results.csv")

    figures_output_folder = output_root / "figures"
    ensure_path_exists(path=figures_output_folder, is_folder=True)

    plot_iterative_complexity_results(
        df_results=df_results,
        output_name=output_name,
        df_complexity_results=sidp.complexity_results,
        output_folder=figures_output_folder,
    )


def plot_iterative_complexity_results(
    df_results: pd.DataFrame,
    df_complexity_results: pd.DataFrame,
    output_name: str,
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
            output_name=output_name,
            metric=metric,
            top_performance=top_performance,
            line_label=line_label,
        )
        fig.savefig(output_folder / f"{metric}_change.pdf")
        plt.close(fig)

        direction = "increase" if metric != "rmse" else "decrease"
        fig_changes = plot_top_changes(
            df=df_results,
            metric=metric,
            metric_change=f"{metric}_change",
            output_name=output_name,
            direction=direction,
        )
        fig_changes.savefig(output_folder / f"{metric}_top_changes.pdf")
        plt.close(fig_changes)


def parse_metrics_changes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df_copy = df.copy()
    skip_cols = ["Unnamed: 0", "model_type", "step", "group"]
    metric_columns = [col for col in df_copy.columns if col not in skip_cols]
    for metric in metric_columns:
        df_copy[f"{metric}_change"] = df_copy[metric].diff().fillna(0)
    return df_copy, metric_columns


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


def plot_top_changes(
    df: pd.DataFrame,
    metric_change: str,
    metric: str,
    output_name: str,
    top_n: int = 3,
    direction: str = "increase",
) -> plt.Figure:
    if direction not in ["increase", "decrease"]:
        raise ValueError("direction must be 'increase' or 'decrease'")

    excluded_terms = {"Tabular", "Genotype", "Genotype + Tabular"}
    df_filtered = df[~df["model_type"].isin(excluded_terms)]

    condition = (
        f"{metric_change} > 0" if direction == "increase" else f"{metric_change} < 0"
    )

    df_relevant_changes = df_filtered.query(condition)

    sort_ascending = direction == "decrease"

    df_top_changes = (
        df_relevant_changes.sort_values(by=[metric_change], ascending=sort_ascending)
        .groupby("group")
        .head(top_n)
    )

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_top_changes,
        x=metric_change,
        y="model_type",
        hue="group",
        orient="h",
        palette="deep",
    )
    title_suffix = "Decreases" if direction == "decrease" else "Improvements"
    xlabel_suffix = "Reduction" if direction == "decrease" else "Improvement"
    plt.title(
        f"Top {top_n} {title_suffix} in {output_name}",
        fontsize=16,
        weight="bold",
    )
    plt.xlabel(f"{metric} {xlabel_suffix}", fontsize=14)
    plt.ylabel("Model Type", fontsize=14)
    plt.tight_layout()
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc="upper left")

    return fig


def plot_metric_figure(
    df: pd.DataFrame,
    metric: str,
    top_performance: float,
    line_label: str,
    output_name: str,
) -> plt.Figure:
    sns.set(style="whitegrid", palette="deep")

    change_col = f"{metric}_change"
    direction = "min" if metric == "rmse" else "max"

    df_parsed = prepare_dataframe_with_other_terms(
        df=df,
        metric_change=change_col,
        metric=metric,
        top_n=3,
        direction=direction,
    )

    y_size = 0.75 * len(df_parsed["model_type"].unique())
    longest_string = max(df_parsed["model_type"], key=len)
    x_size = max(8, int(0.35 * len(longest_string)))
    plt.figure(figsize=(x_size, y_size))

    ax = sns.barplot(
        data=df_parsed,
        x=metric,
        y="model_type",
        orient="h",
        hue="group",
    )

    ax.set_title(
        f"{output_name}",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel(f"{metric.upper()} Change", fontsize=14)
    ax.set_ylabel("Model Type", fontsize=14)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black")

    for p, change in zip(ax.patches, df_parsed[change_col]):
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

    ax.get_legend().remove()

    y_ticks = ax.get_yticks()
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    updated_labels = [
        "Other Terms" if "Other terms" in label else label for label in y_labels
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(updated_labels)

    sns.despine(left=False, bottom=False, right=True, top=True)
    plt.tight_layout()

    return plt.gcf()


def prepare_dataframe_with_other_terms(
    df: pd.DataFrame,
    metric: str,
    metric_change: str = "r2_change",
    top_n: int = 3,
    direction: str = "max",
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[metric] = df_copy[metric].fillna(0)

    df_processed = pd.DataFrame()
    df_copy = df_copy.sort_values(by="step")

    for group in df_copy["group"].unique():
        df_group = df_copy[df_copy["group"] == group]
        if direction == "max":
            top_contributors = df_group.nlargest(n=top_n, columns=metric_change)
        else:
            top_contributors = df_group.nsmallest(n=top_n, columns=metric_change)

        df_other_terms = pd.DataFrame()

        last_step_included = -1
        for i, row in df_group.iterrows():
            if row["step"] <= last_step_included:
                continue

            if row["model_type"] in top_contributors["model_type"].values:
                df_processed = pd.concat(
                    [df_processed, df_other_terms, pd.DataFrame([row])],
                    ignore_index=True,
                )
                df_other_terms = pd.DataFrame()

                last_step_included = row["step"]
            else:
                if df_other_terms.empty:
                    df_other_terms = pd.DataFrame([row])
                else:
                    n_rows = len(df_processed)
                    df_other_terms[metric_change] += row[metric_change]
                    df_other_terms[metric] = row[metric]
                    df_other_terms["model_type"] = f"Other terms {n_rows}"
                    df_other_terms["step"] = row["step"]

        if not df_other_terms.empty:
            df_processed = pd.concat(
                [df_processed, df_other_terms],
                ignore_index=True,
            )

    df_processed["group"] = df_processed["group"].replace(
        {
            "Tabular": "Base",
            "Genotype": "Base",
            "Genotype + Tabular": "Base",
        }
    )

    return df_processed


def get_step_training_iterator(
    post_analysis_object: "PostAnalysisObject",
    step_information_objects: StepInformationObjects,
    max_one_hot: int,
    max_interaction_exe: int,
    max_interaction_gxe: int,
    max_interaction_gxg: int,
    eval_set: str,
    include_genotype: bool = True,
    include_tabular: bool = True,
) -> Generator[tuple[ModelReadyObject, str, str], None, None]:
    """
    1. Only tabular
    2. Only genotype
    3. Genotype + tabular
    4. Genotype + tabular + tabular complex terms
    5. Iteratively one-hot encode SNPs up to max_one_hot
    6. One-hot encode all SNPs
    7. Iteratively add Tabular x Tabular interaction terms up to max_interaction_exe
    8. Iteratively add SNP x Tabular interaction terms up to max_interaction_gxe
    9. Iteratively add SNP x SNP interaction terms up to max_interaction_gxg
    """

    ei = post_analysis_object.experiment_info
    any_tabular = len(ei.all_input_columns) > 0
    run_tabular = include_tabular and any_tabular
    run_genotype = include_genotype

    if not run_tabular and not run_genotype:
        logger.warning(
            "No data to run iterative complexity analysis on as both "
            "tabular and genotype are excluded, skipping."
        )
        return

    # 1
    tabular_feature_importance = None
    running_mro = None

    if run_tabular:
        logger.info("Running iterative complexity analysis: tabular only")
        mro_tabular = convert_split_data_to_model_ready_object(
            split_model_data=post_analysis_object.modelling_data,
            include_genotype=False,
            include_tabular=True,
            one_hot_encode=False,
        )

        yield mro_tabular, "Tabular", "Tabular"

        tabular_results = train_and_evaluate_linear(
            modelling_data=mro_tabular,
            target_type=ei.target_type,
            eval_set=eval_set,
            cv_use_val_split=True,
            with_fallback=True,
        )
        tabular_feature_importance = tabular_results.feature_importance
        running_mro = mro_tabular

    # 2
    if run_genotype:
        logger.info("Running iterative complexity analysis: genotype only")
        mro_genotype = convert_split_data_to_model_ready_object(
            split_model_data=post_analysis_object.modelling_data,
            include_genotype=True,
            include_tabular=False,
            one_hot_encode=False,
        )

        yield mro_genotype, "Genotype", "Genotype"
        running_mro = mro_genotype

    # 3
    if run_tabular and run_genotype:
        logger.info("Running iterative complexity analysis: genotype + tabular")
        mro_genotype_tabular = convert_split_data_to_model_ready_object(
            split_model_data=post_analysis_object.modelling_data,
            include_genotype=True,
            include_tabular=True,
            one_hot_encode=False,
        )

        yield mro_genotype_tabular, "Genotype + Tabular", "Genotype + Tabular"
        running_mro = mro_genotype_tabular

    # 4
    if run_tabular:
        logger.info("Running iterative complexity analysis: tabular complex terms")
        mro_tabular_complex_terms, _ = get_t_term_iterator(
            post_analysis_object=post_analysis_object,
            running_mro=running_mro,
        )

        for mro_t_term, model_type in mro_tabular_complex_terms:
            yield mro_t_term, "Tabular Complex Terms", model_type
            running_mro = mro_t_term

    # 5
    if run_genotype:
        logger.info("Running iterative complexity analysis: one-hot encoding")
        one_hot_iterator = _get_one_hot_iterator(
            post_analysis_object=post_analysis_object,
            running_mro=running_mro,
            df_allele_effects=step_information_objects.allele_effects,
            max_one_hot=max_one_hot,
        )

        for cur_mro_oh, model_type in one_hot_iterator:
            yield cur_mro_oh, "OH", model_type
            running_mro = cur_mro_oh

    # 6
    if run_genotype:
        logger.info("Running iterative complexity analysis: all one-hot encoding")
        mro_all_oh, model_type = one_hot_encode_all_snps(
            running_mro=running_mro,
            df_allele_effects=step_information_objects.allele_effects,
        )

        yield mro_all_oh, "All OH", model_type
        running_mro = mro_all_oh

    # 7
    if run_tabular:
        logger.info("Running iterative complexity analysis: tabular x tabular")
        txt_iterator, _, _ = get_txt_iterator(
            running_mro=running_mro,
            post_analysis_object=post_analysis_object,
            top_n=max_interaction_exe,
        )

        if txt_iterator is not None:
            for mro_txt, model_type in txt_iterator:
                yield mro_txt, "TxT", model_type
                running_mro = mro_txt

    # 8
    if run_tabular and run_genotype:
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
            yield mro_gxt, "GxT", model_type
            running_mro = mro_gxt

    # 9
    if run_genotype:
        logger.info("Running iterative complexity analysis: genotype x genotype")
        gxg_iterator = get_gxg_iterator(
            running_mro=running_mro,
            post_analysis_object=post_analysis_object,
            interaction_effects_df=step_information_objects.interaction_effects,
            top_n=max_interaction_gxg,
        )

        for cur_mro_oh, model_type in gxg_iterator:
            yield cur_mro_oh, "GxG", model_type


def _get_one_hot_iterator(
    post_analysis_object: "PostAnalysisObject",
    running_mro: ModelReadyObject,
    df_allele_effects: pd.DataFrame,
    max_one_hot: int,
) -> Generator:
    pao = post_analysis_object

    genotype_columns = pao.modelling_data.train.df_genotype_input.columns.tolist()
    df_top_non_additive_snps = _find_non_additive_snps(
        df_allele_effects=df_allele_effects,
        genotype_columns=genotype_columns,
        n=max_one_hot,
    )

    for snp in df_top_non_additive_snps["KEY"]:

        def one_hot_encode_snp(
            inputs: pd.DataFrame, targets: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            if snp in inputs.columns:
                inputs = pd.get_dummies(
                    inputs,
                    columns=[snp],
                    prefix=snp + "_OH",
                    drop_first=True,
                )
            return inputs, targets

        running_mro = _merge_operate_and_split(
            model_ready_object=running_mro, function=one_hot_encode_snp
        )

        cur_name = f"+ OH {snp}"

        yield running_mro, cur_name


def _find_non_additive_snps(
    df_allele_effects: pd.DataFrame,
    genotype_columns: list[str],
    n: int,
) -> pd.DataFrame:
    non_additive_effects = []

    df_all_filt = df_allele_effects[
        df_allele_effects["KEY"].isin(genotype_columns)
    ].copy()

    genotype_order = {"REF": 0, "HET": 1, "ALT": 2}
    df_all_filt["genotype_order"] = df_all_filt["Label"].map(genotype_order)

    for key, group in df_all_filt.sort_values(["KEY", "genotype_order"]).groupby("KEY"):
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


def one_hot_encode_all_snps(
    running_mro: ModelReadyObject, df_allele_effects: pd.DataFrame
) -> tuple:
    snps_to_encode = df_allele_effects["KEY"].unique()

    snps_to_encode = [
        snp for snp in snps_to_encode if snp in running_mro.input_train.columns
    ]

    def one_hot_encode_multiple_snps(
        inputs: pd.DataFrame, targets: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if snps_to_encode:
            inputs = pd.get_dummies(
                data=inputs,
                columns=snps_to_encode,
                prefix={snp: snp + "_OH" for snp in snps_to_encode},
                drop_first=True,
            )
        return inputs, targets

    updated_mro = _merge_operate_and_split(
        model_ready_object=running_mro, function=one_hot_encode_multiple_snps
    )

    operation_name = "+ All OH"

    return updated_mro, operation_name


def get_t_term_iterator(
    post_analysis_object: "PostAnalysisObject",
    running_mro: ModelReadyObject,
) -> tuple[Generator, ModelReadyObject]:
    """
    Note: For now we are only considering numerical columns here. This can be extended
          to also include the one-hot-encoded columns from categorical variables,
          by looping over:
          [i for i in running_mro.df_train.columns if i.startswith("_COVAR")]
    """

    scaler = running_mro.transformers.input_scaler
    numerical_columns = running_mro.transformers.numerical_columns

    train_ids = running_mro.input_train.index

    def generator():
        nonlocal running_mro
        for col in numerical_columns:

            def add_squared_term(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                col_name = f"{col}"
                term = f"{col}^2"

                inputs[term] = inputs[col_name] ** 2

                return inputs, targets

            def add_cubed_term(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                col_name = f"{col}"
                term = f"{col}^3"
                inputs[term] = inputs[col_name] ** 3

                return inputs, targets

            def add_log_term(
                inputs: pd.DataFrame,
                targets: pd.DataFrame,
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                """
                We have this kind of roundabout way of doing this because we need to
                inverse scale the column (otherwise won't work due to <=0 values
                from scaling) and then take the log, then scale again for models like
                ElasticNetCV.
                """
                col_name: str = f"{col}"
                term: str = f"log_{col_name}"

                if scaler is not None:
                    col_data = inverse_scale_column(
                        scaler=scaler,
                        inputs=inputs,
                        col_name=col_name,
                        numerical_columns=numerical_columns,
                    )

                    if np.all(col_data > 0):
                        inputs[term] = np.log(col_data)
                    else:
                        raise ValueError(
                            f"Column {col_name} has non-positive values, "
                            f"cannot take the log of this column"
                        )

                    inputs = fit_transform_on_train_only(
                        inputs=inputs,
                        train_ids=train_ids,
                        column=term,
                    )

                return inputs, targets

            def add_inverse_term(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                col_name = f"{col}"
                term = f"1/{col}"

                if scaler is not None:
                    col_data = inverse_scale_column(
                        scaler=scaler,
                        inputs=inputs,
                        col_name=col_name,
                        numerical_columns=numerical_columns,
                    )

                    if np.any(col_data == 0):
                        raise ValueError(
                            f"Column {col_name} contains zero values, "
                            f"cannot take the inverse of this column"
                        )
                    else:
                        inputs[term] = 1 / col_data

                    inputs = fit_transform_on_train_only(
                        inputs=inputs,
                        train_ids=train_ids,
                        column=term,
                    )
                else:
                    if (inputs[col_name] == 0).any():
                        raise ValueError(
                            f"Column {col_name} contains zero values, "
                            f"cannot take the inverse of this column"
                        )
                    else:
                        inputs[term] = 1 / inputs[col_name]

                return inputs, targets

            running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_squared_term,
            )
            yield running_mro, f"+ T {col}^2"

            running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_cubed_term,
            )
            yield running_mro, f"+ T {col}^3"

            maybe_running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_log_term,
            )

            if isinstance(maybe_running_mro, ModelReadyObject):
                running_mro = maybe_running_mro
                yield running_mro, f"+ T log_{col}"

            maybe_running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_inverse_term,
            )

            if isinstance(maybe_running_mro, ModelReadyObject):
                running_mro = maybe_running_mro
                yield running_mro, f"+ T 1/{col}"

    return generator(), running_mro


def inverse_scale_column(
    scaler: StandardScaler,
    inputs: pd.DataFrame,
    col_name: str,
    numerical_columns: list[str],
) -> np.ndarray:
    col_position: int = numerical_columns.index(col_name)

    mean: float = scaler.mean_[col_position]
    scale: float = scaler.scale_[col_position]

    col_data: np.ndarray = inputs[col_name].to_numpy()
    inversely_scaled_data: np.ndarray = col_data * scale + mean
    return inversely_scaled_data


def fit_transform_on_train_only(
    inputs: pd.DataFrame,
    train_ids: pd.Index,
    column: str,
) -> pd.DataFrame:
    scaler = StandardScaler()
    scaler.fit(X=inputs.loc[train_ids, [column]])
    inputs[column] = scaler.transform(X=inputs[[column]])
    return inputs


def get_txt_iterator(
    post_analysis_object: "PostAnalysisObject",
    running_mro: ModelReadyObject,
    top_n: int,
) -> tuple[Optional[Generator], pd.DataFrame, ModelReadyObject]:
    pao = post_analysis_object

    target_type = pao.experiment_info.target_type
    numerical_columns = running_mro.transformers.numerical_columns

    mro_txt_search = convert_split_data_to_model_ready_object(
        split_model_data=pao.modelling_data,
        include_genotype=False,
        include_tabular=True,
        one_hot_encode=False,
    )

    txt_candidates = _find_txt_candidates(
        model_ready_object=mro_txt_search,
        tabular_columns=numerical_columns,
        target_type=target_type,
        top_n=top_n,
    )

    if txt_candidates.empty:
        logger.warning("No TxT candidates found, skipping.")
        return None, pd.DataFrame(), running_mro

    logger.debug(
        "Top %d TxT candidates: %s", top_n, txt_candidates["interaction"].tolist()
    )

    def generator():
        nonlocal running_mro
        for idx, row in txt_candidates.iterrows():
            col1, col2 = row["interaction"].split("_x_")

            def add_interaction(
                inputs: pd.DataFrame, targets: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
                col1_name = f"{col1}"
                col2_name = f"{col2}"
                interaction_term = f"{col1_name}_x_{col2_name}"
                inputs[interaction_term] = inputs[col1_name] * inputs[col2_name]
                return inputs, targets

            running_mro = _merge_operate_and_split(
                model_ready_object=running_mro,
                function=add_interaction,
            )

            yield running_mro, f"+ TxT {col1}_x_{col2}"

    return generator(), txt_candidates, running_mro


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
            col1_name = f"{col1}"
            col2_name = f"{col2}"
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
            eval_set="valid",
            cv_use_val_split=True,
            with_fallback=True,
        )
        df_performance = train_eval_results.performance
        df_performance["interaction"] = f"{col1}_x_{col2}"

        results.append(df_performance)

    if not results:
        return pd.DataFrame()

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
                snp_column_additive = pd.Series(0, index=inputs.index)
                for genotype in [0, 1, 2]:
                    snp_col = f"{snp_column}_OH_{genotype}"
                    if snp_col in inputs.columns:
                        snp_column_additive += inputs[snp_col] * genotype

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

    df_tabular_feature_importance = df_tabular_feature_importance[
        df_tabular_feature_importance["Feature"] != "Intercept"
    ]
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
                eval_set="valid",
                cv_use_val_split=True,
                with_fallback=True,
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
            snp1_additive = pd.Series(0, index=inputs.index)
            snp2_additive = pd.Series(0, index=inputs.index)

            for genotype in [0, 1, 2]:
                snp1_col = f"{snp1}_OH_{genotype}"
                if snp1_col in inputs.columns:
                    snp1_additive += inputs[snp1_col] * genotype

                snp2_col = f"{snp2}_OH_{genotype}"
                if snp2_col in inputs.columns:
                    snp2_additive += inputs[snp2_col] * genotype

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


class FailedMergeOperateAndSplit:
    pass


def _merge_operate_and_split(
    model_ready_object: ModelReadyObject,
    function: Callable[
        [pd.DataFrame, pd.DataFrame],
        tuple[pd.DataFrame, pd.DataFrame],
    ],
) -> ModelReadyObject | FailedMergeOperateAndSplit:
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

    try:
        df_all_inputs, df_all_targets = function(df_all_inputs, df_all_targets)
    except Exception as e:
        logger.error("Error when applying function to model ready object: %s", e)
        return FailedMergeOperateAndSplit()

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
