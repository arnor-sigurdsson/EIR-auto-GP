from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import seaborn as sns
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.common.data_preparation import (
    FitTransformers,
    ModelData,
    SplitModelData,
)
from eir_auto_gp.post_analysis.complexity_analysis.modelling import (
    get_training_eval_iterator,
    train_and_evaluate_linear,
    train_and_evaluate_xgboost,
)
from eir_auto_gp.post_analysis.complexity_analysis.viz import plot_performance

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_post_analysis import PostAnalysisObject

sns.set_theme(style="whitegrid")
sns.set(font_scale=1.2)

logger = get_logger(name=__name__)


def run_complexity_analysis(post_analysis_object: "PostAnalysisObject") -> None:
    pao = post_analysis_object

    eval_sets = ("valid", "test")
    for eval_set in eval_sets:
        eval_set: Literal["valid", "test"]
        df_results = train_and_evaluate_wrapper(
            analysis_object=post_analysis_object,
            eval_set=eval_set,
        )
        metrics_to_plot = _get_metric_columns(
            target_type=pao.experiment_info.target_type
        )
        metrics_to_plot_w_average = metrics_to_plot + ["average_performance"]

        plot_output_root = (
            pao.data_paths.analysis_output_path / f"complexity/{eval_set}/plots"
        )
        for metric in metrics_to_plot_w_average:
            plot_performance(
                df=df_results,
                output_path=plot_output_root,
                metric=metric,
            )


def train_and_evaluate_wrapper(
    analysis_object: "PostAnalysisObject",
    eval_set: Literal["valid", "test"],
) -> pd.DataFrame:
    all_results = []

    any_tabular = len(analysis_object.experiment_info.all_input_columns) > 0
    for conditions in get_training_eval_iterator(any_tabular_input=any_tabular):
        mro = convert_split_data_to_model_ready_object(
            split_model_data=analysis_object.modelling_data,
            include_genotype=conditions["include_genotype"],
            include_tabular=conditions["include_tabular"],
            one_hot_encode=conditions["one_hot_encode"],
            one_hot_drop_first=True,
        )
        combination_string = "_".join(
            f"{k}={v}" for k, v in conditions.items() if k != "one_hot_drop_first"
        )

        if conditions["model_type"] == "xgboost":
            logger.info(f"Running XGBoost with {combination_string}.")
            train_eval_results = train_and_evaluate_xgboost(
                modelling_data=mro,
                target_type=analysis_object.experiment_info.target_type,
                eval_set=eval_set,
            )
        else:
            logger.info(f"Running linear model with {combination_string}.")
            model_type = conditions["model_type"]
            train_eval_results = train_and_evaluate_linear(
                modelling_data=mro,
                target_type=analysis_object.experiment_info.target_type,
                eval_set=eval_set,
                cv_use_val_split=True,
                model_type=model_type,
            )

        df_conditions = pd.DataFrame([conditions])
        df_result = pd.concat(
            objs=[df_conditions, train_eval_results.performance],
            axis=1,
        )

        result = {
            "conditions": conditions,
            "df_predictions": train_eval_results.df_predictions,
            "df_predictions_raw": train_eval_results.df_predictions_raw,
            "performance": df_result,
            "importance": train_eval_results.feature_importance,
        }

        all_results.append(result)

    df_all_results = process_results(
        all_results=all_results,
        target_type=analysis_object.experiment_info.target_type,
        analysis_output_path=analysis_object.data_paths.analysis_output_path,
        eval_set=eval_set,
    )

    return df_all_results


def process_results(
    all_results: list,
    target_type: str,
    analysis_output_path: Path,
    eval_set: Literal["valid", "test"],
) -> pd.DataFrame:
    df_all_results = pd.concat(
        [r["performance"] for r in all_results],
        ignore_index=True,
    )
    numerical_columns = _get_metric_columns(target_type=target_type)

    average = df_all_results[numerical_columns].mean(axis=1)
    df_all_results["average_performance"] = average

    output_root = analysis_output_path / "complexity" / eval_set
    ensure_path_exists(path=output_root, is_folder=True)
    df_all_results.to_csv(output_root / "all_results.csv", index=False)

    for i, result in enumerate(all_results):
        condition_str = "_".join(f"{k}={v}" for k, v in result["conditions"].items())
        file_name = f"predictions_{i}_{condition_str}.csv"

        numerical_output_path = output_root / "predictions/numerical" / file_name
        ensure_path_exists(path=numerical_output_path, is_folder=False)
        result["df_predictions"].to_csv(numerical_output_path)

        raw_output_path = output_root / "predictions/raw" / file_name
        ensure_path_exists(path=raw_output_path, is_folder=False)
        result["df_predictions_raw"].to_csv(raw_output_path)

        importance_output_path = output_root.parent / "importance" / file_name
        ensure_path_exists(path=importance_output_path, is_folder=False)
        if not importance_output_path.exists():
            result["importance"].to_csv(importance_output_path)

    return df_all_results


def _get_metric_columns(target_type: str) -> list[str]:
    if target_type == "classification":
        return ["acc", "mcc", "roc_auc", "ap"]
    elif target_type == "regression":
        return ["r2", "pcc"]
    else:
        raise ValueError()


@dataclass()
class ModelReadyObject:
    input_train: pd.DataFrame
    target_train: pd.DataFrame

    input_val: pd.DataFrame
    target_val: pd.DataFrame

    input_test: pd.DataFrame
    target_test: pd.DataFrame

    transformers: FitTransformers


def convert_split_data_to_model_ready_object(
    split_model_data: "SplitModelData",
    include_genotype: bool = True,
    include_tabular: bool = True,
    one_hot_encode: bool = False,
    one_hot_drop_first: bool = False,
) -> ModelReadyObject:
    genotype_data_combined = pd.concat(
        [
            split_model_data.train.df_genotype_input,
            split_model_data.val.df_genotype_input,
            split_model_data.test.df_genotype_input,
        ]
    )

    if one_hot_encode:
        genotype_data_combined = pd.get_dummies(
            genotype_data_combined.astype(str),
            drop_first=one_hot_drop_first,
        )

    def get_input_data(model_data: ModelData) -> pd.DataFrame:
        data_to_include = []

        if include_genotype:
            genotype_data = genotype_data_combined.loc[
                model_data.df_genotype_input.index
            ]
            data_to_include.append(genotype_data)

        if include_tabular:
            data_to_include.append(model_data.df_tabular_input)

        return pd.concat(data_to_include, axis=1)

    input_train = get_input_data(model_data=split_model_data.train)
    target_train = split_model_data.train.df_target

    input_val = get_input_data(model_data=split_model_data.val)
    target_val = split_model_data.val.df_target

    input_test = get_input_data(model_data=split_model_data.test)
    target_test = split_model_data.test.df_target

    return ModelReadyObject(
        input_train=input_train,
        target_train=target_train,
        input_val=input_val,
        target_val=target_val,
        input_test=input_test,
        target_test=target_test,
        transformers=split_model_data.transformers,
    )
