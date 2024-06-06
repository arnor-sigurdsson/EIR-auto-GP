from pathlib import Path

import pandas as pd
import seaborn as sns
import xgboost
from aislib.misc_utils import ensure_path_exists
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV


def assign_complexity(row):
    if row["include_tabular"] and not row["include_genotype"]:
        return 1
    elif (
        row["include_genotype"]
        and not row["include_tabular"]
        and not row["one_hot_encode"]
    ):
        return 2
    elif (
        row["include_genotype"] and not row["include_tabular"] and row["one_hot_encode"]
    ):
        return 3
    elif (
        row["include_genotype"] and row["include_tabular"] and not row["one_hot_encode"]
    ):
        return 4
    elif row["include_genotype"] and row["include_tabular"] and row["one_hot_encode"]:
        return 5


def plot_performance(
    df: pd.DataFrame,
    output_path: Path,
    metric: str,
) -> None:
    df = df.copy()
    df["complexity_level"] = df.apply(assign_complexity, axis=1)

    model_name_mapping = {
        "linear_model": "Linear Model",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "elasticnet": "ElasticNet",
        "xgboost": "XGBoost",
    }
    df["model_type"] = df["model_type"].map(model_name_mapping)

    plt.figure(figsize=(12, 8))

    hue_order = ["Linear Model", "Ridge", "Lasso", "ElasticNet", "XGBoost"]
    sns.barplot(
        x="complexity_level",
        y=metric,
        hue="model_type",
        hue_order=hue_order,
        data=df,
    )
    plt.title(metric.upper())

    if not df["include_tabular"].any():
        complexity_descriptions = {
            1: "Genotype (G)",
            2: "Genotype (G), OH",
        }
        ticks_range = range(1, 3)
    else:
        complexity_descriptions = {
            1: "Tabular (T)",
            2: "Genotype (G)",
            3: "Genotype (G), OH",
            4: "G + T",
            5: "G + T, OH",
        }
        ticks_range = range(1, 6)

    plt.xticks(
        ticks=[i - 1.0 for i in ticks_range],
        labels=[complexity_descriptions[i] for i in ticks_range],
    )

    plt.xlabel("")
    plt.title("")

    best_performance = df[metric].max()
    plt.axhline(
        best_performance,
        color="r",
        linestyle="--",
        linewidth=1,
    )

    plt.legend(title="Model Type")

    plt.tight_layout()
    ensure_path_exists(path=output_path, is_folder=True)
    plt.savefig(output_path / f"{metric}_performance.pdf")


def plot_xgboost_feature_importance(
    model: xgboost.Booster, output_folder: Path
) -> None:
    xgboost.plot_importance(model)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_folder / "xgboost_feature_importance.pdf")


def plot_linear_coefficients(
    model: LogisticRegressionCV | ElasticNetCV,
    feature_names: list,
    output_folder: Path,
) -> None:
    coefficients = pd.Series(model.coef_, index=feature_names)
    sns.barplot(x=coefficients.values, y=coefficients.index)
    plt.xlabel("Coefficients")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_folder / "linear_coefficients.pdf")
