import warnings
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

import numpy as np
import pandas as pd
import xgboost
from aislib.misc_utils import get_logger
from eir.train_utils import metrics as eir_metrics
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNetCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_complexity_analysis import ModelReadyObject

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = get_logger(name=__name__)


@dataclass()
class TrainEvalResults:
    df_predictions: pd.DataFrame
    df_predictions_raw: pd.DataFrame
    performance: pd.DataFrame
    feature_importance: Optional[pd.DataFrame]


def train_and_evaluate_xgboost(
    modelling_data: "ModelReadyObject",
    target_type: str,
) -> TrainEvalResults:
    df_target = modelling_data.target_train
    num_classes = df_target[df_target.columns[0]].nunique()
    params = get_xgboost_params(task_type=target_type, num_classes=num_classes)

    x_train = modelling_data.input_train.values
    y_train = modelling_data.target_train.values
    x_val = modelling_data.input_val.values
    y_val = modelling_data.target_val.values

    d_train = xgboost.DMatrix(data=x_train, label=y_train)
    d_val = xgboost.DMatrix(data=x_val, label=y_val)

    watchlist = [(d_train, "train"), (d_val, "eval")]
    trained_model = xgboost.train(
        params=params,
        dtrain=d_train,
        num_boost_round=10000,
        evals=watchlist,
        verbose_eval=100,
        early_stopping_rounds=100,
    )

    x_test = modelling_data.input_test.values
    y_test = modelling_data.target_test.values

    d_test = xgboost.DMatrix(data=x_test, label=y_test)
    y_score = trained_model.predict(data=d_test, output_margin=True)
    if target_type == "regression":
        y_score = y_score.reshape(-1, 1)

    performance = evaluate_model_performance(
        y_true=y_test,
        y_score=y_score,
        task_type=target_type,
    )
    df_performance = pd.DataFrame.from_dict(data=performance, orient="index").T

    df_predictions, df_predictions_raw = parse_predictions(
        modelling_data=modelling_data,
        y_test=y_test,
        y_score=y_score,
        eval_set="test",
    )

    feature_importance = pd.DataFrame(
        data=trained_model.get_score(importance_type="gain").items(),
        columns=["Feature", "Importance"],
    )

    results = TrainEvalResults(
        df_predictions=df_predictions,
        df_predictions_raw=df_predictions_raw,
        performance=df_performance,
        feature_importance=feature_importance,
    )

    return results


def train_and_evaluate_linear(
    modelling_data: "ModelReadyObject",
    target_type: str,
    eval_set: str = "test",
    cv_use_val_split: bool = False,
) -> TrainEvalResults:
    if target_type not in ["classification", "regression"]:
        raise ValueError("target_type must be 'classification' or 'regression'")

    if eval_set not in ["test", "valid"]:
        raise ValueError("eval_set must be 'test' or 'valid'")

    x_train, y_train = _select_training_data(
        modelling_data=modelling_data,
        eval_set=eval_set,
    )

    cv = _get_cv_split(
        modelling_data=modelling_data,
        cv_use_val_split=cv_use_val_split,
    )

    model = _initialize_linear_model(target_type=target_type, cv=cv)

    model.fit(X=x_train, y=y_train.ravel())

    x_eval, y_eval = _select_evaluation_set(
        modelling_data=modelling_data,
        eval_set=eval_set,
    )

    y_score = _predict_with_linear(
        model=model,
        x_eval=x_eval,
        target_type=target_type,
    )

    performance = evaluate_model_performance(
        y_true=y_eval,
        y_score=y_score,
        task_type=target_type,
    )

    df_predictions, df_predictions_raw = parse_predictions(
        modelling_data=modelling_data,
        y_test=y_eval,
        y_score=y_score,
        eval_set=eval_set,
    )
    df_performance = pd.DataFrame.from_dict(data=performance, orient="index").T

    feature_names = modelling_data.input_train.columns.tolist()

    df_feature_importance = _extract_linear_feature_importance(
        model=model,
        feature_names=feature_names,
        target_type=target_type,
    )

    results = TrainEvalResults(
        df_predictions=df_predictions,
        df_predictions_raw=df_predictions_raw,
        performance=df_performance,
        feature_importance=df_feature_importance,
    )

    return results


def _select_training_data(modelling_data: "ModelReadyObject", eval_set: str) -> tuple:
    if eval_set == "valid":
        x_train = modelling_data.input_train.values
        y_train = modelling_data.target_train.values
    elif eval_set == "test":
        x_train = pd.concat(
            [
                modelling_data.input_train,
                modelling_data.input_val,
            ]
        ).values
        y_train = pd.concat(
            [
                modelling_data.target_train,
                modelling_data.target_val,
            ]
        ).values
    else:
        raise ValueError("eval_set must be 'test' or 'valid'")

    return x_train, y_train


def _get_cv_split(
    modelling_data: "ModelReadyObject", cv_use_val_split: bool
) -> int | PredefinedSplit:
    if cv_use_val_split:
        train_indices = [0] * len(modelling_data.input_train)
        val_indices = [-1] * len(modelling_data.input_val)
        test_fold = train_indices + val_indices

        cv = PredefinedSplit(test_fold=test_fold)
    else:
        cv = 10
    return cv


def _select_evaluation_set(
    modelling_data: "ModelReadyObject", eval_set: str
) -> tuple[np.ndarray, np.ndarray]:
    if eval_set == "test":
        x_eval = modelling_data.input_test.values
        y_eval = modelling_data.target_test.values
    elif eval_set == "valid":
        x_eval = modelling_data.input_val.values
        y_eval = modelling_data.target_val.values
    else:
        raise ValueError("eval_set must be 'test' or 'valid'")

    return x_eval, y_eval


def _initialize_linear_model(target_type: str, cv: int | PredefinedSplit) -> Any:
    if target_type == "classification":
        return LogisticRegression(
            solver="saga",
            n_jobs=-1,
        )
    elif target_type == "regression":
        return LinearRegression()
    else:
        raise ValueError()


def _predict_with_linear(
    model: LogisticRegressionCV | ElasticNetCV, x_eval: np.ndarray, target_type: str
) -> np.ndarray:
    if target_type == "classification":
        return model.predict_proba(x_eval)
    elif target_type == "regression":
        return model.predict(x_eval).reshape(-1, 1)
    else:
        raise ValueError()


def _extract_linear_feature_importance(
    model: LogisticRegressionCV | ElasticNetCV,
    feature_names: list[str],
    target_type: str,
) -> pd.DataFrame:
    if target_type == "classification":
        coef = model.coef_[0]
        feature_importance = abs(model.coef_[0])
    elif target_type == "regression":
        coef = model.coef_
        feature_importance = abs(model.coef_)
    else:
        raise ValueError()

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_importance,
            "Coef": coef,
        }
    ).sort_values(by="Importance", ascending=False)


def parse_predictions(
    modelling_data: "ModelReadyObject",
    y_test: np.ndarray,
    y_score: np.ndarray,
    eval_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_encoder = modelling_data.transformers.target_label_encoder
    if isinstance(target_encoder, LabelEncoder):
        class_names = target_encoder.classes_
        column_names = [f"Prediction_{class_name}" for class_name in class_names]
    else:
        column_names = ["Prediction"]

    if eval_set == "test":
        index = modelling_data.target_test.index
    elif eval_set == "valid":
        index = modelling_data.target_val.index
    else:
        raise ValueError()

    df_predictions = pd.DataFrame(
        data=np.column_stack((y_test, y_score)),
        columns=["True Label"] + column_names,
        index=index,
    )

    if isinstance(target_encoder, LabelEncoder):
        y_score_raw = np.argmax(y_score, axis=1)
        y_test_raw = target_encoder.inverse_transform(y_test.ravel())
        y_score_raw = target_encoder.inverse_transform(y_score_raw.ravel())
    else:
        y_score_raw = y_score
        y_test_raw = y_test

    df_predictions_raw = pd.DataFrame(
        data=np.column_stack((y_test_raw, y_score_raw)),
        columns=["True Label", "Prediction"],
        index=index,
    )

    return df_predictions, df_predictions_raw


def evaluate_model_performance(
    y_true: np.ndarray, y_score: np.ndarray, task_type: str
) -> dict[str, float]:
    performance: dict[str, float]
    if task_type == "classification":
        acc = eir_metrics.calc_acc(outputs=y_score, labels=y_true)
        mcc = eir_metrics.calc_mcc(outputs=y_score, labels=y_true)
        roc_auc = eir_metrics.calc_roc_auc_ovo(outputs=y_score, labels=y_true)
        ap = eir_metrics.calc_average_precision(outputs=y_score, labels=y_true)
        performance = {
            "acc": acc,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "ap": ap,
        }
    elif task_type == "regression":
        pcc = eir_metrics.calc_pcc(outputs=y_score, labels=y_true)
        r2 = eir_metrics.calc_r2(outputs=y_score, labels=y_true)
        rmse = metrics.root_mean_squared_error(y_true=y_true, y_pred=y_score)
        performance = {
            "pcc": pcc,
            "r2": r2,
            "rmse": rmse,
        }
    else:
        raise ValueError()

    return performance


def get_xgboost_params(task_type: str, num_classes: int) -> Dict[str, Any]:
    assert task_type in {"classification", "regression"}

    params = {
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 4,
        "lambda": 1,
        "alpha": 0.5,
        "booster": "gbtree",
    }

    if task_type == "classification":
        params.update(
            {
                "objective": "multi:softprob",
                "num_class": num_classes,
            }
        )
    elif task_type == "regression":
        params.update(
            {
                "objective": "reg:squarederror",
            }
        )

    return params


def get_training_eval_iterator(
    any_tabular_input: bool,
) -> Generator[Dict[str, Any], None, None]:
    condition_lists = [
        ["xgboost", "linear"],
        [True, False],
        [True, False],
        [True, False],
    ]

    iterator = product(*condition_lists)
    for model_type, include_genotype, include_tabular, one_hot_encode in iterator:
        if not include_genotype and not include_tabular:
            continue
        if not include_genotype and one_hot_encode:
            continue
        if include_tabular and not any_tabular_input:
            continue

        yield {
            "model_type": model_type,
            "include_genotype": include_genotype,
            "include_tabular": include_tabular,
            "one_hot_encode": one_hot_encode,
        }
