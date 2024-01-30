import warnings
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, Generator

import numpy as np
import pandas as pd
import xgboost
from eir.train_utils import metrics as eir_metrics
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_complexity_analysis import ModelReadyObject

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass()
class TrainEvalResults:
    df_predictions: pd.DataFrame
    df_predictions_raw: pd.DataFrame
    performance: pd.DataFrame


def train_and_evaluate_xboost(
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
    )

    results = TrainEvalResults(
        df_predictions=df_predictions,
        df_predictions_raw=df_predictions_raw,
        performance=df_performance,
    )

    return results


def train_and_evaluate_linear(
    modelling_data: "ModelReadyObject",
    target_type: str,
) -> TrainEvalResults:
    if target_type not in ["classification", "regression"]:
        raise ValueError("target_type must be 'classification' or 'regression'")

    x_train = pd.concat([modelling_data.input_train, modelling_data.input_val]).values
    y_train = pd.concat([modelling_data.target_train, modelling_data.target_val]).values

    if target_type == "classification":
        model = LogisticRegressionCV(
            cv=5,
            random_state=0,
            max_iter=1000,
        )
    else:
        model = ElasticNetCV(
            eps=1e-7,
            cv=5,
            random_state=0,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            tol=1e-03,
            selection="random",
        )

    model.fit(X=x_train, y=y_train.ravel())

    x_test = modelling_data.input_test.values
    y_test = modelling_data.target_test.values

    if target_type == "classification":
        y_score = model.predict_proba(x_test)
    else:
        y_score = model.predict(x_test)
        y_score = y_score.reshape(-1, 1)

    performance = evaluate_model_performance(
        y_true=y_test,
        y_score=y_score,
        task_type=target_type,
    )

    df_predictions, df_predictions_raw = parse_predictions(
        modelling_data=modelling_data,
        y_test=y_test,
        y_score=y_score,
    )
    df_performance = pd.DataFrame.from_dict(data=performance, orient="index").T

    results = TrainEvalResults(
        df_predictions=df_predictions,
        df_predictions_raw=df_predictions_raw,
        performance=df_performance,
    )

    return results


def parse_predictions(
    modelling_data: "ModelReadyObject",
    y_test: np.ndarray,
    y_score: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_encoder = modelling_data.transformers.target_label_encoder
    if isinstance(target_encoder, LabelEncoder):
        class_names = target_encoder.classes_
        column_names = [f"Prediction_{class_name}" for class_name in class_names]
    else:
        column_names = ["Prediction"]

    df_predictions = pd.DataFrame(
        data=np.column_stack((y_test, y_score)),
        columns=["True Label"] + column_names,
        index=modelling_data.target_test.index,
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
        index=modelling_data.target_test.index,
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

    if task_type == "classification":
        params = {
            "eta": 0.02,
            "max_depth": 7,
            "objective": "multi:softprob",
            "subsample": 0.5,
            "booster": "gbtree",
            "num_class": num_classes,
        }
    elif task_type == "regression":
        params = {
            "eta": 0.02,
            "max_depth": 7,
            "objective": "reg:squarederror",
            "subsample": 0.5,
            "booster": "gbtree",
        }
    else:
        raise ValueError()

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
