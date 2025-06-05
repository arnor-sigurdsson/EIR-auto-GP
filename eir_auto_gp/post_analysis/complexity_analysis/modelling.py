import warnings
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, Generator, Literal, Optional, Union

import numpy as np
import pandas as pd
import xgboost
from aislib.misc_utils import get_logger
from eir.train_utils import metrics as eir_metrics
from scipy import stats
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeClassifierCV,
    RidgeCV,
)
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_complexity_analysis import ModelReadyObject

al_linear_models = (
    ElasticNetCV
    | LassoCV
    | LinearRegression
    | LogisticRegression
    | LogisticRegressionCV
    | RidgeClassifierCV
    | RidgeCV
)

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
    eval_set: Literal["valid", "test"],
) -> TrainEvalResults:

    assert target_type in {"classification", "regression"}
    assert eval_set in {"valid", "test"}

    df_target = modelling_data.target_train
    num_classes = df_target[df_target.columns[0]].nunique()
    params = get_xgboost_params(task_type=target_type, num_classes=num_classes)

    x_train = modelling_data.input_train.values
    y_train = modelling_data.target_train.values
    d_train = xgboost.DMatrix(data=x_train, label=y_train)

    x_val = modelling_data.input_val.values
    y_val = modelling_data.target_val.values
    d_val = xgboost.DMatrix(data=x_val, label=y_val)

    x_test = modelling_data.input_test.values
    y_test = modelling_data.target_test.values
    d_test = xgboost.DMatrix(data=x_test, label=y_test)

    eval_data = d_test if eval_set == "test" else d_val
    eval_labels = y_test if eval_set == "test" else y_val

    watchlist = [(d_train, "train"), (eval_data, "eval")]
    trained_model = xgboost.train(
        params=params,
        dtrain=d_train,
        num_boost_round=10000,
        evals=watchlist,
        verbose_eval=100,
        early_stopping_rounds=100,
    )

    y_score = trained_model.predict(data=eval_data, output_margin=True)
    if target_type == "regression":
        y_score = y_score.reshape(-1, 1)

    performance = evaluate_model_performance(
        y_true=eval_labels,
        y_score=y_score,
        task_type=target_type,
    )
    df_performance = pd.DataFrame.from_dict(data=performance, orient="index").T

    df_predictions, df_predictions_raw = parse_predictions(
        modelling_data=modelling_data,
        y_test=eval_labels,
        y_score=y_score,
        eval_set=eval_set,
    )

    feature_names = modelling_data.input_train.columns
    booster_fi = trained_model.get_score(importance_type="gain")
    feature_importance = pd.DataFrame(
        [(feature_names[int(k[1:])], v) for k, v in booster_fi.items()],
        columns=["Feature", "Importance"],
    ).sort_values(by="Importance", ascending=False)

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
    eval_set: str,
    cv_use_val_split: bool,
    model_type: str = "linear_model",
    with_fallback: bool = False,
) -> TrainEvalResults:
    if target_type not in ["classification", "regression"]:
        raise ValueError("target_type must be 'classification' or 'regression'")

    if eval_set not in ["test", "valid"]:
        raise ValueError("eval_set must be 'test' or 'valid'")

    x_train, y_train = _get_training_data(
        modelling_data=modelling_data,
        model_type=model_type,
    )

    cv = _get_cv_split(
        modelling_data=modelling_data,
        cv_use_val_split=cv_use_val_split,
    )

    model = _initialize_linear_model(
        target_type=target_type,
        cv=cv,
        model_type=model_type,
    )

    try:
        model.fit(X=x_train, y=y_train.ravel())
    except Exception as e:
        logger.error(f"Encountered error when fitting simple linear model: {e}")
        if with_fallback:
            logger.warning("Falling back to ElasticNetCV.")
            model = _initialize_fallback_model(target_type=target_type, cv=cv)
            model.fit(X=x_train, y=y_train.ravel())
        else:
            raise e

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

    y_score_train = _predict_with_linear(
        model=model,
        x_eval=x_train,
        target_type=target_type,
    )
    df_feature_importance = _extract_linear_feature_importance(
        model=model,
        feature_names=feature_names,
        x=x_train,
        y=y_train,
        model_predictions=y_score_train,
        target_type=target_type,
        model_type=model_type,
    )

    results = TrainEvalResults(
        df_predictions=df_predictions,
        df_predictions_raw=df_predictions_raw,
        performance=df_performance,
        feature_importance=df_feature_importance,
    )

    return results


def _validate_linear_training_data(modelling_data: "ModelReadyObject") -> None:
    input_train = modelling_data.input_train
    input_val = modelling_data.input_val
    input_test = modelling_data.input_test

    names = ["input_train", "input_val", "input_test"]

    for df, name in zip([input_train, input_val, input_test], names):
        _check_and_report_values(df=df, df_name=name)


def _check_and_report_values(df: pd.DataFrame, df_name: str) -> None:
    nan_info = df.isna().sum()
    nan_info_as_dict = nan_info.to_dict()
    has_nan = nan_info.any()

    inf_info = df.replace([np.inf, -np.inf], np.nan).isna().sum() - nan_info
    inf_info_as_dict = inf_info.to_dict()
    has_inf = inf_info.any()

    if has_nan or has_inf:
        for column, nan_count in nan_info_as_dict.items():
            if nan_count > 0:
                logger.error(
                    f"Found NaN in column '{column}': {nan_count} entries in {df_name}."
                )
        for column, inf_count in inf_info_as_dict.items():
            if inf_count > 0:
                logger.error(
                    f"Found Inf in column '{column}': {inf_count} entries in {df_name}."
                )

        error_message = (
            "Data contains invalid values (NaN/Inf), "
            "which are not permitted. "
            "This is a bug as they should have been imputed "
            "or managed already."
        )
        raise AssertionError(error_message)


def _get_training_data(modelling_data: "ModelReadyObject", model_type: str) -> tuple:

    _validate_linear_training_data(modelling_data=modelling_data)

    if model_type == "linear_model":
        x_train = modelling_data.input_train.values
        y_train = modelling_data.target_train.values
    else:
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
    modelling_data: "ModelReadyObject",
    eval_set: str,
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


def _initialize_linear_model(
    target_type: str,
    cv: Union[int, PredefinedSplit],
    model_type: str,
) -> Any:

    alphas = np.logspace(-4, 1, 50)

    if target_type == "classification":
        if model_type == "linear_model":
            return LogisticRegression(solver="saga", n_jobs=-1)
        elif model_type == "ridge":
            return RidgeClassifierCV(cv=cv, alphas=alphas)
        elif model_type == "lasso":
            return LogisticRegressionCV(
                cv=cv,
                penalty="l1",
                solver="saga",
                n_jobs=-1,
            )
        elif model_type == "elasticnet":
            return LogisticRegressionCV(
                cv=cv,
                penalty="elasticnet",
                Cs=10,
                l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                solver="saga",
                n_jobs=-1,
            )
        else:
            raise ValueError("Invalid model type for classification.")

    elif target_type == "regression":
        if model_type == "linear_model":
            return LinearRegression()
        elif model_type == "ridge":
            return RidgeCV(
                cv=cv,
                alphas=alphas,
            )
        elif model_type == "lasso":
            return LassoCV(
                cv=cv,
                alphas=alphas,
            )
        elif model_type == "elasticnet":
            return ElasticNetCV(
                cv=cv,
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                alphas=alphas,
            )
        else:
            raise ValueError("Invalid model type for regression.")

    else:
        raise ValueError("Invalid target type.")


def _initialize_fallback_model(
    target_type: str, cv: int | PredefinedSplit
) -> ElasticNetCV | LogisticRegressionCV:
    if target_type == "classification":
        logger.warning("Falling back to LogisticRegressionCV.")
        model = LogisticRegressionCV(
            cv=cv,
            random_state=0,
            max_iter=1000,
            class_weight="balanced",
            scoring="roc_auc",
            solver="saga",
            penalty="elasticnet",
            Cs=10,
            l1_ratios=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_jobs=-1,
        )
    elif target_type == "regression":
        logger.warning("Falling back to ElasticNetCV.")
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            eps=1e-5,
            n_alphas=100,
            cv=cv,
            tol=1e-4,
            selection="cyclic",
            random_state=0,
            n_jobs=-1,
        )
    else:
        raise ValueError()

    return model


def _predict_with_linear(
    model: Union[LogisticRegressionCV, ElasticNetCV],
    x_eval: np.ndarray,
    target_type: str,
) -> np.ndarray:

    if target_type == "classification":
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x_eval)
        elif hasattr(model, "_predict_proba_lr"):
            return model._predict_proba_lr(x_eval)
        else:
            raise ValueError()

    elif target_type == "regression":
        return model.predict(x_eval).reshape(-1, 1)
    else:
        raise ValueError("Invalid target type.")


def _extract_linear_feature_importance(
    model: al_linear_models,
    x: np.ndarray,
    y: np.ndarray,
    model_predictions: np.ndarray,
    feature_names: list[str],
    target_type: str,
    model_type: str,
) -> pd.DataFrame:
    """
    Adapted from https://stackoverflow.com/questions/27928275
    """
    if target_type == "classification":
        coef = model.coef_
        feature_importance = abs(model.coef_)
        model_predictions = model_predictions[:, 1]
    elif target_type == "regression":
        coef = model.coef_
        feature_importance = abs(model.coef_)
        model_predictions = model_predictions.ravel()
    else:
        raise ValueError()

    dataframe_entries = {
        "Feature": ["Intercept"] + feature_names,
        "Coefficient": np.append(model.intercept_, coef),
        "Importance": np.append(abs(model.intercept_), feature_importance),
    }

    if model_type == "linear_model":

        x = x.astype(float)
        residuals = y.ravel() - model_predictions
        mse = np.sum(residuals**2) / (len(x) - len(feature_names) - 1)

        new_x = np.append(np.ones((len(x), 1)), x, axis=1)

        used_pinv = False
        try:
            var_b = mse * (np.linalg.inv(np.dot(new_x.T, new_x)).diagonal())
        except np.linalg.LinAlgError:
            logger.warning(
                "Singular matrix encountered when calculating standard errors, "
                "falling back to using pseudo-inverse. "
                "This may lead to inaccurate results."
            )
            var_b = mse * (np.linalg.pinv(np.dot(new_x.T, new_x)).diagonal())
            used_pinv = True

        se_b = np.sqrt(var_b)

        coefficients = np.append(model.intercept_, coef)
        t_values = coefficients / se_b
        p_values = 2 * (1 - stats.t.cdf(abs(t_values), len(x) - len(feature_names) - 1))

        ci_lower = coefficients - 1.96 * se_b
        ci_upper = coefficients + 1.96 * se_b

        dataframe_entries.update(
            {
                "Standard Error": se_b,
                "t-Value": t_values,
                "P-Value": p_values,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "Used Pseudo-Inverse": [used_pinv] + [used_pinv] * len(feature_names),
            }
        )

    return pd.DataFrame(dataframe_entries).sort_values(by="Importance", ascending=False)


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
        ["xgboost", "linear_model", "ridge", "lasso", "elasticnet"],
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
