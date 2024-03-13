from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from eir_auto_gp.post_analysis import run_post_analysis
from eir_auto_gp.single_task.run_single_task import (
    get_argument_parser,
    run,
    store_experiment_config,
)


def _get_test_modelling_cl_command(
    folder_path: Path, target_type: str, feature_selection: str
) -> str:
    base = (
        f"--genotype_data_path {folder_path}/ "
        f"--label_file_path {folder_path}/phenotype.csv "
        "--global_output_folder runs/simulated_test "
        f"--output_{target_type}_columns phenotype "
        "--folds 2 "
        f"--feature_selection {feature_selection} "
        "--n_dl_feature_selection_setup_folds 1 "
        "--do_test "
        "--gwas_p_value_threshold 1e-01 "
    )

    return base


@pytest.mark.parametrize(
    "feature_selection",
    [
        "dl",
        "gwas",
        "gwas->dl",
        "gwas+bo",
    ],
)
def test_post_analysis_classification(
    feature_selection: str,
    simulate_genetic_data_to_bed: Callable[[int, int, str], Path],
    tmp_path: Path,
) -> None:
    simulated_path = simulate_genetic_data_to_bed(10000, 12, "binary")

    command = _get_test_modelling_cl_command(
        folder_path=simulated_path,
        target_type="cat",
        feature_selection=feature_selection,
    )

    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)
    store_experiment_config(cl_args=cl_args)
    run(cl_args=cl_args)

    post_analysis_parser = run_post_analysis.get_argument_parser()
    post_command = f"--run_dir {tmp_path} --top_n_snps 128"
    cl_args_post = post_analysis_parser.parse_args(post_command.split())
    post_analysis_object = run_post_analysis.build_post_analysis_object(
        cl_args=cl_args_post
    )
    run_post_analysis.run_complexity_analysis(post_analysis_object=post_analysis_object)
    run_post_analysis.run_effect_analysis(post_analysis_object=post_analysis_object)

    post_analysis_folder = tmp_path / "analysis" / "post_analysis"
    _check_post_analysis_results_wrapper(
        post_analysis_folder=post_analysis_folder,
        include_tabular=False,
        check_effects=True,
        regression_type="logistic",
    )


@pytest.mark.parametrize(
    "feature_selection",
    [
        "dl",
        "gwas",
        "gwas->dl",
        "gwas+bo",
    ],
)
def test_post_analysis_regression(
    feature_selection: str,
    simulate_genetic_data_to_bed: Callable[[int, int, str], Path],
    tmp_path: Path,
) -> None:
    simulated_path = simulate_genetic_data_to_bed(10000, 12, "continuous")

    command = _get_test_modelling_cl_command(
        folder_path=simulated_path,
        target_type="con",
        feature_selection=feature_selection,
    )

    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)
    store_experiment_config(cl_args=cl_args)
    run(cl_args=cl_args)

    post_analysis_parser = run_post_analysis.get_argument_parser()
    post_command = f"--run_dir {tmp_path} --top_n_snps 128"
    cl_args_post = post_analysis_parser.parse_args(post_command.split())
    post_analysis_object = run_post_analysis.build_post_analysis_object(
        cl_args=cl_args_post
    )
    run_post_analysis.run_complexity_analysis(post_analysis_object=post_analysis_object)
    run_post_analysis.run_effect_analysis(post_analysis_object=post_analysis_object)

    post_analysis_folder = tmp_path / "analysis" / "post_analysis"
    _check_post_analysis_results_wrapper(
        post_analysis_folder=post_analysis_folder,
        include_tabular=False,
        check_effects=True,
        regression_type="linear",
    )


def _check_post_analysis_results_wrapper(
    post_analysis_folder: Path,
    include_tabular: bool,
    check_effects: bool,
    regression_type: str,
) -> None:
    _check_complexity_analysis_results(
        complexity_folder=post_analysis_folder / "complexity",
        include_tabular=include_tabular,
    )

    if check_effects:
        effects_folder = post_analysis_folder / "effect_analysis"
        _check_effect_analysis_results(
            effects_folder=effects_folder,
            regression_type=regression_type,
        )


def _check_complexity_analysis_results(
    complexity_folder: Path,
    include_tabular: bool,
) -> None:
    expected_runs = 8 if include_tabular else 4
    df = pd.read_csv(complexity_folder / "all_results.csv")

    assert len(df) == expected_runs

    _check_xgboost_better_than_linear(df=df)
    _check_one_hot_better_in_linear(df=df)

    predictions_folder = complexity_folder / "predictions"
    assert len(list(predictions_folder.glob("numerical/*.csv"))) == expected_runs
    assert len(list(predictions_folder.glob("raw/*.csv"))) == expected_runs


def _check_xgboost_better_than_linear(df: pd.DataFrame) -> None:
    avg_perf_xgboost = df[df["model_type"] == "xgboost"]["average_performance"].mean()
    avg_perf_linear = df[df["model_type"] == "linear"]["average_performance"].mean()
    assert avg_perf_xgboost > avg_perf_linear


def _check_one_hot_better_in_linear(df: pd.DataFrame) -> None:
    df_linear = df[df["model_type"] == "linear"]

    df_linear_one_hot = df_linear[df_linear["one_hot_encode"]]

    df_linear_no_one_hot = df_linear[~df_linear["one_hot_encode"]]

    avg_perf_linear_one_hot = df_linear_one_hot["average_performance"].mean()
    avg_perf_linear_no_one_hot = df_linear_no_one_hot["average_performance"].mean()

    assert avg_perf_linear_one_hot > avg_perf_linear_no_one_hot


def _check_effect_analysis_results(effects_folder: Path, regression_type: str) -> None:
    df_allele_effects = pd.read_csv(
        effects_folder / "allele_effects" / "allele_effects.csv"
    )
    _check_allele_effects(
        df_allele_effects=df_allele_effects, regression_type=regression_type
    )

    df_interaction_effects = pd.read_csv(
        effects_folder / "interaction_effects" / "interaction_effects.csv"
    )
    _check_interaction_effects(df_interactions=df_interaction_effects)


def _check_allele_effects(
    df_allele_effects: pd.DataFrame, regression_type: str
) -> None:
    assert regression_type in ("logistic", "linear")

    df_allele_effects["SNP"] = df_allele_effects["allele"].str.split(" ").str[0]
    _check_basic_snps_significant_p_values(df=df_allele_effects)
    _check_additive_coefficients(df=df_allele_effects, regression_type=regression_type)
    _check_dominant_coefficients(df=df_allele_effects)
    _check_recessive_coefficients(df=df_allele_effects)


def _check_basic_snps_significant_p_values(df: pd.DataFrame) -> None:
    for snp in range(1, 7):
        df_snp = df[df["SNP"] == f"snp{snp}"]
        assert (df_snp["p_value"] < 5e-8).sum() >= 2


def _check_additive_coefficients(df: pd.DataFrame, regression_type: str) -> None:
    for snp in [1, 2]:
        snp_df = df[df["allele"].str.startswith(f"snp{snp}")]
        second_row_coef = snp_df.iloc[1]["Coefficient"]
        third_row_coef = snp_df.iloc[2]["Coefficient"]

        if regression_type == "logistic":
            second_row_coef = np.exp(second_row_coef)
            third_row_coef = np.exp(third_row_coef)

        ratio = abs(third_row_coef - 2 * second_row_coef) / (2 * second_row_coef)
        is_close = ratio < 0.2

        msg = (
            f"SNP{snp}: 2nd row = {second_row_coef}, "
            f"3rd row = {third_row_coef}, ratio = {ratio}"
        )

        assert is_close, msg


def _check_dominant_coefficients(df: pd.DataFrame) -> None:
    df_snp = df[df["SNP"] == "snp3"]
    second_row_coef = df_snp.iloc[1]["Coefficient"]
    third_row_coef = df_snp.iloc[2]["Coefficient"]
    difference = abs(third_row_coef - second_row_coef)

    msg = f"SNP3: 2nd row = {second_row_coef}, 3rd row = {third_row_coef}"
    assert difference < 1.5, msg


def _check_recessive_coefficients(df: pd.DataFrame) -> None:
    df_snp = df[df["SNP"] == "snp4"]

    second_row_coef = df_snp.iloc[1]["Coefficient"]
    third_row_coef = df_snp.iloc[2]["Coefficient"]

    msg1 = f"SNP4: 2nd row = {second_row_coef}"
    assert abs(second_row_coef) < 1.1, msg1

    msg2 = f"SNP4: 2nd row = {second_row_coef}, 3rd row = {third_row_coef}"
    assert third_row_coef > second_row_coef, msg2


def _check_interaction_effects(df_interactions: pd.DataFrame) -> None:
    df_interaction_terms = df_interactions[
        df_interactions["allele"].str.contains("--:--")
    ]
    max_interaction_coefficient = df_interaction_terms.loc[
        df_interaction_terms["Coefficient"].idxmax(),
        "KEY",
    ]
    assert max_interaction_coefficient == "snp5--:--snp6"
