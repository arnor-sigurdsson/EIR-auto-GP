import json
from pathlib import Path
from statistics import mean

import pandas as pd
import pytest
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.multi_task.run_multi_task import (
    get_argument_parser,
    run,
    validate_column_duplicates,
)


def _get_test_cl_commands() -> list[str]:
    base = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--genotype_feature_selection random "
        "--global_output_folder runs/penncath "
        "--output_cat_columns CAD "
        "--output_con_columns tg hdl ldl "
        "--output_groups semirandom "
        "--input_con_columns age "
        "--input_cat_columns sex "
        "--folds 1 "
        "--model_size nano "
        "--modelling_data_format auto "
        "--do_test "
    )

    commands = []

    for data_storage_format in ("disk",):
        data_fmt_str = f"--data_storage_format {data_storage_format} "
        cur_command = base + data_fmt_str
        commands.append(cur_command)

    return commands


def _get_test_cl_commands_genotype_only() -> list[str]:
    base = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--genotype_feature_selection random "
        "--global_output_folder runs/penncath "
        "--output_cat_columns CAD "
        "--output_con_columns tg "
        "--output_groups semirandom "
        "--input_con_columns age "
        "--input_cat_columns sex "
        "--folds 1 "
        "--model_size nano "
        "--modelling_data_format auto "
        "--do_test "
        "--genotype_only_test "
    )

    commands = []

    for data_storage_format in ("disk",):
        data_fmt_str = f"--data_storage_format {data_storage_format} "
        cur_command = base + data_fmt_str
        commands.append(cur_command)

    return commands


@pytest.mark.parametrize("command", _get_test_cl_commands())
def test_modelling(command: str, tmp_path: Path) -> None:
    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)
    run(cl_args=cl_args)

    model_folder = tmp_path / "modelling"
    check_test = True if "do_test" in command else False
    for modelling_run in _iterdir_ignore_hidden(path=model_folder):
        if not modelling_run.name.startswith("fold_"):
            continue

        check_modelling_results(run_folder=modelling_run, check_test=check_test)


def check_modelling_results(run_folder: Path, check_test: bool) -> None:
    for file in ["validation_average_history.log"]:
        check_average_performances(file_path=run_folder / file, threshold=0.1)

    assert (run_folder / "completed_train.txt").exists()

    if check_test:
        test_folder = run_folder / "test_set_predictions"
        assert (test_folder / "test_complete.txt").exists()

        metrics_file = test_folder / "calculated_metrics.json"
        assert metrics_file.exists()

        metrics = json.loads(metrics_file.read_text())
        metric_values = find_numeric_values(input_dict=metrics, accumulator=[])
        avg = mean(metric_values)
        assert avg >= 0.1


def check_average_performances(file_path: Path, threshold: float = 0.5) -> None:
    df = pd.read_csv(filepath_or_buffer=file_path)

    assert df["perf-average"].max() >= threshold


def find_numeric_values(input_dict: dict, accumulator: list) -> list[float | int]:
    for _key, value in input_dict.items():
        match value:
            case dict(value):
                find_numeric_values(input_dict=value, accumulator=accumulator)
            case int(value) | float(value):
                accumulator.append(value)

    return accumulator


@pytest.mark.parametrize("command", _get_test_cl_commands_genotype_only())
def test_modelling_genotype_only_test(command: str, tmp_path: Path) -> None:
    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)
    run(cl_args=cl_args)

    model_folder = tmp_path / "modelling"
    check_test = True if "do_test" in command else False
    for modelling_run in _iterdir_ignore_hidden(path=model_folder):
        if not modelling_run.name.startswith("fold_"):
            continue

        check_modelling_results(run_folder=modelling_run, check_test=check_test)

        test_configs = (
            modelling_run / "test_set_predictions" / "test_predictor_0" / "configs"
        )
        if test_configs.exists():
            tabular_configs = list(test_configs.glob("*tabular*"))
            assert len(tabular_configs) == 0, (
                f"Found tabular configs in genotype-only test: {tabular_configs}"
            )


def test_train_with_tabular_predict_without(tmp_path: Path) -> None:
    command = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv "
        "--genotype_feature_selection random "
        "--global_output_folder runs/penncath "
        "--output_cat_columns CAD "
        "--output_con_columns tg "
        "--output_groups semirandom "
        "--input_con_columns age "
        "--input_cat_columns sex "
        "--folds 1 "
        "--model_size nano "
        "--modelling_data_format auto "
        "--data_storage_format disk "
        "--do_test "
        "--genotype_only_test "
    )

    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)
    run(cl_args=cl_args)

    model_folder = tmp_path / "modelling"
    run_folder = None
    for modelling_run in _iterdir_ignore_hidden(path=model_folder):
        if modelling_run.name.startswith("fold_"):
            run_folder = modelling_run
            break

    assert run_folder is not None
    assert (run_folder / "completed_train.txt").exists()

    test_folder = run_folder / "test_set_predictions"
    assert (test_folder / "test_complete.txt").exists()

    metrics_file = test_folder / "calculated_metrics.json"
    assert metrics_file.exists()

    metrics = json.loads(metrics_file.read_text())
    metric_values = find_numeric_values(input_dict=metrics, accumulator=[])
    avg = mean(metric_values)
    assert avg >= 0.1


def test_duplicate_column_validation() -> None:
    with pytest.raises(
        ValueError, match="Duplicate column names found in input categorical columns"
    ):
        validate_column_duplicates(
            input_cat_columns=["age", "sex", "age"],
            input_con_columns=["height"],
            output_cat_columns=["disease"],
            output_con_columns=["weight"],
        )

    with pytest.raises(
        ValueError, match="Duplicate column names found in input continuous columns"
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["height", "bmi", "height"],
            output_cat_columns=["disease"],
            output_con_columns=["weight"],
        )

    with pytest.raises(
        ValueError, match="Duplicate column names found in output categorical columns"
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["height"],
            output_cat_columns=["disease", "cancer", "disease"],
            output_con_columns=["weight"],
        )

    with pytest.raises(
        ValueError, match="Duplicate column names found in output continuous columns"
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["height"],
            output_cat_columns=["disease"],
            output_con_columns=["weight", "bmi", "weight"],
        )

    with pytest.raises(
        ValueError,
        match="Column names must be unique across all input and output columns",
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["height"],
            output_cat_columns=["age"],
            output_con_columns=["weight"],
        )

    with pytest.raises(
        ValueError,
        match="Column names must be unique across all input and output columns",
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["height"],
            output_cat_columns=["disease"],
            output_con_columns=["height"],
        )

    with pytest.raises(
        ValueError,
        match=r"Duplicate column names found in input categorical "
        r"columns.*case-insensitive",
    ):
        validate_column_duplicates(
            input_cat_columns=["HDL cholesterol", "HDL Cholesterol"],
            input_con_columns=["height"],
            output_cat_columns=["disease"],
            output_con_columns=["weight"],
        )

    with pytest.raises(
        ValueError,
        match=r"Column names must be unique across all input and "
        r"output columns.*case-insensitive",
    ):
        validate_column_duplicates(
            input_cat_columns=["age"],
            input_con_columns=["HDL cholesterol"],
            output_cat_columns=["disease"],
            output_con_columns=["hdl cholesterol"],
        )

    validate_column_duplicates(
        input_cat_columns=["age", "sex"],
        input_con_columns=["height", "bmi"],
        output_cat_columns=["disease", "cancer"],
        output_con_columns=["weight", "cholesterol"],
    )
