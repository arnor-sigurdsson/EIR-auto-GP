from pathlib import Path
import json
from statistics import mean

import pytest
import pandas as pd
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.run import get_argument_parser, run


def _get_test_cl_commands() -> list[str]:
    base = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--global_output_folder runs/penncath "
        "--output_cat_columns CAD "
        "--input_con_columns tg hdl ldl age "
        "--input_cat_columns sex "
        "--folds 0-2 "
        "--do_test"
    )

    feature_selections = [
        "",
        "--feature_selection gwas ",
        "--feature_selection dl --n_dl_feature_selection_setup_folds 1 ",
        "--feature_selection gwas->dl --n_dl_feature_selection_setup_folds 1 ",
        "--feature_selection dl+gwas --n_dl_feature_selection_setup_folds 1 ",
    ]

    commands = []

    for feature_selection in feature_selections:
        commands.append(f"{base} {feature_selection}")

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
        check_modelling_results(run_folder=modelling_run, check_test=check_test)


def check_modelling_results(run_folder: Path, check_test: bool) -> None:
    for file in ["validation_average_history.log"]:
        check_average_performances(file_path=run_folder / file)

    assert (run_folder / "completed_train.txt").exists()

    if check_test:
        test_folder = run_folder / "test_set_predictions"
        assert (test_folder / "test_complete.txt").exists()

        metrics_file = test_folder / "calculated_metrics.json"
        assert metrics_file.exists()

        metrics = json.loads(metrics_file.read_text())
        metric_values = find_numeric_values(input_dict=metrics, accumulator=[])
        avg = mean(metric_values)
        assert avg >= 0.5


def check_average_performances(file_path: Path, threshold: float = 0.6) -> None:
    df = pd.read_csv(filepath_or_buffer=file_path)

    assert df["perf-average"].max() >= threshold


def find_numeric_values(input_dict: dict, accumulator: list) -> list[float | int]:
    for key, value in input_dict.items():
        match value:
            case dict(value):
                find_numeric_values(input_dict=value, accumulator=accumulator)
            case int(value) | float(value):
                accumulator.append(value)

    return accumulator
