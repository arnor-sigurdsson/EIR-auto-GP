import json
import subprocess
from pathlib import Path
from statistics import mean

import pandas as pd
import pytest
import yaml
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.multi_task.run_multi_task import get_argument_parser, run


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

    config_folder = run_folder / "serializations" / "configs_stripped"
    assert config_folder.exists()

    predict_config_folder = tmp_path / "predict_configs"
    predict_config_folder.mkdir(parents=True, exist_ok=True)

    for config_file in config_folder.iterdir():
        if config_file.suffix != ".yaml":
            continue

        if config_file.stem == "input_configs":
            configs = yaml.safe_load(config_file.read_text())
            genotype_only_configs = [
                c for c in configs if c["input_info"]["input_name"] != "eir_tabular"
            ]

            if len(genotype_only_configs) == len(configs):
                pytest.skip("No tabular input found in training config")

            for _idx, config in enumerate(genotype_only_configs):
                input_name = config["input_info"]["input_name"]
                output_path = predict_config_folder / f"{input_name}_input_config.yaml"
                output_path.write_text(yaml.dump(config))
        else:
            output_path = predict_config_folder / config_file.name
            output_path.write_text(config_file.read_text())

    saved_models = list((run_folder / "saved_models").iterdir())
    assert len(saved_models) == 1

    predict_output = tmp_path / "genotype_only_predictions"
    predict_output.mkdir(parents=True, exist_ok=True)

    command_parts = ["eirpredict"]

    for config_file in predict_config_folder.iterdir():
        if config_file.suffix != ".yaml":
            continue
        if "global" in config_file.stem:
            command_parts.extend(["--global_configs", str(config_file)])
        elif "input" in config_file.stem:
            command_parts.extend(["--input_configs", str(config_file)])
        elif "fusion" in config_file.stem:
            command_parts.extend(["--fusion_configs", str(config_file)])
        elif "output" in config_file.stem:
            command_parts.extend(["--output_configs", str(config_file)])

    command_parts.extend(
        [
            "--model_path",
            str(saved_models[0]),
            "--no-strict",
            "--output_folder",
            str(predict_output),
        ]
    )

    result = subprocess.run(
        command_parts,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Prediction failed: {result.stderr}"

    assert (predict_output / "predictions_complete.txt").exists()

    prediction_files = list(predict_output.rglob("*.csv"))
    assert len(prediction_files) > 0, "No prediction files generated"

    for pred_file in prediction_files:
        df = pd.read_csv(pred_file)
        assert len(df) > 0, f"Empty predictions in {pred_file}"
