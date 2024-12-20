import json
import random
import shutil
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import pandas as pd
import pytest
from eir.train_utils.metrics import calc_pcc
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from eir_auto_gp.multi_task.run_multi_task import (
    get_argument_parser,
    run,
    store_experiment_config,
)
from eir_auto_gp.predict.pack import pack_experiment
from eir_auto_gp.predict.run_predict import get_parser, run_sync_and_predict_wrapper


def _get_test_cl_commands(folder_path: Path) -> list[str]:
    base = (
        f"--genotype_data_path {folder_path}/ "
        f"--label_file_path {folder_path}/phenotype.csv "
        "--global_output_folder runs/simulated_test "
        "--output_cat_columns phenotype "
        "--output_con_columns CON_COMPUTED "
        "--model_size nano "
        "--folds 1 "
        "--do_test"
    )

    with_groups = f"{base} " "--output_groups random"

    return [base, with_groups]


def _build_test_predict_data(
    tmp_path: Path,
    input_data_path: Path,
    num_snps: int = 100,
) -> Path:
    output_dir = tmp_path / "subset_data"
    output_dir.mkdir(exist_ok=True)

    input_prefix = input_data_path / "genetic_data"
    output_prefix = output_dir / "simulated_subset"

    bim_file = input_prefix.with_suffix(".bim")
    with open(bim_file, "r") as f:
        all_snps = [line.split()[1] for line in f]

    selected_snps = random.sample(all_snps, min(num_snps, len(all_snps)))

    snp_list_file = output_dir / "selected_snps.txt"
    with open(snp_list_file, "w") as f:
        for snp in selected_snps:
            f.write(f"{snp}\n")

    plink2_command = [
        "plink2",
        "--bfile",
        str(input_prefix),
        "--extract",
        str(snp_list_file),
        "--make-bed",
        "--out",
        str(output_prefix),
    ]

    try:
        subprocess.run(plink2_command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running plink2: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

    shutil.copy2(input_data_path / "phenotype.csv", output_dir / "phenotype.csv")

    id_root = input_data_path / "data/ids"
    for file_name in [
        "test_ids.txt",
        "train_ids.txt",
        "valid_ids.txt",
    ]:
        shutil.copy2(id_root / file_name, output_dir / file_name)

    return output_dir


@pytest.mark.parametrize("command", _get_test_cl_commands(Path("placeholder")))
def test_modelling_pack_and_predict(
    command: str,
    tmp_path: Path,
    simulate_genetic_data_to_bed: Callable[[int, int, str], Path],
) -> None:
    simulated_path = simulate_genetic_data_to_bed(5000, 50, "binary")

    command = command.replace("placeholder", str(simulated_path))

    parser = get_argument_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.global_output_folder = str(tmp_path)

    store_experiment_config(cl_args=cl_args)
    run(cl_args=cl_args)

    model_folder = tmp_path / "modelling"
    check_test = True if "--do_test" in command else False
    for modelling_run in Path(model_folder).iterdir():
        if not modelling_run.name.startswith("fold_"):
            continue

        check_modelling_results(run_folder=modelling_run, check_test=check_test)

    experiment_folder = tmp_path
    packed_path = experiment_folder / "experiment.zip"
    pack_experiment(
        experiment_folder=experiment_folder,
        output_path=packed_path,
    )

    predict_test_parser = get_parser()

    test_predict_subset_folder = _build_test_predict_data(
        tmp_path=tmp_path,
        input_data_path=simulated_path,
        num_snps=25,
    )

    predict_output_folder = tmp_path / "predict_output"
    predict_test_cl_args = predict_test_parser.parse_args(
        f"--genotype_data_path {str(test_predict_subset_folder)} "
        f"--packed_experiment_path {packed_path} "
        f"--output_folder {str(predict_output_folder)}".split()
    )

    run_sync_and_predict_wrapper(cl_args=predict_test_cl_args)

    predict_output_folder = tmp_path / "predict_output" / "results"
    actual_data_path = simulated_path / "phenotype.csv"

    check_predict_results(
        predict_output_folder=predict_output_folder,
        actual_data_path=actual_data_path,
    )


def check_modelling_results(run_folder: Path, check_test: bool) -> None:
    for file in ["validation_average_history.log"]:
        check_average_performances(file_path=run_folder / file, threshold=0.04)

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
    for key, value in input_dict.items():
        match value:
            case dict(value):
                find_numeric_values(input_dict=value, accumulator=accumulator)
            case int(value) | float(value):
                accumulator.append(value)

    return accumulator


def check_predict_results(
    predict_output_folder: Path,
    actual_data_path: Path,
) -> None:

    gathered_results = gather_predict_results(
        predict_output_folder=predict_output_folder,
        actual_data_path=actual_data_path,
    )
    assert gathered_results, "No results gathered"

    for target, metrics in gathered_results.items():
        if target in ["phenotype"]:
            assert (
                0 <= metrics["accuracy"] <= 1
            ), f"Accuracy for {target} should be between 0 and 1"
            assert (
                0 <= metrics["auc"] <= 1
            ), f"AUC for {target} should be between 0 and 1"
            assert (
                metrics["auc"] >= 0.55
            ), f"AUC for {target} should be at least 0.55, got {metrics['auc']}"
        else:
            assert metrics["mse"] >= 0, f"MSE for {target} should be non-negative"
            assert (
                metrics["pcc"] >= 0.03
            ), f"PCC for {target} should be at least 0.03, got {metrics['pcc']}"


def gather_predict_results(
    predict_output_folder: Path,
    actual_data_path: Path,
) -> dict[str, Any]:
    actual_data = pd.read_csv(actual_data_path)
    actual_data["ID"] = actual_data["ID"].astype(str)
    actual_data.set_index("ID", inplace=True)

    results = {}

    cat_targets = ["phenotype"]
    for target in cat_targets:
        if (predict_output_folder / f"{target}.csv").exists():
            pred_data = pd.read_csv(predict_output_folder / f"{target}.csv")
            pred_data["ID"] = pred_data["ID"].astype(str)
            pred_data.set_index("ID", inplace=True)

            merged_data = actual_data.join(pred_data, how="inner")
            merged_data = merged_data.dropna(subset=[target])

            y_true = merged_data[target]
            y_pred = merged_data[f"{target} Predicted Class"]
            y_prob = merged_data[f"{target} Ensemble Prob 1.0"]

            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            auc = roc_auc_score(y_true=y_true, y_score=y_prob)

            results[target] = {"accuracy": accuracy, "auc": auc}

    cont_targets = ["CON_COMPUTED"]
    for target in cont_targets:
        if (predict_output_folder / f"{target}.csv").exists():
            pred_data = pd.read_csv(predict_output_folder / f"{target}.csv")
            pred_data["ID"] = pred_data["ID"].astype(str)
            pred_data.set_index("ID", inplace=True)

            merged_data = actual_data.join(pred_data, how="inner")
            merged_data = merged_data.dropna(subset=[target])

            y_true = merged_data[target]
            y_pred = merged_data[f"{target} Ensemble"]

            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            pcc = calc_pcc(outputs=y_pred, labels=y_true)

            results[target] = {"mse": mse, "pcc": pcc}

    return results
