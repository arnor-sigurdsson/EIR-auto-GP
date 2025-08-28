import argparse
import json
import logging
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from aislib.misc_utils import get_logger
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.multi_task.modelling.run_modelling import (
    get_testing_string_from_config_folder,
)
from eir_auto_gp.predict.data_preparation_utils import get_experiment_bim_file
from eir_auto_gp.predict.pack import unpack_experiment
from eir_auto_gp.predict.prepare_data import run_prepare_data

logger = get_logger(name=__name__)

luigi_logger = logging.getLogger("luigi")
luigi_logger.setLevel(logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genotype_data_path",
        type=str,
        help="The path to the genotype data to predict on.",
    )

    parser.add_argument(
        "--packed_experiment_path",
        type=str,
        help="Path to packed (zipped) experiment.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="The folder to output the results.",
    )

    parser.add_argument(
        "--genotype_processing_chunk_size",
        type=int,
        default=1024,
    )
    return parser


def run_sync_and_predict_wrapper(
    cl_args: argparse.Namespace,
) -> None:

    unpacked_experiment_path = unpack_experiment(
        packed_experiment_path=cl_args.packed_experiment_path,
        output_path=cl_args.output_folder + "/unpacked_experiment",
    )

    data_output_folder = Path(cl_args.output_folder, "data")
    experiment_folder = Path(cl_args.output_folder + "/unpacked_experiment")
    experiment_bim = get_experiment_bim_file(experiment_folder=experiment_folder)

    prepared_folder = run_prepare_data(
        genotype_data_path=Path(cl_args.genotype_data_path),
        output_folder=data_output_folder,
        reference_bim_to_project_to=experiment_bim,
        array_chunk_size=cl_args.genotype_processing_chunk_size,
    )

    run_predict(
        prepared_input_data_folder=prepared_folder.array_folder,
        unpacked_experiment_path=Path(unpacked_experiment_path),
        output_folder=Path(cl_args.output_folder),
    )

    gather_results(
        unpacked_experiment_path=Path(unpacked_experiment_path),
        output_folder=Path(cl_args.output_folder, "results"),
    )

    shutil.copy(
        data_output_folder / "snp_overlap_analysis.txt",
        Path(cl_args.output_folder, "snp_overlap_analysis.txt"),
    )

    shutil.rmtree(path=data_output_folder)
    shutil.rmtree(path=unpacked_experiment_path)


def gather_results(unpacked_experiment_path: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    config_path = unpacked_experiment_path / "config.json"
    config = read_config(config_file=config_path)
    data_types = prepare_data_types(config=config)

    aggregated_data: dict[str, pd.DataFrame] = {}

    for predictions_file in unpacked_experiment_path.rglob("predictions.csv"):
        df = pd.read_csv(predictions_file)

        target_name = predictions_file.parent.name

        fold_path_part = predictions_file.parts[-5]
        assert fold_path_part.startswith("fold_")
        fold_number = fold_path_part.split("_")[-1]

        id_column = "ID"

        data_type = data_types[target_name]

        if data_type == "continuous":
            target_column = f"{target_name} Untransformed"
            new_column_name = f"{target_name} Fold {fold_number}"

            if target_column not in df.columns:
                logger.warning(
                    f"Column '{target_column}' not found in {predictions_file}"
                )
                continue

            df_selected = df[[id_column, target_column]].rename(
                columns={target_column: new_column_name}
            )
        elif data_type == "categorical":
            rename_map = {
                col: f"{target_name}: {col} Fold {fold_number}"
                for col in df.columns
                if col != id_column
            }
            df_selected = df.rename(columns=rename_map)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        if target_name in aggregated_data:
            aggregated_data[target_name] = pd.merge(
                left=aggregated_data[target_name],
                right=df_selected,
                on="ID",
                how="outer",
            )
        else:
            aggregated_data[target_name] = df_selected

    for target_name, df_aggregated in aggregated_data.items():
        output_path = output_folder / f"{target_name}.csv"
        data_type = data_types[target_name]
        df_aggregated = compute_ensemble_and_uncertainty(
            df=df_aggregated,
            target=target_name,
            data_type=data_type,
        )
        df_aggregated.to_csv(output_path, index=False)


def read_config(config_file: Path) -> dict:
    with config_file.open("r") as file:
        config = json.load(file)
    return config


def prepare_data_types(config: dict) -> dict:
    data_types = {}
    for col in config["output_cat_columns"]:
        data_types[col] = "categorical"
    for col in config["output_con_columns"]:
        data_types[col] = "continuous"
    return data_types


def extract_categorical_class_name(column_name: str) -> str:
    class_name = re.sub(r"^.*?:", "", column_name).strip()
    return re.sub(r"\s+Fold\s+\d+$", "", class_name)


def compute_continuous_ensemble(
    df: pd.DataFrame,
    target_columns: list,
    target: str,
) -> pd.DataFrame:
    results = pd.DataFrame()
    results[f"{target} Ensemble"] = df[target_columns].mean(axis=1)

    bootstrap_lower_ci, bootstrap_upper_ci = [], []
    for row in df[target_columns].to_numpy():
        lower_ci, upper_ci = sample_bootstrap_ci(
            data=row,
            n_bootstraps=1000,
            ci=[2.5, 97.5],
        )
        bootstrap_lower_ci.append(lower_ci)
        bootstrap_upper_ci.append(upper_ci)

    results[f"{target} 2.5% CI"] = bootstrap_lower_ci
    results[f"{target} 97.5% CI"] = bootstrap_upper_ci
    return results


import numpy as np
import pandas as pd
from scipy.special import softmax


def compute_categorical_ensemble(
    *,
    df: pd.DataFrame,
    target_columns: list[str],
    target: str,
) -> pd.DataFrame:
    results = pd.DataFrame()
    classes = sorted(
        set(extract_categorical_class_name(column_name=col) for col in target_columns)
    )

    for class_name in classes:
        class_columns = [
            col
            for col in target_columns
            if extract_categorical_class_name(column_name=col) == class_name
        ]
        results[f"{target} Ensemble Raw {class_name}"] = df[class_columns].mean(axis=1)

    raw_outputs = results[[f"{target} Ensemble Raw {c}" for c in classes]].values

    if len(classes) == 1:
        class1_probs = 1 / (1 + np.exp(-raw_outputs))
        softmax_probs = np.hstack([1 - class1_probs, class1_probs])
        classes = ["0"] + classes
    else:
        softmax_probs = softmax(x=raw_outputs, axis=1)

    for i, class_name in enumerate(classes):
        results[f"{target} Ensemble Prob {class_name}"] = softmax_probs[:, i]

    epsilon = 1e-10
    entropies = -np.sum(softmax_probs * np.log(softmax_probs + epsilon), axis=1)
    max_entropy = np.log(len(classes))
    results[f"{target} Uncertainty"] = entropies / max_entropy

    results[f"{target} Predicted Class"] = [
        classes[i] for i in np.argmax(a=softmax_probs, axis=1)
    ]

    return results


def compute_ensemble_and_uncertainty(
    df: pd.DataFrame,
    target: str,
    data_type: str,
) -> pd.DataFrame:

    results = pd.DataFrame({"ID": df["ID"]})
    target_columns = [col for col in df.columns if col.startswith(target)]
    results = pd.concat([results, df[target_columns]], axis=1)

    if data_type == "continuous":
        ensemble_results = compute_continuous_ensemble(
            df=df,
            target_columns=target_columns,
            target=target,
        )
    elif data_type == "categorical":
        ensemble_results = compute_categorical_ensemble(
            df=df,
            target_columns=target_columns,
            target=target,
        )
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    results = pd.concat([results, ensemble_results], axis=1)
    return results


def sample_bootstrap_ci(data, n_bootstraps=1000, ci=(2.5, 97.5)):
    bootstrap_samples = np.random.choice(
        data,
        size=(n_bootstraps, data.shape[0]),
        replace=True,
    )
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_ci, upper_ci = np.percentile(bootstrap_means, ci)
    return lower_ci, upper_ci


def run_predict(
    prepared_input_data_folder: Path,
    unpacked_experiment_path: Path,
    output_folder: Path,
) -> None:
    modelling_folder = unpacked_experiment_path / "modelling"
    assert modelling_folder.exists(), f"Folder {modelling_folder} does not exist."

    tmp_dir = output_folder / "tmp_configs"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    for model_folder in _iterdir_ignore_hidden(path=modelling_folder):

        build_predict_configs(
            modelling_folder=model_folder,
            prepared_input_data_folder=prepared_input_data_folder,
            output_folder=tmp_dir,
        )

        command = get_testing_string_from_config_folder(
            config_folder=tmp_dir,
            train_run_folder=model_folder,
            with_labels=False,
        )
        logger.info(f"Running command: {command}")

        subprocess.run(command, shell=True)

    shutil.rmtree(path=tmp_dir)


def build_predict_configs(
    modelling_folder: Path,
    prepared_input_data_folder: Path,
    output_folder: Path,
) -> dict[str, Path]:
    configs_folder = modelling_folder / "serializations/configs_stripped"

    subset_folder = modelling_folder / "snp_subset_files"

    modified_files = {"output_config": []}

    for f in _iterdir_ignore_hidden(path=configs_folder):
        configs_as_dict = yaml.safe_load(f.read_text())
        match f.stem:
            case "global_config":
                config_device = configs_as_dict["basic_experiment"]["device"]
                if "cuda" in config_device and not torch.cuda.is_available():
                    logger.info(
                        "CUDA was used for original experiments but is not "
                        "available on current system. Changing device to CPU."
                    )
                    configs_as_dict["basic_experiment"]["device"] = "cpu"

                logger.info("Setting dataloader workers to 0 for prediction.")
                configs_as_dict["basic_experiment"]["dataloader_workers"] = 0

                cur_config = configs_as_dict
                new_path = output_folder / f.name
                new_path.write_text(yaml.dump(cur_config))
                modified_files["global_config"] = new_path

            case "input_configs":
                for input_config in configs_as_dict:
                    cur_name = input_config["input_info"]["input_name"]

                    input_config["input_info"]["input_source"] = str(
                        prepared_input_data_folder
                    )

                    input_config = maybe_add_subset_to_input_config(
                        input_config=input_config,
                        subset_folder=subset_folder,
                        modelling_folder=modelling_folder,
                    )

                    new_path = output_folder / f"{cur_name}_input_config.yaml"
                    new_path.write_text(yaml.dump(input_config))
                    modified_files[cur_name] = new_path

            case "fusion_config":
                cur_config = configs_as_dict
                new_path = output_folder / f.name
                new_path.write_text(yaml.dump(cur_config))
                modified_files["fusion_config"] = new_path

            case "output_configs":
                for output_config in configs_as_dict:
                    cur_name = output_config["output_info"]["output_name"]
                    output_config["output_info"]["output_source"] = None
                    new_path = output_folder / f"{cur_name}_output_config.yaml"
                    new_path.write_text(yaml.dump(output_config))
                    modified_files["output_config"].append(new_path)

            case _:
                raise ValueError(f"File {f} is not recognized.")

    return modified_files


def maybe_add_subset_to_input_config(
    input_config: dict[str, Any],
    subset_folder: Path,
    modelling_folder: Path,
) -> dict[str, Any]:

    if not subset_folder.exists():
        return input_config

    input_config_copy = deepcopy(input_config)

    model_name = modelling_folder.name
    assert model_name.startswith("fold_")

    cur_fold = model_name.split("_")[-1]

    cur_subset_file_name = f"random_subset_fold={cur_fold}.txt"

    cur_subset_path = subset_folder / cur_subset_file_name
    assert cur_subset_path.exists(), f"Subset file {cur_subset_path} does not exist."

    logger.info(f"Adding subset file {cur_subset_path} to input config.")

    input_config_copy["input_type_info"]["subset_snps_file"] = str(cur_subset_path)

    return input_config_copy


def main() -> None:
    parser = get_parser()
    cl_args = parser.parse_args()
    run_sync_and_predict_wrapper(cl_args=cl_args)


if __name__ == "__main__":
    main()
