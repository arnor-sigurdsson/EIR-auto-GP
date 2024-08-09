import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from aislib.misc_utils import get_logger
from eir.train_utils.train_handlers import _iterdir_ignore_hidden
from scipy.stats import mode

from eir_auto_gp.multi_task.modelling.run_modelling import (
    get_testing_string_from_config_folder,
)
from eir_auto_gp.predict.pack import unpack_experiment
from eir_auto_gp.predict.prepare_data import run_prepare_data
from eir_auto_gp.predict.sync import run_sync

logger = get_logger(name=__name__)

luigi_logger = logging.getLogger("luigi")
luigi_logger.setLevel(logging.INFO)


def parse_arguments():
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
    return parser.parse_args()


def main(cl_args: argparse.Namespace) -> None:
    unpacked_experiment_path = unpack_experiment(
        packed_experiment_path=cl_args.packed_experiment_path,
        output_path=cl_args.output_folder + "/unpacked_experiment",
    )

    final_genotype_path = run_sync(
        genotype_data_path=Path(cl_args.genotype_data_path),
        experiment_folder=Path(cl_args.output_folder + "/unpacked_experiment"),
        output_folder=Path(cl_args.output_folder),
    )

    prepared_folder = run_prepare_data(
        final_genotype_data_path=final_genotype_path,
        output_folder=cl_args.output_folder,
    )

    run_predict(
        prepared_input_data_folder=prepared_folder,
        unpacked_experiment_path=Path(unpacked_experiment_path),
        output_folder=Path(cl_args.output_folder),
    )

    gather_results(
        unpacked_experiment_path=Path(unpacked_experiment_path),
        output_folder=Path(cl_args.output_folder, "results"),
    )


def gather_results(unpacked_experiment_path: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)

    aggregated_data: dict[str, pd.DataFrame] = {}

    for predictions_file in unpacked_experiment_path.rglob("predictions.csv"):
        df = pd.read_csv(predictions_file)

        target_name = predictions_file.parent.name

        fold_path_part = predictions_file.parts[-5]
        assert fold_path_part.startswith("fold_")
        fold_number = fold_path_part.split("_")[-1]

        id_column = "ID"
        target_column = f"{target_name} Untransformed"
        new_column_name = f"{target_name} Fold {fold_number}"

        if target_column not in df.columns:
            logger.warning(f"Column '{target_column}' not found in {predictions_file}")
            continue

        df_selected = df[[id_column, target_column]].rename(
            columns={target_column: new_column_name}
        )

        if target_name in aggregated_data:
            aggregated_data[target_name] = pd.merge(
                left=aggregated_data[target_name],
                right=df_selected,
                on="ID",
                how="outer",
            )
        else:
            aggregated_data[target_name] = df_selected

    config_path = unpacked_experiment_path / "config.json"
    config = read_config(config_file=config_path)
    data_types = prepare_data_types(config=config)

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


def compute_ensemble_and_uncertainty(
    df: pd.DataFrame,
    target: str,
    data_type: str,
) -> pd.DataFrame:
    results = pd.DataFrame({"ID": df["ID"]})

    target_columns = [col for col in df.columns if col.startswith(target)]
    results = pd.concat([results, df[target_columns]], axis=1)

    if data_type == "continuous":
        ensemble_mean = df[target_columns].mean(axis=1)
        results[f"{target} Ensemble"] = ensemble_mean

        bootstrap_lower_ci = []
        bootstrap_upper_ci = []
        for row in df[target_columns].to_numpy():
            lower_ci, upper_ci = bootstrap_ci(row, n_bootstraps=1000, ci=[2.5, 97.5])
            bootstrap_lower_ci.append(lower_ci)
            bootstrap_upper_ci.append(upper_ci)

        results[f"{target} 2.5% CI"] = bootstrap_lower_ci
        results[f"{target} 97.5% CI"] = bootstrap_upper_ci

    elif data_type == "categorical":
        mode_values, counts = mode(df[target_columns], axis=1)
        results[f"{target} Ensemble"] = mode_values.flatten()
        results[f"{target} Uncertainty"] = counts.flatten() / len(target_columns)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return results


def bootstrap_ci(data, n_bootstraps=1000, ci=(2.5, 97.5)):
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

    for model_folder in _iterdir_ignore_hidden(path=modelling_folder):
        tmp_dir = output_folder / "tmp"
        tmp_dir.mkdir(exist_ok=True, parents=True)

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


def build_predict_configs(
    modelling_folder: Path,
    prepared_input_data_folder: Path,
    output_folder: Path,
) -> dict[str, Path]:
    configs_folder = modelling_folder / "configs"

    modified_files = {}

    for f in _iterdir_ignore_hidden(path=configs_folder):
        configs_as_dict = yaml.safe_load(f.read_text())
        match f.stem:
            case "global_config":
                config_device = configs_as_dict["device"]
                if "cuda" in config_device and not torch.cuda.is_available():
                    logger.info(
                        "CUDA was used for original experiments but is not "
                        "available on current system. Changing device to CPU."
                    )
                    configs_as_dict["device"] = "cpu"

                logger.info("Setting dataloader workers to 0 for prediction.")
                configs_as_dict["dataloader_workers"] = 0

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

                    new_path = output_folder / f"{cur_name}_input_config.yaml"
                    new_path.write_text(yaml.dump(input_config))
                    modified_files[cur_name] = new_path

            case "fusion_config":
                cur_config = configs_as_dict
                new_path = output_folder / f.name
                new_path.write_text(yaml.dump(cur_config))
                modified_files["fusion_config"] = new_path

            case "output_configs":
                assert len(configs_as_dict) == 1
                cur_config = configs_as_dict[0]
                cur_config["output_info"]["output_source"] = None
                new_path = output_folder / f.name
                new_path.write_text(yaml.dump(cur_config))
                modified_files["output_config"] = new_path

            case _:
                raise ValueError(f"File {f} is not recognized.")

    return modified_files


if __name__ == "__main__":
    args = parse_arguments()
    main(cl_args=args)
