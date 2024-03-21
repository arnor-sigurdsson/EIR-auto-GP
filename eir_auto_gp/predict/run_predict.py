import argparse
import subprocess
from pathlib import Path

import torch
import yaml
from aislib.misc_utils import get_logger
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.multi_task.modelling.run_modelling import (
    get_testing_string_from_config_folder,
)
from eir_auto_gp.predict.pack import unpack_experiment
from eir_auto_gp.predict.prepare_data import run_prepare_data
from eir_auto_gp.predict.sync import run_sync

logger = get_logger(name=__name__)


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
        )

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
