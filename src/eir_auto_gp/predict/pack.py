import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import yaml
from aislib.misc_utils import ensure_path_exists, get_logger

logger = get_logger(name=__name__)


def pack_experiment(experiment_folder: Path, output_path: Path) -> None:
    bim_file = get_run_bim_file(experiment_folder=experiment_folder)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        add_modelling_folders_to_zip(
            zipf=zipf,
            root_dir=experiment_folder / "modelling",
            target_dirs=["serializations", "saved_models"],
        )
        modify_and_add_config_to_zip(
            zipf=zipf, config_path=experiment_folder / "config.json"
        )
        add_bim_file_to_zip(zipf=zipf, bim_file=bim_file)

        subset_folder = experiment_folder / "modelling" / "snp_subset_files"
        if subset_folder.exists():
            add_subset_folder_to_zip(zipf=zipf, subset_folder=subset_folder)


def add_modelling_folders_to_zip(
    zipf: zipfile.ZipFile, root_dir: Path, target_dirs: list[str]
) -> None:
    logger.info(f"Writing directories {target_dirs} from {root_dir}.")
    for fold_dir in root_dir.iterdir():
        if fold_dir.is_dir() and fold_dir.name.startswith("fold_"):
            for directory in target_dirs:
                target_dir = fold_dir / directory
                if target_dir.exists() and target_dir.is_dir():
                    for file in target_dir.rglob("*"):
                        if not file.is_file():
                            continue

                        if _should_filter_tabular_from_file(file=file):
                            _add_filtered_configs_to_zip(
                                zipf=zipf,
                                file=file,
                                arcname=file.relative_to(root_dir.parent),
                            )
                        else:
                            zipf.write(file, arcname=file.relative_to(root_dir.parent))
                else:
                    raise ValueError(f"Directory {target_dir} does not exist.")


def _should_filter_tabular_from_file(file: Path) -> bool:
    return (
        "serializations" in file.parts
        and "configs_stripped" in file.parts
        and file.name in ("input_configs.yaml", "fusion_config.yaml")
    )


def _add_filtered_configs_to_zip(
    zipf: zipfile.ZipFile, file: Path, arcname: Path
) -> None:
    config_data = yaml.safe_load(file.read_text())

    if file.name == "input_configs.yaml":
        filtered_data = _filter_input_configs(configs=config_data)
    elif file.name == "fusion_config.yaml":
        filtered_data = _filter_fusion_config(config=config_data)
    else:
        filtered_data = config_data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(filtered_data, tmp)
        tmp_path = Path(tmp.name)

    try:
        zipf.write(tmp_path, arcname=arcname)
    finally:
        tmp_path.unlink()


def _filter_input_configs(configs: list[dict]) -> list[dict]:
    filtered = [c for c in configs if c["input_info"]["input_name"] != "eir_tabular"]

    if len(filtered) < len(configs):
        logger.info(f"Filtered {len(configs) - len(filtered)} tabular input config(s)")

    return filtered


def _filter_fusion_config(config: dict) -> dict:
    if "tensor_broker_config" not in config:
        return config

    tb_config = config["tensor_broker_config"]
    if "message_configs" not in tb_config:
        return config

    original_messages = tb_config["message_configs"]
    filtered_messages = []

    for msg in original_messages:
        if "tabular" in msg.get("name", "").lower():
            logger.info(f"Filtered TB message: {msg.get('name')}")
            continue

        if "use_from_cache" in msg:
            original_cache = msg["use_from_cache"]
            filtered_cache = [c for c in original_cache if c != "tabular_output"]
            if len(filtered_cache) < len(original_cache):
                logger.info(
                    f"Removed 'tabular_output' from cache in message: {msg.get('name')}"
                )
                msg["use_from_cache"] = filtered_cache

        filtered_messages.append(msg)

    config["tensor_broker_config"]["message_configs"] = filtered_messages
    return config


def modify_and_add_config_to_zip(zipf: zipfile.ZipFile, config_path: Path) -> None:
    with open(config_path) as f:
        config_data = json.load(f)
        config_data["genotype_data_path"] = None
        config_data["label_file_path"] = None
        config_data["global_output_folder"] = None

    temp_config_path = config_path.parent / "temp_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(config_data, f)

    logger.info(f"Writing config file to {temp_config_path}.")

    zipf.write(temp_config_path, arcname="config.json")
    temp_config_path.unlink()


def add_bim_file_to_zip(zipf: zipfile.ZipFile, bim_file: Path) -> None:
    if bim_file and bim_file.exists():
        logger.info(f"Writing BIM file to {bim_file}.")
        bim_target_path = "meta/snps.bim"
        zipf.write(bim_file, arcname=bim_target_path)
    else:
        raise ValueError(f"File {bim_file} does not exist.")


def add_subset_folder_to_zip(zipf: zipfile.ZipFile, subset_folder: Path) -> None:
    if subset_folder and subset_folder.exists():
        logger.info(f"Writing subset folder to {subset_folder}.")
        for file in subset_folder.rglob("*"):
            zipf.write(file, arcname=file.relative_to(subset_folder.parent))


def get_run_bim_file(experiment_folder: Path) -> Path:
    bim_file = experiment_folder / "data/genotype/processed/parsed_files/data_final.bim"
    assert bim_file.exists(), f"File {bim_file} does not exist."

    return bim_file


def unpack_experiment(
    packed_experiment_path: str,
    output_path: str,
) -> str:
    ensure_path_exists(Path(output_path), is_folder=True)

    with zipfile.ZipFile(packed_experiment_path, "r") as zipf:
        zipf.extractall(output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_folder",
        type=str,
        help="The folder to pack.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="The path to the output zip file.",
    )

    args = parser.parse_args()

    experiment_folder = Path(args.experiment_folder)
    output_path = Path(args.output_path)

    pack_experiment(experiment_folder=experiment_folder, output_path=output_path)


if __name__ == "__main__":
    main()
