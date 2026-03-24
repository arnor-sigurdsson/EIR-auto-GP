import base64
import signal
import subprocess
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from aislib.misc_utils import get_logger
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.predict.data_preparation_utils import (
    get_experiment_bim_file,
    log_overlap,
    read_bim_and_cast_dtypes,
)
from eir_auto_gp.predict.prepare_data import (
    PlinkFileSet,
    create_snp_mapping,
    get_plink_fileset_from_folder,
    get_projected_snp_stream,
)
from eir_auto_gp.preprocess.genotype import get_encoded_snp_stream

logger = get_logger(name=__name__)


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    base_port: int = 8000
    batch_size: int = 32
    chunk_size: int = 1024
    startup_timeout: float = 120.0
    request_timeout: float = 300.0


def start_serve_process(
    model_path: Path,
    port: int,
    host: str = "127.0.0.1",
    device: str = "cpu",
) -> subprocess.Popen:
    cmd = [
        "eirserve",
        "--model-path",
        str(model_path),
        "--port",
        str(port),
        "--host",
        host,
        "--device",
        device,
    ]

    logger.info(f"Starting serve process: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return process


def wait_for_server_ready(
    host: str,
    port: int,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
) -> bool:
    url = f"http://{host}:{port}/info"
    start_time = time.time()

    logger.info(f"Waiting for server at {url} (timeout: {timeout}s)")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url=url, timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Server ready at {url}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass

        time.sleep(poll_interval)

    logger.error(f"Server at {url} did not become ready within {timeout}s")
    return False


def stop_serve_process(process: subprocess.Popen, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return

    logger.info(f"Stopping serve process (PID: {process.pid})")

    process.send_signal(signal.SIGTERM)

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"Process {process.pid} did not terminate, sending SIGKILL")
        process.kill()
        process.wait(timeout=5.0)


def get_server_info(host: str, port: int, timeout: float = 30.0) -> dict[str, Any]:
    url = f"http://{host}:{port}/info"
    response = requests.get(url=url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def encode_omics_array(array: np.ndarray) -> str:
    array_bool = array.astype(np.bool_)
    array_bytes = array_bool.tobytes()
    return base64.b64encode(array_bytes).decode("utf-8")


def send_predict_request(
    host: str,
    port: int,
    batch: list[dict[str, Any]],
    timeout: float = 300.0,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    url = f"http://{host}:{port}/predict"

    for attempt in range(max_retries):
        try:
            response = requests.post(url=url, json=batch, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            return result["result"]
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s"
                )
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Failed to send predict request after all retries")


def batched(iterable: Iterator, batch_size: int) -> Generator[list]:
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def stream_and_predict(
    plink_fileset: PlinkFileSet,
    reference_bim: Path,
    host: str,
    port: int,
    input_name: str,
    batch_size: int = 32,
    chunk_size: int = 1024,
    request_timeout: float = 300.0,
) -> pd.DataFrame:
    df_bim_input = read_bim_and_cast_dtypes(bim_file_path=plink_fileset.bim)
    df_bim_reference = read_bim_and_cast_dtypes(bim_file_path=reference_bim)

    snp_mapping = create_snp_mapping(
        from_bim=df_bim_input,
        to_reference_bim=df_bim_reference,
    )

    encoded_stream = get_encoded_snp_stream(
        bed_path=plink_fileset.bed,
        chunk_size=chunk_size,
        output_format="disk",
    )

    projected_stream = get_projected_snp_stream(
        from_stream=encoded_stream,
        snp_mapping=snp_mapping,
    )

    all_sample_ids: list[str] = []
    all_predictions: list[dict[str, Any]] = []

    n_samples_processed = 0
    for batch in batched(iterable=projected_stream, batch_size=batch_size):
        batch_requests = []
        batch_ids = []

        for sample_id, projected_array in batch:
            encoded = encode_omics_array(array=projected_array)
            batch_requests.append({input_name: encoded})
            batch_ids.append(sample_id)

        responses = send_predict_request(
            host=host,
            port=port,
            batch=batch_requests,
            timeout=request_timeout,
        )

        all_sample_ids.extend(batch_ids)
        all_predictions.extend(responses)

        n_samples_processed += len(batch)
        if n_samples_processed % 100 == 0:
            logger.info(f"Processed {n_samples_processed} samples")

    logger.info(f"Completed processing {n_samples_processed} samples")

    return build_predictions_dataframe(
        sample_ids=all_sample_ids,
        predictions=all_predictions,
    )


def build_predictions_dataframe(
    sample_ids: list[str],
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []

    for sample_id, prediction in zip(sample_ids, predictions, strict=True):
        row = {"ID": sample_id}

        for output_name, output_value in prediction.items():
            if isinstance(output_value, dict):
                for col_name, col_value in output_value.items():
                    if isinstance(col_value, dict):
                        col_value_keys = list(col_value.keys())
                        is_continuous = (
                            len(col_value_keys) == 1 and col_value_keys[0] == col_name
                        )
                        if is_continuous:
                            row[f"{col_name} Untransformed"] = col_value[col_name]
                        else:
                            is_binary = len(col_value_keys) == 1
                            for class_name, prob in col_value.items():
                                if is_binary:
                                    row[f"{col_name}_1.0"] = prob
                                else:
                                    row[f"{col_name}_{class_name}"] = prob
                    else:
                        row[f"{col_name} Untransformed"] = col_value
            else:
                row[output_name] = output_value

        rows.append(row)

    return pd.DataFrame(rows)


def get_saved_model_path(fold_path: Path) -> Path:
    saved_models_dir = fold_path / "saved_models"
    if not saved_models_dir.exists():
        raise FileNotFoundError(f"No saved_models directory in {fold_path}")

    model_files = list(saved_models_dir.glob("*.pt"))
    if len(model_files) == 0:
        raise FileNotFoundError(f"No .pt files found in {saved_models_dir}")
    if len(model_files) > 1:
        logger.warning(
            f"Multiple model files found in {saved_models_dir}, using first: "
            f"{model_files[0].name}"
        )

    return model_files[0]


def get_genotype_input_name(model_path: Path) -> str:
    run_folder = model_path.parent.parent
    configs_folder = run_folder / "serializations" / "configs_stripped"

    input_configs_file = configs_folder / "input_configs.yaml"
    if not input_configs_file.exists():
        logger.warning(
            f"Could not find input_configs.yaml at {input_configs_file}, "
            "defaulting to 'genotype'"
        )
        return "genotype"

    import yaml

    with open(input_configs_file) as f:
        input_configs = yaml.safe_load(f)

    for config in input_configs:
        input_type = config.get("input_info", {}).get("input_type")
        if input_type == "omics":
            return config["input_info"]["input_name"]

    logger.warning("No omics input found in configs, defaulting to 'genotype'")
    return "genotype"


def reorganize_predictions_for_gather(
    df_predictions: pd.DataFrame,
    fold_path: Path,
) -> None:
    target_cols: dict[str, list[str]] = {}
    for col in df_predictions.columns:
        if col == "ID":
            continue

        if " Untransformed" in col:
            target_name = col.replace(" Untransformed", "")
        elif "_" in col:
            target_name = col.rsplit("_", 1)[0]
        else:
            target_name = col

        if target_name not in target_cols:
            target_cols[target_name] = []
        target_cols[target_name].append(col)

    for target_name, cols in target_cols.items():
        output_dir = fold_path / "results" / "test_set_predictions" / target_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "predictions.csv"
        df_target = df_predictions[["ID"] + cols].copy()

        rename_map = {}
        for col in cols:
            if " Untransformed" in col:
                new_name = f"{target_name} Untransformed"
                rename_map[col] = new_name
            elif "_" in col:
                class_name = col.rsplit("_", 1)[1]
                rename_map[col] = class_name

        if rename_map:
            df_target = df_target.rename(columns=rename_map)

        df_target.to_csv(output_file, index=False)


def run_serve_predict(
    genotype_data_path: Path,
    unpacked_experiment_path: Path,
    output_folder: Path,
    serve_config: ServeConfig,
    device: str = "cpu",
) -> None:
    modelling_folder = unpacked_experiment_path / "modelling"
    if not modelling_folder.exists():
        raise FileNotFoundError(f"Modelling folder not found: {modelling_folder}")

    experiment_bim = get_experiment_bim_file(experiment_folder=unpacked_experiment_path)
    plink_fileset = get_plink_fileset_from_folder(folder_path=genotype_data_path)

    df_bim_input = read_bim_and_cast_dtypes(bim_file_path=plink_fileset.bim)
    df_bim_reference = read_bim_and_cast_dtypes(bim_file_path=experiment_bim)

    log_output_path = output_folder / "snp_overlap_analysis.txt"
    log_overlap(
        df_bim_prd=df_bim_input,
        df_bim_exp=df_bim_reference,
        output_path=log_output_path,
    )

    fold_dirs = sorted(_iterdir_ignore_hidden(path=modelling_folder))
    logger.info(f"Found {len(fold_dirs)} folds to process")

    for fold_idx, fold_path in enumerate(fold_dirs):
        logger.info(
            f"Processing fold {fold_idx + 1}/{len(fold_dirs)}: {fold_path.name}"
        )

        model_path = get_saved_model_path(fold_path=fold_path)
        input_name = get_genotype_input_name(model_path=model_path)
        port = serve_config.base_port + fold_idx

        process = start_serve_process(
            model_path=model_path,
            port=port,
            host=serve_config.host,
            device=device,
        )

        try:
            if not wait_for_server_ready(
                host=serve_config.host,
                port=port,
                timeout=serve_config.startup_timeout,
            ):
                stderr = process.stderr.read() if process.stderr else "N/A"
                raise RuntimeError(
                    f"Server failed to start for {fold_path.name}. Stderr: {stderr}"
                )

            df_predictions = stream_and_predict(
                plink_fileset=plink_fileset,
                reference_bim=experiment_bim,
                host=serve_config.host,
                port=port,
                input_name=input_name,
                batch_size=serve_config.batch_size,
                chunk_size=serve_config.chunk_size,
                request_timeout=serve_config.request_timeout,
            )

            reorganize_predictions_for_gather(
                df_predictions=df_predictions,
                fold_path=fold_path,
            )

            logger.info(f"Completed fold {fold_path.name}")

        finally:
            stop_serve_process(process=process)
