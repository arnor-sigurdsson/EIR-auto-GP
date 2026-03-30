from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import luigi
import pandas as pd
import polars as pl

from eir_auto_gp.multi_task.modelling.hyperparameters import (
    _get_checkpoint_interval,
    _get_compile_model,
    _get_learning_rate,
    _get_supported_precision,
    build_random_snp_subset_file,
    get_gln_kernel_parameters,
)
from eir_auto_gp.preprocess.converge import get_batch_size, get_dynamic_valid_size
from eir_auto_gp.single_task.modelling.run_modelling import lines_in_file
from eir_auto_gp.utils.shared_modelling_utils import (
    get_bim_path,
    get_dataloader_workers,
    get_device,
    get_memory_dataset,
    get_samples_per_epoch,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


@dataclass
class MultiTaskModelInjectionParams:
    fold: int
    modelling_base_output_folder: str
    output_folder: str
    manual_valid_ids_file: str | None
    genotype_input_source: str
    genotype_feature_selection: str
    label_file_path: str
    input_cat_columns: list[str]
    input_con_columns: list[str]
    output_cat_columns: list[str]
    output_con_columns: list[str]
    weighted_sampling_columns: list[str] | None
    modelling_data_format: str
    output_configs: list[dict[str, Any]]
    batch_size: int | None
    optimize_model: bool
    genotype_only_test: bool = False


def build_injection_params(
    fold: int,
    data_input_dict: dict[str, luigi.LocalTarget],
    task: Literal["train", "test"],
    data_config: dict[str, Any],
    modelling_config: dict[str, Any],
    output_configs: list[dict[str, Any]],
) -> MultiTaskModelInjectionParams:
    weighted_sampling_columns = get_weighted_sampling_columns(
        modelling_config=modelling_config
    )

    base_output_folder = modelling_config["modelling_output_folder"]
    cur_run_output_folder = f"{base_output_folder}/fold_{fold}"

    label_file_path = data_input_dict[f"{task}_tabular"].path

    manual_valid_ids_file = get_manual_valid_ids_file(
        task=task,
        data_config=data_config,
        label_file_path=label_file_path,
        base_output_folder=base_output_folder,
    )

    genotype_only_test = task == "test" and modelling_config.get(
        "genotype_only_test", False
    )

    params = MultiTaskModelInjectionParams(
        fold=fold,
        output_folder=cur_run_output_folder,
        modelling_base_output_folder=base_output_folder,
        manual_valid_ids_file=str(manual_valid_ids_file),
        genotype_input_source=data_input_dict[f"{task}_genotype"].path,
        label_file_path=label_file_path,
        genotype_feature_selection=modelling_config["genotype_feature_selection"],
        input_cat_columns=modelling_config["input_cat_columns"],
        input_con_columns=modelling_config["input_con_columns"],
        output_cat_columns=modelling_config["output_cat_columns"],
        output_con_columns=modelling_config["output_con_columns"],
        weighted_sampling_columns=weighted_sampling_columns,
        modelling_data_format=data_config["modelling_data_format"],
        output_configs=output_configs,
        batch_size=modelling_config["batch_size"],
        optimize_model=modelling_config["optimize_model"],
        genotype_only_test=genotype_only_test,
    )

    return params


def get_weighted_sampling_columns(
    modelling_config: dict[str, Any],
) -> list[str] | None:
    weighted_sampling = modelling_config["weighted_sampling"]

    if weighted_sampling == "true":
        return ["all"]
    elif weighted_sampling == "false":
        return None
    elif weighted_sampling == "auto":
        cat_cols = modelling_config["output_cat_columns"]
        is_single_cat = len(cat_cols) == 1
        return ["all"] if is_single_cat else None
    else:
        raise ValueError(
            f"Invalid weighted_sampling value: {weighted_sampling}. "
            f"Expected one of: 'auto', 'true', 'false'"
        )


def get_manual_valid_ids_file(
    task: str,
    data_config: dict[str, Any],
    label_file_path: str,
    base_output_folder: str,
) -> str | None:
    if task != "train":
        return None

    valid_ids_file = Path(data_config["data_output_folder"], "ids/valid_ids.txt")
    if not valid_ids_file.exists():
        return None

    manual_valid_ids = pd.read_csv(valid_ids_file, header=None)[0].tolist()
    df_ids = pd.read_csv(label_file_path, usecols=["ID"]).dropna()
    intersected_ids = list(
        set(manual_valid_ids).intersection(set(df_ids["ID"].tolist()))
    )

    manual_valid_ids_file = Path(base_output_folder, "tmp", "valid_ids.txt")
    manual_valid_ids_file.parent.mkdir(parents=True, exist_ok=True)

    with open(manual_valid_ids_file, "w") as f:
        for id_ in intersected_ids:
            f.write(f"{id_}\n")

    return str(manual_valid_ids_file)


def _get_global_injections(
    fold: int,
    output_folder: str,
    valid_size: int,
    batch_size: int,
    manual_valid_ids_file: str | None,
    n_snps: int,
    n_samples: int,
    iter_per_epoch: int,
    weighted_sampling_columns: list[str] | None,
    modelling_data_format: str,
    optimize_model: bool,
) -> dict[str, Any]:
    mixing_candidates = [0.0]
    cur_mixing = mixing_candidates[fold % len(mixing_candidates)]

    device = get_device()

    if modelling_data_format == "auto":
        memory_dataset = get_memory_dataset(n_snps=n_snps, n_samples=n_samples)
    elif modelling_data_format == "disk":
        memory_dataset = False
    elif modelling_data_format == "memory":
        memory_dataset = True
    else:
        raise ValueError(f"Unknown data format: '{modelling_data_format}'.")

    n_workers = get_dataloader_workers(memory_dataset=memory_dataset, device=device)
    early_stopping_buffer = min(5000, iter_per_epoch * 5)
    early_stopping_buffer = max(early_stopping_buffer, 1000)

    sample_interval = _get_checkpoint_interval(iter_per_epoch=iter_per_epoch)
    logger.info(
        "Evaluation interval set to %d iterations (%.1f evaluations/epoch) "
        "for %d iterations/epoch.",
        sample_interval,
        iter_per_epoch / sample_interval,
        iter_per_epoch,
    )

    lr = _get_learning_rate(n_snps=n_snps)
    precision = _get_supported_precision(optimize_model=optimize_model)
    compile_model = _get_compile_model(optimize_model=optimize_model)

    injections = {
        "basic_experiment": {
            "output_folder": output_folder,
            "device": device,
            "batch_size": batch_size,
            "valid_size": valid_size,
            "manual_valid_ids_file": manual_valid_ids_file,
            "dataloader_workers": n_workers,
            "memory_dataset": memory_dataset,
        },
        "optimization": {
            "lr": lr,
        },
        "evaluation_checkpoint": {
            "sample_interval": sample_interval,
            "checkpoint_interval": sample_interval,
        },
        "training_control": {
            "mixing_alpha": cur_mixing,
            "early_stopping_buffer": early_stopping_buffer,
            "weighted_sampling_columns": weighted_sampling_columns,
            **({"manifold_mixup_layer_groups": None} if cur_mixing <= 0 else {}),
        },
        "model": {
            "compile_model": compile_model,
        },
        "accelerator": {
            "precision": precision,
        },
    }

    return injections


def _get_genotype_injections(
    input_source: str,
    n_snps: int,
    subset_snp_path: Path | None,
    expert_snp_groups_file: str | None = None,
) -> dict[str, Any]:
    base_snp_path = (
        Path(input_source).parent.parent / "processed/parsed_files/data_final.bim"
    )
    assert base_snp_path.exists(), f"SNP file not found at {base_snp_path}"

    kernel_width, first_kernel_expansion = get_gln_kernel_parameters(n_snps=n_snps)

    injections = {
        "input_info": {
            "input_source": input_source,
        },
        "input_type_info": {
            "snp_file": str(base_snp_path),
        },
        "model_config": {
            "model_init_config": {
                "kernel_width": kernel_width,
                "first_kernel_expansion": first_kernel_expansion,
            }
        },
    }

    if expert_snp_groups_file:
        injections["input_type_info"]["expert_snp_groups_file"] = expert_snp_groups_file
    elif subset_snp_path:
        injections["input_type_info"]["subset_snps_file"] = str(subset_snp_path)

    return injections


def _get_tabular_injections(
    input_source: str, input_cat_columns: list[str], input_con_columns: list[str]
) -> dict[str, Any]:
    injections = {
        "input_info": {
            "input_source": input_source,
        },
        "input_type_info": {
            "input_cat_columns": input_cat_columns,
            "input_con_columns": input_con_columns,
        },
    }
    return injections


def _get_output_injections(
    label_file_path: str,
    output_cat_columns: list[str],
    output_con_columns: list[str],
    use_weighted_sampling: bool = False,
) -> dict[str, Any]:
    if output_cat_columns:
        df = pl.scan_csv(source=label_file_path).select(output_cat_columns).collect()

        all_binary = all(is_binary_column(df=df, col=col) for col in output_cat_columns)

        cat_loss = "BCEWithLogitsLoss" if all_binary else "CrossEntropyLoss"
        if all_binary:
            logger.info("Setting categorical loss to BCEWithLogitsLoss.")
    else:
        cat_loss = "CrossEntropyLoss"

    use_cb = cat_loss == "BCEWithLogitsLoss" and not use_weighted_sampling

    injections = {
        "output_info": {
            "output_source": label_file_path,
        },
        "output_type_info": {
            "target_cat_columns": output_cat_columns,
            "target_con_columns": output_con_columns,
            "cat_loss_name": cat_loss,
            "cat_loss_class_balanced": use_cb,
            "uncertainty_weighted_mt_loss": False,
        },
    }

    return injections


def is_binary_column(df: pl.DataFrame, col: str) -> bool:
    n_unique = df.select(pl.col(col)).filter(pl.col(col).is_not_null()).unique().height
    return n_unique <= 2


def _get_all_dynamic_injections(
    injection_params: MultiTaskModelInjectionParams,
    genotype_data_path: str,
    expert_snp_groups_file: str | None = None,
) -> dict[str, Any]:
    mip = injection_params

    spe = get_samples_per_epoch(model_injection_params=mip)

    if mip.batch_size is not None:
        batch_size = mip.batch_size
    else:
        batch_size = get_batch_size(samples_per_epoch=spe.samples_per_epoch)

    valid_size = get_dynamic_valid_size(
        num_samples_per_epoch=spe.samples_per_epoch,
        minimum=batch_size,
    )
    iter_per_epoch = get_num_iter_per_epoch(
        num_samples_per_epoch=spe.samples_per_epoch,
        num_samples_total=spe.num_samples_total,
        batch_size=batch_size,
        valid_size=valid_size,
    )
    bim_path = get_bim_path(genotype_data_path=genotype_data_path)

    subset_folder = Path(mip.modelling_base_output_folder) / "snp_subset_files"

    subset_snp_path = None
    if mip.genotype_feature_selection == "random":
        n_snps, subset_snp_path = build_random_snp_subset_file(
            original_bim_path=Path(bim_path),
            output_folder=subset_folder,
            fold=mip.fold,
            fraction_per_chr=0.1,
        )
    else:
        assert not mip.genotype_feature_selection
        n_snps = lines_in_file(file_path=bim_path)

    injections = {
        "global_config": _get_global_injections(
            fold=mip.fold,
            output_folder=mip.output_folder,
            batch_size=batch_size,
            manual_valid_ids_file=mip.manual_valid_ids_file,
            valid_size=valid_size,
            iter_per_epoch=iter_per_epoch,
            n_snps=n_snps,
            n_samples=spe.num_samples_total,
            weighted_sampling_columns=mip.weighted_sampling_columns,
            modelling_data_format=mip.modelling_data_format,
            optimize_model=mip.optimize_model,
        ),
        "input_genotype_config": _get_genotype_injections(
            input_source=mip.genotype_input_source,
            n_snps=n_snps,
            subset_snp_path=subset_snp_path,
            expert_snp_groups_file=expert_snp_groups_file,
        ),
        "fusion_config": {},
        "output_config": {},
    }

    for output_config in mip.output_configs:
        cat_cols = output_config["output_type_info"]["target_cat_columns"]
        con_cols = output_config["output_type_info"]["target_con_columns"]

        cur_config_name = output_config["output_info"]["output_name"]

        cur_injections = _get_output_injections(
            label_file_path=mip.label_file_path,
            output_cat_columns=cat_cols,
            output_con_columns=con_cols,
            use_weighted_sampling=mip.weighted_sampling_columns is not None,
        )

        injections["output_config"][cur_config_name] = cur_injections

    if (mip.input_cat_columns or mip.input_con_columns) and not mip.genotype_only_test:
        injections["input_tabular_config"] = _get_tabular_injections(
            input_source=mip.label_file_path,
            input_cat_columns=list(mip.input_cat_columns),
            input_con_columns=list(mip.input_con_columns),
        )

    return injections


def get_num_iter_per_epoch(
    num_samples_per_epoch: int,
    num_samples_total: int,
    batch_size: int,
    valid_size: int,
) -> int:
    min_iter_per_epoch = 500
    if num_samples_total < 10_000:
        min_iter_per_epoch = 100

    iter_per_epoch = (num_samples_per_epoch - valid_size) // batch_size
    iter_per_epoch = max(min_iter_per_epoch, iter_per_epoch)

    logger.info(
        "Setting iter_per_epoch to %d with %d samples and %d valid samples.",
        iter_per_epoch,
        num_samples_per_epoch,
        valid_size,
    )

    return iter_per_epoch
