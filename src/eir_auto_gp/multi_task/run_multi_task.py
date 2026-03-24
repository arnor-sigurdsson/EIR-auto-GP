import argparse
import json
import logging
import shutil
from argparse import RawTextHelpFormatter
from collections.abc import Sequence
from copy import copy
from pathlib import Path
from typing import Any

import luigi
import pandas as pd
from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.multi_task.analysis.run_analysis import RunAnalysisWrapper
from eir_auto_gp.multi_task.custom_config import CustomConfig
from eir_auto_gp.preprocess.converge import ParseDataWrapper
from eir_auto_gp.preprocess.gwas_pre_selection import validate_geno_data_path
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)

luigi_logger = logging.getLogger("luigi")
luigi_logger.setLevel(logging.INFO)


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--genotype_data_path",
        type=str,
        required=True,
        help="Root path to raw genotype data to be processed\n"
        "(e.g., containing my_data.bed, my_data.fam, my_data.bim).\n"
        "For this example, this parameter should be\n"
        "'/path/to/raw/genotype/data/'.\n"
        "Note that the file names are not included in this path,\n"
        "only the root folder. The file names are inferred, and\n"
        "*only one* set of files is expected.",
    )

    parser.add_argument(
        "--genotype_processing_chunk_size",
        type=int,
        default=1024,
        help="Chunk size for processing genotype data. Increasing"
        "this value will increase the memory usage, but will"
        "likely speed up the processing.",
    )

    parser.add_argument(
        "--label_file_path",
        type=str,
        required=True,
        help="File path to label file with tabular inputs and labels to predict.",
    )

    parser.add_argument(
        "--only_data",
        action="store_true",
        required=False,
        help="If this flag is set, only the data processing step will be run.",
    )

    parser.add_argument(
        "--data_storage_format",
        type=str,
        choices=["disk"],
        default="disk",
    )

    parser.add_argument(
        "--global_output_folder",
        type=str,
        required=False,
        help="Common root folder to save data, feature selection and modelling results"
        " in.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="genotype",
        help="Name used for dataset.",
    )

    parser.add_argument(
        "--pre_split_folder",
        type=str,
        required=False,
        help="If there is a pre-split folder, this will be used to\n"
        "split the data into train/val and test sets. If not,\n"
        "the data will be split randomly.\n"
        "The folder should contain the following files:\n"
        "  - train_ids.txt: List of sample IDs to use for training.\n"
        "  - test_ids.txt: List of sample IDs to use for testing.\n"
        "  - (Optional): valid_ids.txt: List of sample IDs to use for "
        "validation.\n"
        "If this option is not specified, the data will be split randomly\n"
        "into 90/10 (train+val)/test sets.",
    )

    parser.add_argument(
        "--freeze_validation_set",
        help="If this flag is set, the validation set will be frozen\n"
        "and not changed between DL training folds.\n"
        "This only has an effect if the validation set is not specified\n"
        "in as a valid_ids.txt in file the pre_split_folder.\n"
        "If this flag is not set, the validation set will be randomly\n"
        "selected from the training set each time in each DL training run fold.\n"
        "This also has an effect when GWAS is used in feature selection.\n"
        "If the validation set is not specified manually or this flag is set,\n"
        "the GWAS will be performed on the training *and* validation set.\n"
        "This might potentially inflate the results on the validation set,\n"
        "particularly if the dataset is small. To turn off this behavior,\n"
        "you can use the --no-freeze_validation_set flag.",
        default=True,
        action="store_true",
    )

    parser.add_argument(
        "--genotype_feature_selection",
        type=str,
        default="",
        help="Feature selection method to use for genotype data.\n"
        "Options are:\n"
        "  - 'random': Randomly choose 10%% of each chromosome.\n",
    )

    parser.add_argument(
        "--folds",
        type=str,
        default="0-5",
        help="Training runs / folds to run, can be a single fold (e.g. 0),\n"
        "a range of folds (e.g. 0-5), or a comma-separated list of \n"
        "folds (e.g. 0,1,2,3,4,5).",
    )

    parser.add_argument(
        "--input_cat_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of categorical columns to use as input.",
    )

    parser.add_argument(
        "--input_con_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of continuous columns to use as input.",
    )

    parser.add_argument(
        "--output_cat_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of categorical columns to use as output.",
    )

    parser.add_argument(
        "--output_con_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of continuous columns to use as output.",
    )

    parser.add_argument(
        "--output_groups",
        type=str,
        required=False,
        help="A .yaml file containing the groups for output targets, each group "
        "will use a shared output branch in the model. The file should be in the "
        "following format:\n"
        "group_1:\n"
        "  - target_1\n"
        "  - target_2\n"
        "group_2:\n"
        "  - target_3\n"
        "  - target_4\n",
    )

    parser.add_argument(
        "--n_random_output_groups",
        type=int,
        default=2,
        help="Number of random groups to create when using 'random' for output_groups.",
    )

    parser.add_argument(
        "--model_size",
        type=str,
        default="mini",
        help="Model size to use for training.",
        choices=[
            "nano",
            "mini",
            "small",
            "medium",
            "large",
            "xlarge",
        ],
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to run test set prediction.",
    )

    parser.add_argument(
        "--genotype_only_test",
        action="store_true",
        help="When testing (--do_test), predict using only genotype data,\n"
        "excluding tabular covariates. This simulates deployment to new cohorts\n"
        "where covariates may not be available. Useful for validating that the\n"
        "dropout strategy enables robust genotype-only predictions.",
    )

    parser.add_argument(
        "--categorical_as_survival",
        action="store_true",
        help="Treat categorical output columns as survival (time-to-event) targets.\n"
        "When set, each categorical column 'X' is treated as an event indicator\n"
        "(0=censored, 1=event) and a corresponding time column 'X_Time' must\n"
        "exist in the label file. Uses Cox PH loss by default.",
    )

    parser.add_argument(
        "--custom_config",
        type=str,
        required=False,
        help="Path to a YAML file with advanced model and training configuration.\n"
        "See CustomConfig class documentation for available keys and defaults.",
    )

    return parser


def get_cl_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    cl_args = parser.parse_args()

    return cl_args


def load_custom_config(cl_args: argparse.Namespace) -> CustomConfig:
    if cl_args.custom_config:
        return CustomConfig.from_yaml(path=cl_args.custom_config)
    return CustomConfig()


def validate_column_duplicates(
    input_cat_columns: list[str],
    input_con_columns: list[str],
    output_cat_columns: list[str],
    output_con_columns: list[str],
) -> None:
    def _check_duplicates_within(columns: list[str], category: str) -> None:
        seen_lower = {}
        for col in columns:
            col_lower = col.lower()
            if col_lower in seen_lower:
                raise ValueError(
                    f"Duplicate column names found in {category} "
                    f"(case-insensitive): {sorted([seen_lower[col_lower], col])}"
                )
            seen_lower[col_lower] = col

    _check_duplicates_within(input_cat_columns, "input categorical columns")
    _check_duplicates_within(input_con_columns, "input continuous columns")
    _check_duplicates_within(output_cat_columns, "output categorical columns")
    _check_duplicates_within(output_con_columns, "output continuous columns")

    all_columns = (
        input_cat_columns + input_con_columns + output_cat_columns + output_con_columns
    )
    seen_global_lower = {}
    for col in all_columns:
        col_lower = col.lower()
        if col_lower in seen_global_lower:
            raise ValueError(
                f"Column names must be unique across all input and output columns "
                f"(case-insensitive). Found duplicates: "
                f"{sorted([seen_global_lower[col_lower], col])}"
            )
        seen_global_lower[col_lower] = col


def validate_label_file(
    label_file_path: str,
    input_cat_columns: list[str],
    input_con_columns: list[str],
    output_cat_columns: list[str],
    output_con_columns: list[str],
    categorical_as_survival: bool = False,
) -> None:
    if not Path(label_file_path).exists():
        raise ValueError(
            f"Label file path {label_file_path} is invalid. "
            f"Expected to find {label_file_path}."
        )

    columns = pd.read_csv(label_file_path, nrows=1).columns
    if "ID" not in columns:
        raise ValueError(
            f"Label file path {label_file_path} is invalid. "
            f"Expected to find 'ID' column."
        )

    all_columns = set(
        input_cat_columns + input_con_columns + output_cat_columns + output_con_columns
    )

    if categorical_as_survival:
        time_columns = {f"{col}_Time" for col in output_cat_columns}
        all_columns |= time_columns

    missing_columns = all_columns - set(columns)
    if len(missing_columns) > 0:
        raise ValueError(
            f"Label file path {label_file_path} is invalid. "
            f"Expected to find columns {missing_columns}."
        )


def validate_targets(
    output_con_columns: list[str], output_cat_columns: list[str], only_data: bool
) -> None:
    if not only_data and len(output_con_columns) == 0 and len(output_cat_columns) == 0:
        raise ValueError(
            "At least one output column must be specified as continuous or categorical."
        )


def validate_plink2_exists_in_path() -> None:
    if shutil.which("plink2") is None:
        raise ValueError(
            "plink2 is not installed or not in the path. "
            "Please install plink2 and try again."
        )


def validate_pre_split_folder(pre_split_folder: str | None) -> None:
    if not pre_split_folder:
        return

    ids = {}
    for file in ["train_ids.txt", "valid_ids.txt", "test_ids.txt"]:
        if not Path(pre_split_folder, file).exists():
            continue
        with open(Path(pre_split_folder, file)) as f:
            ids[file] = set(f.read().splitlines())

    if len(ids) == 0:
        raise ValueError(
            f"Pre-split folder {pre_split_folder} is invalid. "
            f"Expected to find at least train_ids.txt and test_ids.txt."
        )

    assert "train_ids.txt" in ids
    assert "test_ids.txt" in ids

    if "valid_ids.txt" in ids:
        assert ids["valid_ids.txt"].isdisjoint(ids["train_ids.txt"])
        assert ids["valid_ids.txt"].isdisjoint(ids["test_ids.txt"])

    assert ids["train_ids.txt"].isdisjoint(ids["test_ids.txt"])


def validate_model_architecture_params(
    model_size: str,
    custom_config: CustomConfig,
) -> None:
    granular_param_names = [
        "n_fusion_layers",
        "fusion_dim",
        "skip_to_every_n_fusion_layers",
        "n_output_layers",
        "output_dim",
    ]
    granular_params = [getattr(custom_config, name) for name in granular_param_names]

    any_granular_specified = any(p is not None for p in granular_params)
    all_granular_specified = all(p is not None for p in granular_params)

    if any_granular_specified and model_size != "mini":
        raise ValueError(
            "Cannot use model_size with granular architecture parameters. "
            "Either use model_size alone or specify all granular parameters "
            "in the custom config file "
            "(n_fusion_layers, fusion_dim, skip_to_every_n_fusion_layers, "
            "n_output_layers, output_dim)."
        )

    if any_granular_specified and not all_granular_specified:
        missing = [
            name
            for name, val in zip(granular_param_names, granular_params, strict=False)
            if val is None
        ]
        raise ValueError(
            f"When using granular architecture parameters, all must be specified. "
            f"Missing: {', '.join(missing)}"
        )


def run(cl_args: argparse.Namespace, custom_config: CustomConfig) -> None:
    validate_geno_data_path(geno_data_path=cl_args.genotype_data_path)
    validate_column_duplicates(
        input_cat_columns=cl_args.input_cat_columns,
        input_con_columns=cl_args.input_con_columns,
        output_cat_columns=cl_args.output_cat_columns,
        output_con_columns=cl_args.output_con_columns,
    )
    validate_label_file(
        label_file_path=cl_args.label_file_path,
        input_cat_columns=cl_args.input_cat_columns,
        input_con_columns=cl_args.input_con_columns,
        output_cat_columns=cl_args.output_cat_columns,
        output_con_columns=cl_args.output_con_columns,
        categorical_as_survival=cl_args.categorical_as_survival,
    )
    validate_targets(
        output_con_columns=cl_args.output_con_columns,
        output_cat_columns=cl_args.output_cat_columns,
        only_data=cl_args.only_data,
    )
    validate_plink2_exists_in_path()
    validate_pre_split_folder(pre_split_folder=cl_args.pre_split_folder)
    validate_model_architecture_params(
        model_size=cl_args.model_size,
        custom_config=custom_config,
    )

    cl_args = parse_output_folders(cl_args=cl_args)
    cl_args = _add_pre_split_folder_if_present(cl_args=cl_args)

    data_config = build_data_config(cl_args=cl_args, custom_config=custom_config)
    modelling_config = build_modelling_config(
        cl_args=cl_args, custom_config=custom_config
    )
    analysis_config = build_analysis_config(cl_args=cl_args)
    root_task = get_root_task(
        folds=cl_args.folds,
        data_config=data_config,
        modelling_config=modelling_config,
        analysis_config=analysis_config,
    )

    luigi.build(
        tasks=[root_task],
        workers=1,
        local_scheduler=True,
        log_level="INFO",
    )


def get_root_task(
    data_config: dict,
    folds: int,
    modelling_config: dict,
    analysis_config: dict,
) -> RunAnalysisWrapper | ParseDataWrapper:
    if data_config.get("only_data"):
        return ParseDataWrapper(
            data_config=data_config, modelling_config=modelling_config
        )

    return RunAnalysisWrapper(
        folds=folds,
        data_config=data_config,
        modelling_config=modelling_config,
        analysis_config=analysis_config,
    )


def main():
    parser = get_argument_parser()
    cl_args = get_cl_args(parser=parser)
    custom_config = load_custom_config(cl_args=cl_args)
    store_experiment_config(cl_args=cl_args, custom_config=custom_config)

    run(cl_args=cl_args, custom_config=custom_config)


def store_experiment_config(
    cl_args: argparse.Namespace,
    custom_config: CustomConfig,
) -> None:
    output_folder = Path(cl_args.global_output_folder)
    ensure_path_exists(path=output_folder, is_folder=True)

    if cl_args.custom_config:
        custom_config_copy = output_folder / "custom_config.yaml"
        shutil.copy2(src=cl_args.custom_config, dst=custom_config_copy)
        logger.info("Copied custom config to %s", custom_config_copy)

    config_dict = {**vars(cl_args), **custom_config.to_dict()}
    output_path = output_folder / "config.json"

    if output_path.exists():
        logger.warning(
            f"Output config file {output_path} already exists. Overwriting it."
        )

    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=4)


def parse_output_folders(cl_args: argparse.Namespace) -> argparse.Namespace:
    cl_args_copy = copy(cl_args)
    if cl_args_copy.global_output_folder:
        gof = cl_args_copy.global_output_folder.rstrip("/")
        cl_args_copy.data_output_folder = gof + "/data"
        cl_args_copy.modelling_output_folder = gof + "/modelling"
        cl_args_copy.analysis_output_folder = gof + "/analysis"
    else:
        raise ValueError(
            "Global output folder must be specified. "
            "Please specify the --global_output_folder parameter."
        )

    return cl_args_copy


def _add_pre_split_folder_if_present(cl_args: argparse.Namespace) -> argparse.Namespace:
    cl_args_copy = copy(cl_args)
    genotype_path = Path(cl_args_copy.genotype_data_path)

    id_path = genotype_path / "ids"
    if id_path.exists():
        cl_args_copy.pre_split_folder = str(genotype_path / "ids")

        found_files = [
            i.name
            for i in id_path.iterdir()
            if i.is_file() and i.name.endswith(".txt") and "_plink" not in i.name
        ]

        logger.info(
            f"Found pre-split folder '{cl_args_copy.pre_split_folder}'. "
            f"in root genotype folder. "
            f"Using files: {found_files} for respective splits."
        )

    return cl_args_copy


def build_data_config(
    cl_args: argparse.Namespace,
    custom_config: CustomConfig,
) -> dict[str, Any]:
    data_keys = [
        "genotype_data_path",
        "label_file_path",
        "data_output_folder",
        "output_name",
        "only_data",
        "pre_split_folder",
        "freeze_validation_set",
        "genotype_processing_chunk_size",
        "data_storage_format",
    ]

    base = extract_from_namespace(namespace=cl_args, keys=data_keys)
    base["modelling_data_format"] = custom_config.modelling_data_format

    return base


def build_modelling_config(
    cl_args: argparse.Namespace,
    custom_config: CustomConfig,
) -> dict[str, Any]:
    modelling_keys = [
        "modelling_output_folder",
        "input_cat_columns",
        "input_con_columns",
        "output_cat_columns",
        "output_con_columns",
        "output_groups",
        "n_random_output_groups",
        "genotype_feature_selection",
        "model_size",
        "do_test",
        "genotype_only_test",
        "categorical_as_survival",
    ]

    config = extract_from_namespace(namespace=cl_args, keys=modelling_keys)
    config.update(custom_config.to_dict())

    return config


def build_analysis_config(cl_args: argparse.Namespace) -> dict[str, Any]:
    analysis_keys = [
        "analysis_output_folder",
    ]

    return extract_from_namespace(namespace=cl_args, keys=analysis_keys)


def extract_from_namespace(
    namespace: argparse.Namespace, keys: Sequence[str]
) -> dict[str, Any]:
    return {key: getattr(namespace, key) for key in keys}


if __name__ == "__main__":
    main()
