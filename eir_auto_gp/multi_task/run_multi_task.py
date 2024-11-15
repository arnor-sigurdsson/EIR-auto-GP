import argparse
import json
import logging
import shutil
from argparse import RawTextHelpFormatter
from copy import copy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import luigi
import pandas as pd
from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.multi_task.analysis.run_analysis import RunAnalysisWrapper
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
        choices=["disk", "deeplake"],
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
        "  - (Optional): valid_ids.txt: List of sample IDs to use for validation.\n"
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
        "--modelling_data_format",
        type=str,
        default="disk",
        help="Which format to store the data in during modelling.",
        choices=["disk", "memory", "auto"],
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to run test set prediction.",
    )

    return parser


def get_cl_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    cl_args = parser.parse_args()

    return cl_args


def validate_label_file(
    label_file_path: str,
    input_cat_columns: list[str],
    input_con_columns: list[str],
    output_cat_columns: list[str],
    output_con_columns: list[str],
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


def validate_pre_split_folder(pre_split_folder: Optional[str]) -> None:
    if not pre_split_folder:
        return

    ids = {}
    for file in ["train_ids.txt", "valid_ids.txt", "test_ids.txt"]:
        if not Path(pre_split_folder, file).exists():
            continue
        with open(Path(pre_split_folder, file), "r") as f:
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


def run(cl_args: argparse.Namespace) -> None:
    validate_geno_data_path(geno_data_path=cl_args.genotype_data_path)
    validate_label_file(
        label_file_path=cl_args.label_file_path,
        input_cat_columns=cl_args.input_cat_columns,
        input_con_columns=cl_args.input_con_columns,
        output_cat_columns=cl_args.output_cat_columns,
        output_con_columns=cl_args.output_con_columns,
    )
    validate_targets(
        output_con_columns=cl_args.output_con_columns,
        output_cat_columns=cl_args.output_cat_columns,
        only_data=cl_args.only_data,
    )
    validate_plink2_exists_in_path()
    validate_pre_split_folder(pre_split_folder=cl_args.pre_split_folder)

    cl_args = parse_output_folders(cl_args=cl_args)
    cl_args = _add_pre_split_folder_if_present(cl_args=cl_args)

    data_config = build_data_config(cl_args=cl_args)
    modelling_config = build_modelling_config(cl_args=cl_args)
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
    data_config: Dict,
    folds: int,
    modelling_config: Dict,
    analysis_config: Dict,
) -> RunAnalysisWrapper | ParseDataWrapper:
    if data_config.get("only_data"):
        return ParseDataWrapper(data_config=data_config)

    return RunAnalysisWrapper(
        folds=folds,
        data_config=data_config,
        modelling_config=modelling_config,
        analysis_config=analysis_config,
    )


def main():
    parser = get_argument_parser()
    cl_args = get_cl_args(parser=parser)
    store_experiment_config(cl_args=cl_args)

    run(cl_args=cl_args)


def store_experiment_config(
    cl_args: argparse.Namespace,
) -> None:
    config_dict = vars(cl_args)

    output_folder = Path(cl_args.global_output_folder)

    ensure_path_exists(path=output_folder, is_folder=True)
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


def build_data_config(cl_args: argparse.Namespace) -> Dict[str, Any]:
    data_keys = [
        "genotype_data_path",
        "label_file_path",
        "data_output_folder",
        "output_name",
        "only_data",
        "pre_split_folder",
        "freeze_validation_set",
        "genotype_processing_chunk_size",
        "modelling_data_format",
        "data_storage_format",
    ]

    base = extract_from_namespace(namespace=cl_args, keys=data_keys)

    return base


def build_modelling_config(cl_args: argparse.Namespace) -> Dict[str, Any]:
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
    ]

    return extract_from_namespace(namespace=cl_args, keys=modelling_keys)


def build_analysis_config(cl_args: argparse.Namespace) -> Dict[str, Any]:
    analysis_keys = [
        "analysis_output_folder",
    ]

    return extract_from_namespace(namespace=cl_args, keys=analysis_keys)


def extract_from_namespace(
    namespace: argparse.Namespace, keys: Sequence[str]
) -> Dict[str, Any]:
    return {key: getattr(namespace, key) for key in keys}


if __name__ == "__main__":
    main()
