from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from aislib.misc_utils import ensure_path_exists

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save


def get_tutorial_03_run_data_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/03_feature_selection_tutorial/data_run"

    command = [
        "eirautogp",
        "--genotype_data_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath",
        "--label_file_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
        "--global_output_folder",
        "eir_auto_gp_tutorials/tutorial_runs/03_feature_selection_data",
        "--only_data",
        "--freeze_validation_set",
    ]

    file_copy_mapping = []

    data_output_path = Path("eir_auto_gp_tutorials/01_basic_tutorial/data/penncath.zip")

    get_data_folder = (
        run_capture_and_save,
        {
            "command": ["tree", str(data_output_path.parent), "-L", "2", "--noreport"],
            "output_path": Path(base_path) / "commands/input_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="AUTO_1_DATA",
        data_url="https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p",
        data_output_path=data_output_path,
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=file_copy_mapping,
        post_run_functions=(get_data_folder,),
    )

    return ade


def get_feature_selection_tutorial_info(
    feature_selection_method: Optional[str],
) -> AutoDocExperimentInfo:
    feature_selection_method_str = (
        feature_selection_method if feature_selection_method is not None else "none"
    )

    base_path = (
        f"docs/tutorials/tutorial_files/03_feature_selection_tutorial/"
        f"{feature_selection_method_str}"
    )

    fsms = _parse_feature_selection_string(feature_selection_method_str)

    command = [
        "eirautogp",
        "--genotype_data_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath",
        "--label_file_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
        "--data_output_folder",
        "eir_auto_gp_tutorials/tutorial_runs/03_feature_selection_data/data",
        "--feature_selection_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/"
        f"03_feature_selection_{fsms}/feature_selection",
        "--modelling_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/"
        f"03_feature_selection_{fsms}/modelling",
        "--analysis_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/03_feature_selection_{fsms}/analysis",
        "--output_cat_columns",
        "CAD",
        "--input_con_columns",
        "tg",
        "hdl",
        "ldl",
        "age",
        "--input_cat_columns",
        "sex",
        "--folds",
        "10",
        "--feature_selection",
        feature_selection_method,
        "--n_dl_feature_selection_setup_folds",
        "3",
        "--do_test",
    ]

    file_copy_mapping = []

    post_process_csvs = (
        _post_copy_test_results,
        {
            "file_path": Path(
                f"eir_auto_gp_tutorials/tutorial_runs/"
                f"03_feature_selection_{fsms}/"
                "analysis/CAD_test_results.csv"
            ),
            "output_folder": Path(base_path) / "figures",
        },
    )

    data_output_path = Path("eir_auto_gp_tutorials/01_basic_tutorial/data/penncath.zip")

    ade = AutoDocExperimentInfo(
        name=f"AUTO_FS_{feature_selection_method_str.upper()}",
        data_url="https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p",
        data_output_path=data_output_path,
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=file_copy_mapping,
        post_run_functions=(post_process_csvs,),
    )

    return ade


def _post_copy_test_results(file_path: Path, output_folder: Path) -> None:
    df = pd.read_csv(file_path)
    df = df.round(4)

    ensure_path_exists(path=output_folder, is_folder=True)
    df.to_csv(output_folder / file_path.name, index=False)


def _parse_feature_selection_string(feature_selection: str) -> str:
    mapping = {
        "gwas": "gwas",
        "gwas->dl": "gwas_then_dl",
        "gwas+bo": "gwas_and_bo",
    }

    return mapping[feature_selection]


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    """
    For nice folders and also some regex matching used later.
    """
    feature_selection_methods = [
        "gwas",
        "gwas->dl",
        "gwas+bo",
    ]

    experiments_data = [get_tutorial_03_run_data_info()]

    experiments_runs = [
        get_feature_selection_tutorial_info(method)
        for method in feature_selection_methods
    ]

    experiments = experiments_data + experiments_runs

    return experiments
