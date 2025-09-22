from collections.abc import Sequence
from pathlib import Path

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.utils import post_process_csv_files


def get_tutorial_02_run_data_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/02_multi_tutorial"

    command = [
        "eirautogp",
        "--genotype_data_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath",
        "--label_file_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
        "--global_output_folder",
        "eir_auto_gp_tutorials/tutorial_runs/02_multi_tutorial",
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
        name="AUTO_2_DATA",
        data_url="https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p",
        data_output_path=data_output_path,
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=file_copy_mapping,
        post_run_functions=(get_data_folder,),
    )

    return ade


def get_tutorial_02_run_single_target(target: str) -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/02_multi_tutorial"

    command = get_command(target=target)

    t = target
    file_copy_mapping = [
        (f"{t}_test_results.csv", f"figures/{t}_test_results.csv"),
    ]

    data_output_path = Path("eir_auto_gp_tutorials/01_basic_tutorial/data/penncath.zip")

    post_process_csvs = (
        post_process_csv_files,
        {
            "folder": Path("docs/tutorials/tutorial_files/02_multi_tutorial/figures"),
        },
    )

    ade = AutoDocExperimentInfo(
        name=f"AUTO_2_{target}",
        data_url="https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p",
        data_output_path=data_output_path,
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=file_copy_mapping,
        post_run_functions=(post_process_csvs,),
    )

    return ade


def get_tutorial_02_run_multi_targets() -> Sequence[AutoDocExperimentInfo]:
    for target in ["tg", "ldl", "hdl", "age"]:
        yield get_tutorial_02_run_single_target(target=target)


def get_command(target: str) -> list[str]:
    t = target
    command = [
        "eirautogp",
        "--genotype_data_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath",
        "--label_file_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
        "--data_output_folder",
        "eir_auto_gp_tutorials/tutorial_runs/02_multi_tutorial/data",
        "--feature_selection_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/02_multi_tutorial/{t}/feature_selection",
        "--modelling_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/02_multi_tutorial/{t}/modelling",
        "--analysis_output_folder",
        f"eir_auto_gp_tutorials/tutorial_runs/02_multi_tutorial/{t}/analysis",
        "--output_con_columns",
        f"{t}",
        "--input_cat_columns",
        "sex",
        "--folds",
        "5",
        "--feature_selection",
        "gwas",
        "--do_test",
    ]

    input_con_columns = ["--input_con_columns"]
    for candidate in ["tg", "ldl", "hdl", "age"]:
        if candidate == t:
            continue
        input_con_columns.append(candidate)

    command += input_con_columns

    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_tutorial_02_run_data_info()
    exp_2 = list(get_tutorial_02_run_multi_targets())

    return [exp_1] + exp_2
