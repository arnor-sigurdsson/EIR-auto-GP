from pathlib import Path
from typing import Sequence, List

from docs.doc_modules.experiments import AutoDocExperimentInfo, run_capture_and_save
from docs.doc_modules.utils import get_saved_model_path, post_process_csv_files


def get_tutorial_01_run_1_gln_info() -> AutoDocExperimentInfo:
    base_path = "docs/tutorials/tutorial_files/01_basic_tutorial"

    command = [
        "eirautogp",
        "--genotype_data_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath",
        "--label_file_path",
        "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
        "--global_output_folder",
        "eir_auto_gp_tutorials/tutorial_runs/01_basic_tutorial",
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
        "0-5",
        "--feature_selection",
        "gwas->dl",
        "--n_dl_feature_selection_folds",
        "2",
        "--do_test",
    ]

    file_copy_mapping = [
        ("hybrid_manhattan", "figures/tutorial_01_manhattan.png"),
        ("hybrid_qq", "figures/tutorial_01_hybrid_qq.png"),
        (
            ".*feature_selection.*Aggregated CAD_manhattan.*",
            "figures/dl_manhattan.png",
        ),
        ("CAD_validation_results.csv", "figures/CAD_validation_results.csv"),
        ("CAD_feature_selection.pdf", "figures/CAD_feature_selection.pdf"),
        ("CAD_test_results.csv", "figures/CAD_test_results.csv"),
    ]

    post_process_csvs = (
        post_process_csv_files,
        {
            "folder": Path("docs/tutorials/tutorial_files/01_basic_tutorial/figures"),
        },
    )

    data_output_path = Path("eir_auto_gp_tutorials/01_basic_tutorial/data/penncath.zip")

    get_data_folder = (
        run_capture_and_save,
        {
            "command": ["tree", str(data_output_path.parent), "-L", "2", "--noreport"],
            "output_path": Path(base_path) / "commands/input_folder.txt",
        },
    )

    head_label_file = (
        run_capture_and_save,
        {
            "command": [
                "head",
                "eir_auto_gp_tutorials/01_basic_tutorial/data/penncath/penncath.csv",
            ],
            "output_path": Path(base_path) / "commands/label_file.txt",
        },
    )

    get_eir_train_auto_gp_help = (
        run_capture_and_save,
        {
            "command": [
                "eirautogp",
                "--help",
            ],
            "output_path": Path(base_path) / "commands/eirautogp_help.txt",
        },
    )

    get_tutorial_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_auto_gp_tutorials/tutorial_runs/01_basic_tutorial/",
                "-L",
                "2",
                "-I",
                "*01b*",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/tutorial_folder.txt",
        },
    )

    get_feature_selection_folder = (
        run_capture_and_save,
        {
            "command": [
                "tree",
                "eir_auto_gp_tutorials/tutorial_runs/"
                "01_basic_tutorial/feature_selection",
                "-L",
                "3",
                "--noreport",
            ],
            "output_path": Path(base_path) / "commands/feature_selection_folder.txt",
        },
    )

    ade = AutoDocExperimentInfo(
        name="AUTO_1",
        data_url="https://drive.google.com/file/d/15Kgcxxm1CntoxH6Gq7Ev_KBj24izKg3p",
        data_output_path=data_output_path,
        base_path=Path(base_path),
        command=command,
        files_to_copy_mapping=file_copy_mapping,
        post_run_functions=(
            get_data_folder,
            head_label_file,
            get_eir_train_auto_gp_help,
            get_tutorial_folder,
            get_feature_selection_folder,
            post_process_csvs,
        ),
    )

    return ade


def _get_model_path_for_predict() -> str:
    run_1_output_path = "eir_auto_gp_tutorials/tutorial_runs/tutorial_01_run"
    model_path = get_saved_model_path(run_folder=Path(run_1_output_path))

    return model_path


def _add_model_path_to_command(command: List[str]) -> List[str]:
    model_path = _get_model_path_for_predict()
    command = [x.replace("FILL_MODEL", model_path) for x in command]
    return command


def get_experiments() -> Sequence[AutoDocExperimentInfo]:
    exp_1 = get_tutorial_01_run_1_gln_info()

    return [exp_1]
