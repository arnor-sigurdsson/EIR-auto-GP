import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

from aislib.misc_utils import ensure_path_exists, get_logger
from pdf2image import convert_from_path
from PIL.Image import Image

from docs.doc_modules.data import get_data

logger = get_logger(name=__name__)


@dataclass
class AutoDocExperimentInfo:
    name: str
    data_url: str
    data_output_path: Path
    base_path: Path
    command: list[str]
    files_to_copy_mapping: Sequence[tuple[str, str]]
    pre_run_command_modifications: Sequence[Callable[[list[str]], list[str]]] = ()
    post_run_functions: Sequence[tuple[Callable, dict]] = ()
    force_run_command: bool = False


def make_tutorial_data(auto_doc_experiment_info: AutoDocExperimentInfo) -> None:
    ade = auto_doc_experiment_info

    get_data(url=ade.data_url, output_path=ade.data_output_path)

    command = ade.command
    for command_modification in ade.pre_run_command_modifications:
        command = command_modification(ade.command)

    run_folder = run_experiment_from_command(
        command=command, force_run=ade.force_run_command
    )

    save_command_as_text(
        command=command,
        output_path=(ade.base_path / "commands" / ade.name).with_suffix(".txt"),
    )

    find_and_copy_files(
        run_folder=run_folder,
        output_folder=ade.base_path,
        patterns=ade.files_to_copy_mapping,
    )

    for func, kwargs in ade.post_run_functions:
        func(**kwargs)


def save_command_as_text(command: list[str], output_path: Path) -> None:
    ensure_path_exists(path=output_path)

    command_as_str = command[0] + " \\\n"

    cur_str = command[1]

    for part in command[2:]:
        if part.startswith("--"):
            command_as_str += cur_str + " \\\n"
            cur_str = part
        else:
            cur_str += " " + part

    command_as_str += cur_str

    output_path.write_text(command_as_str)


def set_up_conf_files(base_path: Path, conf_output_path: Path):
    ensure_path_exists(path=conf_output_path, is_folder=True)
    for path in base_path.rglob("*"):
        if path.suffix == ".yaml":
            copy2(path, conf_output_path)


def find_and_copy_files(
    run_folder: Path,
    output_folder: Path,
    patterns: Sequence[tuple[str, str]],
    strict: bool = True,
):
    matched_patterns = {}

    for path in run_folder.rglob("*"):
        for pattern, target in patterns:
            if re.match(pattern=pattern, string=str(path)) or pattern in str(path):
                output_destination = output_folder / target
                ensure_path_exists(path=output_destination, is_folder=False)
                copy2(path, output_destination)

                if output_destination.suffix == ".pdf":
                    files = convert_from_path(
                        pdf_path=output_destination,
                        fmt="png",
                    )

                    assert len(files) == 1
                    pil_image: Image = files[0]

                    pil_image.save(output_destination.with_suffix(".png"))
                    output_destination.unlink()

                matched_patterns[pattern] = True

    for pattern, _ in patterns:
        if pattern not in matched_patterns:
            if strict:
                raise FileNotFoundError(
                    f"No files found for pattern {pattern} in {run_folder}."
                )
            else:
                logger.warning(f"No files found for pattern {pattern} in {run_folder}.")


def run_experiment_from_command(command: list[str], force_run: bool = False) -> Path:
    output_folders = _find_all_outputs(command=command)
    assert output_folders is not None

    all_exist = all(folder.exists() for folder in output_folders.values())
    if "--global_output_folder" in output_folders:
        final_output_folder = output_folders["--global_output_folder"]
    else:
        assert len(output_folders) == 4
        final_output_folder = output_folders["--data_output_folder"].parent

    if not force_run and all_exist:
        return final_output_folder

    subprocess.run(args=command)

    return final_output_folder


def _find_all_outputs(command: list[str]) -> dict[str, Path]:
    targets = [
        "--global_output_folder",
        "--data_output_folder",
        "--feature_selection_output_folder",
        "--modelling_output_folder",
        "--analysis_output_folder",
    ]

    values = {}
    command_str = " ".join(command)
    for target in targets:
        regex = re.escape(target) + r"\s+([\w/]+)"
        match = re.search(regex, command_str)
        if match:
            values[target] = Path(match.group(1))

    return values


def run_capture_and_save(command: list[str], output_path: Path, *args, **kwargs):
    result_text = run_subprocess_and_capture_output(command=command)

    output_path.write_text(result_text)


def run_subprocess_and_capture_output(command: list[str]):
    result = subprocess.run(args=command, capture_output=True, text=True).stdout

    return result
