from pathlib import Path

import pandas as pd


def get_saved_model_path(run_folder: Path) -> str:
    model_path = next((run_folder / "saved_models").iterdir())

    assert model_path.suffix == ".pt"

    return str(model_path)


def post_process_csv_files(folder: Path) -> None:
    for path in folder.rglob("*.csv"):
        df = pd.read_csv(path)
        df = df.round(4)
        df.to_csv(path, index=False)
