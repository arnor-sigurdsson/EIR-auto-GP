from pathlib import Path

import pandas as pd
from eir.train_utils.train_handlers import _iterdir_ignore_hidden


def read_gwas_df(gwas_output_folder: Path) -> pd.DataFrame:
    dfs = []
    for gwas_file in gwas_output_folder.iterdir():
        if "logistic" not in gwas_file.name and "linear" not in gwas_file.name:
            continue

        df_gwas = pd.read_csv(filepath_or_buffer=gwas_file, sep="\t")
        df_gwas = df_gwas.rename(columns={"ID": "VAR_ID"})
        df_gwas = df_gwas.set_index("VAR_ID")
        dfs.append(df_gwas)

    assert len(dfs) == 1
    df_gwas = dfs[0]

    return df_gwas


def gather_fractions_and_performances(
    folder_with_runs: Path, feature_selection_output_folder: Path
) -> pd.DataFrame:
    df_val_performances = _gather_best_val_performances(
        folder_with_runs=folder_with_runs
    )
    df_snp_fractions = _get_snp_fractions(
        feature_selection_output_folder=feature_selection_output_folder
    )

    df = df_val_performances.merge(df_snp_fractions, left_index=True, right_index=True)

    df = df.sort_index()

    return df


def _gather_best_val_performances(folder_with_runs: Path) -> pd.DataFrame:
    results = {}
    for fold in _iterdir_ignore_hidden(path=folder_with_runs):
        if not (fold / "completed_train.txt").exists():
            continue

        df_val = pd.read_csv(filepath_or_buffer=fold / "validation_average_history.log")
        best_performance = df_val["perf-average"].max()
        fold_as_int = int(fold.stem.split("_")[-1])
        results[fold_as_int] = best_performance

    df = pd.DataFrame.from_dict(
        results, orient="index", columns=["best_val_performance"]
    )
    return df


def _get_snp_fractions(feature_selection_output_folder: Path) -> pd.DataFrame:
    results = {}
    for f in _iterdir_ignore_hidden(
        path=feature_selection_output_folder / "snp_importance" / "snp_subsets"
    ):
        if "_fraction" not in f.name:
            continue

        fold_as_int = int(f.stem.split("_")[-1])
        fraction = f.read_text()
        results[fold_as_int] = float(fraction)

    df = pd.DataFrame.from_dict(results, orient="index", columns=["fraction"])
    return df
