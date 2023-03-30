from pathlib import Path
from typing import Optional, Tuple, Iterator, Literal

import pandas as pd
from aislib.misc_utils import ensure_path_exists
from eir.train_utils.train_handlers import _iterdir_ignore_hidden
from eir.visualization.interpretation_visualization import plot_snp_manhattan_plots
from skopt import Optimizer

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def get_genotype_subset_snps_file(
    fold: int,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    bim_file: str | Path,
    feature_selection_approach: Literal["dl", "gwas", "gwas->dl", None],
    n_dl_feature_selection_setup_folds: int,
    manual_subset_from_gwas: Optional[str | Path],
) -> Optional[Path]:
    match feature_selection_approach:
        case None:
            return None
        case "gwas":
            return manual_subset_from_gwas
        case "dl" | "gwas->dl":
            if feature_selection_approach == "gwas->dl":
                assert manual_subset_from_gwas is not None

            computed_subset_file = run_dl_bo_selection(
                fold=fold,
                folder_with_runs=folder_with_runs,
                feature_selection_output_folder=feature_selection_output_folder,
                bim_file=bim_file,
                n_dl_feature_selection_setup_folds=n_dl_feature_selection_setup_folds,
                manual_subset_from_gwas=manual_subset_from_gwas,
            )
            return computed_subset_file


def run_dl_bo_selection(
    fold: int,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    bim_file: str | Path,
    n_dl_feature_selection_setup_folds: int,
    manual_subset_from_gwas: Optional[str | Path],
) -> Optional[Path]:
    fs_out_folder = feature_selection_output_folder
    subsets_out_folder = fs_out_folder / "dl_importance" / "snp_subsets"
    snp_subset_file = subsets_out_folder / f"dl_snps_{fold}.txt"
    if snp_subset_file.exists():
        return snp_subset_file

    fractions_file = subsets_out_folder / f"dl_snps_fraction_{fold}.txt"
    if fold < n_dl_feature_selection_setup_folds:
        gwas_snps_or_none = _handle_dl_feature_selection_options(
            bim_file=bim_file,
            manual_subset_from_gwas_file=manual_subset_from_gwas,
            snp_subset_file=snp_subset_file,
            fractions_file=fractions_file,
        )
        return gwas_snps_or_none

    df_attributions = gather_eir_snp_attributions(folder_with_runs=folder_with_runs)

    importance_file = fs_out_folder / "dl_importance" / "dl_attributions.csv"
    ensure_path_exists(path=importance_file)
    df_attributions.to_csv(path_or_buf=importance_file)

    plot_snp_manhattan_plots(
        df_snp_grads=df_attributions,
        outfolder=importance_file.parent,
        title_extra="Aggregated",
    )

    top_n, fraction = get_auto_top_n(
        df_attributions=df_attributions,
        folder_with_runs=folder_with_runs,
        feature_selection_output_folder=feature_selection_output_folder,
        fold=fold,
    )
    logger.info("Top %d SNPs selected.", top_n)

    df_top_n = get_top_n_snp_list_df(df_attributions=df_attributions, top_n_snps=top_n)
    ensure_path_exists(path=snp_subset_file)
    df_top_n.to_csv(path_or_buf=snp_subset_file, index=False, header=False)
    fractions_file.write_text(str(fraction))

    return snp_subset_file


def _handle_dl_feature_selection_options(
    bim_file: str | Path,
    manual_subset_from_gwas_file: Optional[str | Path],
    snp_subset_file: Path,
    fractions_file: Path,
) -> Optional[Path]:
    ensure_path_exists(path=snp_subset_file)
    all_snps = _get_snps_in_bim_file(bim_file_path=bim_file)
    base_fraction = "1.0"

    if manual_subset_from_gwas_file is not None:
        logger.info("Using manual subset of SNPs.")
        gwas_snps = Path(manual_subset_from_gwas_file).read_text().splitlines()
        all_snps = gwas_snps

    snp_subset_file.write_text("\n".join(all_snps))

    ensure_path_exists(path=fractions_file)
    fractions_file.write_text(base_fraction)

    if manual_subset_from_gwas_file is not None:
        return snp_subset_file

    return None


def get_top_n_snp_list_df(
    df_attributions: pd.DataFrame, top_n_snps: int
) -> pd.DataFrame:
    target_columns = [i for i in df_attributions.columns if "Aggregated" in i]
    assert len(target_columns) == 1
    target_column = target_columns[0]

    df_top_n = df_attributions.nlargest(n=top_n_snps, columns=target_column)
    df_top_n = df_top_n.reset_index()
    df_top_n = df_top_n.rename(columns={"VAR_ID": "SNP"})
    df_top_n = df_top_n[["SNP"]]

    return df_top_n


def gather_eir_snp_attributions(folder_with_runs: Path) -> pd.DataFrame:
    """
    Currently this assumes that we have only computed the attributions on the full
    sets. Later we can add support for subset runs with attributions as well.
    """

    df_final = None
    gathered_series = {}
    counts = {}
    for fold in _iterdir_ignore_hidden(path=folder_with_runs):
        results_folder = fold / "results/eir_auto_gp"

        if not results_folder.exists():
            logger.warning("Results folder %s does not exist.", str(results_folder))
            continue

        for target_folder in _iterdir_ignore_hidden(path=results_folder):
            cur_act_iter = _get_attribution_iterator(target_folder=target_folder)

            for df_act, aggregate_name in cur_act_iter:
                if df_final is None:
                    df_final = df_act.drop(columns=[aggregate_name])

                series_act = df_act[aggregate_name]

                if aggregate_name not in gathered_series:
                    gathered_series[aggregate_name] = series_act
                    counts[aggregate_name] = 1
                else:
                    gathered_series[aggregate_name] += series_act
                    counts[aggregate_name] += 1

    for aggregate_name, series_act in gathered_series.items():
        cur_average = series_act / counts[aggregate_name]
        df_final[aggregate_name] = cur_average

    return df_final


def _get_attribution_iterator(
    target_folder: Path,
) -> Iterator[tuple[pd.DataFrame, str]]:
    for p in target_folder.rglob("*"):
        if p.name == "snp_attributions.csv":
            cur_csv = p

            df_cur = pd.read_csv(filepath_or_buffer=cur_csv)
            df_cur = df_cur.set_index("VAR_ID")
            df_cur = df_cur.drop(columns=["Unnamed: 0"])

            act_cols = [i for i in df_cur.columns if "_attributions" in i]

            aggregate_column = f"Aggregated {target_folder.name}_attributions"
            df_cur[aggregate_column] = abs(df_cur[act_cols]).sum(axis=1)

            df_cur = df_cur.drop(columns=act_cols)

            yield df_cur, aggregate_column


def _get_snps_in_bim_file(bim_file_path: str | Path) -> list[str]:
    with open(bim_file_path, "r") as f:
        snps = [line.split()[1] for line in f]
    return snps


def get_auto_top_n(
    df_attributions: pd.DataFrame,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    fold: int,
) -> Tuple[int, float]:
    manual_fractions = _get_manual_fractions(
        total_n_snps=len(df_attributions), min_snps_cutoff=16
    )

    if fold < len(manual_fractions):
        next_fraction = manual_fractions[fold]
    else:
        opt = Optimizer(dimensions=[(0.0, 1.0)])
        df_history = _gather_fractions_and_performances(
            folder_with_runs=folder_with_runs,
            feature_selection_output_folder=feature_selection_output_folder,
        )

        for t in df_history.itertuples():
            opt.tell([t.fraction], t.best_val_performance)

        next_fraction = opt.ask()[0]

    top_n = int(next_fraction * len(df_attributions))
    if top_n < 16:
        top_n = 16
        next_fraction = top_n / len(df_attributions)

    return top_n, next_fraction


def _get_manual_fractions(total_n_snps: int, min_snps_cutoff: int) -> list[float]:
    base = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]

    manual_fractions = []
    for i in base:
        cur = i * total_n_snps
        if cur >= min_snps_cutoff:
            manual_fractions.append(i)

    return manual_fractions


def _gather_fractions_and_performances(
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
        path=feature_selection_output_folder / "dl_importance" / "snp_subsets"
    ):
        if "_fraction" not in f.name:
            continue

        fold_as_int = int(f.stem.split("_")[-1])
        fraction = f.read_text()
        results[fold_as_int] = float(fraction)

    df = pd.DataFrame.from_dict(results, orient="index", columns=["fraction"])
    return df
