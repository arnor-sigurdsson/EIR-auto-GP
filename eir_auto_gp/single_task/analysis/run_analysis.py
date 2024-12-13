import json
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, Iterator, Literal, Sequence, Tuple

import luigi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from aislib.misc_utils import ensure_path_exists
from eir.experiment_io.experiment_io import load_serialized_train_experiment
from eir.train_utils.metrics import al_metric_record_dict
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.single_task.modelling.run_modelling import RunModellingWrapper
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


class RunAnalysisWrapper(luigi.WrapperTask):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()
    analysis_config = luigi.DictParameter()

    def requires(self):
        yield RunModellingWrapper(
            folds=self.folds,
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
        )
        yield GatherValidationResults(
            folds=self.folds,
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
            analysis_config=self.analysis_config,
        )
        if self.feature_selection_config["feature_selection"] in (
            "dl",
            "gwas->dl",
            "gwas+bo",
        ):
            yield GatherFeatureSelectionResults(
                folds=self.folds,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
                analysis_config=self.analysis_config,
            )
        if self.modelling_config["do_test"]:
            yield GatherTestResults(
                folds=self.folds,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
                analysis_config=self.analysis_config,
            )


class GatherFeatureSelectionResults(luigi.Task):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()
    analysis_config = luigi.DictParameter()

    def requires(self):
        return {
            "runs": RunModellingWrapper(
                folds=self.folds,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
            ),
            "validation_results": GatherValidationResults(
                folds=self.folds,
                data_config=self.data_config,
                feature_selection_config=self.feature_selection_config,
                modelling_config=self.modelling_config,
                analysis_config=self.analysis_config,
            ),
        }

    def run(self):
        val_results = self.input()["validation_results"]
        feature_selection_folder = Path(
            self.feature_selection_config["feature_selection_output_folder"]
        )
        df_fractions = gather_fractions(
            feature_selection_folder=feature_selection_folder
        )
        for target_name, local_target in val_results.items():
            df_val_results = pd.read_csv(local_target.path, index_col=0)
            df_val_parsed = _parse_val_results(df=df_val_results)

            df_combined = df_val_parsed.join(df_fractions)

            output_folder = Path(self.analysis_config["analysis_output_folder"])
            ensure_path_exists(path=output_folder, is_folder=True)

            figure = _get_performance_fraction_figure(
                df=df_combined, target_name=target_name
            )
            output_path = output_folder / f"{target_name}_feature_selection.pdf"
            figure.savefig(output_path)

            df_output_path = output_folder / f"{target_name}_feature_selection.csv"
            df_combined.to_csv(path_or_buf=df_output_path)

    def output(self):
        output_folder = Path(self.analysis_config["analysis_output_folder"])
        ensure_path_exists(path=output_folder, is_folder=True)

        targets = {}
        for target in chain(
            self.modelling_config["output_cat_columns"],
            self.modelling_config["output_con_columns"],
        ):
            targets[target] = luigi.LocalTarget(
                path=Path(output_folder, f"{target}_feature_selection.pdf")
            )

        return targets


def _parse_val_results(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    return df_copy[["Best Average Performance"]]


def _get_performance_fraction_figure(df: pd.DataFrame, target_name: str) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    df_mean = df.groupby("Fraction SNPs").mean().reset_index()
    ax = sns.lineplot(
        data=df_mean,
        x="Fraction SNPs",
        y="Best Average Performance",
        color="black",
        linewidth=1,
    )

    ax.set_xscale("log")
    ax.set(xlabel="Fraction SNPs", ylabel="Average Val. Performance", title=target_name)

    return ax.figure


def gather_fractions(feature_selection_folder: Path) -> pd.DataFrame:
    fractions = []

    for file in _iterdir_ignore_hidden(
        path=feature_selection_folder / "snp_importance" / "snp_subsets"
    ):
        if "fraction" in file.stem:
            cur_fraction = float(file.read_text().strip())
            cur_fold = int(file.stem.split("_")[-1])
            fractions.append(
                {"Fold": f"Fold {cur_fold}", "Fraction SNPs": cur_fraction}
            )

    df_fractions = pd.DataFrame(data=fractions).set_index("Fold")

    return df_fractions


class GatherTestResults(luigi.Task):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()
    analysis_config = luigi.DictParameter()

    def requires(self):
        return RunModellingWrapper(
            folds=self.folds,
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
        )

    def run(self):
        output_folder = Path(self.analysis_config["analysis_output_folder"])
        ensure_path_exists(path=output_folder, is_folder=True)

        run_folder = Path(self.modelling_config["modelling_output_folder"])
        cat_targets = self.modelling_config["output_cat_columns"]
        con_targets = self.modelling_config["output_con_columns"]
        test_results_iterator = gather_test_predictions(
            folder_with_folds=run_folder,
            cat_targets=cat_targets,
            con_targets=con_targets,
        )

        for tr in test_results_iterator:
            tr.df_ensemble.to_csv(
                path_or_buf=output_folder / f"{tr.target_name}_test_predictions.csv",
                index=True,
            )
            tr.results.to_csv(
                path_or_buf=output_folder / f"{tr.target_name}_test_results.csv",
                index=True,
            )

    def output(self):
        analysis_output_folder = Path(self.analysis_config["analysis_output_folder"])
        aof = analysis_output_folder

        targets = {}
        for target in chain(
            self.modelling_config["output_cat_columns"],
            self.modelling_config["output_con_columns"],
        ):
            targets[f"{target}_test_results"] = luigi.LocalTarget(
                aof / f"{target}_test_results.csv"
            )
            targets[f"{target}_test_predictions"] = luigi.LocalTarget(
                aof / f"{target}_test_predictions.csv"
            )

        return targets


@dataclass
class TestResults:
    target_name: str
    df_ensemble: pd.DataFrame
    results: pd.DataFrame


def gather_test_predictions(
    folder_with_folds: Path,
    cat_targets: Sequence[str],
    con_targets: Sequence[str],
) -> Iterator[TestResults]:
    """
    TODO: Add mapping of numerical labels to original strings.
    """
    dfs = {}
    results = {}

    metric_funcs = None
    for fold in _iterdir_ignore_hidden(path=folder_with_folds):
        targets_folder = Path(fold, "test_set_predictions", "eir_auto_gp")

        if metric_funcs is None:
            logger.debug("Loading metric functions for ensemble from '%s'.", fold)
            loaded_experiment = load_serialized_train_experiment(
                run_folder=fold,
                device="cpu",
            )
            metric_funcs = loaded_experiment.metrics

        for target_folder in _iterdir_ignore_hidden(path=targets_folder):
            target_name = target_folder.name

            if target_name not in dfs:
                dfs[target_name] = []

            if target_name not in results:
                results[target_name] = []

            predictions_file = Path(target_folder, "predictions.csv")
            df = pd.read_csv(filepath_or_buffer=predictions_file)
            dfs[target_name].append(df)

            cur_metrics = get_single_fold_results(fold=fold, target_name=target_name)
            results[target_name].append(cur_metrics)

    for target_name, dfs in dfs.items():
        df_ensemble = get_ensemble_predictions(dfs=dfs)

        target_type = get_target_type(
            target_name=target_name, cat_targets=cat_targets, con_targets=con_targets
        )
        ensemble_metrics = compute_metrics(
            df=df_ensemble,
            metrics=metric_funcs,
            target_type=target_type,
            target_name=target_name,
        )
        results[target_name].append(ensemble_metrics)

        df_results = pd.DataFrame(results[target_name])
        df_results = df_results.sort_values(by="Fold").set_index("Fold")

        yield TestResults(
            target_name=target_name,
            df_ensemble=df_ensemble,
            results=df_results,
        )


def get_ensemble_predictions(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    df_combined = pd.concat(dfs, axis=0)
    df_ensemble = df_combined.groupby(["ID", "True Label Untransformed"]).mean()
    df_ensemble = df_ensemble.reset_index().set_index("ID")

    return df_ensemble


def get_single_fold_results(fold: Path, target_name: str) -> Dict[str, float | str]:
    cur_metrics_file = Path(fold, "test_set_predictions", "calculated_metrics.json")
    cur_fold_metrics = json.load(open(cur_metrics_file, "r"))
    cur_metrics = cur_fold_metrics["eir_auto_gp"][target_name]

    parsed_metrics = {"Fold": fold.name.title().replace("_", " ")}

    for metric_name, metric_value in cur_metrics.items():
        parsed_metric_name = metric_name.split("_")[-1].upper()
        parsed_metrics[parsed_metric_name] = metric_value

    return parsed_metrics


def compute_metrics(
    df: pd.DataFrame,
    metrics: al_metric_record_dict,
    target_type: Literal["con", "cat"],
    target_name: str,
) -> Dict[str, float | str]:
    cur_metrics = metrics[target_type]

    metrics = {}
    for metric_record in cur_metrics:
        cur_labels = df["True Label"].values
        output_columns = [
            col
            for col in df.columns
            if "True Label" not in col and " Untransformed" not in col
        ]
        cur_outputs = df[output_columns].values

        cur_metric_func = metric_record.function

        cur_metric_value = cur_metric_func(
            labels=cur_labels,
            outputs=cur_outputs,
            output_name="eir_auto_gp",
            column_name=target_name,
        )

        cur_metric_name = metric_record.name.upper()
        metrics[cur_metric_name] = cur_metric_value

    metrics["Fold"] = "Ensemble"
    return metrics


def get_target_type(
    target_name: str, cat_targets: Sequence[str], con_targets: Sequence[str]
) -> Literal["con", "cat"]:
    if target_name in cat_targets:
        return "cat"
    elif target_name in con_targets:
        return "con"


class GatherValidationResults(luigi.Task):
    folds = luigi.Parameter()
    data_config = luigi.DictParameter()
    feature_selection_config = luigi.DictParameter()
    modelling_config = luigi.DictParameter()
    analysis_config = luigi.DictParameter()

    def requires(self):
        return RunModellingWrapper(
            folds=self.folds,
            data_config=self.data_config,
            feature_selection_config=self.feature_selection_config,
            modelling_config=self.modelling_config,
        )

    def run(self):
        output_folder = Path(self.analysis_config["analysis_output_folder"])
        ensure_path_exists(path=output_folder, is_folder=True)

        folder_with_folds = Path(self.modelling_config["modelling_output_folder"])

        val_results_iter = _gather_validation_results(
            folder_with_folds=folder_with_folds,
        )
        feature_selection_folder = Path(
            self.feature_selection_config["feature_selection_output_folder"]
        )
        dl_fs = self.feature_selection_config["feature_selection"] in ("dl", "gwas->dl")

        for target_name, df_results in val_results_iter:
            output_file = Path(output_folder, f"{target_name}_validation_results.csv")

            if dl_fs:
                df_fractions = gather_fractions(
                    feature_selection_folder=feature_selection_folder
                )
                df_results = df_results.join(df_fractions, how="left")

            df_results.to_csv(path_or_buf=output_file)

    def output(self):
        output_folder = Path(self.analysis_config["analysis_output_folder"])
        ensure_path_exists(path=output_folder, is_folder=True)

        targets = {}
        for target in chain(
            self.modelling_config["output_cat_columns"],
            self.modelling_config["output_con_columns"],
        ):
            targets[target] = luigi.LocalTarget(
                path=Path(output_folder, f"{target}_validation_results.csv")
            )

        return targets


def _gather_validation_results(
    folder_with_folds: Path,
) -> Iterator[Tuple[str, pd.DataFrame]]:
    results = {}
    for fold_folder in _iterdir_ignore_hidden(path=folder_with_folds):
        targets_folder = Path(fold_folder, "results", "eir_auto_gp")

        avg_file = Path(fold_folder, "validation_average_history.log")
        df_average = pd.read_csv(filepath_or_buffer=avg_file)
        best_iter_idx = df_average["perf-average"].idxmax()
        best_val_average = df_average.loc[best_iter_idx]["perf-average"].item()

        for target_folder in _iterdir_ignore_hidden(path=targets_folder):
            target_name = target_folder.name

            if target_name not in results:
                results[target_name] = []

            val_file = Path(target_folder, f"validation_{target_name}_history.log")
            df_val_metrics = pd.read_csv(filepath_or_buffer=val_file)
            best_val_metrics = df_val_metrics.loc[best_iter_idx].to_dict()

            parsed_metrics = {
                "Fold": fold_folder.name.title().replace("_", " "),
                "Best Average Performance": best_val_average,
            }
            for metric_name, metric_value in best_val_metrics.items():
                parsed_metric_name = metric_name.split("_")[-1].upper()
                parsed_metrics[parsed_metric_name] = metric_value

            results[target_name].append(parsed_metrics)

    for target_name, results_list in results.items():
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values(by="Fold").set_index("Fold")

        yield target_name, df_results
