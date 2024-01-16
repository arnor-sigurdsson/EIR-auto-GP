import argparse
from dataclasses import dataclass
from pathlib import Path

from eir_auto_gp.post_analysis.common.data_preparation import (
    DataPaths,
    ExperimentInfo,
    SplitModelData,
    build_data_paths,
    extract_experiment_info_from_config,
    set_up_split_model_data,
)
from eir_auto_gp.post_analysis.run_complexity_analysis import run_complexity_analysis
from eir_auto_gp.post_analysis.run_effect_analysis import run_effect_analysis


@dataclass()
class PostAnalysisObject:
    data_paths: DataPaths
    experiment_info: ExperimentInfo
    modelling_data: SplitModelData


def build_post_analysis_object(cl_args: argparse.Namespace) -> PostAnalysisObject:
    run_dir = Path(cl_args.run_dir)
    data_paths = build_data_paths(run_dir=run_dir)

    experiment_info = extract_experiment_info_from_config(
        config_path=data_paths.experiment_config
    )

    modelling_data = set_up_split_model_data(
        data_paths=data_paths,
        experiment_info=experiment_info,
        top_snps=cl_args.top_n_snps,
    )

    complexity_object = PostAnalysisObject(
        data_paths=data_paths,
        experiment_info=experiment_info,
        modelling_data=modelling_data,
    )

    return complexity_object


def get_cl_args() -> argparse.Namespace:
    parser = get_argument_parser()
    args = parser.parse_args()
    return args


def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Path to the run directory.",
    )

    parser.add_argument(
        "--top_n_snps",
        type=int,
        default=128,
        help="Number of SNPs to use for the analysis.",
    )

    return parser


def run_all():
    cl_args = get_cl_args()
    post_analysis_object = build_post_analysis_object(cl_args=cl_args)
    run_complexity_analysis(post_analysis_object=post_analysis_object)
    run_effect_analysis(post_analysis_object=post_analysis_object)


def main():
    run_all()


if __name__ == "__main__":
    main()
