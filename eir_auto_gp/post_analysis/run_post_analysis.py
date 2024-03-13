import argparse
from dataclasses import dataclass
from pathlib import Path

from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.post_analysis.common.data_preparation import (
    DataPaths,
    ExperimentInfo,
    SplitModelData,
    build_data_paths,
    extract_experiment_info_from_config,
    set_up_split_model_data,
)
from eir_auto_gp.post_analysis.iterative_complexity_analysis import (
    run_iterative_complexity_analysis,
)
from eir_auto_gp.post_analysis.run_complexity_analysis import (
    convert_split_data_to_model_ready_object,
    run_complexity_analysis,
)
from eir_auto_gp.post_analysis.run_effect_analysis import run_effect_analysis


@dataclass()
class PostAnalysisObject:
    data_paths: DataPaths
    experiment_info: ExperimentInfo
    modelling_data: SplitModelData
    top_n_genotype_snps_effects_to_plot: int
    top_n_interaction_pairs: int
    allow_within_chr_interaction: bool
    min_interaction_pair_distance: int
    sets_for_effect_analysis: list[str]


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

    sets_for_effect_analysis = cl_args.sets_for_effect_analysis.split(",")

    complexity_object = PostAnalysisObject(
        data_paths=data_paths,
        experiment_info=experiment_info,
        modelling_data=modelling_data,
        top_n_genotype_snps_effects_to_plot=cl_args.top_n_genotype_snps_effects_to_plot,
        top_n_interaction_pairs=cl_args.top_n_interaction_pairs,
        allow_within_chr_interaction=cl_args.allow_within_chr_interaction,
        min_interaction_pair_distance=cl_args.min_interaction_pair_distance,
        sets_for_effect_analysis=sets_for_effect_analysis,
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

    parser.add_argument(
        "--top_n_genotype_snps_effects_to_plot",
        type=int,
        default=10,
        help="Number of interaction pairs to use for the analysis.",
    )

    parser.add_argument(
        "--top_n_interaction_pairs",
        type=int,
        default=10,
        help="Number of interaction pairs to use for the analysis.",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--allow_within_chr_interaction",
        action="store_true",
        default=True,
        help="Allows within chromosome interactions. This is the default behavior.",
    )
    group.add_argument(
        "--disallow_within_chr_interaction",
        dest="allow_within_chr_interaction",
        action="store_false",
        help="Disallows within chromosome interactions. "
        "Include this flag to override the default behavior.",
    )

    parser.add_argument(
        "--min_interaction_pair_distance",
        type=int,
        default=0,
        help="Minimum distance between interaction pairs if they are within chr.",
    )

    parser.add_argument(
        "--sets_for_effect_analysis",
        type=str,
        default="train",
        help="What parts of the data to use for the effect analysis. "
        "Options: 'train', 'test', 'validation', or a comma separated "
        "list of these.",
    )

    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help="Save the data (genotype and tabular) used for the post analysis.",
    )

    return parser


def _save_data(post_analysis_object: PostAnalysisObject) -> None:
    mro = convert_split_data_to_model_ready_object(
        split_model_data=post_analysis_object.modelling_data,
        include_genotype=True,
        include_tabular=True,
        one_hot_encode=False,
    )

    output_folder = (
        post_analysis_object.data_paths.analysis_output_path / "post_analysis_data"
    )
    ensure_path_exists(path=output_folder, is_folder=True)

    train_input_and_target = mro.input_train.join(mro.target_train)
    train_input_and_target.to_csv(output_folder / "train_input_and_target.csv")

    val_input_and_target = mro.input_val.join(mro.target_val)
    val_input_and_target.to_csv(output_folder / "val_input_and_target.csv")

    test_input_and_target = mro.input_test.join(mro.target_test)
    test_input_and_target.to_csv(output_folder / "test_input_and_target.csv")

    return None


def run_all():
    cl_args = get_cl_args()
    post_analysis_object = build_post_analysis_object(cl_args=cl_args)

    if cl_args.save_data:
        _save_data(post_analysis_object=post_analysis_object)

    run_complexity_analysis(post_analysis_object=post_analysis_object)
    run_effect_analysis(post_analysis_object=post_analysis_object)
    run_iterative_complexity_analysis(post_analysis_object=post_analysis_object)


def main():
    run_all()


if __name__ == "__main__":
    main()
