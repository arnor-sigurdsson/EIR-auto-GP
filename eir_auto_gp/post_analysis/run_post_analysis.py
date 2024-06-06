import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.common.data_preparation import (
    DataPaths,
    ExperimentInfo,
    SplitModelData,
    build_data_paths,
    extract_experiment_info_from_config,
    set_up_split_model_data,
)
from eir_auto_gp.post_analysis.effect_analysis.genotype_effects import (
    get_snp_allele_maps,
    read_bim,
)
from eir_auto_gp.post_analysis.iterative_complexity_analysis import (
    run_iterative_complexity_analysis,
)
from eir_auto_gp.post_analysis.run_complexity_analysis import (
    convert_split_data_to_model_ready_object,
    run_complexity_analysis,
)
from eir_auto_gp.post_analysis.run_effect_analysis import run_effect_analysis

logger = get_logger(name=__name__)


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
        "--n_iterative_complexity_candidates",
        type=int,
        default=5,
        help="Number of candidates to consider for the iterative complexity analysis. "
        "This value is commonly applied to the number of features to consider in "
        "the number of one-hot encoded SNPs as well as tested interaction"
        "terms for ExE, GxE and GxG interactions.",
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

    bim_path = str(post_analysis_object.data_paths.snp_bim_path)
    df_bim = read_bim(bim_file_path=bim_path)
    variants = post_analysis_object.modelling_data.train.df_genotype_input.columns
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=variants)

    with open(output_folder / "allele_maps.json", "w") as f:
        json.dump(obj=allele_maps, fp=f, indent=4)

    return None


def run_all():
    cl_args = get_cl_args()
    post_analysis_object = build_post_analysis_object(cl_args=cl_args)

    _serialize_post_analysis_config(
        cl_args=cl_args,
        analysis_output_path=post_analysis_object.data_paths.analysis_output_path,
    )

    if cl_args.save_data:
        _save_data(post_analysis_object=post_analysis_object)

    run_complexity_analysis(post_analysis_object=post_analysis_object)
    run_effect_analysis(post_analysis_object=post_analysis_object)

    should_run_iter_test = _should_run_iterative_complexity_analysis(
        post_analysis_object=post_analysis_object,
        eval_set="test",
        force_unsafe=False,
    )
    if should_run_iter_test:
        run_iterative_complexity_analysis(
            post_analysis_object=post_analysis_object,
            n_iterative_complexity_candidates=cl_args.n_iterative_complexity_candidates,
            eval_set="test",
        )

    should_run_iter_valid = _should_run_iterative_complexity_analysis(
        post_analysis_object=post_analysis_object,
        eval_set="valid",
        force_unsafe=False,
    )
    if should_run_iter_valid:
        run_iterative_complexity_analysis(
            post_analysis_object=post_analysis_object,
            n_iterative_complexity_candidates=cl_args.n_iterative_complexity_candidates,
            eval_set="valid",
        )


def _serialize_post_analysis_config(
    cl_args: argparse.Namespace, analysis_output_path: Path
) -> None:
    config_path = analysis_output_path / "config.json"
    ensure_path_exists(path=config_path, is_folder=False)
    with open(config_path, "w") as f:
        json.dump(vars(cl_args), f)
    return None


def _should_run_iterative_complexity_analysis(
    post_analysis_object: PostAnalysisObject,
    eval_set: str,
    force_unsafe: bool,
) -> bool:
    sfea = post_analysis_object.sets_for_effect_analysis

    if force_unsafe:
        logger.warning(
            "Forcing the iterative complexity analysis to run. "
            "This is unsafe and may lead to data leakage. "
            "This is intended for testing purposes only."
        )
        return True

    if "valid" in sfea and "test" in sfea:
        logger.warning(
            "Both 'valid' and 'test' are in the sets for effect analysis. "
            "As results from analyses performed sets are used to inform e.g. feature "
            "selection in the iterative complexity analysis, having the effects "
            "computed on the whole set introduces a risk of data leakage. "
            "Iterative complexity analysis will not be run."
        )
        return False

    if "test" in sfea and eval_set == "test":
        logger.warning(
            "Test set was used for the effect analysis. "
            "As results from analyses performed sets are used to inform e.g. feature "
            "selection in the iterative complexity analysis, having the effects "
            "computed on the whole set introduces a risk of data leakage. "
            "Iterative complexity analysis will not be run."
        )
        return False

    if "valid" in sfea and eval_set == "valid":
        logger.warning(
            "Validation set was used for the effect analysis. "
            "As results from analyses performed sets are used to inform e.g. feature "
            "selection in the iterative complexity analysis, having the effects "
            "computed on the whole set introduces a risk of data leakage. "
            "Iterative complexity analysis will not be run."
        )
        return False

    return True


def main():
    run_all()


if __name__ == "__main__":
    main()
