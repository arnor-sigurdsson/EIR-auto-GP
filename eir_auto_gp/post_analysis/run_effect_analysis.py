from typing import TYPE_CHECKING

import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.effect_analysis.genotype_effects import (
    get_allele_effects,
)
from eir_auto_gp.post_analysis.effect_analysis.interaction_effects import (
    get_interaction_effects,
)
from eir_auto_gp.post_analysis.effect_analysis.viz_genotype_effects import plot_top_snps
from eir_auto_gp.post_analysis.effect_analysis.viz_interaction_effects_graph import (
    generate_interaction_snp_graph_figure,
)
from eir_auto_gp.post_analysis.effect_analysis.viz_interaction_effects_point import (
    run_grouped_interaction_analysis,
)
from eir_auto_gp.post_analysis.run_complexity_analysis import (
    ModelReadyObject,
    convert_split_data_to_model_ready_object,
)

if TYPE_CHECKING:
    from eir_auto_gp.post_analysis.run_post_analysis import PostAnalysisObject

logger = get_logger(name=__name__)


def run_effect_analysis(post_analysis_object: "PostAnalysisObject") -> None:
    mro_genotype = convert_split_data_to_model_ready_object(
        split_model_data=post_analysis_object.modelling_data,
        include_genotype=True,
        include_tabular=False,
        one_hot_encode=False,
    )

    df_genotype, df_target = _build_effect_inputs(
        model_ready_object=mro_genotype,
        sets_for_effect_analysis=post_analysis_object.sets_for_effect_analysis,
    )

    pao = post_analysis_object

    output_root = pao.data_paths.analysis_output_path / "effect_analysis"
    ensure_path_exists(path=output_root, is_folder=True)

    effects_output = output_root / "allele_effects"
    ensure_path_exists(path=effects_output, is_folder=True)
    df_allele_effects = get_allele_effects(
        df_genotype=df_genotype,
        df_target=df_target,
        bim_file=pao.data_paths.snp_bim_path,
        target_type=pao.experiment_info.target_type,
    )
    df_allele_effects.to_csv(effects_output / "allele_effects.csv")

    plot_top_snps(
        df=df_allele_effects,
        p_value_threshold=0.05,
        top_n=pao.top_n_genotype_snps_effects_to_plot,
        output_dir=effects_output / "figures",
    )

    interaction_output = output_root / "interaction_effects"
    ensure_path_exists(path=interaction_output, is_folder=True)
    df_interaction_effects = get_interaction_effects(
        df_genotype=df_genotype,
        df_target=df_target,
        bim_file=pao.data_paths.snp_bim_path,
        target_type=pao.experiment_info.target_type,
        allow_within_chr_interaction=pao.allow_within_chr_interaction,
        min_interaction_pair_distance=pao.min_interaction_pair_distance,
    )
    df_interaction_effects.to_csv(interaction_output / "interaction_effects.csv")

    if len(df_interaction_effects) > 0:
        trait_name = df_target.columns[0]
        generate_interaction_snp_graph_figure(
            df_interaction_effects=df_interaction_effects,
            bim_file_path=pao.data_paths.snp_bim_path,
            df_target=df_target,
            trait=trait_name,
            plot_output_root=interaction_output,
            top_n_snps=pao.top_n_interaction_pairs,
        )

        run_grouped_interaction_analysis(
            df_genotype=df_genotype,
            df_target=df_target,
            df_interaction_effects=df_interaction_effects,
            top_n_snps=pao.top_n_interaction_pairs,
            bim_file=pao.data_paths.snp_bim_path,
            output_folder=output_root / "grouped_interaction_analysis",
        )


def _build_effect_inputs(
    model_ready_object: ModelReadyObject,
    sets_for_effect_analysis: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_sets = {"train", "valid", "test"}
    if not all(set_name in valid_sets for set_name in sets_for_effect_analysis):
        raise ValueError(
            "Invalid set names in sets_for_effect_analysis. "
            "Valid options are 'train', 'valid', 'test'."
        )

    input_dfs = []
    target_dfs = []

    if "train" in sets_for_effect_analysis:
        input_dfs.append(model_ready_object.input_train)
        target_dfs.append(model_ready_object.target_train)
    if "valid" in sets_for_effect_analysis:
        input_dfs.append(model_ready_object.input_val)
        target_dfs.append(model_ready_object.target_val)
    if "test" in sets_for_effect_analysis:
        input_dfs.append(model_ready_object.input_test)
        target_dfs.append(model_ready_object.target_test)

    concatenated_input = pd.concat(input_dfs, ignore_index=True)
    concatenated_target = pd.concat(target_dfs, ignore_index=True)

    return concatenated_input, concatenated_target
