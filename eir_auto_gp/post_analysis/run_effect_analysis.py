from typing import TYPE_CHECKING

from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.post_analysis.effect_analysis.genotype_effects import (
    get_allele_effects,
)
from eir_auto_gp.post_analysis.effect_analysis.interaction_effects import (
    get_interaction_effects,
)
from eir_auto_gp.post_analysis.effect_analysis.viz_genotype_effects import plot_top_snps
from eir_auto_gp.post_analysis.effect_analysis.viz_interaction_effects import (
    generate_interaction_snp_graph_figure,
)
from eir_auto_gp.post_analysis.run_complexity_analysis import (
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

    output_root = (
        post_analysis_object.data_paths.analysis_output_path / "effect_analysis"
    )
    ensure_path_exists(path=output_root, is_folder=True)

    df_allele_effects = get_allele_effects(
        df_genotype=mro_genotype.input_train,
        df_target=mro_genotype.target_train,
        bim_file=post_analysis_object.data_paths.snp_bim_path,
        target_type=post_analysis_object.experiment_info.target_type,
    )
    df_allele_effects.to_csv(output_root / "allele_effects.csv")

    plot_top_snps(
        df=df_allele_effects,
        p_value_threshold=0.05,
        top_n=10,
        output_dir=output_root / "top_snps",
    )

    df_interaction_effects = get_interaction_effects(
        df_genotype=mro_genotype.input_train,
        df_target=mro_genotype.target_train,
        bim_file=post_analysis_object.data_paths.snp_bim_path,
        target_type=post_analysis_object.experiment_info.target_type,
    )
    df_interaction_effects.to_csv(output_root / "interaction_effects.csv")

    if len(df_interaction_effects) > 0:
        trait_name = mro_genotype.target_train.columns[0]
        generate_interaction_snp_graph_figure(
            df=df_interaction_effects,
            bim_file_path=post_analysis_object.data_paths.snp_bim_path,
            df_target=mro_genotype.target_train,
            trait=trait_name,
            plot_output_root=output_root / "interaction_graphs",
            top_n_snps=10,
        )
