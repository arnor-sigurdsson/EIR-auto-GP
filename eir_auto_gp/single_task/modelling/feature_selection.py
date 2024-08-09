from pathlib import Path
from typing import Literal, Optional

from eir_auto_gp.single_task.modelling.dl_feature_selection import (
    run_dl_bo_selection,
    run_dl_plus_gwas_bo_selection,
)
from eir_auto_gp.single_task.modelling.gwas_bo_feature_selection import (
    run_gwas_bo_feature_selection,
)


def get_genotype_subset_snps_file(
    fold: int,
    folder_with_runs: Path,
    feature_selection_output_folder: Path,
    bim_file: str | Path,
    feature_selection_approach: Literal["dl", "gwas", "gwas->dl", "dl+gwas", None],
    n_dl_feature_selection_setup_folds: int,
    manual_subset_from_gwas: Optional[str | Path],
    gwas_p_value_threshold: Optional[float],
) -> Optional[Path]:
    match feature_selection_approach:
        case None:
            return None
        case "gwas":
            return manual_subset_from_gwas
        case "gwas+bo":
            assert manual_subset_from_gwas is not None
            computed_subset_file = run_gwas_bo_feature_selection(
                fold=fold,
                folder_with_runs=folder_with_runs,
                feature_selection_output_folder=feature_selection_output_folder,
                gwas_output_folder=Path(manual_subset_from_gwas).parent,
                gwas_p_value_threshold=gwas_p_value_threshold,
            )

            return computed_subset_file
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
        case "dl+gwas":
            if fold > n_dl_feature_selection_setup_folds:
                assert manual_subset_from_gwas is not None

            gwas_output_folder = manual_subset_from_gwas.parent
            computed_subset_file = run_dl_plus_gwas_bo_selection(
                fold=fold,
                folder_with_runs=folder_with_runs,
                feature_selection_output_folder=feature_selection_output_folder,
                gwas_output_folder=gwas_output_folder,
                bim_file=bim_file,
                n_dl_feature_selection_setup_folds=n_dl_feature_selection_setup_folds,
            )
            return computed_subset_file

        case _:
            raise ValueError()
