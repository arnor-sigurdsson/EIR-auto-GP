import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Generator

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.predict.data_preparation_utils import (
    log_overlap,
    read_bim_and_cast_dtypes,
)
from eir_auto_gp.preprocess.genotype import get_encoded_snp_stream

logger = get_logger(name=__name__)


@dataclass
class PlinkFileSet:
    bed: Path
    bim: Path
    fam: Path


def get_plink_fileset_from_folder(folder_path: Path) -> PlinkFileSet:
    bed_files = [i for i in folder_path.iterdir() if i.suffix == ".bed"]
    bim_files = [i for i in folder_path.iterdir() if i.suffix == ".bim"]
    fam_files = [i for i in folder_path.iterdir() if i.suffix == ".fam"]

    if len(bed_files) != 1:
        raise ValueError(
            f"Expected one .bed file in {folder_path}, but" f"found {bed_files}."
        )
    if len(bim_files) != 1:
        raise ValueError(
            f"Expected one .bim file in {folder_path}, but" f"found {bim_files}."
        )
    if len(fam_files) != 1:
        raise ValueError(
            f"Expected one .fam file in {folder_path}, but" f"found {fam_files}."
        )

    return PlinkFileSet(
        bed=bed_files[0],
        bim=bim_files[0],
        fam=fam_files[0],
    )


@dataclass
class PreparedData:
    array_folder: Path
    bim_file: Path
    log_file: Path


def run_prepare_data(
    genotype_data_path: Path,
    array_chunk_size: int,
    reference_bim_to_project_to: Path,
    output_folder: Path,
) -> PreparedData:

    plink_fileset = get_plink_fileset_from_folder(folder_path=genotype_data_path)

    ref_bim = reference_bim_to_project_to
    df_bim_reference = read_bim_and_cast_dtypes(bim_file_path=ref_bim)

    df_bim = read_bim_and_cast_dtypes(bim_file_path=plink_fileset.bim)

    log_output_path = output_folder / "snp_overlap_analysis.txt"
    log_overlap(
        df_bim_prd=df_bim,
        df_bim_exp=df_bim_reference,
        output_path=log_output_path,
    )

    snp_mapping = create_snp_mapping(from_bim=df_bim, to_reference_bim=df_bim_reference)

    from_stream = get_encoded_snp_stream(
        bed_path=plink_fileset.bed,
        chunk_size=array_chunk_size,
        output_format="disk",
    )

    projected_stream = get_projected_snp_stream(
        from_stream=from_stream,
        snp_mapping=snp_mapping,
    )

    copyfile(src=ref_bim, dst=output_folder / "projected.bim")

    array_outfolder = output_folder / "encoded_arrays"
    ensure_path_exists(path=array_outfolder, is_folder=True)

    for sample_id, projected_array in projected_stream:
        np.save(file=array_outfolder / f"{sample_id}.npy", arr=projected_array)

    return PreparedData(
        array_folder=array_outfolder,
        bim_file=ref_bim,
        log_file=log_output_path,
    )


@dataclass
class SNPMapping:
    direct_match_source_indices: np.ndarray
    direct_match_target_indices: np.ndarray
    flip_match_source_indices: np.ndarray
    flip_match_target_indices: np.ndarray
    missing_target_indices: np.ndarray
    n_source_snps: int
    n_target_snps: int


def create_snp_mapping(
    from_bim: pd.DataFrame,
    to_reference_bim: pd.DataFrame,
) -> SNPMapping:
    """
    from_bim: DataFrame with SNPs from the dataset we want to project
    to_reference_bim: DataFrame with SNPs in the reference space we want to
    project into
    """
    from_bim["pos_key"] = from_bim["CHR_CODE"] + ":" + from_bim["BP_COORD"].astype(str)
    to_reference_bim["pos_key"] = (
        to_reference_bim["CHR_CODE"] + ":" + to_reference_bim["BP_COORD"].astype(str)
    )

    from_bim["allele_key"] = from_bim.apply(
        lambda x: tuple(sorted([x["REF"], x["ALT"]])), axis=1
    )
    to_reference_bim["allele_key"] = to_reference_bim.apply(
        lambda x: tuple(sorted([x["REF"], x["ALT"]])), axis=1
    )

    matches = pd.merge(
        from_bim.reset_index().rename(columns={"index": "source_idx"}),
        to_reference_bim.reset_index().rename(columns={"index": "target_idx"}),
        on="pos_key",
        suffixes=("_source", "_target"),
    )

    direct_mask = (matches["REF_source"] == matches["REF_target"]) & (
        matches["ALT_source"] == matches["ALT_target"]
    )
    flip_mask = (matches["REF_source"] == matches["ALT_target"]) & (
        matches["ALT_source"] == matches["REF_target"]
    )

    direct_matches = matches[direct_mask]
    direct_source_indices = direct_matches["source_idx"].to_numpy()
    direct_target_indices = direct_matches["target_idx"].to_numpy()

    flip_matches = matches[flip_mask]
    flip_source_indices = flip_matches["source_idx"].to_numpy()
    flip_target_indices = flip_matches["target_idx"].to_numpy()

    all_matched_targets = set(direct_target_indices) | set(flip_target_indices)
    missing_target_indices = np.array(
        [i for i in range(len(to_reference_bim)) if i not in all_matched_targets]
    )

    return SNPMapping(
        direct_match_source_indices=direct_source_indices,
        direct_match_target_indices=direct_target_indices,
        flip_match_source_indices=flip_source_indices,
        flip_match_target_indices=flip_target_indices,
        missing_target_indices=missing_target_indices,
        n_source_snps=len(from_bim),
        n_target_snps=len(to_reference_bim),
    )


def get_projected_snp_stream(
    from_stream: Generator[tuple[str, np.ndarray], None, None],
    snp_mapping: SNPMapping,
) -> Generator[tuple[str, np.ndarray], None, None]:
    for sample_id, source_array in from_stream:
        projected_array = np.zeros((4, snp_mapping.n_target_snps), dtype=np.uint8)
        projected_array[3, :] = 1

        projected_array[:, snp_mapping.direct_match_target_indices] = source_array[
            :, snp_mapping.direct_match_source_indices
        ]

        if len(snp_mapping.flip_match_source_indices) > 0:
            flipped_data = source_array[:, snp_mapping.flip_match_source_indices].copy()
            flipped_data[[0, 2]] = flipped_data[[2, 0]]
            projected_array[:, snp_mapping.flip_match_target_indices] = flipped_data

        # assert all columns sum to exactly 1
        assert np.allclose(projected_array.sum(axis=0), 1.0)
        yield sample_id, projected_array


def get_prepared_folder_path(output_folder: str) -> Path:
    return Path(output_folder, "processed/full_inds/full_chrs/encoded_outputs")


def run_prepare_wrapper(cl_args: argparse.Namespace) -> None:
    genotype_data_path = Path(cl_args.genotype_data_path)
    reference_bim_path = Path(cl_args.reference_bim_path)
    output_folder = Path(cl_args.output_folder)
    array_chunk_size = cl_args.array_chunk_size

    prepared_data = run_prepare_data(
        genotype_data_path=genotype_data_path,
        array_chunk_size=array_chunk_size,
        reference_bim_to_project_to=reference_bim_path,
        output_folder=output_folder,
    )

    logger.info(
        f"Data preparation completed. Results stored in: {prepared_data.array_folder}"
    )
    logger.info(f"BIM file: {prepared_data.bim_file}")
    logger.info(f"Log file: {prepared_data.log_file}")


def get_cl_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare genotype data by projecting it into a reference space and "
            "saving the resulting NumPy arrays."
        )
    )

    parser.add_argument(
        "--genotype_data_path",
        type=str,
        help="Path to the folder containing the genotype data to prepare "
        "(should contain .bed, .bim, and .fam files).",
        required=True,
    )

    parser.add_argument(
        "--reference_bim_path",
        type=str,
        help="Path to reference BIM file to project the genotype data into.",
        required=True,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        help="Folder where the prepared data will be stored.",
        required=True,
    )

    parser.add_argument(
        "--array_chunk_size",
        type=int,
        default=256,
        help="Number of SNPs to process in each chunk (default: 1000).",
    )

    return parser


def main() -> None:
    args = get_cl_args()
    run_prepare_wrapper(cl_args=args)


if __name__ == "__main__":
    main()
