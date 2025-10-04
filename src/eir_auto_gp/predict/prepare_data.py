import argparse
import shutil
import subprocess
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.predict.data_preparation_utils import (
    log_overlap,
    read_bim_and_cast_dtypes,
)
from eir_auto_gp.preprocess.genotype import get_encoded_snp_stream

logger = get_logger(name=__name__)


def check_plink_availability() -> str | None:
    for plink_cmd in ["plink2", "plink"]:
        if shutil.which(plink_cmd) is not None:
            return plink_cmd
    return None


def should_apply_plink_prefilter(
    n_matched_snps: int, n_total_snps: int, threshold: float = 0.8
) -> bool:
    if n_total_snps == 0:
        return False
    match_ratio = n_matched_snps / n_total_snps
    return match_ratio < threshold


def create_var_id_extract_list(
    df_bim: pd.DataFrame,
    direct_indices: np.ndarray,
    flip_indices: np.ndarray,
) -> pd.Series:
    combined_indices = np.concatenate([direct_indices, flip_indices])
    var_ids = (
        df_bim.iloc[combined_indices]["VAR_ID"].dropna().astype(str).drop_duplicates()
    )

    logger.info(
        f"Created extract list with {len(var_ids)} unique VAR_IDs from "
        f"{len(combined_indices)} matched SNPs"
    )

    if len(var_ids) != len(combined_indices):
        n_dropped = len(combined_indices) - len(var_ids)
        logger.warning(f"Dropped {n_dropped} SNPs due to missing or duplicate VAR_IDs")

    return var_ids


def run_plink_prefilter(
    original_fileset: "PlinkFileSet",
    df_bim: pd.DataFrame,
    df_bim_reference: pd.DataFrame,
    work_dir: Path,
    enable_prefilter: bool = True,
) -> Optional["PlinkFileSet"]:
    if not enable_prefilter:
        logger.info("PLINK pre-filtering disabled")
        return None

    plink_cmd = check_plink_availability()
    if plink_cmd is None:
        logger.warning("PLINK2/PLINK not available, skipping pre-filtering")
        return None

    snp_mapping = create_snp_mapping(from_bim=df_bim, to_reference_bim=df_bim_reference)

    n_matched_snps = len(snp_mapping.direct_match_source_indices) + len(
        snp_mapping.flip_match_source_indices
    )
    n_total_snps = len(df_bim)

    threshold = 0.8
    if not should_apply_plink_prefilter(n_matched_snps, n_total_snps, threshold):
        match_ratio = n_matched_snps / n_total_snps if n_total_snps > 0 else 0
        logger.info(
            f"Skipping PLINK pre-filtering "
            f"(match ratio: {match_ratio:.2f} >= {threshold:.2f})"
        )
        return None

    logger.info(
        f"Applying PLINK pre-filtering ({n_matched_snps}/{n_total_snps} SNPs matched)"
    )

    var_ids = create_var_id_extract_list(
        df_bim=df_bim,
        direct_indices=snp_mapping.direct_match_source_indices,
        flip_indices=snp_mapping.flip_match_source_indices,
    )

    if len(var_ids) == 0:
        logger.warning("No valid VAR_IDs to extract, skipping pre-filtering")
        return None

    run_tag = uuid4().hex[:8]
    temp_dir = work_dir / f"plink_temp_{run_tag}"
    ensure_path_exists(temp_dir, is_folder=True)

    extract_file = temp_dir / "extract_list.txt"
    output_prefix = temp_dir / "filtered"

    try:
        var_ids.to_csv(extract_file, index=False, header=False)

        input_prefix = original_fileset.bed.stem
        input_dir = original_fileset.bed.parent

        cmd = [
            plink_cmd,
            "--bfile",
            str(input_dir / input_prefix),
            "--extract",
            str(extract_file),
            "--make-bed",
            "--keep-allele-order",
            "--out",
            str(output_prefix),
        ]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info("PLINK filtering completed successfully")
        if result.stdout.strip():
            logger.info(f"PLINK stdout: {result.stdout.strip()}")
        if result.stderr and result.stderr.strip():
            logger.info(f"PLINK stderr: {result.stderr.strip()}")

        bed_file = output_prefix.with_suffix(".bed")
        bim_file = output_prefix.with_suffix(".bim")
        fam_file = output_prefix.with_suffix(".fam")

        for file_path, name in [
            (bed_file, "BED"),
            (bim_file, "BIM"),
            (fam_file, "FAM"),
        ]:
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.error(
                    f"PLINK output {name} file is missing or empty: {file_path}"
                )
                return None

        filtered_fileset = PlinkFileSet(bed=bed_file, bim=bim_file, fam=fam_file)

        persistent_dir = work_dir / f"plink_filtered_{run_tag}"
        ensure_path_exists(persistent_dir, is_folder=True)

        persistent_fileset = PlinkFileSet(
            bed=persistent_dir / "filtered.bed",
            bim=persistent_dir / "filtered.bim",
            fam=persistent_dir / "filtered.fam",
        )

        copyfile(filtered_fileset.bed, persistent_fileset.bed)
        copyfile(filtered_fileset.bim, persistent_fileset.bim)
        copyfile(filtered_fileset.fam, persistent_fileset.fam)

        plink_log = output_prefix.with_suffix(".log")
        if plink_log.exists():
            persistent_log = persistent_dir / "filtered.log"
            copyfile(plink_log, persistent_log)
            logger.info(f"PLINK log saved to: {persistent_log}")

        logger.info(f"Filtered files saved to: {persistent_dir}")
        return persistent_fileset

    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK command failed: {e}")
        logger.error(f"STDERR: {e.stderr}")
        return None
    finally:
        try:
            if extract_file.exists():
                extract_file.unlink()
            for suffix in [".bed", ".bim", ".fam"]:
                temp_file = output_prefix.with_suffix(suffix)
                if temp_file.exists():
                    temp_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


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
            f"Expected one .bed file in {folder_path}, butfound {bed_files}."
        )
    if len(bim_files) != 1:
        raise ValueError(
            f"Expected one .bim file in {folder_path}, butfound {bim_files}."
        )
    if len(fam_files) != 1:
        raise ValueError(
            f"Expected one .fam file in {folder_path}, butfound {fam_files}."
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
    enable_plink_prefilter: bool = True,
) -> PreparedData:
    plink_fileset = get_plink_fileset_from_folder(folder_path=genotype_data_path)

    ref_bim = reference_bim_to_project_to
    df_bim_reference = read_bim_and_cast_dtypes(bim_file_path=ref_bim)

    df_bim = read_bim_and_cast_dtypes(bim_file_path=plink_fileset.bim)

    filtered_fileset = run_plink_prefilter(
        original_fileset=plink_fileset,
        df_bim=df_bim,
        df_bim_reference=df_bim_reference,
        work_dir=output_folder,
        enable_prefilter=enable_plink_prefilter,
    )

    if filtered_fileset is not None:
        plink_fileset = filtered_fileset
        df_bim = read_bim_and_cast_dtypes(bim_file_path=plink_fileset.bim)
        logger.info("Using PLINK pre-filtered dataset for processing")

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
    from_df = from_bim.reset_index().rename(columns={"index": "source_idx"})
    to_df = to_reference_bim.reset_index().rename(columns={"index": "target_idx"})

    from_pos_key = from_df["CHR_CODE"] + ":" + from_df["BP_COORD"].astype(str)
    to_pos_key = to_df["CHR_CODE"] + ":" + to_df["BP_COORD"].astype(str)

    matches = pd.merge(
        from_df.assign(pos_key=from_pos_key),
        to_df.assign(pos_key=to_pos_key),
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
    from_stream: Generator[tuple[str, np.ndarray]],
    snp_mapping: SNPMapping,
) -> Generator[tuple[str, np.ndarray]]:
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


def run_prepare_wrapper(cl_args: argparse.Namespace) -> None:
    genotype_data_path = Path(cl_args.genotype_data_path)
    reference_bim_path = Path(cl_args.reference_bim_path)
    output_folder = Path(cl_args.output_folder)
    array_chunk_size = cl_args.array_chunk_size

    enable_plink_prefilter = True
    if hasattr(cl_args, "disable_plink_prefilter") and cl_args.disable_plink_prefilter:
        enable_plink_prefilter = False

    prepared_data = run_prepare_data(
        genotype_data_path=genotype_data_path,
        array_chunk_size=array_chunk_size,
        reference_bim_to_project_to=reference_bim_path,
        output_folder=output_folder,
        enable_plink_prefilter=enable_plink_prefilter,
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
        default=1024,
        help="Number of SNPs to process in each chunk (default: 1024).",
    )

    parser.add_argument(
        "--disable_plink_prefilter",
        action="store_true",
        help="Disable PLINK pre-filtering optimization (default: enabled).",
    )

    return parser


def main() -> None:
    args = get_cl_args()
    run_prepare_wrapper(cl_args=args)


if __name__ == "__main__":
    main()
