import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from eir_auto_gp.predict.prepare_data import run_prepare_data


def create_test_genotype_data(
    output_path: Path,
    n_samples: int,
    n_snps: int,
) -> None:
    command = [
        "plink2",
        "--dummy",
        str(n_samples),
        str(n_snps),
        "--out",
        str(output_path),
        "--make-bed",
    ]
    subprocess.run(command, check=True, capture_output=True)


def modify_bim_file(bim_path: Path, changes: dict) -> None:
    df_bim = pd.read_csv(
        bim_path,
        sep="\t",
        header=None,
        names=["CHR", "ID", "CM", "POS", "A1", "A2"],
    )

    for idx, mods in changes.items():
        for col, val in mods.items():
            df_bim.loc[idx, col] = val

    df_bim.to_csv(bim_path, sep="\t", header=False, index=False)


def read_encoded_arrays(array_folder: Path) -> tuple[np.ndarray, list[str]]:
    array_files = sorted(array_folder.glob("*.npy"))
    arrays = []
    sample_ids = []

    for f in array_files:
        arrays.append(np.load(f))
        sample_ids.append(f.stem)

    return np.stack(arrays, axis=0), sample_ids


def test_prepare_module() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create separate folders for each dataset
        dataset1_dir = tmpdir / "reference_data"
        dataset2_dir = tmpdir / "data_to_prepare"
        dataset1_dir.mkdir()
        dataset2_dir.mkdir()

        # Create the datasets
        dataset1_path = dataset1_dir / "dataset1"
        dataset2_path = dataset2_dir / "dataset2"

        create_test_genotype_data(dataset1_path, n_samples=100, n_snps=1000)
        create_test_genotype_data(dataset2_path, n_samples=100, n_snps=950)

        # Modify the second dataset's BIM file
        bim2_path = dataset2_path.with_suffix(".bim")
        modifications = {
            0: {"A1": "G", "A2": "A"},  # Flip alleles
            1: {"POS": 1000},  # Change position
            2: {"ID": "rs_new_name"},  # Change SNP name
        }
        modify_bim_file(bim_path=bim2_path, changes=modifications)

        output_dir = tmpdir / "prepare_output"
        output_dir.mkdir()

        prepared_data = run_prepare_data(
            genotype_data_path=dataset2_dir,
            array_chunk_size=100,
            reference_bim_to_project_to=dataset1_path.with_suffix(".bim"),
            output_folder=output_dir,
        )

        arrays, _ = read_encoded_arrays(prepared_data.array_folder)

        n_samples, n_encodings, n_snps = arrays.shape
        assert n_samples == 100, f"Expected 100 samples, got {n_samples}"
        assert n_encodings == 4, f"Expected 4 encodings, got {n_encodings}"
        assert n_snps == 1000, f"Expected 1000 SNPs, got {n_snps}"

        assert np.allclose(arrays.sum(axis=1), 1.0), "Arrays not one-hot encoded"

        df_prepared = pd.read_csv(
            prepared_data.bim_file,
            sep="\t",
            names=[
                "CHR_CODE",
                "VAR_ID",
                "POS_CM",
                "BP_COORD",
                "ALT",
                "REF",
            ],
        )
        df_ref = pd.read_csv(
            dataset1_path.with_suffix(".bim"),
            sep="\t",
            names=[
                "CHR_CODE",
                "VAR_ID",
                "POS_CM",
                "BP_COORD",
                "ALT",
                "REF",
            ],
        )

        pd.testing.assert_frame_equal(
            df_prepared[
                [
                    "CHR_CODE",
                    "BP_COORD",
                    "ALT",
                    "REF",
                ]
            ],
            df_ref[
                [
                    "CHR_CODE",
                    "BP_COORD",
                    "ALT",
                    "REF",
                ]
            ],
        )


def test_prepare_flipped_data() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        original_dir = tmpdir / "original_data"
        flipped_dir = tmpdir / "flipped_data"
        original_dir.mkdir()
        flipped_dir.mkdir()

        original_path = original_dir / "dataset"
        create_test_genotype_data(original_path, n_samples=100, n_snps=1000)

        for suffix in [".bed", ".bim", ".fam"]:
            shutil.copy(
                original_path.with_suffix(suffix), flipped_dir / f"dataset{suffix}"
            )

        flipped_bim = flipped_dir / "dataset.bim"
        df_bim = pd.read_csv(
            flipped_bim,
            sep="\t",
            header=None,
            names=["CHR", "ID", "CM", "POS", "A1", "A2"],
        )
        df_bim[["A1", "A2"]] = df_bim[["A2", "A1"]]
        df_bim.to_csv(flipped_bim, sep="\t", header=False, index=False)

        output_dir = tmpdir / "prepare_output"
        output_dir.mkdir()

        prepared_data = run_prepare_data(
            genotype_data_path=flipped_dir,
            array_chunk_size=100,
            reference_bim_to_project_to=original_path.with_suffix(".bim"),
            output_folder=output_dir,
        )

        arrays, _ = read_encoded_arrays(array_folder=prepared_data.array_folder)

        # For flipped alleles, encoding should be flipped
        # Original: [1,0,0,0] -> Flipped: [0,0,1,0]
        assert np.allclose(arrays.sum(axis=1), 1.0), "Arrays not one-hot encoded"

        original_arrays, _ = read_encoded_arrays(
            array_folder=output_dir / "encoded_arrays"
        )

        # When alleles are flipped, the 0 and 2 encodings should be swapped
        # This means summing the 0th and 2nd encoding positions should give same results
        assert np.allclose(
            arrays[:, [0, 2]].sum(axis=1), original_arrays[:, [0, 2]].sum(axis=1)
        ), "Flipped allele encoding mismatch"


def test_plink_prefilter_optimization() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        reference_dir = tmpdir / "reference_data"
        reference_dir.mkdir()
        reference_path = reference_dir / "reference"
        create_test_genotype_data(reference_path, n_samples=100, n_snps=1000)

        target_dir = tmpdir / "target_data"
        target_dir.mkdir()
        target_path = target_dir / "target"
        create_test_genotype_data(target_path, n_samples=100, n_snps=2000)

        output_dir = tmpdir / "prepare_output"
        output_dir.mkdir()

        # This should trigger PLINK pre-filtering
        prepared_data = run_prepare_data(
            genotype_data_path=target_dir,
            array_chunk_size=100,
            reference_bim_to_project_to=reference_path.with_suffix(".bim"),
            output_folder=output_dir,
            enable_plink_prefilter=True,
        )

        arrays, _ = read_encoded_arrays(prepared_data.array_folder)
        n_samples, n_encodings, n_snps = arrays.shape

        assert n_samples == 100
        assert n_encodings == 4
        assert n_snps == 1000
        assert np.allclose(arrays.sum(axis=1), 1.0)

        plink_filtered_dirs = list(output_dir.glob("plink_filtered_*"))
        assert len(plink_filtered_dirs) >= 1, "PLINK filtered directory should exist"


def test_plink_prefilter_disabled() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        reference_dir = tmpdir / "reference_data"
        target_dir = tmpdir / "target_data"
        reference_dir.mkdir()
        target_dir.mkdir()

        reference_path = reference_dir / "reference"
        target_path = target_dir / "target"

        create_test_genotype_data(reference_path, n_samples=100, n_snps=1000)
        create_test_genotype_data(target_path, n_samples=100, n_snps=2000)

        output_dir = tmpdir / "prepare_output"
        output_dir.mkdir()

        prepared_data = run_prepare_data(
            genotype_data_path=target_dir,
            array_chunk_size=100,
            reference_bim_to_project_to=reference_path.with_suffix(".bim"),
            output_folder=output_dir,
            enable_plink_prefilter=False,
        )

        arrays, _ = read_encoded_arrays(prepared_data.array_folder)

        assert arrays.shape == (100, 4, 1000)
        assert np.allclose(arrays.sum(axis=1), 1.0)

        plink_filtered_dirs = list(output_dir.glob("plink_filtered_*"))
        assert (
            len(plink_filtered_dirs) == 0
        ), "No PLINK filtered directories should exist when disabled"


def test_high_overlap_skips_prefilter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        reference_dir = tmpdir / "reference_data"
        target_dir = tmpdir / "target_data"
        reference_dir.mkdir()
        target_dir.mkdir()

        reference_path = reference_dir / "reference"
        target_path = target_dir / "target"

        create_test_genotype_data(reference_path, n_samples=100, n_snps=1000)
        create_test_genotype_data(target_path, n_samples=100, n_snps=1000)

        output_dir = tmpdir / "prepare_output"
        output_dir.mkdir()

        prepared_data = run_prepare_data(
            genotype_data_path=target_dir,
            array_chunk_size=100,
            reference_bim_to_project_to=reference_path.with_suffix(".bim"),
            output_folder=output_dir,
            enable_plink_prefilter=True,
        )

        arrays, _ = read_encoded_arrays(prepared_data.array_folder)
        assert arrays.shape == (100, 4, 1000)
        assert np.allclose(arrays.sum(axis=1), 1.0)

        plink_filtered_dirs = list(output_dir.glob("plink_filtered_*"))
        assert (
            len(plink_filtered_dirs) == 0
        ), "High overlap should skip PLINK pre-filtering"


if __name__ == "__main__":
    test_prepare_module()
    test_prepare_flipped_data()
    test_plink_prefilter_optimization()
    test_plink_prefilter_disabled()
    test_high_overlap_skips_prefilter()
