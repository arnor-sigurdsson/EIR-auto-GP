import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.preprocess.gwas_pre_selection import (
    _get_plink_filter_snps_command,
    gather_all_snps_to_keep,
    get_covariate_names,
    get_gwas_parser,
    get_gwas_pre_filter_config,
    get_pheno_names,
    get_plink_gwas_command,
    run_gwas_pre_filter_wrapper,
    run_ld_clumping,
)
from eir_auto_gp.single_task.modelling.run_modelling import lines_in_file


def _get_test_cl_commands() -> list[str]:
    base = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--output_path runs/penncath_gwas "
        "--covariate_names age sex ldl hdl tg "
        "--gwas_p_value_threshold 1e-04 "
        "--target_names CAD "
        "--do_plot"
    )

    base_with_ld_clumping = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--output_path runs/penncath_gwas_ld "
        "--covariate_names age sex ldl hdl tg "
        "--gwas_p_value_threshold 1e-04 "
        "--target_names CAD "
        "--do_plot "
        "--ld_clump "
        "--ld_clump_r2 0.2 "
        "--ld_clump_kb 500"
    )

    commands = [
        base,
        base_with_ld_clumping,
    ]

    return commands


@pytest.mark.parametrize("command", _get_test_cl_commands())
def test_run_gwas_pre_selection(command: str, tmp_path: Path) -> None:
    parser = get_gwas_parser()
    cl_args = parser.parse_args(command.split())
    cl_args.output_path = str(tmp_path)

    filter_config = get_gwas_pre_filter_config(cl_args=cl_args)
    with patch.object(Path, "unlink") as _:
        run_gwas_pre_filter_wrapper(filter_config=filter_config)

    output_folder = tmp_path

    expected_files = [
        "gwas_output",
        "ids",
        "penncath.bed",
        "penncath.bim",
        "penncath.fam",
        "penncath.log",
    ]

    for expected_file in expected_files:
        assert (output_folder / expected_file).exists()

    if "--ld_clump" in command:
        gwas_output_folder = output_folder / "gwas_output"
        snps_to_keep_file = gwas_output_folder / "snps_to_keep.txt"
        assert snps_to_keep_file.exists(), (
            "SNPs to keep file should exist when LD clumping is enabled"
        )

    orig_snps_file = Path("tests/test_data/penncath.bim")
    n_orig_snps = lines_in_file(file_path=orig_snps_file)
    n_gwas_snps = lines_in_file(file_path=output_folder / "penncath.bim")
    assert n_gwas_snps < n_orig_snps

    df_gwas = pd.read_csv(output_folder / "gwas_label_file.csv", sep="\t")
    assert set(df_gwas.columns) == {
        "FID",
        "IID",
        "age",
        "sex",
        "ldl",
        "hdl",
        "tg",
        "CAD",
    }

    for col in df_gwas.columns:
        if col not in ["FID", "IID"]:
            assert df_gwas[col].dtype in [float, int]

    gwas_files = list(_iterdir_ignore_hidden(path=output_folder / "gwas_output"))

    assert "gwas.CAD.glm.logistic.hybrid" in (i.name for i in gwas_files)

    if "--ld_clump" in command:
        expected_min_files = 6  # Original 4 + clumping files
        assert len(gwas_files) >= expected_min_files, (
            f"Expected at least {expected_min_files} files with LD clumping, "
            f"got {len(gwas_files)}"
        )

        file_names = [f.name for f in gwas_files]
        assert "snps_to_keep.txt" in file_names, (
            "snps_to_keep.txt should be created with LD clumping"
        )
        assert "clumped_snps.txt" in file_names, (
            "clumped_snps.txt should be created with LD clumping"
        )
    else:
        assert len(gwas_files) == 4, (
            f"Expected 4 files without LD clumping, got {len(gwas_files)}"
        )


def mocked_read_csv(*args, **kwargs):
    df_mock = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    return df_mock


def test_get_plink_filter_snps_command():
    base_path = "base_path"
    snps_to_keep_path = "snps_to_keep_path"
    output_path = "output_path"

    result = _get_plink_filter_snps_command(
        base_path=base_path,
        snps_to_keep_path=snps_to_keep_path,
        output_path=output_path,
    )

    expected_result = [
        "plink2",
        "--bfile",
        "base_path",
        "--extract",
        "snps_to_keep_path",
        "--make-bed",
        "--out",
        "output_path",
    ]

    assert result == expected_result


def test_get_plink_gwas_command():
    base_path = "base_path"
    label_file_path = "label_file_path"
    target_names = ["A", "B"]
    covariate_names = ["C", "D"]
    output_path = "output_path"
    ids_file = None

    with patch("eir_auto_gp.preprocess.gwas_pre_selection.ensure_path_exists") as _:
        with patch(
            "eir_auto_gp.preprocess.gwas_pre_selection.get_covariate_names"
        ) as mocked_get_covar_names:
            mocked_get_covar_names.return_value = covariate_names

            result = get_plink_gwas_command(
                base_path=base_path,
                label_file_path=label_file_path,
                target_names=target_names,
                covariate_names=covariate_names,
                output_path=output_path,
                ids_file=ids_file,
                one_hot_mappings_file=None,
            )

    expected_result = [
        "plink2",
        "--bfile",
        "base_path",
        "--1",
        "--pheno",
        "label_file_path",
        "--pheno-name",
        "A",
        "B",
        "--no-input-missing-phenotype",
        "--glm",
        "firth-fallback",
        "hide-covar",
        "omit-ref",
        "no-x-sex",
        "allow-no-covars",
        "qt-residualize",
        "cc-residualize",
        "skip-invalid-pheno",
        "--out",
        "output_path/gwas",
        "--covar",
        "label_file_path",
        "--covar-name",
        "C",
        "D",
        "--covar-variance-standardize",
    ]

    assert result == expected_result


@pytest.mark.parametrize(
    "target_names,covariate_names,expected_result",
    [
        (
            ["A", "B"],
            ["C", "D"],
            ["A", "B"],
        ),
        (
            None,
            ["C", "D"],
            ["A", "B"],
        ),
    ],
)
def test_get_pheno_names(target_names, covariate_names, expected_result):
    label_file_path = Path("label_file_path")

    with patch("pandas.read_csv", side_effect=mocked_read_csv):
        result = get_pheno_names(
            label_file_path=label_file_path,
            target_names=target_names,
            covariate_names=covariate_names,
        )

    assert result == expected_result


def mocked_read_csv_covariate(*args, **kwargs):
    df_mock = pd.DataFrame(
        {
            "A": [1],
            "B": [2],
            "C": [3],
            "D": [4],
            "ID": [5],
            "FID": [6],
            "IID": [7],
        }
    )
    return df_mock


@pytest.mark.parametrize(
    "target_names,covariate_names,expected_result",
    [
        (
            ["A", "B"],
            ["C", "D"],
            ["C", "D"],
        ),
        (
            ["A", "B"],
            None,
            [],
        ),
        (
            ["A", "B"],
            [],
            [],
        ),
    ],
)
def test_get_covariate_names_basic(target_names, covariate_names, expected_result):
    label_file_path = Path("label_file_path")

    with patch("pandas.read_csv", side_effect=mocked_read_csv_covariate):
        result = get_covariate_names(
            label_file_path=label_file_path,
            target_names=target_names,
            covariate_names=covariate_names,
            one_hot_mappings_file=None,
        )

    assert result == expected_result


@pytest.fixture
def mock_csv(tmp_path):
    sample_csv = """
    ID FID IID Age Gender
    1 1 1 30 M
    2 1 2 35 F
    """

    d = tmp_path / "sub"
    d.mkdir(exist_ok=True)
    p = d / "sample.csv"
    p.write_text(sample_csv)
    return p


@pytest.fixture
def mock_one_hot_mappings(tmp_path):
    sample_one_hot_mappings = {
        "Gender": [
            "Gender_M",
            "Gender_F",
        ]
    }

    d = tmp_path / "sub"
    d.mkdir(exist_ok=True)
    p = d / "one_hot_mappings.json"
    p.write_text(json.dumps(sample_one_hot_mappings))
    return p


def test_get_covariate_names_with_one_hot(mock_csv, mock_one_hot_mappings):
    target_names = ["Age"]
    covariate_names = ["Gender"]
    result = get_covariate_names(
        label_file_path=mock_csv,
        target_names=target_names,
        covariate_names=covariate_names,
        one_hot_mappings_file=mock_one_hot_mappings,
    )
    assert sorted(result) == sorted(["Gender_Gender_M", "Gender_Gender_F"])


def test_get_covariate_names_no_one_hot(mock_csv):
    target_names = ["Age"]
    covariate_names = ["Gender"]
    result = get_covariate_names(
        label_file_path=mock_csv,
        target_names=target_names,
        covariate_names=covariate_names,
        one_hot_mappings_file=None,
    )
    assert result == ["Gender"]


def test_get_covariate_names_no_inputs(mock_csv, mock_one_hot_mappings):
    result = get_covariate_names(
        label_file_path=mock_csv,
        target_names=[],
        covariate_names=[],
        one_hot_mappings_file=mock_one_hot_mappings,
    )
    assert result == []


@pytest.fixture
def mock_gwas_output_folder(tmp_path):
    gwas_folder = tmp_path / "gwas_output"
    gwas_folder.mkdir()

    gwas_file = gwas_folder / "gwas.CAD.glm.logistic"
    gwas_content = (
        "ID\tP\tCHR\tPOS\n"
        "rs1\t0.001\t1\t1000\n"
        "rs2\t0.5\t1\t2000\n"
        "rs3\t0.0001\t2\t3000\n"
    )
    gwas_file.write_text(gwas_content)

    return gwas_folder


@pytest.fixture
def mock_clumps_file(tmp_path):
    clumps_file = tmp_path / "clumped_snps.clumps"
    clumps_content = (
        "#CHROM POS ID P TOTAL NONSIG S0.05 S0.01 S0.001 S0.0001 SP2\n"
        "1 1000 rs1 0.001 5 2 2 1 1 0 rs4,rs5\n"
        "2 3000 rs3 0.0001 3 1 1 1 1 1 rs6\n"
    )
    clumps_file.write_text(clumps_content)
    return clumps_file


def test_run_ld_clumping_cache_existing(mock_gwas_output_folder, tmp_path):
    base_path = tmp_path / "genotype_data"

    existing_clumped_file = mock_gwas_output_folder / "clumped_snps.txt"
    existing_clumped_file.write_text("rs1\nrs2\nrs3\n")

    result = run_ld_clumping(
        gwas_output_folder=mock_gwas_output_folder,
        base_path=base_path,
        r2_threshold=0.1,
        kb_window=250,
    )

    assert result == existing_clumped_file
    assert result.exists()


def test_run_ld_clumping_no_gwas_files(tmp_path):
    gwas_folder = tmp_path / "gwas_output"
    gwas_folder.mkdir()
    base_path = tmp_path / "genotype_data"

    with pytest.raises(ValueError, match="No GWAS result files found"):
        run_ld_clumping(
            gwas_output_folder=gwas_folder,
            base_path=base_path,
        )


@patch("subprocess.run")
def test_run_ld_clumping_command_construction(
    mock_subprocess, mock_gwas_output_folder, mock_clumps_file, tmp_path
):
    base_path = tmp_path / "genotype_data"

    mock_subprocess.return_value = None

    expected_clumps_file = mock_gwas_output_folder / "clumped_snps.clumps"
    mock_clumps_file.rename(expected_clumps_file)

    run_ld_clumping(
        gwas_output_folder=mock_gwas_output_folder,
        base_path=base_path,
        r2_threshold=0.2,
        kb_window=500,
    )

    mock_subprocess.assert_called_once()
    call_args = mock_subprocess.call_args[0][0]

    expected_command = [
        "plink2",
        "--bfile",
        str(base_path),
        "--clump",
        str(mock_gwas_output_folder / "gwas.CAD.glm.logistic"),
        "--clump-p1",
        "1",
        "--clump-r2",
        "0.2",
        "--clump-kb",
        "500",
        "--out",
        str(mock_gwas_output_folder / "clumped_snps"),
    ]

    assert call_args == expected_command


def test_run_ld_clumping_parsing_clumps_file(
    mock_gwas_output_folder, mock_clumps_file, tmp_path
):
    base_path = tmp_path / "genotype_data"

    expected_clumps_file = mock_gwas_output_folder / "clumped_snps.clumps"
    mock_clumps_file.rename(expected_clumps_file)

    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = None

        result = run_ld_clumping(
            gwas_output_folder=mock_gwas_output_folder,
            base_path=base_path,
        )

    assert result.exists()
    with open(result) as f:
        snps = f.read().strip().split("\n")

    assert set(snps) == {"rs1", "rs3"}


def test_run_ld_clumping_missing_snp_column(mock_gwas_output_folder, tmp_path):
    base_path = tmp_path / "genotype_data"

    clumps_file = mock_gwas_output_folder / "clumped_snps.clumps"
    clumps_content = "CHR F VARIANT BP P\n1 1 rs1 1000 0.001\n"
    clumps_file.write_text(clumps_content)

    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = None

        with pytest.raises(KeyError, match="Required column 'SNP' or 'ID' not found"):
            run_ld_clumping(
                gwas_output_folder=mock_gwas_output_folder,
                base_path=base_path,
            )


def test_gather_all_snps_to_keep_with_ld_clumping(mock_gwas_output_folder, tmp_path):
    base_path = tmp_path / "genotype_data"

    clumped_snps_file = mock_gwas_output_folder / "clumped_snps.txt"
    clumped_snps_file.write_text("rs1\nrs2\nrs3\n")

    with patch(
        "eir_auto_gp.preprocess.gwas_pre_selection.run_ld_clumping"
    ) as mock_clump:
        mock_clump.return_value = clumped_snps_file

        result_path = gather_all_snps_to_keep(
            gwas_output_folder=mock_gwas_output_folder,
            p_value_threshold=0.01,
            ld_clump=True,
            ld_clump_r2=0.1,
            ld_clump_kb=250,
            base_path=base_path,
        )

    assert result_path.exists()

    mock_clump.assert_called_once_with(
        gwas_output_folder=mock_gwas_output_folder,
        base_path=base_path,
        r2_threshold=0.1,
        kb_window=250,
    )


def test_gather_all_snps_to_keep_ld_clump_no_base_path():
    with pytest.raises(ValueError, match="base_path is required when ld_clump=True"):
        gather_all_snps_to_keep(
            gwas_output_folder=Path("dummy"),
            p_value_threshold=0.01,
            ld_clump=True,
            base_path=None,
        )


def test_gwas_parser_ld_clump_arguments():
    parser = get_gwas_parser()

    args = parser.parse_args(
        [
            "--genotype_data_path",
            "test_path",
            "--label_file_path",
            "test_labels.csv",
            "--output_path",
            "test_output",
            "--target_names",
            "CAD",
            "--ld_clump",
            "--ld_clump_r2",
            "0.2",
            "--ld_clump_kb",
            "500",
        ]
    )

    assert args.ld_clump is True
    assert args.ld_clump_r2 == 0.2
    assert args.ld_clump_kb == 500

    args_default = parser.parse_args(
        [
            "--genotype_data_path",
            "test_path",
            "--label_file_path",
            "test_labels.csv",
            "--output_path",
            "test_output",
            "--target_names",
            "CAD",
        ]
    )

    assert args_default.ld_clump is False
    assert args_default.ld_clump_r2 == 0.5
    assert args_default.ld_clump_kb == 250


def test_gwas_pre_filter_config_ld_clump():
    parser = get_gwas_parser()
    args = parser.parse_args(
        [
            "--genotype_data_path",
            "test_path",
            "--label_file_path",
            "test_labels.csv",
            "--output_path",
            "test_output",
            "--target_names",
            "CAD",
            "--ld_clump",
            "--ld_clump_r2",
            "0.15",
            "--ld_clump_kb",
            "300",
        ]
    )

    config = get_gwas_pre_filter_config(cl_args=args)

    assert config.ld_clump is True
    assert config.ld_clump_r2 == 0.15
    assert config.ld_clump_kb == 300
