import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from eir.train_utils.train_handlers import _iterdir_ignore_hidden

from eir_auto_gp.preprocess.gwas_pre_selection import (
    _get_plink_filter_snps_command,
    get_covariate_names,
    get_gwas_parser,
    get_gwas_pre_filter_config,
    get_pheno_names,
    get_plink_gwas_command,
    run_gwas_pre_filter_wrapper,
)
from eir_auto_gp.single_task.modelling.run_modelling import _lines_in_file


def _get_test_cl_commands() -> list[str]:
    base = (
        "--genotype_data_path tests/test_data/ "
        "--label_file_path tests/test_data/penncath.csv  "
        "--output_path runs/penncath_gwas "
        "--covariate_names age sex ldl hdl tg "
        "--gwas_p_value_threshold 1e-04 "
        "--target_names CAD"
    )

    commands = [
        base,
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

    for expected_file in [
        "gwas_output",
        "ids",
        "penncath.bed",
        "penncath.bim",
        "penncath.fam",
        "penncath.log",
    ]:
        assert (output_folder / expected_file).exists()

    orig_snps_file = Path("tests/test_data/penncath.bim")
    n_orig_snps = _lines_in_file(file_path=orig_snps_file)
    n_gwas_snps = _lines_in_file(file_path=output_folder / "penncath.bim")
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
    assert len(gwas_files) == 4
    assert "gwas.CAD.glm.logistic.hybrid" in (i.name for i in gwas_files)


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
        "--glm",
        "firth-fallback",
        "hide-covar",
        "omit-ref",
        "no-x-sex",
        "allow-no-covars",
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
    assert sorted(result) == sorted(["Gender_M", "Gender_F"])


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
