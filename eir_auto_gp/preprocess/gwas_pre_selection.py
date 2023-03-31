import argparse
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from qmplot import manhattanplot, qqplot
from qmplot.modules._qq import ppoints
from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.preprocess.converge import (
    gather_ids_from_csv_file,
    _id_setup_wrapper,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def main():
    parser = get_gwas_parser()
    cl_args = parser.parse_args()
    validate_geno_data_path(geno_data_path=cl_args.genotype_data_path)

    filter_config_ = get_gwas_pre_filter_config(cl_args=cl_args)
    run_gwas_pre_filter_wrapper(filter_config=filter_config_)


def run_gwas_pre_filter_wrapper(filter_config: "GWASPreFilterConfig") -> None:
    fam_file_path = next(Path(filter_config.genotype_data_path).glob("*.fam"))

    gwas_label_path = Path(filter_config.output_path, "gwas_label_file.csv")
    prepare_gwas_label_file(
        label_file_path=filter_config.label_file_path,
        fam_file_path=fam_file_path,
        output_path=gwas_label_path,
    )

    common_ids_to_keep = gather_all_ids(
        fam_file_path=fam_file_path,
        label_file_path=filter_config.label_file_path,
    )

    _id_setup_wrapper(
        common_ids_to_keep=common_ids_to_keep,
        output_root=Path(filter_config.output_path),
        pre_split_folder=filter_config.pre_split_folder,
        freeze_validation_set=True,
    )
    train_ids_plink, *_ = add_plink_format_train_test_files(
        fam_file=fam_file_path,
        ids_folder=Path(filter_config.output_path, "ids"),
    )

    target_names = parse_gwas_label_file_column_names(
        target_names=filter_config.target_names, gwas_label_file=gwas_label_path
    )

    base_path = fam_file_path.with_suffix("")
    gwas_output_path = Path(filter_config.output_path, "gwas_output")
    command = get_plink_gwas_command(
        base_path=base_path,
        label_file_path=gwas_label_path,
        target_names=target_names,
        covariate_names=filter_config.covariate_names,
        output_path=gwas_output_path,
        ids_file=train_ids_plink,
    )

    logger.info("Running GWAS with command: %s", " ".join(command))
    subprocess.run(command, check=True)
    plot_gwas_results(
        gwas_output_path=gwas_output_path, p_value_line=filter_config.p_value_threshold
    )
    gwas_label_path.unlink()

    snps_to_keep_path = gather_all_snps_to_keep(
        gwas_output_folder=gwas_output_path,
        p_value_threshold=filter_config.p_value_threshold,
    )

    if filter_config.only_gwas:
        logger.info("Only running GWAS, not filtering genotype data.")
        return

    filter_output_path = Path(filter_config.output_path, fam_file_path.stem)
    filter_command = _get_plink_filter_snps_command(
        base_path=base_path,
        snps_to_keep_path=snps_to_keep_path,
        output_path=filter_output_path,
    )

    logger.info("Running GWAS filter with command: %s", " ".join(filter_command))
    subprocess.run(filter_command, check=True)


def parse_gwas_label_file_column_names(
    target_names: list[str], gwas_label_file: Path
) -> list[str]:
    assert target_names
    assert gwas_label_file.exists()

    gwas_columns = pd.read_csv(gwas_label_file, nrows=0, sep=r"\s+").columns.tolist()
    parsed_names = []

    for target_name in target_names:
        if target_name in gwas_columns:
            parsed_names.append(target_name)
            continue

        prefix_matches = [
            col for col in gwas_columns if col.startswith(f"{target_name}_")
        ]
        assert prefix_matches
        parsed_names.extend(prefix_matches)

    return parsed_names


def _get_plink_filter_snps_command(
    base_path: str | Path,
    snps_to_keep_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    command_base = (
        f"plink2"
        f" --bfile {base_path}"
        f" --extract {snps_to_keep_path}"
        f" --make-bed"
        f" --out {output_path}"
    )

    return command_base.split()


def get_plink_gwas_command(
    base_path: str | Path,
    label_file_path: str | Path,
    target_names: list[str],
    covariate_names: list[str],
    output_path: str | Path,
    ids_file: Optional[str | Path],
) -> list[str]:
    ensure_path_exists(path=output_path, is_folder=True)

    pheno_names = get_pheno_names(
        label_file_path=label_file_path,
        target_names=target_names,
        covariate_names=covariate_names,
    )

    command_base = (
        f"plink2"
        f" --bfile {base_path}"
        " --1"
        f" --pheno {label_file_path}"
        f" --pheno-name {' '.join(pheno_names)}"
        f" --glm skip-invalid-pheno "
        f"firth-fallback hide-covar omit-ref no-x-sex allow-no-covars"
        f" --out {output_path}/gwas"
    )

    if covariate_names:
        command_base += f" --covar {label_file_path}"
        command_base += f" --covar-name {' '.join(covariate_names)}"
        command_base += " --covar-variance-standardize"

    if ids_file is not None:
        command_base += f" --keep {ids_file}"

    return command_base.split()


def get_pheno_names(
    label_file_path: Path, target_names: list[str], covariate_names: list[str]
) -> list[str]:
    if target_names:
        return target_names

    id_columns = ["ID", "FID", "IID"]
    all_columns = pd.read_csv(label_file_path, nrows=1, sep=r"\s+").columns.tolist()
    to_skip = id_columns + covariate_names
    inferred_target_names = [col for col in all_columns if col not in to_skip]

    logger.info(
        "No phenotype target names provided, "
        "inferring target names from label file: %s",
        inferred_target_names,
    )

    return inferred_target_names


def add_plink_format_train_test_files(
    fam_file: Path, ids_folder: Path
) -> tuple[Path, Path, Path | None]:
    df_fam = _read_fam(fam_path=fam_file)
    df_fam = df_fam[[0, 1]]

    train_ids = (
        pd.read_csv(Path(ids_folder, "train_ids.txt"), header=None)[0]
        .astype(str)
        .tolist()
    )
    train_ids = set(train_ids)

    test_ids = (
        pd.read_csv(Path(ids_folder, "test_ids.txt"), header=None)[0]
        .astype(str)
        .tolist()
    )
    test_ids = set(test_ids)

    train_output_path = Path(ids_folder, "train_ids_plink.txt")
    extract_and_save_wrapper(
        df_fam=df_fam,
        output_path=train_output_path,
        ids=train_ids,
    )

    test_output_path = Path(ids_folder, "test_ids_plink.txt")
    extract_and_save_wrapper(
        df_fam=df_fam,
        output_path=test_output_path,
        ids=test_ids,
    )

    valid_output_path = None
    if Path(ids_folder, "valid_ids.txt").exists():
        valid_ids = (
            pd.read_csv(Path(ids_folder, "valid_ids.txt"), header=None)[0]
            .astype(str)
            .tolist()
        )
        valid_ids = set(valid_ids)

        valid_output_path = Path(ids_folder, "valid_ids_plink.txt")
        extract_and_save_wrapper(
            df_fam=df_fam,
            output_path=valid_output_path,
            ids=valid_ids,
        )

    return train_output_path, test_output_path, valid_output_path


def extract_and_save_wrapper(df_fam: pd.DataFrame, output_path: Path, ids: set[str]):
    df_fam = df_fam[df_fam[1].isin(ids)]
    df_fam.to_csv(output_path, sep="\t", header=False, index=False)


def _read_fam(fam_path: Path) -> pd.DataFrame:
    df_fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        dtype={0: str, 1: str},
    )

    return df_fam


def gather_all_snps_to_keep(gwas_output_folder: Path, p_value_threshold: float) -> Path:
    snps_to_keep_path = Path(gwas_output_folder, "snps_to_keep.txt")
    if snps_to_keep_path.exists():
        logger.info("Found existing %s file, not overwriting.", snps_to_keep_path)
        return snps_to_keep_path

    snps_to_keep = set()
    for gwas_file in gwas_output_folder.iterdir():
        if "logistic" not in gwas_file.name and "linear" not in gwas_file.name:
            continue

        cur_snps = _gather_snps_to_keep_from_gwas_output(
            gwas_file_path=gwas_file,
            p_value_threshold=p_value_threshold,
        )
        snps_to_keep.update(cur_snps)

    snps_to_keep = list(snps_to_keep)
    logger.info("Keeping %d SNPs in total.", len(snps_to_keep))

    with open(snps_to_keep_path, "w") as f:
        f.write("\n".join(snps_to_keep))

    return snps_to_keep_path


def _gather_snps_to_keep_from_gwas_output(
    gwas_file_path: str | Path,
    p_value_threshold: float,
) -> list[str]:
    df_gwas = pd.read_csv(filepath_or_buffer=gwas_file_path, sep="\t")

    snps_to_keep = df_gwas[df_gwas["P"] < p_value_threshold]["ID"].tolist()
    logger.info(
        "Keeping %d SNPs from %s with p-value < %f (total: %d).",
        len(snps_to_keep),
        gwas_file_path,
        p_value_threshold,
        len(df_gwas),
    )

    return snps_to_keep


def gather_all_ids(fam_file_path: str | Path, label_file_path: str | Path) -> list[str]:
    df_fam = _read_fam(fam_path=fam_file_path)
    genotype_ids = set(df_fam[1].astype(str))

    labelled_ids = set(gather_ids_from_csv_file(file_path=label_file_path))

    common_ids_to_keep = set().union(genotype_ids, labelled_ids)
    common_ids = genotype_ids.intersection(labelled_ids)

    logger.info(
        "Keeping %d common IDs among %d total (difference: %d).",
        len(common_ids),
        len(common_ids_to_keep),
        len(common_ids_to_keep) - len(common_ids),
    )

    return list(common_ids)


def _get_train_ids_file(filter_config: "GWASPreFilterConfig") -> Path:
    if filter_config.pre_split_folder:
        train_ids_file = Path(filter_config.pre_split_folder, "train_ids.txt")
    else:
        train_ids_file = Path(filter_config.output_path, "ids/train_ids.txt")
    assert train_ids_file.exists(), f"Train ids file {train_ids_file} does not exist."

    return train_ids_file


def prepare_gwas_label_file(
    label_file_path: str | Path, fam_file_path: str | Path, output_path: str | Path
) -> Path:
    df = pd.read_csv(filepath_or_buffer=label_file_path, index_col=0, dtype={"ID": str})

    df_fam = _read_fam(fam_path=fam_file_path)

    iid_to_fid = dict(zip(df_fam[1], df_fam[0]))

    df.insert(0, "FID", df.index.map(iid_to_fid))
    df.insert(1, "IID", df.index)

    df, one_hot_mappings = _prepare_df_columns_for_gwas(df=df)

    df.to_csv(path_or_buf=output_path, sep="\t", index=False)
    json_out = output_path.parent / "one_hot_mappings.json"
    with open(json_out, "w") as f:
        json.dump(one_hot_mappings, f)

    return Path(output_path)


def _prepare_df_columns_for_gwas(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Note: We replace spaces with underscores in column names due to plink2 assuming
    tab/space separated columns.

    Note: For now we just have some simple heuristics for determining which columns
    to one-hot encode. We can make this more sophisticated/configurable later.
    """
    one_hot_target_columns = []
    one_hot_mappings = {}
    for column in df.columns:
        if column in ["FID", "IID"]:
            continue

        if df[column].dtype in ("object", "category"):
            one_hot_target_columns.append(column)
            one_hot_mappings[column] = list(df[column].unique())

        elif df[column].dtype == "int":
            n_unique = df[column].nunique()
            if 2 < n_unique < 10:
                one_hot_target_columns.append(column)
                one_hot_mappings[column] = list(df[column].unique())
            else:
                logger.debug(
                    "Integer Column %s has %d unique values, not one-hot encoding.",
                    column,
                    n_unique,
                )

    df = pd.get_dummies(df, columns=one_hot_target_columns)

    df.columns = df.columns.str.replace(" ", "_")
    df = df.fillna(-9)

    return df, one_hot_mappings


def plot_gwas_results(
    gwas_output_path: Path, p_value_line: Optional[float] = None
) -> None:
    for f in gwas_output_path.iterdir():
        if "linear" not in f.name and "logistic" not in f.name:
            continue

        df = pd.read_csv(f, sep="\t")
        df = df.dropna(how="any", axis=0)

        fig_manhattan = get_manhattan_plot(df=df, p_value_line=p_value_line)
        manhattan_output_path = Path(
            gwas_output_path, "plots", f"{f.name}_manhattan.png"
        )
        ensure_path_exists(path=manhattan_output_path)
        fig_manhattan.savefig(fname=manhattan_output_path, dpi=300)

        fig_qq = get_qq_plot(df=df)
        qq_output_path = Path(gwas_output_path, "plots", f"{f.name}_qq.png")
        ensure_path_exists(path=qq_output_path)
        fig_qq.savefig(fname=qq_output_path, dpi=300)


def get_manhattan_plot(df: pd.DataFrame, p_value_line: Optional[float]) -> plt.Figure:
    f, ax = plt.subplots(figsize=(12, 4), facecolor="w", edgecolor="k")
    ax = manhattanplot(
        data=df,
        marker=".",
        suggestiveline=p_value_line,
        genomewideline=None,
        sign_marker_p=None,
        sign_marker_color="r",
        snp="ID",
        title="",
        xlabel="Chromosome",
        ylabel=r"$-log_{10}{(P)}$",
        sign_line_cols=["#D62728", "#2CA02C"],
        hline_kws={"linestyle": "--", "lw": 1.3},
        is_annotate_topsnp=False,
        ld_block_size=500000,
        text_kws={
            "fontsize": 10,
            "arrowprops": dict(arrowstyle="-", color="k", alpha=0.6),
        },
        xticklabel_kws={"rotation": "vertical"},
        ax=ax,
    )

    fig = ax.get_figure()

    return fig


def get_qq_plot(df: pd.DataFrame) -> plt.Figure:
    f, ax = plt.subplots(figsize=(6, 6), facecolor="w", edgecolor="k")
    ax = qqplot(
        data=df["P"],
        marker="o",
        title="",
        xlabel=r"Expected $-log_{10}{(P)}$",
        ylabel=r"Observed $-log_{10}{(P)}$",
        ablinecolor="red",
        ax=ax,
    )

    data = df["P"]

    e = ppoints(len(data))
    o = -np.log10(sorted(data))
    e = -np.log10(e)

    ax.plot(
        [e.min(), o.min()],
        [e.max(), o.max()],
        color="red",
        linestyle="-",
    )

    fig = ax.get_figure()

    return fig


@dataclass
class GWASPreFilterConfig:
    genotype_data_path: str
    label_file_path: str
    output_path: str
    target_names: list[str]
    covariate_names: list[str]
    pre_split_folder: Optional[str]
    only_gwas: bool
    p_value_threshold: float = 1e-04


def get_gwas_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument(
        "--genotype_data_path",
        type=str,
        required=True,
        help="Root path to raw genotype data to be processed"
        "(e.g., containing my_data.bed, my_data.fam, my_data.bim). "
        "For this example, this parameter should be "
        "'/path/to/data/raw/genotype/'."
        "Note that the file names is not included in this path, only the root folder."
        "The file names are inferred, and *only one* set of files is expected.",
    )

    parser.add_argument(
        "--label_file_path",
        type=str,
        help="Path to label file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output folder.",
    )

    parser.add_argument(
        "--target_names",
        type=str,
        nargs="+",
        default=[],
        help="Which target columns to perform GWAS on.",
    )

    parser.add_argument(
        "--covariate_names",
        type=str,
        nargs="+",
        default=[],
        help="Covariate names to include in GWAS.",
    )

    parser.add_argument(
        "--pre_split_folder",
        type=str,
        required=False,
        help="Path to folder containing train.txt and test.txt.",
    )

    parser.add_argument(
        "--gwas_p_value_threshold",
        type=float,
        required=False,
        default=1e-04,
        help="GWAS p-value threshold for filtering.",
    )

    parser.add_argument(
        "--only_gwas",
        action="store_true",
        help="Only perform GWAS and do not filter genotype data.",
    )

    return parser


def get_gwas_pre_filter_config(cl_args: argparse.Namespace) -> GWASPreFilterConfig:
    return GWASPreFilterConfig(
        genotype_data_path=cl_args.genotype_data_path,
        label_file_path=cl_args.label_file_path,
        output_path=cl_args.output_path,
        target_names=cl_args.target_names,
        covariate_names=cl_args.covariate_names,
        pre_split_folder=cl_args.pre_split_folder,
        p_value_threshold=cl_args.gwas_p_value_threshold,
        only_gwas=cl_args.only_gwas,
    )


if __name__ == "__main__":
    main()


def validate_geno_data_path(geno_data_path: str) -> None:
    for suffix in [".bed", ".fam", ".bim"]:
        files = list(Path(geno_data_path).glob(f"*{suffix}"))
        if len(files) != 1:
            raise ValueError(
                f"Genotype data path {geno_data_path} is invalid. "
                f"Expected to find *exactly one* file with suffix {suffix}. "
                f"Found {files} ({len(files)} files) instead."
            )
