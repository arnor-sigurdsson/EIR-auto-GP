import argparse
import shutil
import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import Optional

import pandas as pd
from aislib.misc_utils import ensure_path_exists, get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

logger = get_logger(name=__name__)


def run_sync_wrapper(cl_args: argparse.Namespace) -> None:
    genotype_data_path = Path(cl_args.genotype_data_path)
    reference_bim_path = Path(cl_args.reference_bim_path)
    data_output_folder = Path(cl_args.output_folder)

    run_sync(
        genotype_data_path=genotype_data_path,
        reference_bim_path=reference_bim_path,
        data_output_folder=data_output_folder,
    )

    shutil.rmtree(data_output_folder / "filtered_genotype_data")
    shutil.rmtree(data_output_folder / "dummy_genotype_data")
    shutil.rmtree(data_output_folder / "with_missing")

    final_data = Path(data_output_folder / "reordered_genotype_data")

    final_data_renamed = final_data.parent / "synced_data"
    final_data.rename(final_data_renamed)

    for f in final_data_renamed.iterdir():
        if f.suffix in (".bed", ".bim", ".fam", ".log"):
            new_name = f.name.replace("_with_missing_reordered", "")
            f.rename(f.parent / new_name)


def get_cl_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genotype_data_path",
        type=str,
        help="The path to the genotype data to sync, this ultimately will get"
        "projected / mapped into the reference_bim_path.",
    )

    parser.add_argument(
        "--reference_bim_path",
        type=str,
        help="Path to reference bim fild to sync genotype_data_path to.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        help="The folder to output the results.",
    )

    return parser


def run_sync(
    genotype_data_path: Path,
    reference_bim_path: Path,
    data_output_folder: Path,
) -> str:
    with Progress() as progress:
        genotype_base_name = check_genotype_folder(genotype_folder=genotype_data_path)

        # this is the reference / target bim file, which we project one below into
        df_bim_exp = read_bim_and_cast_dtypes(bim_file_path=reference_bim_path)
        # this is the one we are predicting on generally
        df_bim_prd = get_predict_bim_file(genotype_folder=genotype_data_path)

        log_output_path = data_output_folder / "snp_overlap_analysis.txt"
        log_overlap(
            df_bim_prd=df_bim_prd,
            df_bim_exp=df_bim_exp,
            output_path=log_output_path,
        )

        sync_task = progress.add_task(
            description="[green]Synchronizing genotype data...",
            total=4,
        )

        cols = ["CHR_CODE", "BP_COORD", "ALT", "REF"]
        df_bim_prd["key"] = df_bim_prd[cols].astype(str).agg("-".join, axis=1)
        df_bim_exp["key"] = df_bim_exp[cols].astype(str).agg("-".join, axis=1)
        to_remove = df_bim_prd[~df_bim_prd["key"].isin(df_bim_exp["key"])]
        df_remove = df_bim_prd.loc[to_remove.index]
        df_remove = df_remove.drop(columns=["key"])

        progress.advance(sync_task)

        filtered_genotype_data_path = remove_extra_snps(
            df_remove=df_remove,
            genotype_data_path=str(genotype_data_path),
            genotype_base_name=genotype_base_name,
            output_folder=data_output_folder,
        )
        progress.advance(sync_task)

        to_add = df_bim_exp[~df_bim_exp["key"].isin(df_bim_prd["key"])]
        df_add = df_bim_exp.loc[to_add.index]

        added_genotype_data = add_missing_snps(
            df_add=df_add,
            genotype_data_path=filtered_genotype_data_path,
            genotype_base_name=genotype_base_name,
            output_folder=data_output_folder,
        )
        progress.advance(sync_task)

        reordered_genotype_data = update_and_reorder(
            genotype_input_folder=added_genotype_data,
            df_exp_bim=df_bim_exp,
            output_folder=data_output_folder,
        )
        progress.advance(sync_task)

        df_bim_exp = df_bim_exp.drop(columns=["key"])
        df_bim_final = get_predict_bim_file(
            genotype_folder=Path(reordered_genotype_data)
        )
        assert df_bim_final[cols].equals(df_bim_exp[cols])
        progress.advance(sync_task)

    print("\n")

    return str(reordered_genotype_data)


def log_overlap(
    df_bim_prd: pd.DataFrame,
    df_bim_exp: pd.DataFrame,
    output_path: Optional[str | Path] = None,
) -> None:
    logger.info(
        "Examining SNP overlap between current prediction and previous experiment."
    )

    console = Console()

    overlap = df_bim_prd.merge(df_bim_exp, on=["CHR_CODE", "BP_COORD", "ALT", "REF"])
    prd_count = len(df_bim_prd)
    exp_count = len(df_bim_exp)
    overlap_count = len(overlap)
    to_remove_count = prd_count - overlap_count
    to_add_count = exp_count - overlap_count
    overlap_percentage = (overlap_count / prd_count * 100) if prd_count > 0 else 0

    color = (
        "green"
        if overlap_percentage > 75
        else "yellow" if overlap_percentage > 50 else "red"
    )

    table = Table(title="SNP Overlap Analysis", title_style="bold blue")
    table.add_column("Metric", style="dim", width=40)
    table.add_column("Value", style="bold")

    results = {
        "overlapping_snps": overlap_count,
        "overlap_percentage": f"{overlap_percentage:.2f}%",
        "previous_experiment_snps": exp_count,
        "current_prediction_snps": prd_count,
        "snps_to_remove": to_remove_count,
        "snps_to_add": to_add_count,
    }

    table.add_row("Number of overlapping SNPs", f"[{color}]{overlap_count:,}[/]")
    table.add_row(
        "Percentage of current prediction SNPs overlapping",
        f"[{color}]{overlap_percentage:.2f}%[/]",
    )
    table.add_row("Number of SNPs in previous experiment", f"{exp_count:,}")
    table.add_row("Number of SNPs for current prediction", f"{prd_count:,}")
    table.add_row("Number of SNPs to remove", f"{to_remove_count:,}")
    table.add_row("Number of SNPs to add", f"{to_add_count:,}")

    console.print("\n")
    console.print(table)
    console.print("\n")

    if output_path:
        try:
            overlap_quality = (
                "High"
                if overlap_percentage > 75
                else "Medium" if overlap_percentage > 50 else "Low"
            )
            results_formatted = [
                f"{k.replace('_', ' ').title()}: {v}" for k, v in results.items()
            ]
            output_text = [
                "SNP Overlap Analysis Results",
                "=========================\n",
                *results_formatted,
                "\nDetailed Information:",
                f"- The overlap between current prediction and previous experiment"
                f" is {overlap_percentage:.2f}%",
                f"- {to_remove_count:,} SNPs need to be removed from the current "
                f"prediction",
                f"- {to_add_count:,} SNPs need to be added from the previous "
                f"experiment",
                "\nQuality Assessment:",
                f"- Overlap Quality: {overlap_quality}",
            ]

            with open(output_path, "w") as f:
                f.write("\n".join(output_text).replace("Snps", "SNPs"))

            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {str(e)}")


def check_genotype_folder(genotype_folder: Path) -> str:
    files = [
        i for i in genotype_folder.iterdir() if i.suffix in [".bed", ".bim", ".fam"]
    ]
    assert len(files) == 3

    name = files[0].stem
    assert all(i.stem == name for i in files)

    return name


def remove_extra_snps(
    df_remove: pd.DataFrame,
    genotype_data_path: str,
    genotype_base_name: str,
    output_folder: Path,
) -> str:
    if len(df_remove) == 0:
        return genotype_data_path

    filtered_path = output_folder / "filtered_genotype_data"
    filtered_path.mkdir(exist_ok=True, parents=True)

    snps_to_remove_filepath = str(filtered_path / "snps_to_remove.txt")

    df_remove[["VAR_ID"]].to_csv(
        snps_to_remove_filepath,
        index=False,
        header=False,
        sep="\t",
    )

    remove_command = [
        "plink2",
        "--bfile",
        genotype_data_path + "/" + genotype_base_name,
        "--exclude",
        snps_to_remove_filepath,
        "--keep-allele-order",
        "--make-bed",
        "--out",
        filtered_path / genotype_base_name,
    ]
    run_subprocess(command=remove_command)

    return str(filtered_path)


def add_missing_snps(
    df_add: pd.DataFrame,
    genotype_data_path: str,
    genotype_base_name: str,
    output_folder: Path,
) -> str:
    if len(df_add) == 0:
        return genotype_data_path

    num_snps_to_add = len(df_add)

    fam_file_path = Path(genotype_data_path) / f"{genotype_base_name}.fam"
    num_samples = sum(1 for _ in open(fam_file_path))

    dummy_path = output_folder / "dummy_genotype_data"
    dummy_path.mkdir(exist_ok=True, parents=True)

    dummy_command = [
        "plink2",
        "--dummy",
        str(num_samples),
        str(num_snps_to_add),
        "1",
        "--out",
        str(dummy_path / "dummy_dataset"),
        "--make-bed",
    ]
    run_subprocess(command=dummy_command)

    bim_file_path = dummy_path / "dummy_dataset.bim"
    if bim_file_path.exists():
        bim_file_path.unlink()

    df_add.to_csv(bim_file_path, index=False, header=False, sep="\t")

    fam_file_destination = dummy_path / "dummy_dataset.fam"
    shutil.copy(fam_file_path, fam_file_destination)

    output_folder_missing_imputed = output_folder / "with_missing"
    ensure_path_exists(path=output_folder_missing_imputed, is_folder=True)

    merge_command = [
        "plink",
        "--bfile",
        str(Path(genotype_data_path) / genotype_base_name),
        "--bmerge",
        str(dummy_path / "dummy_dataset"),
        "--keep-allele-order",
        "--make-bed",
        "--out",
        str(output_folder_missing_imputed / (genotype_base_name + "_with_missing")),
    ]
    run_subprocess(command=merge_command)

    return str(output_folder_missing_imputed)


def get_experiment_bim_file(experiment_folder: Path) -> Path:
    bim_file = experiment_folder / "meta" / "snps.bim"
    assert bim_file.exists()

    return bim_file


def get_predict_bim_file(genotype_folder: Path) -> pd.DataFrame:
    bim_files = [i for i in genotype_folder.iterdir() if i.suffix == ".bim"]
    assert len(bim_files) == 1

    bim_file = bim_files[0]

    df_bim_predict = read_bim_and_cast_dtypes(bim_file_path=str(bim_file))

    return df_bim_predict


def update_and_reorder(
    genotype_input_folder: str,
    df_exp_bim: pd.DataFrame,
    output_folder: Path,
) -> Path:
    genotype_input_path = Path(genotype_input_folder)
    output_path = Path(output_folder, "reordered_genotype_data")
    output_path.mkdir(parents=True, exist_ok=True)

    bim_files = [i for i in genotype_input_path.iterdir() if i.suffix == ".bim"]
    assert len(bim_files) == 1, "There should be exactly one .bim file in the folder."
    bim_file = bim_files[0]
    df_bim_original = read_bim_and_cast_dtypes(bim_file_path=str(bim_file))
    df_bim = df_bim_original.copy()

    mapping = df_exp_bim.set_index(
        [
            "CHR_CODE",
            "BP_COORD",
            "ALT",
            "REF",
        ]
    )["VAR_ID"].to_dict()

    df_bim["VAR_ID"] = df_bim.apply(
        lambda row: mapping.get(
            (row["CHR_CODE"], row["BP_COORD"], row["ALT"], row["REF"]), row["VAR_ID"]
        ),
        axis=1,
    )
    df_bim.to_csv(bim_file, sep="\t", header=False, index=False)

    merged_output_path = create_and_merge_dummy_fileset(
        genotype_input_folder=genotype_input_folder,
        df_exp_bim=df_exp_bim,
        output_folder=output_path,
    )

    df_bim_original.to_csv(bim_file, sep="\t", header=False, index=False)

    return merged_output_path


def create_and_merge_dummy_fileset(
    genotype_input_folder: str,
    df_exp_bim: pd.DataFrame,
    output_folder: Path,
) -> Path:
    genotype_input_path = Path(genotype_input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    base_name = check_genotype_folder(genotype_folder=Path(genotype_input_folder))
    num_snps = len(df_exp_bim)

    dummy_path = output_folder / "dummy_genotype_data"
    dummy_path.mkdir(exist_ok=True, parents=True)
    dummy_command = [
        "plink2",
        "--dummy",
        "1",
        str(num_snps),
        "1",
        "--out",
        str(dummy_path / "dummy_dataset"),
        "--make-bed",
    ]
    run_subprocess(command=dummy_command)

    bim_file_path = dummy_path / "dummy_dataset.bim"
    df_exp_bim[
        [
            "CHR_CODE",
            "VAR_ID",
            "POS_CM",
            "BP_COORD",
            "ALT",
            "REF",
        ]
    ].to_csv(bim_file_path, sep="\t", header=False, index=False)

    merged_output_path = output_folder / f"{base_name}_merged"
    merge_command = [
        "plink",
        "--bfile",
        str(genotype_input_path / base_name),
        "--bmerge",
        str(dummy_path / "dummy_dataset"),
        "--keep-allele-order",
        "--make-bed",
        "--out",
        str(merged_output_path),
    ]
    run_subprocess(command=merge_command)

    remove_command = [
        "plink",
        "--bfile",
        str(merged_output_path),
        "--remove",
        str(dummy_path / "dummy_dataset.fam"),
        "--keep-allele-order",
        "--make-bed",
        "--out",
        str(output_folder / f"{base_name}_reordered"),
    ]
    run_subprocess(command=remove_command)

    shutil.rmtree(output_folder / "dummy_genotype_data")

    keep_name = f"{base_name}_reordered"
    for f in output_folder.iterdir():
        if f.is_file() and not f.stem.startswith(keep_name):
            f.unlink()

    return output_folder


def run_subprocess(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error: {e.stderr}")
        raise


def read_bim_and_cast_dtypes(bim_file_path: Path | str) -> pd.DataFrame:

    dtypes = {
        "CHR_CODE": str,
        "VAR_ID": str,
        "POS_CM": float,
        "BP_COORD": int,
        "ALT": str,
        "REF": str,
    }

    df_bim = read_bim(bim_file_path=str(bim_file_path))

    df_bim = df_bim.astype(dtypes)

    return df_bim


def main() -> None:
    parser = get_parser()
    cl_args = parser.parse_args()
    run_sync_wrapper(cl_args=cl_args)


if __name__ == "__main__":
    main()
