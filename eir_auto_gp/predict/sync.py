import shutil
import subprocess
from pathlib import Path

import pandas as pd
from aislib.misc_utils import ensure_path_exists
from eir.setup.input_setup_modules.setup_omics import read_bim


def run_sync(
    genotype_data_path: Path,
    experiment_folder: Path,
    output_folder: Path,
) -> str:
    genotype_base_name = check_genotype_folder(genotype_folder=genotype_data_path)

    df_bim_exp = get_experiment_bim_file(experiment_folder=experiment_folder)
    df_bim_prd = get_predict_bim_file(genotype_folder=genotype_data_path)

    cols = ["CHR_CODE", "BP_COORD"]
    df_bim_prd["key"] = df_bim_prd[cols].astype(str).agg("-".join, axis=1)
    df_bim_exp["key"] = df_bim_exp[cols].astype(str).agg("-".join, axis=1)

    to_remove = df_bim_prd[~df_bim_prd["key"].isin(df_bim_exp["key"])]
    df_remove = df_bim_prd.loc[to_remove.index]

    df_remove = df_remove.drop(columns=["key"])

    filtered_genotype_data_path = remove_extra_snps(
        df_remove=df_remove,
        genotype_data_path=str(genotype_data_path),
        genotype_base_name=genotype_base_name,
        output_folder=output_folder,
    )

    to_add = df_bim_exp[~df_bim_exp["key"].isin(df_bim_prd["key"])]
    df_add = df_bim_exp.loc[to_add.index]

    final_genotype_data_path = add_missing_snps(
        df_add=df_add,
        genotype_data_path=filtered_genotype_data_path,
        genotype_base_name=genotype_base_name,
        output_folder=output_folder,
    )

    df_bim_exp = df_bim_exp.drop(columns=["key"])
    df_bim_final = get_predict_bim_file(genotype_folder=Path(final_genotype_data_path))
    assert df_bim_final.equals(df_bim_exp)

    return final_genotype_data_path


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
        "--make-bed",
        "--out",
        filtered_path / genotype_base_name,
    ]
    subprocess.run(remove_command, check=True)

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
    subprocess.run(dummy_command, check=True)

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
        "--make-bed",
        "--out",
        str(output_folder_missing_imputed / (genotype_base_name + "_with_missing")),
    ]
    subprocess.run(merge_command, check=True)

    return str(output_folder_missing_imputed)


def get_experiment_bim_file(experiment_folder: Path) -> pd.DataFrame:
    bim_file = experiment_folder / "meta" / "snps.bim"
    assert bim_file.exists()

    df_bim_experiment = read_bim(bim_file_path=str(bim_file))

    return df_bim_experiment


def get_predict_bim_file(genotype_folder: Path) -> pd.DataFrame:
    bim_files = [i for i in genotype_folder.iterdir() if i.suffix == ".bim"]
    assert len(bim_files) == 1

    bim_file = bim_files[0]

    df_bim_predict = read_bim(bim_file_path=str(bim_file))

    return df_bim_predict
