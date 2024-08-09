import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from aislib.misc_utils import ensure_path_exists, get_logger

from eir_auto_gp.preprocess import gwas_pre_selection as gps

logger = get_logger(name=__name__)


def get_gwas_feature_selection_filter_config(
    genotype_data_path: str,
    data_config: Dict[str, Any],
    modelling_config: Dict[str, Any],
    feature_selection_config: Dict[str, Any],
) -> gps.GWASPreFilterConfig:
    target_names = (
        modelling_config["output_con_columns"] + modelling_config["output_cat_columns"]
    )
    covariate_names = (
        modelling_config["input_con_columns"] + modelling_config["input_cat_columns"]
    )

    fam_file_path = next(Path(genotype_data_path).glob("*.fam"))
    id_split_folder = _get_id_split_folder(
        pre_split_folder=data_config["pre_split_folder"],
        data_output_folder=data_config["data_output_folder"],
        fam_file=fam_file_path,
    )

    config = gps.GWASPreFilterConfig(
        genotype_data_path=genotype_data_path,
        label_file_path=data_config["label_file_path"],
        output_path=feature_selection_config["feature_selection_output_folder"],
        target_names=target_names,
        covariate_names=covariate_names,
        pre_split_folder=id_split_folder,
        only_gwas=True,
        p_value_threshold=feature_selection_config["gwas_p_value_threshold"],
    )

    return config


def _get_id_split_folder(
    pre_split_folder: Optional[str],
    data_output_folder: str,
    fam_file: Optional[Path] = None,
) -> str:
    if pre_split_folder is not None and Path(pre_split_folder).exists():
        return pre_split_folder

    ids_folder = Path(data_output_folder, "ids")
    if ids_folder.exists():
        gps.add_plink_format_train_test_files(fam_file=fam_file, ids_folder=ids_folder)
        return str(ids_folder)

    raise ValueError(f"Could not find ids folder in {data_output_folder}")


def run_gwas_feature_selection(
    genotype_data_path: str,
    data_config: Dict[str, Any],
    modelling_config: Dict[str, Any],
    feature_selection_config: Dict[str, Any],
) -> str | Path:
    filter_config = get_gwas_feature_selection_filter_config(
        genotype_data_path=genotype_data_path,
        data_config=data_config,
        modelling_config=modelling_config,
        feature_selection_config=feature_selection_config,
    )

    fam_file_path = next(Path(genotype_data_path).glob("*.fam"))
    fs_output_folder = feature_selection_config["feature_selection_output_folder"]
    gwas_output_path = Path(fs_output_folder, "gwas_output")

    gwas_label_path = Path(filter_config.output_path, "gwas_label_file.csv")
    ensure_path_exists(path=gwas_label_path)
    one_hot_mappings_file = None

    gwas_finished = all_gwas_already_finished(
        target_names=filter_config.target_names,
        gwas_output_folder=gwas_output_path,
    )

    if not gwas_label_path.exists() and not gwas_finished:
        _, one_hot_mappings_file = gps.prepare_gwas_label_file(
            label_file_path=filter_config.label_file_path,
            fam_file_path=fam_file_path,
            output_path=gwas_label_path,
            target_names=filter_config.target_names,
            covariate_names=filter_config.covariate_names,
        )

    if not gwas_finished:
        base_path = fam_file_path.with_suffix("")

        target_names = gps.parse_gwas_label_file_column_names(
            target_names=filter_config.target_names,
            gwas_label_file=gwas_label_path,
        )

        maybe_prepare_plink_split_files(pre_split_folder=filter_config.pre_split_folder)
        train_ids_file = Path(filter_config.pre_split_folder, "train_ids_plink.txt")
        assert (
            train_ids_file.exists()
        ), f"Could not find train ids file at {train_ids_file}"

        command = gps.get_plink_gwas_command(
            base_path=base_path,
            label_file_path=gwas_label_path,
            target_names=target_names,
            covariate_names=filter_config.covariate_names,
            output_path=gwas_output_path,
            ids_file=train_ids_file,
            one_hot_mappings_file=one_hot_mappings_file,
        )

        logger.info("Running GWAS with command: %s", " ".join(command))
        subprocess.run(command, check=True)
        gps.plot_gwas_results(
            gwas_output_path=gwas_output_path,
            p_value_line=filter_config.p_value_threshold,
        )
        gwas_label_path.unlink()

    snps_to_keep_path = gps.gather_all_snps_to_keep(
        gwas_output_folder=gwas_output_path,
        p_value_threshold=filter_config.p_value_threshold,
    )

    return snps_to_keep_path


def maybe_prepare_plink_split_files(pre_split_folder: str) -> None:
    maybe_make_plink_train_ids_file(pre_split_folder=pre_split_folder)
    maybe_make_plink_test_ids_file(pre_split_folder=pre_split_folder)


def maybe_make_plink_train_ids_file(pre_split_folder: str) -> None:
    pre_split_folder_path = Path(pre_split_folder)

    train_ids_file = pre_split_folder_path / "train_ids_plink.txt"

    if train_ids_file.exists():
        return

    train_ids = pre_split_folder_path / "train_ids.txt"
    val_ids = pre_split_folder_path / "valid_ids.txt"

    if not train_ids.exists():
        raise ValueError(f"Could not find train ids file at {train_ids}")

    if not val_ids.exists():
        raise ValueError(f"Could not find val ids file at {val_ids}")

    with open(train_ids_file, "w") as f:
        with open(train_ids, "r") as train_f:
            for line in train_f:
                line = line.strip()
                f.write(f"{line}\t{line}\n")

        with open(val_ids, "r") as val_f:
            for line in val_f:
                line = line.strip()
                f.write(f"{line}\t{line}\n")

    logger.info(
        "Could not find plink train ids file at %s, so created it.", train_ids_file
    )


def maybe_make_plink_test_ids_file(pre_split_folder: str) -> None:
    pre_split_folder_path = Path(pre_split_folder)

    test_ids_file = pre_split_folder_path / "test_ids_plink.txt"

    if test_ids_file.exists():
        return

    test_ids = pre_split_folder_path / "test_ids.txt"

    if not test_ids.exists():
        raise ValueError(f"Could not find test ids file at {test_ids}")

    with open(test_ids_file, "w") as f:
        with open(test_ids, "r") as test_f:
            for line in test_f:
                line = line.strip()
                f.write(f"{line}\t{line}\n")


def all_gwas_already_finished(
    target_names: Sequence[str], gwas_output_folder: str | Path
) -> bool:
    for target_name in target_names:
        parsed_target_name = gps.parse_target_for_plink(target=target_name)
        gwas_file = tuple(
            i for i in Path(gwas_output_folder).glob(f"gwas.{parsed_target_name}*")
        )
        if not gwas_file:
            return False

    return True
