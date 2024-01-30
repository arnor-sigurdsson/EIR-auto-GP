import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger
from eir.data_load.data_source_modules.deeplake_ops import load_deeplake_dataset
from eir.setup.input_setup_modules import setup_omics as eir_setup_omics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir_auto_gp.modelling.dl_feature_selection import (
    get_dl_gwas_top_n_snp_list_df,
    get_dl_top_n_snp_list_df,
)
from eir_auto_gp.modelling.gwas_bo_feature_selection import get_gwas_top_n_snp_list_df

logger = get_logger(name=__name__)


@dataclass()
class DataPaths:
    experiment_config: Path
    train_data_path: Path
    test_data_path: Path
    train_labels_path: Path
    test_labels_path: Path
    snp_bim_path: Path
    dl_attribution_path: Path
    gwas_attribution_path: Optional[Path]
    analysis_output_path: Path


def build_data_paths(run_dir: Path) -> DataPaths:
    gwas_folder = run_dir / "feature_selection/gwas_output/"
    if gwas_folder.exists():
        gwas_attributions = next(
            i
            for i in gwas_folder.iterdir()
            if "glm.logistic" in i.name or "glm.linear" in i.name
        )
        assert gwas_attributions.exists()
    else:
        gwas_attributions = None

    config_path = run_dir / "config.json"
    config_dict = json.load(config_path.open("r"))

    data_dir = config_dict["data_output_folder"]
    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = run_dir / "data"

    paths = DataPaths(
        experiment_config=run_dir / "config.json",
        train_data_path=data_dir / "genotype/final/train",
        test_data_path=data_dir / "genotype/final/test",
        train_labels_path=data_dir / "tabular/final/labels_train.csv",
        test_labels_path=data_dir / "tabular/final/labels_test.csv",
        snp_bim_path=data_dir / "genotype/processed/parsed_files/data_final.bim",
        dl_attribution_path=run_dir
        / "feature_selection/snp_importance/dl_attributions.csv",
        gwas_attribution_path=gwas_attributions,
        analysis_output_path=run_dir / "analysis/post_analysis/",
    )

    return paths


@dataclass()
class ExperimentInfo:
    input_cat_columns: list[str]
    input_con_columns: list[str]
    all_input_columns: list[str]
    output_cat_columns: list[str]
    output_con_columns: list[str]
    target_type: str


@dataclass()
class ModelData:
    df_genotype_input: pd.DataFrame
    df_tabular_input: pd.DataFrame
    df_target: pd.DataFrame


def get_subset_indices_and_names(
    bim_file_path: Path,
    dl_attributions_path: Optional[Path],
    gwas_attributions_path: Optional[Path],
    top_n_snps: int,
) -> tuple[np.ndarray, list[str]]:
    df_bim = eir_setup_omics.read_bim(bim_file_path=str(bim_file_path))
    validate_file_paths(dl_path=dl_attributions_path, gwas_path=gwas_attributions_path)

    df_dl_attributions: Optional[pd.DataFrame] = None
    df_gwas_attributions: Optional[pd.DataFrame] = None

    if dl_attributions_path is not None and dl_attributions_path.exists():
        logger.info("DL attributions %s exist.", dl_attributions_path)
        df_dl_attributions = pd.read_csv(filepath_or_buffer=dl_attributions_path)

    if gwas_attributions_path is not None and gwas_attributions_path.exists():
        logger.info("GWAS attributions %s exist.", gwas_attributions_path)
        df_gwas_attributions = _read_gwas_attributions(
            gwas_attributions_path=gwas_attributions_path
        )

    match (df_dl_attributions, df_gwas_attributions):
        case (pd.DataFrame(), None):
            df_top_snps = get_dl_top_n_snp_list_df(
                df_attributions=df_dl_attributions,
                df_bim=df_bim,
                top_n_snps=top_n_snps,
            )
        case (None, pd.DataFrame()):
            df_top_snps = get_gwas_top_n_snp_list_df(
                df_gwas=df_gwas_attributions,
                top_n_snps=top_n_snps,
            )
        case (pd.DataFrame(), pd.DataFrame()):
            df_dl_gwas = df_dl_attributions.join(other=df_gwas_attributions)
            df_top_snps = get_dl_gwas_top_n_snp_list_df(
                df_dl_gwas=df_dl_gwas,
                top_n_snps=top_n_snps,
            )
        case _:
            raise ValueError("Both DL and GWAS attributions are None.")

    top_snps_list_importance_order = df_top_snps["SNP"].tolist()
    top_snps_list_ordered = _ensure_alignment_of_top_snps_with_genetic_position(
        df_bim=df_bim, top_snps_list=top_snps_list_importance_order
    )

    _check_bim(df_bim=df_bim, top_snps_list=top_snps_list_ordered)

    _ensure_all_snps_present(df_bim=df_bim, top_snps_list=top_snps_list_ordered)

    subset_indices = eir_setup_omics._setup_snp_subset_indices(
        df_bim=df_bim,
        snps_to_subset=top_snps_list_ordered,
    )

    _ensure_order_of_snps_matches(
        df_bim=df_bim,
        subset_indices=subset_indices,
        top_snps_list=top_snps_list_ordered,
    )

    return subset_indices, top_snps_list_ordered


def _ensure_all_snps_present(df_bim: pd.DataFrame, top_snps_list: list[str]) -> None:
    assert all(
        snp in df_bim["VAR_ID"].values for snp in top_snps_list
    ), "All top SNPs should be present in df_bim"


def _ensure_order_of_snps_matches(
    df_bim: pd.DataFrame, subset_indices: np.ndarray, top_snps_list: list[str]
) -> None:
    sorted_snps_in_subset = df_bim.loc[subset_indices, "VAR_ID"].tolist()
    assert (
        sorted_snps_in_subset == top_snps_list
    ), "The order of SNPs in subset_indices does not match top_snps_list"


def _ensure_alignment_of_top_snps_with_genetic_position(
    df_bim: pd.DataFrame, top_snps_list: list[str]
) -> list[str]:
    snp_to_position = dict(zip(df_bim["VAR_ID"], df_bim.index))
    top_snps_list_sorted = sorted(
        top_snps_list, key=lambda x: snp_to_position.get(x, float("inf"))
    )

    return top_snps_list_sorted


def validate_file_paths(dl_path: Optional[Path], gwas_path: Optional[Path]) -> None:
    if (dl_path is None or not dl_path.exists()) and (
        gwas_path is None or not gwas_path.exists()
    ):
        raise ValueError(
            f"Neither {dl_path} nor {gwas_path} exist. "
            "Post-analysis requires at least one of these files to exist."
        )


def _read_gwas_attributions(gwas_attributions_path: Path) -> pd.DataFrame:
    df_gwas = pd.read_csv(filepath_or_buffer=gwas_attributions_path, sep="\t")
    df_gwas = df_gwas.rename(columns={"ID": "VAR_ID"})
    df_gwas = df_gwas.set_index("VAR_ID")
    df_gwas = df_gwas.rename(columns={"P": "GWAS P-VALUE"})
    df_gwas = df_gwas[["GWAS P-VALUE"]]

    return df_gwas


def _check_bim(df_bim: pd.DataFrame, top_snps_list: list[str]) -> None:
    assert df_bim["VAR_ID"].nunique() == df_bim.shape[0]
    assert pd.Series(top_snps_list).isin(df_bim["VAR_ID"]).all()


def set_up_model_data(
    genotype_input_path: Path,
    labels_input_path: Path,
    experiment_info: ExperimentInfo,
    genotype_indices_to_load: np.ndarray,
    top_snps_list: list[str],
) -> ModelData:
    df_genotype_input = load_deeplake_samples_into_df(
        genotype_input_path=genotype_input_path,
        genotype_indices_to_load=genotype_indices_to_load,
        top_snps_list=top_snps_list,
    )

    df_tabular_input, df_target = load_tabular_data_into_df(
        labels_input_path=labels_input_path,
        experiment_info=experiment_info,
    )

    df_genotype_input.sort_index(inplace=True)
    df_tabular_input.sort_index(inplace=True)
    df_target.sort_index(inplace=True)

    df_target, df_genotype_input, df_tabular_input = _handle_missing_target_values(
        df_target=df_target,
        df_genotype_input=df_genotype_input,
        df_tabular_input=df_tabular_input,
    )

    assert df_genotype_input.shape[0] == df_target.shape[0] == df_tabular_input.shape[0]
    assert df_genotype_input.index.equals(df_target.index)
    assert df_genotype_input.index.equals(df_tabular_input.index)

    model_data = ModelData(
        df_genotype_input=df_genotype_input,
        df_tabular_input=df_tabular_input,
        df_target=df_target,
    )

    return model_data


def _handle_missing_target_values(
    df_target: pd.DataFrame,
    df_genotype_input: pd.DataFrame,
    df_tabular_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Handling missing target values, target shape: %s", df_target.shape)

    df_target = df_target.dropna()
    df_genotype_input = df_genotype_input.loc[df_target.index]
    df_tabular_input = df_tabular_input.loc[df_target.index]

    assert df_genotype_input.shape[0] == df_target.shape[0] == df_tabular_input.shape[0]

    logger.info("After dropping missing values, target shape: %s", df_target.shape)

    return df_target, df_genotype_input, df_tabular_input


def process_split_model_data(
    split_model_data: "SplitModelData", task_type: str
) -> "SplitModelData":
    transformers = _set_up_transformers(
        df_train_tab_input=split_model_data.train.df_tabular_input,
        df_train_tab_target=split_model_data.train.df_target,
        task_type=task_type,
    )

    def process_model_data(
        model_data: ModelData,
    ) -> ModelData:
        df_tabular_input_encoded = model_data.df_tabular_input
        if len(df_tabular_input_encoded.columns) > 0:
            df_tabular_input_encoded = pd.get_dummies(
                data=model_data.df_tabular_input,
                drop_first=False,
            )

            input_scaler = transformers.input_scaler
            if input_scaler is not None:
                numerical_columns = transformers.numerical_columns
                transformed_numerical = input_scaler.transform(
                    df_tabular_input_encoded[numerical_columns]
                )
                df_tabular_input_encoded[numerical_columns] = transformed_numerical

        target_data = model_data.df_target.values
        if task_type == "classification":
            label_encoder = transformers.target_label_encoder
            target_data_1d = target_data.ravel()
            target_data = label_encoder.transform(target_data_1d)

        df_target_standardized = pd.DataFrame(
            data=target_data,
            columns=model_data.df_target.columns,
            index=model_data.df_target.index,
        )

        return ModelData(
            df_genotype_input=model_data.df_genotype_input,
            df_tabular_input=df_tabular_input_encoded,
            df_target=df_target_standardized,
        )

    return SplitModelData(
        train=process_model_data(model_data=split_model_data.train),
        val=process_model_data(model_data=split_model_data.val),
        test=process_model_data(model_data=split_model_data.test),
        transformers=transformers,
    )


@dataclass()
class FitTransformers:
    input_scaler: Optional[StandardScaler]
    target_label_encoder: Optional[LabelEncoder]
    numerical_columns: list[str]


def _set_up_transformers(
    df_train_tab_input: pd.DataFrame, df_train_tab_target: pd.DataFrame, task_type: str
) -> FitTransformers:
    numerical_columns = df_train_tab_input.select_dtypes(include=[np.number]).columns

    if len(numerical_columns) > 0:
        input_scaler = StandardScaler()
        input_scaler.fit(X=df_train_tab_input[numerical_columns])
    else:
        input_scaler = None

    if task_type == "classification":
        target_transformer = LabelEncoder()
        target_transformer.fit(df_train_tab_target.values.ravel())
    elif task_type == "regression":
        target_transformer = None
    else:
        raise ValueError()

    return FitTransformers(
        input_scaler=input_scaler,
        target_label_encoder=target_transformer,
        numerical_columns=numerical_columns,
    )


def load_tabular_data_into_df(
    labels_input_path: Path,
    experiment_info: ExperimentInfo,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = list(
        ["ID"]
        + experiment_info.input_cat_columns
        + experiment_info.input_con_columns
        + experiment_info.output_cat_columns
        + experiment_info.output_con_columns,
    )

    df_labels = pd.read_csv(
        filepath_or_buffer=labels_input_path, usecols=all_cols, index_col="ID"
    )
    df_labels.index = df_labels.index.astype(str)

    df_input = df_labels[
        experiment_info.input_cat_columns + experiment_info.input_con_columns
    ]
    df_target = df_labels[
        experiment_info.output_cat_columns + experiment_info.output_con_columns
    ]
    assert df_target.shape[1] == 1

    df_input.columns = ["COVAR_" + col for col in df_input.columns]

    return df_input, df_target


def load_deeplake_samples_into_df(
    genotype_input_path: Path,
    genotype_indices_to_load: np.ndarray,
    top_snps_list: list[str],
) -> pd.DataFrame:
    deeplake_ds = load_deeplake_dataset(data_source=str(genotype_input_path))
    ids = []
    genotype_arrays = []

    for deeplake_sample in deeplake_ds:
        sample_id = deeplake_sample["ID"].text()
        sample_genotype = deeplake_sample["genotype"].numpy()
        sample_genotype_subset = sample_genotype[:, genotype_indices_to_load]

        array_maxed = sample_genotype_subset.argmax(0)
        array_maxed[array_maxed == 3] = -1

        ids.append(sample_id)
        genotype_arrays.append(array_maxed)

    genotype_array = np.stack(genotype_arrays, axis=0)
    df_genotype = pd.DataFrame(data=genotype_array, columns=top_snps_list, index=ids)
    df_genotype.index.name = "ID"
    df_genotype.index = df_genotype.index.astype(str)

    return df_genotype


@dataclass()
class SplitModelData:
    train: ModelData
    val: ModelData
    test: ModelData

    transformers: Optional[FitTransformers] = None


def set_up_split_model_data(
    data_paths: DataPaths,
    experiment_info: ExperimentInfo,
    top_snps: int,
) -> SplitModelData:
    subset_indices, top_snps_list = get_subset_indices_and_names(
        bim_file_path=data_paths.snp_bim_path,
        dl_attributions_path=data_paths.dl_attribution_path,
        gwas_attributions_path=data_paths.gwas_attribution_path,
        top_n_snps=top_snps,
    )

    train_and_valid_data = set_up_model_data(
        genotype_input_path=data_paths.train_data_path,
        labels_input_path=data_paths.train_labels_path,
        experiment_info=experiment_info,
        genotype_indices_to_load=subset_indices,
        top_snps_list=top_snps_list,
    )

    train_data, valid_data = split_model_data_object(
        model_data=train_and_valid_data,
        test_size=0.1,
    )

    test_data = set_up_model_data(
        genotype_input_path=data_paths.test_data_path,
        labels_input_path=data_paths.test_labels_path,
        experiment_info=experiment_info,
        genotype_indices_to_load=subset_indices,
        top_snps_list=top_snps_list,
    )

    split_model_data_raw = SplitModelData(
        train=train_data,
        val=valid_data,
        test=test_data,
        transformers=None,
    )
    validate_split_model_data(split_model_data=split_model_data_raw)

    split_model_data_processed = process_split_model_data(
        split_model_data=split_model_data_raw,
        task_type=experiment_info.target_type,
    )
    validate_split_model_data(split_model_data=split_model_data_processed)

    return split_model_data_processed


def validate_split_model_data(split_model_data: "SplitModelData") -> None:
    train_ids = split_model_data.train.df_genotype_input.index
    val_ids = split_model_data.val.df_genotype_input.index
    test_ids = split_model_data.test.df_genotype_input.index

    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0

    for split in [split_model_data.train, split_model_data.val, split_model_data.test]:
        assert split.df_genotype_input.index.equals(split.df_tabular_input.index)
        assert split.df_genotype_input.index.equals(split.df_target.index)


def parse_target_type(
    output_cat_columns: list[str], output_con_columns: list[str]
) -> str:
    if len(output_cat_columns) > 0:
        assert len(output_con_columns) == 0
        target_type = "classification"
    elif len(output_con_columns) > 0:
        assert len(output_cat_columns) == 0
        target_type = "regression"
    else:
        raise ValueError()

    return target_type


def split_model_data_object(
    model_data: ModelData, test_size: float = 0.2
) -> tuple[ModelData, ModelData]:
    df_input = pd.concat(
        objs=[model_data.df_genotype_input, model_data.df_tabular_input], axis=1
    )
    x_train, x_val, y_train, y_val = train_test_split(
        df_input,
        model_data.df_target,
        test_size=test_size,
        random_state=42,
    )

    train_data = ModelData(
        df_genotype_input=x_train[model_data.df_genotype_input.columns],
        df_tabular_input=x_train[model_data.df_tabular_input.columns],
        df_target=y_train,
    )

    val_data = ModelData(
        df_genotype_input=x_val[model_data.df_genotype_input.columns],
        df_tabular_input=x_val[model_data.df_tabular_input.columns],
        df_target=y_val,
    )

    return train_data, val_data


def extract_experiment_info_from_config(config_path: Path) -> ExperimentInfo:
    config_dict = json.load(config_path.open("r"))

    input_cat_columns = config_dict["input_cat_columns"]
    input_con_columns = config_dict["input_con_columns"]
    all_input_columns = input_cat_columns + input_con_columns

    output_cat_columns = config_dict["output_cat_columns"]
    output_con_columns = config_dict["output_con_columns"]

    target_type = parse_target_type(
        output_cat_columns=output_cat_columns, output_con_columns=output_con_columns
    )

    experiment_info = ExperimentInfo(
        input_cat_columns=input_cat_columns,
        input_con_columns=input_con_columns,
        all_input_columns=all_input_columns,
        output_cat_columns=output_cat_columns,
        output_con_columns=output_con_columns,
        target_type=target_type,
    )

    return experiment_info
