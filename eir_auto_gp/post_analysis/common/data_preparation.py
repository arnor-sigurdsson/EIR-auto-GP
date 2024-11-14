import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger
from eir.data_load.data_source_modules.deeplake_ops import (
    is_deeplake_dataset,
    load_deeplake_dataset,
)
from eir.setup.input_setup_modules import setup_omics as eir_setup_omics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eir_auto_gp.single_task.modelling.dl_feature_selection import (
    get_dl_gwas_top_n_snp_list_df,
    get_dl_top_n_snp_list_df,
)
from eir_auto_gp.single_task.modelling.gwas_bo_feature_selection import (
    get_gwas_top_n_snp_list_df,
)

logger = get_logger(name=__name__)


@dataclass()
class DataPaths:
    experiment_config: Path
    train_data_path: Path
    test_data_path: Path
    train_labels_path: Path
    test_labels_path: Path
    snp_bim_path: Path
    split_ids_path: Optional[Path]
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

    ids_path = data_dir / "ids"
    if not ids_path.exists():
        ids_path = None

    paths = DataPaths(
        experiment_config=run_dir / "config.json",
        train_data_path=data_dir / "genotype/final/train",
        test_data_path=data_dir / "genotype/final/test",
        train_labels_path=data_dir / "tabular/final/labels_train.csv",
        test_labels_path=data_dir / "tabular/final/labels_test.csv",
        split_ids_path=ids_path,
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

    df_genotype_nan_mask: pd.DataFrame


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
    data_name: str,
) -> ModelData:
    """
    TODO:
        Seems that we run into trashing / freezing issues on macOS GHA runners here
        after bumping Deep Lake to V4.
    """

    df_genotype_input = load_genotype_samples_into_df(
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
        data_name=data_name,
    )

    df_tabular_input = _log_and_replace_inf(
        df=df_tabular_input,
        data_name=f"{data_name} tabular input",
    )
    df_genotype_input = _log_and_replace_inf(
        df=df_genotype_input,
        data_name=f"{data_name} genotype input",
    )

    assert df_genotype_input.shape[0] == df_target.shape[0] == df_tabular_input.shape[0]
    assert df_genotype_input.index.equals(df_target.index)
    assert df_genotype_input.index.equals(df_tabular_input.index)

    df_genotype_test_nan_mask = df_genotype_input.isnull().astype(int)
    model_data = ModelData(
        df_genotype_input=df_genotype_input,
        df_tabular_input=df_tabular_input,
        df_target=df_target,
        df_genotype_nan_mask=df_genotype_test_nan_mask,
    )

    return model_data


def load_genotype_samples_into_df(
    genotype_input_path: Path,
    genotype_indices_to_load: np.ndarray,
    top_snps_list: list[str],
) -> pd.DataFrame:
    ids = []
    genotype_arrays = []

    sample_id_iter = _get_genotype_id_iterator(genotype_input_path=genotype_input_path)

    for sample_id, sample_genotype in sample_id_iter:
        sample_genotype_subset = sample_genotype[:, genotype_indices_to_load]

        array_maxed = sample_genotype_subset.argmax(0).astype(np.float32)
        array_maxed[array_maxed == 3] = np.nan

        ids.append(sample_id)
        genotype_arrays.append(array_maxed)

    genotype_array = np.stack(genotype_arrays, axis=0)
    df_genotype = pd.DataFrame(data=genotype_array, columns=top_snps_list, index=ids)
    df_genotype.index.name = "ID"
    df_genotype.index = df_genotype.index.astype(str)

    return df_genotype


def _get_genotype_id_iterator(genotype_input_path: Path) -> Generator:
    if is_deeplake_dataset(data_source=str(genotype_input_path)):
        deeplake_ds = load_deeplake_dataset(data_source=str(genotype_input_path))
        for deeplake_sample in deeplake_ds:
            sample_id = deeplake_sample["ID"]
            sample_genotype = deeplake_sample["genotype"]
            yield sample_id, sample_genotype
    else:
        for f in genotype_input_path.iterdir():
            sample_id = f.stem
            sample_genotype = np.load(f)

            yield sample_id, sample_genotype


def load_tabular_data_into_df(
    labels_input_path: Path,
    experiment_info: ExperimentInfo,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    ei = experiment_info

    all_cols = list(
        ["ID"]
        + ei.input_cat_columns
        + ei.input_con_columns
        + ei.output_cat_columns
        + ei.output_con_columns,
    )

    df_labels = pd.read_csv(
        filepath_or_buffer=labels_input_path,
        usecols=all_cols,
        index_col="ID",
    )
    df_labels.index = df_labels.index.astype(str)

    input_cols = ei.input_cat_columns + ei.input_con_columns
    df_input = df_labels[input_cols]

    target_cols = ei.output_cat_columns + ei.output_con_columns
    df_target = df_labels[target_cols]
    assert df_target.shape[1] == 1

    df_input.columns = ["COVAR_" + col for col in df_input.columns]

    return df_input, df_target


def _handle_missing_target_values(
    df_target: pd.DataFrame,
    df_genotype_input: pd.DataFrame,
    df_tabular_input: pd.DataFrame,
    data_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Handling missing/Inf target values, target shape: %s", df_target.shape)

    df_target = _log_and_replace_inf(df=df_target, data_name=f"{data_name} target")

    df_target = df_target.dropna()
    df_genotype_input = df_genotype_input.loc[df_target.index]
    df_tabular_input = df_tabular_input.loc[df_target.index]

    assert df_genotype_input.shape[0] == df_target.shape[0] == df_tabular_input.shape[0]

    logger.info("After dropping missing values, target shape: %s", df_target.shape)

    return df_target, df_genotype_input, df_tabular_input


def process_split_model_data(
    split_model_data: "SplitModelData",
    task_type: str,
) -> "SplitModelData":
    transformers = _set_up_transformers(
        df_train_tab_input=split_model_data.train.df_tabular_input,
        df_train_tab_target=split_model_data.train.df_target,
        task_type=task_type,
    )

    numerical_columns_ = transformers.numerical_columns
    categorical_columns_ = transformers.categorical_columns

    df_tabular_combined = pd.concat(
        [
            split_model_data.train.df_tabular_input,
            split_model_data.val.df_tabular_input,
            split_model_data.test.df_tabular_input,
        ]
    )

    continuous_imputer, categorical_imputer = setup_tabular_imputers(
        df_train=split_model_data.train.df_tabular_input,
        df_combined=df_tabular_combined,
        categorical_columns=categorical_columns_,
        numerical_columns=numerical_columns_,
    )

    categorical_maps = compute_categorical_maps(
        df_combined=df_tabular_combined,
        categorical_columns=categorical_columns_,
    )

    df_genotype_combined = pd.concat(
        [
            split_model_data.train.df_genotype_input,
            split_model_data.val.df_genotype_input,
            split_model_data.test.df_genotype_input,
        ]
    )
    genotype_imputer = setup_genotype_imputers(
        df_train=split_model_data.train.df_genotype_input,
        df_combined=df_genotype_combined,
    )

    def process_model_data(
        model_data: ModelData,
    ) -> ModelData:
        df_tabular_input_encoded = model_data.df_tabular_input
        if len(df_tabular_input_encoded.columns) > 0:

            df_tabular_input_imputed = apply_tabular_imputers(
                df=df_tabular_input_encoded,
                continuous_imputer=continuous_imputer,
                categorical_imputer=categorical_imputer,
                categorical_columns=categorical_columns_,
                numerical_columns=numerical_columns_,
            )

            df_tabular_input_encoded = apply_one_hot_encoding(
                df_base=df_tabular_input_imputed,
                category_maps=categorical_maps,
                drop_first=True,
            )

            input_scaler = transformers.input_scaler
            if input_scaler is not None:
                numerical_columns = transformers.numerical_columns
                transformed_numerical = input_scaler.transform(
                    df_tabular_input_encoded[numerical_columns]
                )
                df_tabular_input_encoded[numerical_columns] = transformed_numerical

        df_genotype_encoded = apply_genotype_imputers(
            df=model_data.df_genotype_input,
            genotype_imputer=genotype_imputer,
        )
        df_genotype_encoded = df_genotype_encoded.astype(int)

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
            df_genotype_input=df_genotype_encoded,
            df_tabular_input=df_tabular_input_encoded,
            df_target=df_target_standardized,
            df_genotype_nan_mask=model_data.df_genotype_nan_mask,
        )

    processed = SplitModelData(
        train=process_model_data(model_data=split_model_data.train),
        val=process_model_data(model_data=split_model_data.val),
        test=process_model_data(model_data=split_model_data.test),
        transformers=transformers,
    )

    validate_processed_split_model_data(split_model_data=processed)

    return processed


def compute_categorical_maps(
    df_combined: pd.DataFrame, categorical_columns: list[str]
) -> dict[str, list[str]]:
    category_maps = {
        col: sorted(df_combined[col].dropna().unique().tolist())
        for col in categorical_columns
    }
    return category_maps


def apply_one_hot_encoding(
    df_base: pd.DataFrame,
    category_maps: dict[str, list[str]],
    drop_first: bool = True,
) -> pd.DataFrame:

    df = df_base.copy()
    for col, categories in category_maps.items():
        for category in categories:
            df[f"{col}_{category}"] = np.array(df[col] == category).astype(int)
        if drop_first and len(categories) > 1:
            drop_category = categories[0]
            df = df.drop(f"{col}_{drop_category}", axis=1)

    return df.drop(columns=category_maps.keys())


def validate_processed_split_model_data(split_model_data: "SplitModelData") -> None:
    train = split_model_data.train
    val = split_model_data.val
    test = split_model_data.test

    train_tab = train.df_tabular_input
    val_tab = val.df_tabular_input
    test_tab = test.df_tabular_input

    assert train_tab.shape[1] == val_tab.shape[1] == test_tab.shape[1]
    assert train_tab.columns.equals(val_tab.columns)
    assert train_tab.columns.equals(test_tab.columns)

    train_gen = train.df_genotype_input
    val_gen = val.df_genotype_input
    test_gen = test.df_genotype_input

    assert train_gen.shape[1] == val_gen.shape[1] == test_gen.shape[1]
    assert train_gen.columns.equals(val_gen.columns)
    assert train_gen.columns.equals(test_gen.columns)

    train_target = train.df_target
    val_target = val.df_target
    test_target = test.df_target

    assert train_target.shape[1] == val_target.shape[1] == test_target.shape[1]
    assert train_target.columns.equals(val_target.columns)
    assert train_target.columns.equals(test_target.columns)


def setup_genotype_imputers(
    df_train: pd.DataFrame,
    df_combined: pd.DataFrame,
) -> Optional[SimpleImputer]:
    categorical_imputer = None
    if df_combined.isnull().any().any():
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        missing_info = df_combined.isnull().sum()
        logger.info(
            f"Missing values in genotype columns before imputation:\n{missing_info}.\n"
            f"These missing values will be imputed for model training, but "
            f"skipped during effect analysis."
        )
        categorical_imputer.fit(df_train)

    return categorical_imputer


def apply_genotype_imputers(
    df: pd.DataFrame,
    genotype_imputer: Optional[SimpleImputer],
) -> pd.DataFrame:
    if genotype_imputer:
        imputed_values = genotype_imputer.transform(X=df)
        df = pd.DataFrame(data=imputed_values, columns=df.columns, index=df.index)

    return df


def setup_tabular_imputers(
    df_train: pd.DataFrame,
    df_combined: pd.DataFrame,
    categorical_columns: list,
    numerical_columns: list,
) -> tuple[Optional[SimpleImputer], Optional[SimpleImputer]]:
    continuous_imputer = None
    categorical_imputer = None

    logger.info(
        "Checking for missing values and setting up imputation based "
        "on training data. Imputation based on training set statistics will "
        "be applied to training, validation and test sets."
    )

    if numerical_columns:
        logger.info(f"Setting up continuous imputer for columns: {numerical_columns}")
        continuous_imputer = SimpleImputer(strategy="mean")
        if df_combined[numerical_columns].isnull().any().any():
            missing_info = df_combined[numerical_columns].isnull().sum()
            logger.info(
                f"Missing values in numerical columns "
                f"before imputation:\n{missing_info}"
            )
        continuous_imputer.fit(df_train[numerical_columns])

    if categorical_columns:
        logger.info(
            f"Setting up categorical imputer for columns: {categorical_columns}"
        )
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        if df_combined[categorical_columns].isnull().any().any():
            missing_info = df_combined[categorical_columns].isnull().sum()
            logger.info(
                f"Missing values in categorical columns before"
                f" imputation:\n{missing_info}"
            )
        categorical_imputer.fit(df_train[categorical_columns])

    return continuous_imputer, categorical_imputer


def apply_tabular_imputers(
    df: pd.DataFrame,
    continuous_imputer: Optional[SimpleImputer],
    categorical_imputer: Optional[SimpleImputer],
    categorical_columns: list[str],
    numerical_columns: list[str],
) -> pd.DataFrame:
    if continuous_imputer and numerical_columns:
        df[numerical_columns] = continuous_imputer.transform(X=df[numerical_columns])

    if categorical_imputer and categorical_columns:
        df[categorical_columns] = categorical_imputer.transform(
            X=df[categorical_columns]
        )

    return df


@dataclass()
class FitTransformers:
    input_scaler: Optional[StandardScaler]
    target_label_encoder: Optional[LabelEncoder]
    numerical_columns: list[str]
    categorical_columns: list[str]


def _set_up_transformers(
    df_train_tab_input: pd.DataFrame,
    df_train_tab_target: pd.DataFrame,
    task_type: str,
) -> FitTransformers:
    numerical_columns = df_train_tab_input.select_dtypes(include=[np.number]).columns
    categorical_columns = df_train_tab_input.select_dtypes(
        include=[
            "object",
            "category",
        ]
    ).columns

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
        numerical_columns=numerical_columns.tolist(),
        categorical_columns=categorical_columns.tolist(),
    )


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
        data_name="Train and valid",
    )

    train_ids, valid_ids, _ = maybe_gather_ids(split_ids_path=data_paths.split_ids_path)

    train_data, valid_data = split_model_data_object(
        model_data=train_and_valid_data,
        eval_size=0.1,
        ids_train=train_ids,
        ids_val=valid_ids,
    )

    test_data = set_up_model_data(
        genotype_input_path=data_paths.test_data_path,
        labels_input_path=data_paths.test_labels_path,
        experiment_info=experiment_info,
        genotype_indices_to_load=subset_indices,
        top_snps_list=top_snps_list,
        data_name="Test",
    )

    split_model_data_raw = SplitModelData(
        train=train_data,
        val=valid_data,
        test=test_data,
        transformers=None,
    )
    validate_split_model_data(split_model_data=split_model_data_raw)

    split_model_data_processed: SplitModelData = process_split_model_data(
        split_model_data=split_model_data_raw,
        task_type=experiment_info.target_type,
    )
    validate_split_model_data(split_model_data=split_model_data_processed)

    return split_model_data_processed


def maybe_gather_ids(
    split_ids_path: Optional[Path],
) -> tuple[Optional[list], Optional[list], Optional[list]]:
    if split_ids_path is None:
        return None, None, None

    train_file = split_ids_path / "train_ids.txt"
    val_file = split_ids_path / "valid_ids.txt"
    test_file = split_ids_path / "test_ids.txt"

    train_ids = list(train_file.read_text().split())

    val_ids = []
    if val_file.exists():
        val_ids = list(val_file.read_text().split())

    test_ids = list(test_file.read_text().split())

    return train_ids, val_ids, test_ids


def _log_and_replace_inf(df: pd.DataFrame, data_name: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_cols)

    if inf_mask.any().any():
        inf_summary = inf_mask.sum()
        logger.error(
            f"Found Inf values in '{data_name}': {inf_summary}. "
            f"Replacing Inf values with NaN."
        )
        for col in numeric_cols.columns[inf_mask.any()]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


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
    model_data: ModelData,
    ids_train: list[str],
    ids_val: list[str],
    eval_size: float = 0.2,
) -> tuple[ModelData, ModelData]:
    df_input = pd.concat(
        objs=[model_data.df_genotype_input, model_data.df_tabular_input],
        axis=1,
    )

    if not ids_val:
        logger.info("Randomly splitting data into train and validation sets.")
        x_train, x_val, y_train, y_val = train_test_split(
            df_input,
            model_data.df_target,
            test_size=eval_size,
            random_state=42,
        )
    else:
        logger.info("Using provided IDs to split data into train and validation sets.")
        index_set = set(df_input.index)

        valid_ids_train = [id_ for id_ in ids_train if id_ in index_set]
        valid_ids_val = [id_ for id_ in ids_val if id_ in index_set]

        logger.info(f"Train IDs: {len(valid_ids_train)}/{len(ids_train)} available.")
        logger.info(f"Validation IDs: {len(valid_ids_val)}/{len(ids_val)} available.")

        x_train = df_input.loc[valid_ids_train]
        x_val = df_input.loc[valid_ids_val]
        y_train = model_data.df_target.loc[valid_ids_train]
        y_val = model_data.df_target.loc[valid_ids_val]

    df_genotype_input_train = x_train[model_data.df_genotype_input.columns]
    df_genotype_train_nan_mask = df_genotype_input_train.isnull().astype(int)
    train_data = ModelData(
        df_genotype_input=df_genotype_input_train,
        df_tabular_input=x_train[model_data.df_tabular_input.columns],
        df_target=y_train,
        df_genotype_nan_mask=df_genotype_train_nan_mask,
    )

    df_genotype_input_val = x_val[model_data.df_genotype_input.columns]
    val_data = ModelData(
        df_genotype_input=df_genotype_input_val,
        df_tabular_input=x_val[model_data.df_tabular_input.columns],
        df_target=y_val,
        df_genotype_nan_mask=df_genotype_input_val.isnull().astype(int),
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
        output_cat_columns=output_cat_columns,
        output_con_columns=output_con_columns,
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
