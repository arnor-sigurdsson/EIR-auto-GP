import math
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Dict, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", message=".*newer version of deeplake.*")

import deeplake
import eir.data_load.label_setup
import luigi
import pandas as pd
from aislib.misc_utils import ensure_path_exists
from eir.data_load.data_source_modules.deeplake_ops import load_deeplake_dataset

from eir_auto_gp.preprocess.genotype import FinalizeGenotypeParsing
from eir_auto_gp.preprocess.tabular import ParseLabelFile
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


class CommonSplitIntoTestSet(luigi.Task):
    genotype_data_path = luigi.Parameter()
    label_file_path = luigi.Parameter()
    data_output_folder = luigi.Parameter()
    output_format = luigi.Parameter()
    output_name = luigi.Parameter()
    pre_split_folder = luigi.Parameter()
    freeze_validation_set = luigi.BoolParameter()
    only_data = luigi.BoolParameter()
    genotype_processing_chunk_size = luigi.IntParameter()
    data_format = luigi.Parameter(default="disk")

    def requires(self):
        """
        Will require 3 tasks:
            - genotype processing,
            - label file processing.

        Will gather IDs from all of these, then generate a split from that.

        One option for configuration is to allow custom overwriting of QC command.
        """

        geno_task = FinalizeGenotypeParsing(
            raw_data_path=self.genotype_data_path,
            output_folder=str(self.data_output_folder) + "/genotype",
            output_name=self.output_name,
            output_format=self.output_format,
            genotype_processing_chunk_size=self.genotype_processing_chunk_size,
        )

        label_file_task = ParseLabelFile(
            output_folder=self.data_output_folder,
            label_file_path=self.label_file_path,
        )
        return {
            "genotype": geno_task,
            "label_file": label_file_task,
        }

    def run(self):
        inputs = self.input()

        genotype_path = Path(str(inputs["genotype"][1].path)) / "genotype"
        assert genotype_path.exists()
        label_file_path = Path(str(inputs["label_file"].path))

        output_root = Path(str(self.data_output_folder))

        common_ids_to_keep = _gather_all_ids(
            genotype_path=genotype_path, label_file=label_file_path
        )

        train_ids, valid_ids, test_ids = _id_setup_wrapper(
            output_root=output_root,
            pre_split_folder=self.pre_split_folder,
            common_ids_to_keep=common_ids_to_keep,
            freeze_validation_set=self.freeze_validation_set,
        )
        train_and_valid_ids = train_ids + valid_ids

        _split_csv_into_train_and_test(
            train_ids=train_and_valid_ids,
            test_ids=test_ids,
            source=label_file_path,
            destination=output_root / "tabular" / "final",
        )

        _split_deeplake_ds_into_train_and_test(
            train_ids=train_and_valid_ids,
            test_ids=test_ids,
            source=genotype_path,
            destination=output_root / "genotype" / "final",
        )

    def output(self):
        output_root = Path(str(self.data_output_folder))
        outputs = {}

        for split in ("train", "test"):
            cur_target = luigi.LocalTarget(str(output_root / "genotype/final" / split))
            outputs[f"{split}_genotype"] = cur_target

            cur_tabular_path = (
                output_root / "tabular/final" / f"labels_{split}"
            ).with_suffix(".csv")
            cur_tabular_target = luigi.LocalTarget(str(cur_tabular_path))
            outputs[f"{split}_tabular"] = cur_tabular_target

        return outputs


def _id_setup_wrapper(
    output_root: Path,
    common_ids_to_keep: Sequence[str],
    freeze_validation_set: bool,
    pre_split_folder: Optional[str] = None,
    check_valid_and_test_ids: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    valid_path = None
    valid_path_exists = False

    if pre_split_folder is not None:
        pre_split_folder = Path(pre_split_folder)
        logger.info("Using pre-split folder: %s", pre_split_folder)

        train_path = pre_split_folder / "train_ids.txt"
        if not train_path.exists():
            raise FileNotFoundError(f"Could not find train IDs file: {train_path}")

        test_path = pre_split_folder / "test_ids.txt"
        if not test_path.exists():
            raise FileNotFoundError(f"Could not find test IDs file: {test_path}")

        train_ids = (
            pd.read_csv(train_path, header=None).astype(str).squeeze("columns").tolist()
        )
        check_extra_ids(
            ids_to_check=train_ids,
            id_set_name="train",
            common_ids=common_ids_to_keep,
        )

        test_ids = (
            pd.read_csv(test_path, header=None).astype(str).squeeze("columns").tolist()
        )
        if check_valid_and_test_ids:
            check_extra_ids(
                ids_to_check=test_ids,
                id_set_name="test",
                common_ids=common_ids_to_keep,
            )

        valid_path = pre_split_folder / "valid_ids.txt"
        valid_path_exists = valid_path.exists()

    else:
        logger.info("Generating train+valid and test IDs.")

        train_ids, test_ids = _split_ids(all_ids=common_ids_to_keep)

    valid_ids = []
    if freeze_validation_set:
        if not valid_path_exists:
            logger.info("Creating new frozen validation set.")

            batch_size = get_batch_size(samples_per_epoch=len(train_ids))
            valid_size = get_dynamic_valid_size(
                num_samples_per_epoch=len(train_ids),
                minimum=batch_size,
            )
            train_ids, valid_ids = _split_ids(
                all_ids=train_ids, valid_or_test_size=valid_size
            )
        else:
            logger.info("Using frozen validation set from %s.", valid_path)
            valid_ids = (
                pd.read_csv(valid_path, header=None)
                .astype(str)
                .squeeze("columns")
                .tolist()
            )
            if check_valid_and_test_ids:
                check_extra_ids(
                    ids_to_check=valid_ids,
                    id_set_name="valid",
                    common_ids=common_ids_to_keep,
                )

        _save_ids_to_text_file(
            ids=valid_ids, path=output_root / "ids" / "valid_ids.txt"
        )

    _save_ids_to_text_file(ids=train_ids, path=output_root / "ids" / "train_ids.txt")
    _save_ids_to_text_file(ids=test_ids, path=output_root / "ids" / "test_ids.txt")

    logger.info("Train and valid IDs: %d", len(train_ids) + len(valid_ids))
    logger.info("Test IDs: %d", len(test_ids))

    check_missing_ids(
        train_ids=train_ids,
        valid_ids=valid_ids,
        test_ids=test_ids,
        common_ids=common_ids_to_keep,
    )

    return train_ids, valid_ids, test_ids


def check_extra_ids(
    ids_to_check: Sequence[str],
    id_set_name: str,
    common_ids: Sequence[str],
    preview_limit: int = 10,
) -> None:
    common_ids_set = set(common_ids)

    extra_ids = list(set(ids_to_check) - common_ids_set)
    if extra_ids:
        preview = extra_ids[:preview_limit]
        raise ValueError(
            f"{id_set_name} contains {len(extra_ids)} IDs not in common IDs from "
            f"genotype data and label file."
            f"Preview of extra IDs: {preview}."
            f"Please check that the IDs in {id_set_name} "
            f"are available in both genotype data and label file."
        )


def check_missing_ids(
    train_ids: Sequence[str],
    valid_ids: Sequence[str],
    test_ids: Sequence[str],
    common_ids: Sequence[str],
    preview_limit: int = 10,
) -> None:
    common_ids_set = set(common_ids)

    combined_ids = set(train_ids) | set(valid_ids) | set(test_ids)
    missing_ids = list(common_ids_set - combined_ids)
    if missing_ids:
        preview = missing_ids[:preview_limit]
        logger.warning(
            f"Some common IDs are missing in the final sets. "
            f"Preview of missing IDs: {preview}"
        )


def _save_ids_to_text_file(ids: Sequence[str], path: Path) -> None:
    ensure_path_exists(path=path, is_folder=False)
    with open(path, "w") as f:
        for cur_id in ids:
            f.write(f"{cur_id}\n")


def _gather_all_ids(
    genotype_path: Path, label_file: Path, filter_common: bool = True
) -> Sequence[str]:
    genotype_ids = set(
        eir.data_load.label_setup.gather_ids_from_data_source(data_source=genotype_path)
    )
    logger.info(
        "Gathered %d IDs from genotype data: %s", len(genotype_ids), genotype_path
    )

    label_file_ids = set(gather_ids_from_csv_file(file_path=label_file))
    logger.info("Gathered %d IDs from label file: %s", len(label_file_ids), label_file)

    all_ids = set().union(genotype_ids, label_file_ids)
    if filter_common:
        common_ids = label_file_ids.intersection(genotype_ids)
        logger.info(
            "Keeping %d common IDs among %d total (difference: %d).",
            len(common_ids),
            len(all_ids),
            len(all_ids) - len(common_ids),
        )
        return list(common_ids)

    return list(all_ids)


def gather_ids_from_csv_file(file_path: Path, drop_nas: bool = False):
    logger.debug("Gathering IDs from %s.", file_path)
    df = pd.read_csv(filepath_or_buffer=file_path)

    if drop_nas:
        df = df.dropna(how="any", axis=0)

    all_ids = tuple(df["ID"].astype(str))

    return all_ids


def _split_ids(
    all_ids: Sequence[str], valid_or_test_size: float | int = 0.1
) -> Tuple[Sequence[str], Sequence[str]]:
    assert len(all_ids) > 0
    train_ids, test_or_valid_ids = eir.data_load.label_setup.split_ids(
        ids=all_ids, valid_size=valid_or_test_size
    )

    train_ids_set = set(train_ids)
    test_ids_set = set(test_or_valid_ids)
    assert len(train_ids_set.intersection(test_ids_set)) == 0

    return train_ids, test_or_valid_ids


def _split_deeplake_ds_into_train_and_test(
    train_ids: Sequence[str], test_ids: Sequence[str], source: Path, destination: Path
) -> None:
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)
    assert len(train_ids_set.intersection(test_ids_set)) == 0

    train_path = destination / "train"
    test_path = destination / "test"
    if train_path.exists() and test_path.exists():
        return

    ensure_path_exists(path=train_path, is_folder=True)
    ensure_path_exists(path=test_path, is_folder=True)

    full_deeplake_ds = load_deeplake_dataset(data_source=str(source))

    ds_train = deeplake.empty(path=train_path)
    ds_test = deeplake.empty(path=test_path)

    for name, tensor in full_deeplake_ds.tensors.items():
        ds_train.create_tensor(
            name=name,
            htype=tensor.htype,
            dtype=tensor.dtype,
            sample_compression=tensor.meta.sample_compression,
        )
        ds_test.create_tensor(
            name=name,
            htype=tensor.htype,
            dtype=tensor.dtype,
            sample_compression=tensor.meta.sample_compression,
        )

    skipped_samples = 0
    with full_deeplake_ds, ds_train, ds_test:
        for idx, sample in enumerate(full_deeplake_ds):
            cur_id = sample["ID"].numpy().item()
            if cur_id in train_ids_set:
                parsed_sample = _deeplake_sample_to_dict(sample=sample)
                ds_train.append(parsed_sample)
            elif cur_id in test_ids_set:
                parsed_sample = _deeplake_sample_to_dict(sample=sample)
                ds_test.append(parsed_sample)
            else:
                skipped_samples += 1

            if idx % 10000 == 0:
                logger.info(
                    "Iterated over %d samples while splitting deeplake dataset into "
                    "train and test. Skipped %d samples.",
                    idx,
                    skipped_samples,
                )


def _deeplake_sample_to_dict(sample: deeplake.Dataset) -> Dict:
    """
    This is a workaround for the fact that deeplake does creates empty samples for
    *some* dtypes when we try to add them directly when iterating over samples.
    For example, numpy arrays are converted correctly, but strings result in empty
    for some reason (possibly a bug in deeplake).

    Note that this is possibly because deeplake .append is type to
    support Dict[str, Any], so maybe it is not a bug, but it just partially works
    when we feed it a sample.
    """

    parsed = {}

    for name, tensor in sample.tensors.items():
        if tensor.htype == "text":
            parsed[name] = tensor.numpy().item()
        else:
            parsed[name] = tensor.numpy()

    return parsed


def _split_arrays_into_train_and_test(
    train_ids: Sequence[str], test_ids: Sequence[str], source: Path, destination: Path
) -> None:
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)
    assert len(train_ids_set.intersection(test_ids_set)) == 0

    train_path = destination / "train"
    test_path = destination / "test"
    if train_path.exists() and test_path.exists():
        return

    ensure_path_exists(path=train_path, is_folder=True)
    ensure_path_exists(path=test_path, is_folder=True)

    for array_file in source.iterdir():
        if array_file.stem in train_ids_set:
            copyfile(src=array_file, dst=train_path / array_file.name)
        elif array_file.stem in test_ids_set:
            copyfile(src=array_file, dst=test_path / array_file.name)


def _split_csv_into_train_and_test(
    train_ids: Sequence[str], test_ids: Sequence[str], source: Path, destination: Path
) -> None:
    ensure_path_exists(path=destination, is_folder=True)
    df_labels = pd.read_csv(filepath_or_buffer=source)
    df_labels["ID"] = df_labels["ID"].astype(str)

    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    df_train = df_labels[df_labels["ID"].isin(train_ids_set)].copy()
    df_test = df_labels[df_labels["ID"].isin(test_ids_set)].copy()

    df_train.to_csv(destination / "labels_train.csv", index=False)
    df_test.to_csv(destination / "labels_test.csv", index=False)


class ParseDataWrapper(luigi.Task):
    data_config = luigi.DictParameter()

    def requires(self):
        yield CommonSplitIntoTestSet(**self.data_config)

    def output(self):
        return self.input()


def get_dynamic_valid_size(
    num_samples_per_epoch: int,
    minimum: int,
    valid_size_upper_bound: int = 20000,
) -> int:
    valid_size = int(0.1 * num_samples_per_epoch)

    if valid_size < minimum:
        valid_size = minimum

    if valid_size > valid_size_upper_bound:
        valid_size = valid_size_upper_bound

    return valid_size


def get_batch_size(
    samples_per_epoch: int,
    upper_bound: int = 64,
    lower_bound: int = 4,
) -> int:
    batch_size = 2 ** int(math.log2(samples_per_epoch / 40))

    if batch_size > upper_bound:
        return upper_bound
    elif batch_size < lower_bound:
        return lower_bound

    logger.info("Batch size set to: %d", batch_size)

    if batch_size <= 8:
        logger.warning(
            "Computed batch size based on number of training"
            " samples per epoch (%d)"
            " is very small (%d). This may cause issues with training."
            " This is likely due to a small number of samples in the dataset."
            " Consider increasing the number of samples in the dataset if possible.",
            samples_per_epoch,
            batch_size,
        )

    return batch_size
