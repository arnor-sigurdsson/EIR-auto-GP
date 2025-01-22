from pathlib import Path
from shutil import copyfile
from typing import Generator, List, Literal, Optional

import luigi
import numpy as np
from plink_pipelines.make_dataset import (
    RenameOnFailureMixin,
    _get_one_hot_encoded_generator,
    get_sample_generator_from_bed,
)

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


class Config(luigi.Task, RenameOnFailureMixin):
    output_folder = luigi.Parameter()

    @property
    def input_name(self):
        bed_files = [
            i for i in Path(str(self.raw_data_path)).iterdir() if i.suffix == ".bed"
        ]
        if len(bed_files) != 1:
            raise ValueError(
                f"Expected one .bed file in {self.raw_data_path}, but"
                f"found {bed_files}."
            )
        return str(bed_files[0])

    @property
    def file_name(self):
        raise NotImplementedError

    def output_target(self, file_name: str):
        output_path = Path(str(self.output_folder), file_name)

        return luigi.LocalTarget(str(output_path))

    def output(self):
        return self.output_target(self.file_name)


def _get_plink_inputs_from_folder(folder_path: Path) -> List[Path]:
    files = [i.with_suffix("") for i in folder_path.iterdir() if i.suffix == ".bed"]

    return files


class ExternalRawData(luigi.ExternalTask):
    raw_data_path = luigi.Parameter()

    @property
    def input_name(self):
        bed_files = [
            i for i in Path(str(self.raw_data_path)).iterdir() if i.suffix == ".bed"
        ]
        if len(bed_files) != 1:
            raise ValueError(
                f"Expected one .bed file in {self.raw_data_path}, but"
                f"found {bed_files}."
            )
        return str(bed_files[0])

    def output(self):
        return luigi.LocalTarget(str(self.input_name))


def get_encoded_snp_stream(
    bed_path: Path,
    chunk_size: int,
    output_format: Literal["disk", "deeplake"],
) -> Generator[tuple[str, np.ndarray], None, None]:
    chunk_generator = get_sample_generator_from_bed(
        bed_path=bed_path,
        chunk_size=chunk_size,
    )

    yield from _get_one_hot_encoded_generator(
        chunked_sample_generator=chunk_generator,
        output_format=output_format,
    )


def copy_bim_file(
    source_folder: Path,
    output_folder: Path,
    ensure_folder_exists: bool = True,
) -> Optional[Path]:
    bim_files = list(source_folder.glob("*.bim"))

    assert_msg = f"Expected one .bim file in {source_folder}, found {len(bim_files)}"
    assert len(bim_files) == 1, assert_msg

    bim_path = bim_files[0]
    output_path = output_folder / "data_final.bim"

    if ensure_folder_exists:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    copyfile(src=bim_path, dst=output_path)
    return output_path
