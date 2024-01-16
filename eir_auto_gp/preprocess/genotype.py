from pathlib import Path
from shutil import copyfile
from typing import List

import luigi
from aislib.misc_utils import ensure_path_exists
from luigi.util import requires
from plink_pipelines.make_dataset import (
    RenameOnFailureMixin,
    _get_one_hot_encoded_generator,
    get_sample_generator_from_bed,
    write_one_hot_outputs,
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


@requires(ExternalRawData)
class OneHotAutoSNPs(Config):
    """
    Generates one hot encodings from a individuals x SNPs file.
    """

    include_text = luigi.BoolParameter()
    output_folder = luigi.Parameter()
    output_format = luigi.Parameter()
    output_name = luigi.Parameter()
    file_name = str(output_name)
    genotype_processing_chunk_size = luigi.IntParameter()

    def run(self):
        input_path = Path(self.input().path)
        assert input_path.suffix == ".bed"

        output_path = Path(self.output().path)
        ensure_path_exists(output_path, is_folder=True)

        chunk_generator = get_sample_generator_from_bed(
            bed_path=input_path, chunk_size=int(self.genotype_processing_chunk_size)
        )
        sample_id_one_hot_array_generator = _get_one_hot_encoded_generator(
            chunked_sample_generator=chunk_generator
        )
        write_one_hot_outputs(
            id_array_generator=sample_id_one_hot_array_generator,
            output_folder=output_path,
            output_format=str(self.output_format),
            output_name=str(self.output_name),
        )

    def output_target(self, file_name: str):
        return f"{self.output_folder}/processed/encoded_outputs/{self.output_name}"

    def output(self):
        target = self.output_target(self.file_name)
        return luigi.LocalTarget(path=target)


@requires(OneHotAutoSNPs)
class FinalizeGenotypeParsing(luigi.Task):
    raw_data_path = luigi.Parameter()
    output_folder = luigi.Parameter()
    output_format = luigi.Parameter()
    output_name = luigi.Parameter()
    genotype_processing_chunk_size = luigi.IntParameter()

    def run(self):
        raw_path = Path(str(self.raw_data_path))
        bim_files = [i for i in raw_path.iterdir() if i.suffix == ".bim"]
        assert len(bim_files) == 1
        bim_path = bim_files[0]

        output_path = Path(self.output()[0].path)
        ensure_path_exists(output_path, is_folder=False)

        copyfile(bim_path, output_path)

    def output(self):
        output_path = Path(
            str(self.output_folder), "processed/parsed_files/data_final.bim"
        )

        one_hot_outputs = self.input()
        return [luigi.LocalTarget(str(output_path)), one_hot_outputs]
