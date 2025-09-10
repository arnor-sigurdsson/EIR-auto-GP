from pathlib import Path
from shutil import copyfile

import luigi
import pandas as pd
from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


class ParseLabelFile(luigi.Task):
    label_file_path = luigi.Parameter()
    output_folder = luigi.Parameter()
    only_data = luigi.BoolParameter(default=True)
    input_cat_columns = luigi.ListParameter(default=[])
    input_con_columns = luigi.ListParameter(default=[])
    output_cat_columns = luigi.ListParameter(default=[])
    output_con_columns = luigi.ListParameter(default=[])

    def run(self) -> None:
        ensure_path_exists(path=self.output_path())

        if self.only_data or not any(
            [
                self.input_cat_columns,
                self.input_con_columns,
                self.output_cat_columns,
                self.output_con_columns,
            ]
        ):
            copyfile(src=str(self.label_file_path), dst=self.output_path())
        else:
            required_columns = {"ID"}
            if self.input_cat_columns:
                required_columns.update(self.input_cat_columns)
            if self.input_con_columns:
                required_columns.update(self.input_con_columns)
            if self.output_cat_columns:
                required_columns.update(self.output_cat_columns)
            if self.output_con_columns:
                required_columns.update(self.output_con_columns)

            logger.info(
                "Direct modelling mode: copying only required columns: %s",
                sorted(required_columns),
            )

            df = pd.read_csv(self.label_file_path, usecols=list(required_columns))
            df.to_csv(self.output_path(), index=False)

    def output_path(self) -> Path:
        return Path(str(self.output_folder), "tabular/labels.csv")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path())
