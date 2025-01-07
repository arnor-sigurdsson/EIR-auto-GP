from pathlib import Path

import luigi
import pandas as pd
from aislib.misc_utils import ensure_path_exists

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


class ParseLabelFile(luigi.Task):
    label_file_path = luigi.Parameter()
    output_folder = luigi.Parameter()

    def run(self) -> None:
        ensure_path_exists(path=self.output_path())

        df = pd.read_csv(filepath_or_buffer=str(self.label_file_path))
        df_parsed = _parse_label_df(df=df)

        df_parsed.to_csv(
            path_or_buf=self.output_path().with_suffix(".csv"),
            index=False,
        )

    def output_path(self) -> Path:
        return Path(str(self.output_folder), "tabular/labels.csv")

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(self.output_path())


def _parse_label_df(
    df: pd.DataFrame,
    remove_na: bool = False,
) -> pd.DataFrame:
    if remove_na:
        df = _remove_any_na_from_label_df(df=df)

    return df


def _remove_any_na_from_label_df(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    number_before = len(df_copy)
    df_no_na = df_copy.dropna(how="any")
    number_after = len(df_no_na)

    difference = number_before - number_after
    logger.info("Dropped %d rows that had NA in any column.", difference)

    return df_no_na
