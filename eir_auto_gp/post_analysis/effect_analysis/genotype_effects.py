import warnings
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from joblib import Parallel, delayed
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.formula import api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logger = get_logger(name=__name__)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def get_allele_effects(
    df_genotype: pd.DataFrame,
    df_target: pd.DataFrame,
    bim_file: Path,
    target_type: str,
) -> pd.DataFrame:
    target_name = df_target.columns[0]
    df_combined = pd.concat(objs=[df_target, df_genotype], axis=1)

    df_bim = read_bim(bim_file_path=str(bim_file))
    rs_ids = df_genotype.columns
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=rs_ids)

    df_results = get_all_linear_results(
        df=df_combined,
        target_name=target_name,
        allele_maps=allele_maps,
        target_type=target_type,
    )

    if "P>|t|" in df_results.columns:
        df_results = df_results.rename(columns={"P>|t|": "p_value"})
    elif "P>|z|" in df_results.columns:
        df_results = df_results.rename(columns={"P>|z|": "p_value"})
    else:
        raise ValueError("Could not find p value column.")

    return df_results


def get_all_linear_results(
    df: pd.DataFrame,
    target_name: str,
    allele_maps: dict[str, dict[str, str]],
    target_type: str,
) -> pd.DataFrame:
    logger.info("Gathering all linear allele effect results for '%s'.", target_name)

    parallel_worker = Parallel(n_jobs=-1)
    all_results = parallel_worker(
        delayed(_compute_single_snp_effect_wrapper)(
            df=df,
            target_name=target_name,
            allele_maps=allele_maps,
            snp=snp,
            model=target_type,
        )
        for snp in df.columns
        if snp != target_name
    )
    all_results = [i for i in all_results if i is not None]

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results)


def _compute_single_snp_effect_wrapper(
    df: pd.DataFrame,
    target_name: str,
    snp: str,
    allele_maps: dict[str, dict[str, str]],
    model: str,
) -> Optional[pd.DataFrame]:
    formula = build_basic_fit_formula(target=target_name, snp=snp)

    df_cur = df[[target_name, snp]]
    df_cur_no_na = df_cur[df_cur[snp] != -1]

    fit_func = get_statsmodels_fit_function(target_type=model)

    try:
        result = fit_func(formula=formula, data=df_cur_no_na).fit(disp=0)
        if hasattr(result, "mle_retvals"):
            if not result.mle_retvals["converged"]:
                raise RuntimeError(f"Model did not converge on '{snp}'.")
        elif not np.all(np.isfinite(result.normalized_cov_params)):
            raise RuntimeError(f"Model did not converge on '{snp}'.")

        result_summary = result.summary2().tables[1]
        if not np.all(np.isfinite(result_summary)):
            raise RuntimeError(f"Model resulted in non-finite values on '{snp}'.")

    except Exception as exception:
        logger.info(
            "Failed OLS on '%s' due to exception '%s'. Skipping.",
            snp,
            exception,
        )
        return

    df_result = build_df_from_basic_results(
        results=result,
        allele_maps=allele_maps,
        snp=snp,
    )

    return df_result


def get_statsmodels_fit_function(target_type: str) -> Callable:
    if target_type == "regression":
        func = smf.ols
    elif target_type == "classification":
        func = smf.logit
    else:
        raise ValueError()

    return func


def build_basic_fit_formula(target: str, snp: str) -> str:
    return f"Q('{target}') ~ C(Q('{snp}'))"


def get_snp_allele_maps(
    df_bim: pd.DataFrame, snp_ids: Iterable[str]
) -> dict[str, dict[str, str]]:
    snp_allele_maps = {}
    for rs_id in snp_ids:
        snp_allele_maps[rs_id] = get_snp_allele_nucleotide_map(
            df_bim=df_bim, rs_id=rs_id
        )

    return snp_allele_maps


def get_snp_allele_nucleotide_map(df_bim: pd.DataFrame, rs_id: str) -> dict[str, str]:
    nucleotide_map = {}

    ref = df_bim[df_bim["VAR_ID"] == rs_id]["REF"].item()
    alt = df_bim[df_bim["VAR_ID"] == rs_id]["ALT"].item()

    nucleotide_map["REF"] = ref * 2
    nucleotide_map["HET"] = f"{ref}{alt}"
    nucleotide_map["ALT"] = alt * 2

    return nucleotide_map


def build_df_from_basic_results(
    results: LikelihoodModelResults,
    allele_maps: dict[str, dict[str, str]],
    snp: str,
) -> pd.DataFrame:
    results_as_html = results.summary().tables[1].as_html()
    html_buffer = StringIO(results_as_html)
    df_linear = pd.read_html(io=html_buffer, header=0, index_col=0)[0]
    df_linear.index.name = "allele"

    df_linear_renamed = _rename_linear_regression_index(
        df_results=df_linear,
        allele_maps=allele_maps,
        snp=snp,
    )

    df_linear_column_renamed = df_linear_renamed.rename(
        columns={
            "coef": "Coefficient",
            "std err": "STD ERR",
            "[0.025": "0.025 CI",
            "0.975]": "0.975 CI",
        }
    )

    return df_linear_column_renamed


def _rename_linear_regression_index(
    df_results: pd.DataFrame, allele_maps: dict[str, dict[str, str]], snp: str
) -> pd.DataFrame:
    snp_allele_map = allele_maps[snp]
    cur_mapping = {}
    for index in df_results.index:
        if index == "Intercept":
            key = "REF"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele} (Intercept)"
        elif "[T.1]" in index:
            key = "HET"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele}"
        elif "[T.2]" in index:
            key = "ALT"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele}"
        else:
            raise ValueError(f"Unexpected index format: {index}")

    df_results = df_results.rename(index=cur_mapping)

    return df_results
