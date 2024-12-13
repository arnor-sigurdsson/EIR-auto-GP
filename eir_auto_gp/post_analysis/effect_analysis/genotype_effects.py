import warnings
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
    df_inputs: pd.DataFrame,
    df_target: pd.DataFrame,
    df_genotype_missing: pd.DataFrame,
    bim_file: Path,
    target_type: str,
) -> pd.DataFrame:
    target_name = df_target.columns[0]
    df_combined = pd.concat(objs=[df_target, df_inputs], axis=1)

    df_bim = read_bim(bim_file_path=str(bim_file))
    rs_ids = [i for i in df_inputs.columns if not i.startswith("COVAR_")]
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=rs_ids)

    df_results = get_all_linear_results(
        df=df_combined,
        df_genotype_missing=df_genotype_missing,
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
    df_genotype_missing: pd.DataFrame,
    target_name: str,
    allele_maps: dict[str, dict[str, str]],
    target_type: str,
) -> pd.DataFrame:

    covar_columns = [i for i in df.columns if i.startswith("COVAR_")]
    to_check_columns = [
        i for i in df.columns if i != target_name and not i.startswith("COVAR_")
    ]

    logger.info(
        "Gathering all linear allele effect results for '%s'. Checking %d SNPs.",
        target_name,
        len(to_check_columns),
    )

    if len(to_check_columns) == 0:
        return pd.DataFrame()

    parallel_worker = Parallel(n_jobs=-1)
    all_results = parallel_worker(
        delayed(_compute_single_snp_effect_wrapper)(
            df=df,
            df_genotype_missing=df_genotype_missing,
            target_name=target_name,
            allele_maps=allele_maps,
            snp=col,
            model=target_type,
            covar_columns=covar_columns,
        )
        for col in to_check_columns
    )

    all_results = [i for i in all_results if i is not None]

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results)


def _compute_single_snp_effect_wrapper(
    df: pd.DataFrame,
    df_genotype_missing: pd.DataFrame,
    target_name: str,
    snp: str,
    allele_maps: dict[str, dict[str, str]],
    model: str,
    covar_columns: list[str],
) -> Optional[pd.DataFrame]:
    formula = build_basic_fit_formula(
        target=target_name,
        snp=snp,
        covar_columns=covar_columns,
    )

    df_cur = df[[target_name, snp, *covar_columns]]
    df_cur_no_na = df_cur[df_genotype_missing[snp] == 0]

    n_per_group_dict = df_cur_no_na[snp].value_counts().to_dict()
    n_per_group_dict = {
        key: n_per_group_dict.get(idx, 0)
        for idx, key in enumerate(["REF", "HET", "ALT"])
    }

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
        n_per_group_dict=n_per_group_dict,
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


def build_basic_fit_formula(target: str, snp: str, covar_columns: list[str]) -> str:
    formula = f"Q('{target}') ~ C(Q('{snp}'))"

    for covar in covar_columns:
        formula += f" + Q('{covar}')"

    return formula


def get_snp_allele_maps(
    df_bim: pd.DataFrame,
    snp_ids: Iterable[str],
) -> dict[str, dict[str, str]]:
    snp_allele_maps = {}
    for rs_id in snp_ids:
        snp_allele_maps[rs_id] = get_snp_allele_nucleotide_map(
            df_bim=df_bim,
            rs_id=rs_id,
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
    n_per_group_dict: dict[str, int],
) -> pd.DataFrame:
    coef = results.params
    std_err = results.bse
    p_values = results.pvalues
    conf_int = results.conf_int()

    df = pd.DataFrame(
        {
            "Coefficient": coef,
            "STD ERR": std_err,
            "P>|t|": p_values,
            "0.025 CI": conf_int[0],
            "0.975 CI": conf_int[1],
        }
    )

    df.index.name = "allele"

    df_renamed = _rename_linear_regression_index(
        df_results=df,
        allele_maps=allele_maps,
        snp=snp,
    )

    df_renamed["n"] = [n_per_group_dict.get(i, np.nan) for i in df_renamed["Label"]]

    df_renamed = df_renamed.rename(
        columns={
            "coef": "Coefficient",
            "std err": "STD ERR",
            "[0.025": "0.025 CI",
            "0.975]": "0.975 CI",
        }
    )

    df_renamed["KEY"] = snp

    return df_renamed


def _rename_linear_regression_index(
    df_results: pd.DataFrame,
    allele_maps: dict[str, dict[str, str]],
    snp: str,
) -> pd.DataFrame:
    df_results_copy = df_results.copy()

    snp_allele_map = allele_maps[snp]
    cur_mapping = {}
    labels = []
    for index in df_results.index:
        if index == "Intercept":
            key = "REF"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele} (Intercept)"
            labels.append(key)
        elif "[T.1]" in index:
            key = "HET"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele}"
            labels.append(key)
        elif "[T.2]" in index:
            key = "ALT"
            cur_allele = snp_allele_map[key]
            cur_mapping[index] = f"{snp} {cur_allele}"
            labels.append(key)
        elif "Q(" in index:
            start_pos = index.find("'") + 1
            end_pos = index.find("'", start_pos)
            covar_name = index[start_pos:end_pos]

            if "[T." in index and "]" in index:
                start = index.find("[T.") + 3
                end = index.find("]")
                group = index[start:end]
                group_name = f"{covar_name} (Group {group})"
            else:
                group_name = covar_name

            cur_mapping[index] = group_name
            labels.append(group_name)
        else:
            raise ValueError(f"Unexpected index format: {index}")

    df_results_copy = df_results_copy.rename(index=cur_mapping)
    df_results_copy["Label"] = labels

    return df_results_copy
