import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from aislib.misc_utils import get_logger
from eir.setup.input_setup_modules.setup_omics import read_bim
from joblib import Parallel, delayed
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from eir_auto_gp.post_analysis.effect_analysis.genotype_effects import (
    get_snp_allele_maps,
    get_statsmodels_fit_function,
)

logger = get_logger(name=__name__)

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def get_gxe_interaction_effects(
    df_inputs: pd.DataFrame,
    df_target: pd.DataFrame,
    df_genotype_missing: pd.DataFrame,
    bim_file: Path,
    target_type: str,
) -> pd.DataFrame:
    target_name = df_target.columns[0]
    df_combined = pd.concat([df_target, df_inputs], axis=1)

    df_bim = read_bim(bim_file_path=str(bim_file))
    snp_ids = [i for i in df_inputs.columns if not i.startswith("COVAR_")]
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=snp_ids)

    df_results = compute_interactions(
        df=df_combined,
        df_genotype_missing=df_genotype_missing,
        target_name=target_name,
        allele_maps=allele_maps,
        target_type=target_type,
    )

    return df_results


def compute_interactions(
    df: pd.DataFrame,
    df_genotype_missing: pd.DataFrame,
    target_name: str,
    allele_maps: dict[str, dict[str, str]],
    target_type: str,
) -> pd.DataFrame:
    logger.info(
        "Gathering GxE interaction allele effect results for '%s'.", target_name
    )

    snps = [i for i in df.columns if not i.startswith("COVAR_") and i != target_name]
    covar_columns = [i for i in df.columns if i.startswith("COVAR_")]

    parallel_worker = Parallel(n_jobs=-1)
    all_results = parallel_worker(
        delayed(_compute_interaction_effect_wrapper)(
            df=df,
            target_name=target_name,
            df_genotype_missing=df_genotype_missing,
            allele_maps=allele_maps,
            snp=snp,
            covar=covar,
            target_type=target_type,
            covar_columns=covar_columns,
        )
        for snp, covar in product(snps, covar_columns)
    )

    all_results = [i for i in all_results if i is not None]

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results)


def _compute_interaction_effect_wrapper(
    df: pd.DataFrame,
    target_name: str,
    df_genotype_missing: pd.DataFrame,
    snp: str,
    covar: str,
    target_type: str,
    allele_maps: dict[str, dict[str, str]],
    covar_columns: list[str],
) -> pd.DataFrame | None:

    other_covar_columns = [i for i in covar_columns if i != covar]

    formula = build_gxe_interaction_formula(
        target=target_name,
        snp=snp,
        covar=covar,
        other_covar_columns=other_covar_columns,
    )

    df_cur = df[[target_name, snp, *covar_columns]].dropna()
    mask = df_genotype_missing[snp] == 0
    df_cur_no_na = df_cur[mask]

    fit_func = get_statsmodels_fit_function(target_type=target_type)

    try:
        result = fit_func(formula=formula, data=df_cur_no_na).fit(disp=0)
        if hasattr(result, "mle_retvals"):
            if not result.mle_retvals["converged"]:
                raise RuntimeError("Model did not converge.")
        elif not np.all(np.isfinite(result.normalized_cov_params)):
            raise RuntimeError("Model did not converge.")

        result_summary = result.summary2().tables[1]
        if not np.all(np.isfinite(result_summary)):
            raise RuntimeError("Model resulted in non-finite values")

    except Exception as e:
        logger.error(
            f"Failed regression on '{snp}' and '{covar}' due to exception '{e}'. "
            f"Skipping."
        )
        return None

    df_result = build_df_from_interaction_results(
        results=result,
        allele_maps=allele_maps,
        snp=snp,
        covar=covar,
    )

    return df_result


def build_gxe_interaction_formula(
    target: str,
    snp: str,
    covar: str,
    other_covar_columns: list[str],
) -> str:
    formula = f"Q('{target}') ~ C(Q('{snp}')) + Q('{covar}') + Q('{snp}'):Q('{covar}')"

    for other_covar in other_covar_columns:
        formula += f" + Q('{other_covar}')"

    return formula


def build_df_from_interaction_results(
    results: LikelihoodModelResults,
    allele_maps: dict[str, dict[str, str]],
    snp: str,
    covar: str,
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

    df.index.name = "Interaction"
    df_renamed = _rename_interaction_regression_index(
        df_results=df,
        allele_maps=allele_maps,
        snp=snp,
        covar=covar,
    )

    df_renamed = df_renamed.rename(index={f"{snp}:{covar}": f"{snp}--:--{covar}"})
    df_renamed["KEY"] = f"{snp}--:--{covar}"

    return df_renamed


def _rename_interaction_regression_index(
    df_results: pd.DataFrame,
    allele_maps: dict[str, dict[str, str]],
    snp: str,
    covar: str,
) -> pd.DataFrame:
    snp_allele_map = allele_maps[snp]
    cur_mapping = {}

    for index in df_results.index:
        if snp in index:
            if "C(Q(" in index:
                if "[T.1]" in index:
                    cur_allele = snp_allele_map["HET"]
                    new_index = f"{snp} {cur_allele}"
                elif "[T.2]" in index:
                    cur_allele = snp_allele_map["ALT"]
                    new_index = f"{snp} {cur_allele}"
                else:
                    cur_allele = snp_allele_map["REF"]
                    new_index = f"{snp} {cur_allele}"
                cur_mapping[index] = new_index

            elif ":Q('" in index:
                new_index = f"{snp}:{covar.strip()}"
                cur_mapping[index] = new_index

        elif "Q(" in index:
            start_pos = index.find("'") + 1
            end_pos = index.find("'", start_pos)
            covar_name = index[start_pos:end_pos]
            group_suffix = ""
            if "[T." in index and "]" in index:
                start = index.find("[T.") + 3
                end = index.find("]")
                group = index[start:end]

                group_suffix = f" (Group {group})"
            new_index = f"{covar_name}{group_suffix}"
            cur_mapping[index] = new_index

    df_results = df_results.rename(index=cur_mapping)
    return df_results
