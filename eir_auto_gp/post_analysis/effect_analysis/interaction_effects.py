import warnings
from io import StringIO
from itertools import combinations
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


def get_interaction_effects(
    df_genotype: pd.DataFrame,
    df_target: pd.DataFrame,
    bim_file: Path,
    target_type: str,
    allow_within_chr_interaction: bool,
    min_interaction_pair_distance: int,
) -> pd.DataFrame:
    target_name = df_target.columns[0]
    df_combined = pd.concat(objs=[df_target, df_genotype], axis=1)

    df_bim = read_bim(bim_file_path=str(bim_file))
    snp_ids = df_genotype.columns
    allele_maps = get_snp_allele_maps(df_bim=df_bim, snp_ids=snp_ids)

    df_results = compute_interactions(
        df=df_combined,
        target_name=target_name,
        allele_maps=allele_maps,
        target_type=target_type,
        p_threshold="auto",
        df_bim=df_bim,
        allow_within_chr_interaction=allow_within_chr_interaction,
        min_interaction_pair_distance=min_interaction_pair_distance,
    )

    return df_results


def compute_interactions(
    df: pd.DataFrame,
    target_name: str,
    allele_maps: dict[str, dict[str, str]],
    target_type: str,
    df_bim: pd.DataFrame,
    allow_within_chr_interaction: bool,
    min_interaction_pair_distance: int,
    p_threshold: str | float = "auto",
) -> pd.DataFrame:
    logger.info(
        "Gathering all interaction allele effect results for '%s'.",
        target_name,
    )

    snps = [i for i in df.columns if i != target_name]

    if p_threshold == "auto":
        p_threshold = 0.05 / (len(snps) ** 2)

        logger.info(
            "p value threshold set to %f due to 'auto' option and %d SNPs.",
            p_threshold,
            len(snps),
        )

    snp_iter = filter_snp_pairs(
        df_bim=df_bim,
        snps=snps,
        allow_within_chr_interaction=allow_within_chr_interaction,
        min_interaction_pair_distance=min_interaction_pair_distance,
    )

    parallel_worker = Parallel(n_jobs=-1)
    all_results = parallel_worker(
        delayed(_compute_interaction_snp_effect_wrapper)(
            df=df,
            target_name=target_name,
            allele_maps=allele_maps,
            snp_1=snp_1,
            snp_2=snp_2,
            p_threshold=p_threshold,
            target_type=target_type,
        )
        for snp_1, snp_2 in snp_iter
    )
    all_results = [i for i in all_results if i is not None]

    if len(all_results) == 0:
        return pd.DataFrame()

    return pd.concat(all_results)


def filter_snp_pairs(
    df_bim: pd.DataFrame,
    snps: list[str],
    allow_within_chr_interaction: bool,
    min_interaction_pair_distance: int,
) -> list[tuple[str, str]]:
    snp_to_chr = df_bim.set_index("VAR_ID")["CHR_CODE"].to_dict()
    snp_to_pos = df_bim.set_index("VAR_ID")["BP_COORD"].to_dict()

    total_pairs = 0
    filtered_within_chr = 0
    accepted_pairs = 0

    filtered_pairs = []
    for snp_1, snp_2 in combinations(iterable=snps, r=2):
        total_pairs += 1
        chr_1, chr_2 = snp_to_chr.get(snp_1), snp_to_chr.get(snp_2)
        pos_1, pos_2 = snp_to_pos.get(snp_1), snp_to_pos.get(snp_2)

        if chr_1 == chr_2:
            over_min_distance = abs(pos_1 - pos_2) >= min_interaction_pair_distance
            if allow_within_chr_interaction and over_min_distance:
                filtered_pairs.append((snp_1, snp_2))
                accepted_pairs += 1
            else:
                filtered_within_chr += 1
        else:
            filtered_pairs.append((snp_1, snp_2))
            accepted_pairs += 1

    logger.info("Filtering SNP pairs for interaction effects.")
    logger.info(f"Total SNP pairs considered: {total_pairs}")
    logger.info(
        f"SNP pairs filtered out due to being on the same chromosome "
        f"but not meeting the distance criteria: {filtered_within_chr}"
    )
    logger.info(
        f"SNP pairs accepted (both within and between chromosomes): {accepted_pairs}"
    )

    return filtered_pairs


def _compute_interaction_snp_effect_wrapper(
    df: pd.DataFrame,
    target_name: str,
    snp_1: str,
    snp_2: str,
    p_threshold: float,
    allele_maps: dict[str, dict[str, str]],
    target_type: str,
) -> pd.DataFrame | None:
    formula = build_multiplicative_interaction_snp_formula(
        target=target_name,
        snp_1=snp_1,
        snp_2=snp_2,
    )

    df_cur = df[[target_name, snp_1, snp_2]]
    df_cur_no_na = df_cur[df_cur[snp_1] != -1]
    df_cur_no_na = df_cur_no_na[df_cur_no_na[snp_2] != -1]

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

    except Exception as exception:
        logger.info(
            "Failed OLS on '%s' and '%s' due to exception '%s'. Skipping.",
            snp_1,
            snp_2,
            exception,
        )
        return None

    p_value = _extract_p_value_from_interaction_results(results=result)
    if p_value > p_threshold:
        return None

    try:
        df_result = build_df_from_interaction_results(
            results=result, allele_maps=allele_maps, snp_1=snp_1, snp_2=snp_2
        )
        return df_result
    except Exception as e:
        logger.info(
            "Failed parsing interaction results on '%s' and '%s' due to exception "
            "'%s'. Skipping.",
            snp_1,
            snp_2,
            e,
        )
        return None


def _extract_p_value_from_interaction_results(
    results: LikelihoodModelResults,
) -> float:
    p_values = results.pvalues
    last_row_name = p_values.index[-1]
    assert ":" in last_row_name
    p_value = p_values.iloc[-1]
    return p_value


def build_multiplicative_interaction_snp_formula(
    target: str,
    snp_1: str,
    snp_2: str,
) -> str:
    return (
        f"Q('{target}') ~ C(Q('{snp_1}')) + C(Q('{snp_2}')) + Q('{snp_1}'):Q('{snp_2}')"
    )


def build_df_from_interaction_results(
    results: LikelihoodModelResults,
    allele_maps: dict[str, dict[str, str]],
    snp_1: str,
    snp_2: str,
) -> pd.DataFrame:
    results_as_html = results.summary().tables[1].as_html()
    html_buffer = StringIO(results_as_html)
    df_linear = pd.read_html(io=html_buffer, header=0, index_col=0)[0]
    df_linear.index.name = "allele"
    df_linear.index = df_linear.index.str.replace(r"C\(Q\(", "", regex=True)
    df_linear.index = df_linear.index.str.replace(r"Q\(", "", regex=True)
    df_linear.index = df_linear.index.str.replace(r"\)", "", regex=True)
    df_linear.index = df_linear.index.str.replace(r"'", "", regex=True)

    df_linear_renamed = _rename_interaction_regression_index(
        df_results=df_linear, allele_maps=allele_maps, snp=snp_1
    )

    df_linear_renamed = _rename_interaction_regression_index(
        df_results=df_linear_renamed, allele_maps=allele_maps, snp=snp_2
    )

    df_linear_renamed["KEY"] = f"{snp_1}--:--{snp_2}"

    df_linear_column_renamed = df_linear_renamed.rename(
        columns={
            "coef": "Coefficient",
            "std err": "STD ERR",
            "[0.025": "0.025 CI",
            "0.975]": "0.975 CI",
        }
    )

    return df_linear_column_renamed


def _rename_interaction_regression_index(
    df_results: pd.DataFrame, allele_maps: dict[str, dict[str, str]], snp: str
):
    snp_allele_map = allele_maps[snp]
    cur_mapping = {}
    for index in df_results.index:
        if "[T.1]" in index and snp in index:
            key = "HET"
        elif "[T.2]" in index and snp in index:
            key = "ALT"
        else:
            continue

        cur_allele = snp_allele_map[key]
        cur_mapping[index] = f"{snp} {cur_allele}"

    df_results = df_results.rename(index=cur_mapping)

    return df_results
