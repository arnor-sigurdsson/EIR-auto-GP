from pathlib import Path
from typing import Callable, Dict, List

import bed_reader
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simulate_genetic_data_to_bed(tmp_path: Path) -> Callable[[int, int, str], Path]:
    def _wrapper(n_individuals: int, n_snps: int, phenotype: str) -> Path:
        df_snp, df_pheno = simulate_genetic_data(
            n_individuals=n_individuals,
            n_snps=n_snps,
            phenotype=phenotype,
        )

        ind_ids = [f"ind{i+1}" for i in range(n_individuals)]
        properties: Dict[str, List[str]] = {
            "sid": df_snp.columns.tolist(),
            "iid": ind_ids,
            "allele_1": ["A"] * n_snps,  # minor, -> alt -> 2
            "allele_2": ["T"] * n_snps,  # major, -> ref -> 0
            "chromosome": ["1"] * n_snps,
            "bp_position": [i for i in range(n_snps)],
        }

        file_path = tmp_path / "genetic_data.bed"

        bed_reader.to_bed(filepath=file_path, val=df_snp.values, properties=properties)
        df_pheno.index = ind_ids
        df_pheno.index.name = "ID"
        df_pheno.to_csv(tmp_path / "phenotype.csv", index=True)

        return file_path.parent

    return _wrapper


def simulate_genetic_data(
    n_individuals: int, n_snps: int, phenotype: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snp_data = np.random.randint(0, 3, size=(n_individuals, n_snps))

    snp_list = _get_snp_list(n_snps=n_snps)

    df_snp = pd.DataFrame(snp_data, columns=snp_list)

    causal_snps = ["snp1", "snp2"]

    dominant_snp = "snp3"
    recessive_snp = "snp4"

    interaction_pair = ("snp5", "snp6")

    additive_factor = 10
    interaction_factor = 10
    dominant_factor = 20
    recessive_factor = 20

    latent_phenotype = np.zeros(n_individuals)

    latent_phenotype += additive_factor * df_snp[causal_snps[0]]
    latent_phenotype += additive_factor * df_snp[causal_snps[1]]

    latent_phenotype += (
        df_snp[list(interaction_pair)].product(axis=1) * interaction_factor
    )
    latent_phenotype += dominant_factor * (df_snp[dominant_snp] > 0).astype(int)
    latent_phenotype += recessive_factor * (df_snp[recessive_snp] == 2).astype(int)

    if phenotype == "continuous":
        df_snp["phenotype"] = latent_phenotype

    elif phenotype == "binary":
        threshold = np.median(latent_phenotype)
        df_snp["phenotype"] = (latent_phenotype >= threshold).astype(int)

    else:
        raise ValueError("Phenotype must be 'continuous' or 'binary'")

    df_snp, df_pheno = df_snp.drop(columns=["phenotype"]), df_snp["phenotype"]

    df_pheno_with_covars = add_covars_to_phenotype_df(df_pheno=df_pheno)

    return df_snp, df_pheno_with_covars


def add_covars_to_phenotype_df(df_pheno: pd.Series) -> pd.DataFrame:
    n = len(df_pheno)

    categories = ["GroupA", "GroupB", "GroupC"]
    cov_cat_random = np.random.choice(categories, size=n)
    cov_cat_random[np.random.rand(n) < 0.1] = np.nan

    cov_con_random = np.random.normal(loc=0, scale=1, size=n)
    cov_con_random[np.random.rand(n) < 0.1] = np.nan

    quantiles = pd.cut(df_pheno, 3, labels=["Low", "Medium", "High"])
    cov_cat_computed = quantiles.astype(str)
    cov_cat_computed[np.random.rand(n) < 0.1] = np.nan

    cov_con_computed = df_pheno * 0.1 + np.random.normal(loc=0, scale=0.5, size=n)
    cov_con_computed[np.random.rand(n) < 0.1] = np.nan

    df_covars = pd.DataFrame(
        {
            "CAT_RANDOM": cov_cat_random,
            "CON_RANDOM": cov_con_random,
            "CAT_COMPUTED": cov_cat_computed,
            "CON_COMPUTED": cov_con_computed,
        },
        index=df_pheno.index,
    )

    df_pheno_with_covars = pd.concat([df_pheno, df_covars], axis=1)

    return df_pheno_with_covars


def _get_snp_list(n_snps: int) -> list[str]:
    snp_edge_cases = [
        "1:rs7551801",
        "rs7551801_A_T",
        "1:rs7551801_A_T",
        "rs7551801_A_TTT",
        "123456789",
        "rs75-51#801",
    ]

    snp_base_cases = [f"snp{i+1}" for i in range(n_snps - len(snp_edge_cases))]

    snp_list = snp_base_cases + snp_edge_cases
    assert len(snp_list) == n_snps

    return snp_list
