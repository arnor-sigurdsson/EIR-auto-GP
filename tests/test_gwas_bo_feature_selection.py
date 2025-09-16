from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from eir_auto_gp.single_task.modelling.gwas_bo_feature_selection import (
    calculate_dynamic_bounds,
    fraction_to_log_p_value,
    get_gwas_bo_auto_top_n,
    get_gwas_top_n_snp_list_df,
    run_gwas_bo_feature_selection,
)


@pytest.fixture
def sample_gwas_df():
    np.random.seed(42)
    n_snps = 1000

    p_values = []
    p_values.extend(np.random.uniform(1e-8, 1e-6, 20))
    p_values.extend(np.random.uniform(1e-6, 1e-4, 80))
    p_values.extend(np.random.uniform(1e-4, 0.05, 200))
    p_values.extend(np.random.uniform(0.05, 1.0, n_snps - 300))

    np.random.shuffle(p_values)

    df = pd.DataFrame({"GWAS P-VALUE": p_values})
    df.index.name = "SNP"
    df.index = [f"rs{i}" for i in range(len(df))]

    return df


@pytest.fixture
def small_gwas_df():
    df = pd.DataFrame({"GWAS P-VALUE": [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5]})
    df.index.name = "SNP"
    df.index = [f"rs{i}" for i in range(len(df))]
    return df


@pytest.fixture
def mock_history_df():
    return pd.DataFrame(
        {
            "fraction": [0.01, 0.05, 0.1, 0.2, 0.3],
            "best_val_performance": [0.75, 0.82, 0.85, 0.83, 0.80],
        }
    )


class TestCalculateDynamicBounds:
    def test_calculate_dynamic_bounds_with_threshold(self, sample_gwas_df):
        threshold = 1e-4
        min_n_snps = 16

        min_log_p, max_log_p = calculate_dynamic_bounds(
            df_gwas=sample_gwas_df,
            gwas_p_value_threshold=threshold,
            min_n_snps=min_n_snps,
        )

        assert max_log_p == np.log10(threshold)

        assert min_log_p < max_log_p
        assert isinstance(min_log_p, float)
        assert isinstance(max_log_p, float)

    def test_calculate_dynamic_bounds_without_threshold(self, sample_gwas_df):
        min_n_snps = 16

        min_log_p, max_log_p = calculate_dynamic_bounds(
            df_gwas=sample_gwas_df, gwas_p_value_threshold=None, min_n_snps=min_n_snps
        )

        median_p = sample_gwas_df["GWAS P-VALUE"].quantile(0.5)
        assert max_log_p == np.log10(median_p)

        assert min_log_p < max_log_p

    def test_calculate_dynamic_bounds_small_dataset(self, small_gwas_df):
        min_n_snps = 16
        threshold = 0.01

        min_log_p, max_log_p = calculate_dynamic_bounds(
            df_gwas=small_gwas_df,
            gwas_p_value_threshold=threshold,
            min_n_snps=min_n_snps,
        )

        min_p = small_gwas_df["GWAS P-VALUE"].min()
        assert min_log_p == np.log10(min_p)
        assert max_log_p == np.log10(threshold)

    def test_calculate_dynamic_bounds_with_zero_pvalues(self):
        df_with_zeros = pd.DataFrame({"GWAS P-VALUE": [0.0, 1e-6, 1e-5, 1e-4, 0.01]})
        df_with_zeros.index.name = "SNP"

        min_log_p, max_log_p = calculate_dynamic_bounds(
            df_gwas=df_with_zeros, gwas_p_value_threshold=1e-4, min_n_snps=2
        )

        assert min_log_p < max_log_p
        assert not np.isnan(min_log_p)
        assert not np.isnan(max_log_p)

    def test_calculate_dynamic_bounds_invalid_bounds(self):
        df_invalid = pd.DataFrame({"GWAS P-VALUE": [1e-8] * 10})
        df_invalid.index.name = "SNP"

        min_log_p, max_log_p = calculate_dynamic_bounds(
            df_gwas=df_invalid, gwas_p_value_threshold=1e-9, min_n_snps=5
        )

        assert min_log_p == -8.0
        assert max_log_p == -3.0


class TestFractionToLogPValue:
    def test_fraction_to_log_p_value_normal_case(self, sample_gwas_df):
        fraction = 0.1

        log_p = fraction_to_log_p_value(fraction, sample_gwas_df)

        assert isinstance(log_p, float)
        assert not np.isnan(log_p)
        assert log_p < 0

    def test_fraction_to_log_p_value_zero_fraction(self, sample_gwas_df):
        fraction = 0.0

        log_p = fraction_to_log_p_value(fraction, sample_gwas_df)

        assert log_p == -8.0

    def test_fraction_to_log_p_value_full_fraction(self, small_gwas_df):
        fraction = 1.0

        log_p = fraction_to_log_p_value(fraction, small_gwas_df)

        max_p = small_gwas_df["GWAS P-VALUE"].max()
        expected_log_p = np.log10(max_p)
        assert abs(log_p - expected_log_p) < 1e-10

    def test_fraction_to_log_p_value_with_zeros(self):
        df_with_zeros = pd.DataFrame({"GWAS P-VALUE": [0.0, 1e-6, 1e-4, 0.01]})
        df_with_zeros.index.name = "SNP"

        log_p = fraction_to_log_p_value(0.5, df_with_zeros)

        assert isinstance(log_p, float)
        assert not np.isnan(log_p)


class TestGetGwasBoAutoTopN:

    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
        "gather_fractions_and_performances"
    )
    def test_get_gwas_bo_auto_top_n_basic(
        self, mock_gather, sample_gwas_df, mock_history_df, tmp_path
    ):
        mock_gather.return_value = mock_history_df

        top_n, fraction = get_gwas_bo_auto_top_n(
            df_gwas=sample_gwas_df,
            folder_with_runs=tmp_path,
            feature_selection_output_folder=tmp_path,
            fold=0,
            gwas_p_value_threshold=1e-4,
            min_n_snps=16,
        )

        assert isinstance(top_n, int)
        assert isinstance(fraction, float)
        assert top_n >= 16
        assert 0.0 <= fraction <= 1.0
        assert top_n == int(fraction * len(sample_gwas_df))

    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
        "gather_fractions_and_performances"
    )
    def test_get_gwas_bo_auto_top_n_small_dataset(
        self, mock_gather, small_gwas_df, tmp_path
    ):
        mock_gather.return_value = pd.DataFrame(
            {"fraction": [], "best_val_performance": []}
        )

        min_n_snps = 16

        top_n, fraction = get_gwas_bo_auto_top_n(
            df_gwas=small_gwas_df,
            folder_with_runs=tmp_path,
            feature_selection_output_folder=tmp_path,
            fold=0,
            gwas_p_value_threshold=1e-4,
            min_n_snps=min_n_snps,
        )

        assert top_n == len(small_gwas_df)
        assert fraction == 1.0

    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
        "gather_fractions_and_performances"
    )
    def test_get_gwas_bo_auto_top_n_enforces_min_snps(
        self, mock_gather, sample_gwas_df, tmp_path
    ):
        mock_gather.return_value = pd.DataFrame(
            {"fraction": [], "best_val_performance": []}
        )

        min_n_snps = 50

        with patch(
            "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection.Optimizer"
        ) as mock_opt:
            mock_optimizer = Mock()
            mock_optimizer.ask.return_value = [-10.0]
            mock_opt.return_value = mock_optimizer

            top_n, fraction = get_gwas_bo_auto_top_n(
                df_gwas=sample_gwas_df,
                folder_with_runs=tmp_path,
                feature_selection_output_folder=tmp_path,
                fold=0,
                gwas_p_value_threshold=1e-4,
                min_n_snps=min_n_snps,
            )

        assert top_n >= min_n_snps


class TestOptimizationConvergence:

    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
        "gather_fractions_and_performances"
    )
    def test_optimization_finds_optimum_simulated(self, mock_gather, tmp_path):
        n_snps = 1000
        np.random.seed(123)

        optimal_threshold = 1e-2
        p_values = np.random.uniform(0, 1, n_snps)

        n_optimal = int(0.05 * n_snps)
        p_values[:n_optimal] = np.random.uniform(1e-8, optimal_threshold, n_optimal)

        df_gwas = pd.DataFrame({"GWAS P-VALUE": p_values})
        df_gwas.index.name = "SNP"
        df_gwas.index = [f"rs{i}" for i in range(len(df_gwas))]

        historical_fractions = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
        historical_performances = [0.70, 0.75, 0.90, 0.85, 0.75, 0.65]

        mock_history = pd.DataFrame(
            {
                "fraction": historical_fractions,
                "best_val_performance": historical_performances,
            }
        )
        mock_gather.return_value = mock_history

        suggestions = []
        fractions = []

        for fold in range(10):
            top_n, fraction = get_gwas_bo_auto_top_n(
                df_gwas=df_gwas,
                folder_with_runs=tmp_path,
                feature_selection_output_folder=tmp_path,
                fold=fold,
                gwas_p_value_threshold=0.05,
                min_n_snps=10,
            )

            df_sorted = df_gwas.sort_values("GWAS P-VALUE")
            p_threshold = (
                df_sorted.iloc[top_n - 1]["GWAS P-VALUE"] if top_n > 0 else 1e-8
            )

            suggestions.append(p_threshold)
            fractions.append(fraction)

        assert all(0.0 <= f <= 1.0 for f in fractions)
        assert all(p > 0 for p in suggestions)

        very_stringent = sum(1 for p in suggestions if p < 1e-6)
        very_lenient = sum(1 for p in suggestions if p > 0.1)

        assert very_stringent < len(suggestions)
        assert very_lenient < len(suggestions)

    def test_optimizer_initialization_parameters(self, sample_gwas_df, tmp_path):
        with patch(
            "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
            "gather_fractions_and_performances"
        ) as mock_gather:
            mock_gather.return_value = pd.DataFrame(
                {
                    "fraction": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "best_val_performance": [0.75, 0.82, 0.85, 0.83, 0.80],
                }
            )

            with patch(
                "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection.Optimizer"
            ) as mock_opt_class:
                mock_optimizer = Mock()
                mock_optimizer.ask.return_value = [-5.0]
                mock_optimizer.tell = Mock()
                mock_opt_class.return_value = mock_optimizer

                get_gwas_bo_auto_top_n(
                    df_gwas=sample_gwas_df,
                    folder_with_runs=tmp_path,
                    feature_selection_output_folder=tmp_path,
                    fold=0,
                    gwas_p_value_threshold=1e-4,
                    min_n_snps=16,
                )

                mock_opt_class.assert_called_once()
                call_kwargs = mock_opt_class.call_args[1]

                assert "dimensions" in call_kwargs
                assert "n_initial_points" in call_kwargs
                assert "initial_point_generator" in call_kwargs
                assert call_kwargs["initial_point_generator"] == "sobol"


class TestGetGwasTopNSnpListDf:
    def test_get_top_n_snps(self, sample_gwas_df):
        top_n = 50

        result_df = get_gwas_top_n_snp_list_df(sample_gwas_df, top_n)

        assert len(result_df) == top_n
        assert "SNP" in result_df.columns
        assert "GWAS P-VALUE" in result_df.columns

        p_values = result_df["GWAS P-VALUE"].values
        assert all(p_values[i] <= p_values[i + 1] for i in range(len(p_values) - 1))

    def test_get_top_n_snps_all(self, small_gwas_df):
        top_n = len(small_gwas_df)

        result_df = get_gwas_top_n_snp_list_df(small_gwas_df, top_n)

        assert len(result_df) == len(small_gwas_df)
        p_values = result_df["GWAS P-VALUE"].values
        assert all(p_values[i] <= p_values[i + 1] for i in range(len(p_values) - 1))

    def test_get_top_n_snps_edge_cases(self, small_gwas_df):
        result_df = get_gwas_top_n_snp_list_df(small_gwas_df, 0)
        assert len(result_df) == 0

        result_df = get_gwas_top_n_snp_list_df(small_gwas_df, 1)
        assert len(result_df) == 1
        assert result_df["GWAS P-VALUE"].iloc[0] == small_gwas_df["GWAS P-VALUE"].min()


class TestRunGwasBoFeatureSelection:
    @patch("eir_auto_gp.single_task.modelling.gwas_bo_feature_selection.read_gwas_df")
    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection"
        ".get_gwas_bo_auto_top_n"
    )
    def test_run_gwas_bo_feature_selection_success(
        self, mock_get_top_n, mock_read_gwas, sample_gwas_df, tmp_path
    ):
        mock_read_gwas.return_value = sample_gwas_df.rename(
            columns={"GWAS P-VALUE": "P"}
        )
        mock_get_top_n.return_value = (100, 0.1)

        fs_output_folder = tmp_path / "feature_selection"
        subsets_folder = fs_output_folder / "snp_importance" / "snp_subsets"
        subsets_folder.mkdir(parents=True)

        gwas_output_folder = tmp_path / "gwas_output"
        gwas_output_folder.mkdir()

        result_path = run_gwas_bo_feature_selection(
            fold=0,
            folder_with_runs=tmp_path,
            feature_selection_output_folder=fs_output_folder,
            gwas_output_folder=gwas_output_folder,
            gwas_p_value_threshold=1e-4,
        )

        assert result_path is not None
        assert result_path.exists()

        fraction_file = subsets_folder / "chosen_snps_fraction_0.txt"
        assert fraction_file.exists()
        assert fraction_file.read_text() == "0.1"

        mock_get_top_n.assert_called_once()

    def test_run_gwas_bo_feature_selection_existing_file(self, tmp_path):
        fs_output_folder = tmp_path / "feature_selection"
        subsets_folder = fs_output_folder / "snp_importance" / "snp_subsets"
        subsets_folder.mkdir(parents=True)

        existing_file = subsets_folder / "chosen_snps_0.txt"
        existing_file.write_text("rs1\nrs2\nrs3\n")

        result_path = run_gwas_bo_feature_selection(
            fold=0,
            folder_with_runs=tmp_path,
            feature_selection_output_folder=fs_output_folder,
            gwas_output_folder=tmp_path / "gwas_output",
            gwas_p_value_threshold=1e-4,
        )

        assert result_path == existing_file

    def test_run_gwas_bo_feature_selection_no_gwas_folder(self, tmp_path):
        fs_output_folder = tmp_path / "feature_selection"

        with pytest.raises(AssertionError):
            run_gwas_bo_feature_selection(
                fold=0,
                folder_with_runs=tmp_path,
                feature_selection_output_folder=fs_output_folder,
                gwas_output_folder=None,
                gwas_p_value_threshold=1e-4,
            )


class TestIntegrationScenarios:
    @patch(
        "eir_auto_gp.single_task.modelling.gwas_bo_feature_selection."
        "gather_fractions_and_performances"
    )
    def test_realistic_optimization_scenario(self, mock_gather, tmp_path):
        np.random.seed(42)

        p_values = np.concatenate(
            [
                np.random.uniform(1e-8, 1e-5, 50),  # 1% truly significant
                np.random.uniform(1e-5, 1e-3, 200),  # 4% moderately significant
                np.random.uniform(1e-3, 1.0, 4750),  # 95% not significant
            ]
        )
        np.random.shuffle(p_values)

        df_gwas = pd.DataFrame({"GWAS P-VALUE": p_values})
        df_gwas.index.name = "SNP"
        df_gwas.index = [f"rs{i:06d}" for i in range(len(df_gwas))]

        historical_data = []
        for fraction in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
            if 0.008 <= fraction <= 0.025:  # High performance
                performance = 0.85 + np.random.normal(0, 0.02)
            elif 0.002 <= fraction <= 0.05:  # Good performance
                performance = 0.80 + np.random.normal(0, 0.03)
            else:  # Low performance
                performance = 0.70 + np.random.normal(0, 0.05)

            historical_data.append(
                {
                    "fraction": fraction,
                    "best_val_performance": max(0.6, min(0.9, performance)),
                }
            )

        mock_history = pd.DataFrame(historical_data)
        mock_gather.return_value = mock_history

        results = []
        for fold in range(5):
            top_n, fraction = get_gwas_bo_auto_top_n(
                df_gwas=df_gwas,
                folder_with_runs=tmp_path,
                feature_selection_output_folder=tmp_path,
                fold=fold,
                gwas_p_value_threshold=0.05,
                min_n_snps=20,
            )
            results.append((top_n, fraction))

        for top_n, fraction in results:
            assert isinstance(top_n, int)
            assert isinstance(fraction, float)
            assert top_n >= 20
            assert 0.0 <= fraction <= 1.0
            assert top_n <= len(df_gwas)

        fractions = [f for _, f in results]
        assert len(set(fractions)) > 1, "Should explore different fractions"

        avg_fraction = np.mean(fractions)
        assert (
            0.001 <= avg_fraction <= 0.2
        ), f"Average fraction {avg_fraction} should be reasonable"
