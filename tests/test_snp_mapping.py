import numpy as np
import pandas as pd
import pytest

from eir_auto_gp.predict.prepare_data import create_snp_mapping


def test_exact_match():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 1
    assert len(mapping.flip_match_source_indices) == 0
    assert mapping.direct_match_source_indices[0] == 0
    assert mapping.direct_match_target_indices[0] == 0


def test_allele_reversal():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["G"],
            "ALT": ["A"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 1
    assert mapping.flip_match_source_indices[0] == 0
    assert mapping.flip_match_target_indices[0] == 0


def test_strand_flip():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["T"],
            "ALT": ["C"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 1
    assert len(mapping.flip_match_source_indices) == 0
    assert mapping.direct_match_source_indices[0] == 0
    assert mapping.direct_match_target_indices[0] == 0


def test_strand_flip_and_reversal():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["C"],
            "ALT": ["T"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 1
    assert mapping.flip_match_source_indices[0] == 0
    assert mapping.flip_match_target_indices[0] == 0


def test_ambiguous_at_palindrome_dropped():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["T"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["T"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 0
    assert len(mapping.missing_target_indices) == 1
    assert mapping.missing_target_indices[0] == 0


def test_ambiguous_cg_palindrome_dropped():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["C"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["C"],
            "ALT": ["G"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 0
    assert len(mapping.missing_target_indices) == 1


def test_true_mismatch_dropped():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["C"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 0
    assert len(mapping.missing_target_indices) == 1


def test_no_positional_overlap():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [1000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1"],
            "BP_COORD": [2000],
            "REF": ["A"],
            "ALT": ["G"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 0
    assert len(mapping.flip_match_source_indices) == 0
    assert len(mapping.missing_target_indices) == 1


def test_mixed_scenario():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1", "1", "1", "1", "1"],
            "BP_COORD": [100, 200, 300, 400, 500, 600],
            "REF": ["A", "A", "A", "A", "A", "A"],
            "ALT": ["G", "G", "G", "G", "T", "G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1", "1", "1", "1", "1"],
            "BP_COORD": [100, 200, 300, 400, 500, 600],
            "REF": ["A", "G", "T", "C", "A", "C"],
            "ALT": ["G", "A", "C", "T", "T", "G"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 2
    assert len(mapping.flip_match_source_indices) == 2
    assert len(mapping.missing_target_indices) == 2

    assert 0 in mapping.direct_match_source_indices
    assert 2 in mapping.direct_match_source_indices

    assert 1 in mapping.flip_match_source_indices
    assert 3 in mapping.flip_match_source_indices

    assert 4 in mapping.missing_target_indices
    assert 5 in mapping.missing_target_indices


def test_multiple_chromosomes():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "2", "3"],
            "BP_COORD": [1000, 1000, 1000],
            "REF": ["A", "A", "A"],
            "ALT": ["G", "G", "G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "2", "3"],
            "BP_COORD": [1000, 1000, 1000],
            "REF": ["A", "G", "T"],
            "ALT": ["G", "A", "C"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 2
    assert len(mapping.flip_match_source_indices) == 1


def test_index_ordering_preserved():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1", "1"],
            "BP_COORD": [100, 200, 300],
            "REF": ["A", "A", "A"],
            "ALT": ["G", "G", "G"],
        }
    )

    target_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1", "1"],
            "BP_COORD": [300, 100, 200],
            "REF": ["T", "A", "G"],
            "ALT": ["C", "G", "A"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    assert len(mapping.direct_match_source_indices) == 2
    assert len(mapping.flip_match_source_indices) == 1

    source_idx_0 = np.where(mapping.direct_match_source_indices == 0)[0]
    assert len(source_idx_0) == 1
    target_idx = mapping.direct_match_target_indices[source_idx_0[0]]
    assert target_idx == 1

    source_idx_2 = np.where(mapping.direct_match_source_indices == 2)[0]
    assert len(source_idx_2) == 1
    target_idx = mapping.direct_match_target_indices[source_idx_2[0]]
    assert target_idx == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
