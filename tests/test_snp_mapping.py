import numpy as np
import pandas as pd
import pytest

from eir_auto_gp.predict.prepare_data import (
    SNPMapping,
    create_snp_mapping,
    get_projected_snp_stream,
)


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


def test_duplicate_source_positions_same_target():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1"],
            "BP_COORD": [1000, 1000],
            "REF": ["A", "A"],
            "ALT": ["G", "G"],
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

    assert len(mapping.direct_match_target_indices) == 1, (
        "Duplicate source SNPs at same position should deduplicate to one "
        "target mapping"
    )
    assert mapping.direct_match_target_indices[0] == 0


def test_duplicate_target_positions_same_source():
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
            "CHR_CODE": ["1", "1"],
            "BP_COORD": [1000, 1000],
            "REF": ["A", "A"],
            "ALT": ["G", "G"],
        }
    )

    mapping = create_snp_mapping(from_bim=source_bim, to_reference_bim=target_bim)

    unique_targets = set(mapping.direct_match_target_indices)
    assert len(unique_targets) == len(mapping.direct_match_target_indices), (
        "Each target index should appear at most once"
    )


def test_cross_type_conflict_prefers_direct():
    source_bim = pd.DataFrame(
        {
            "CHR_CODE": ["1", "1"],
            "BP_COORD": [1000, 1000],
            "REF": ["A", "G"],
            "ALT": ["G", "A"],
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

    assert len(mapping.direct_match_target_indices) == 1
    assert len(mapping.flip_match_target_indices) == 0, (
        "Cross-type conflict should resolve in favor of direct match"
    )
    assert mapping.direct_match_source_indices[0] == 0


def _make_one_hot(genotype: int) -> np.ndarray:
    arr = np.zeros(3, dtype=np.uint8)
    arr[genotype] = 1
    return arr


def _single_sample_stream(
    sample_id: str,
    genotypes: list[int],
):
    arr = np.stack([_make_one_hot(genotype=g) for g in genotypes], axis=1)
    yield sample_id, arr


def test_projection_exact_match_preserves_values():
    source_genotypes = [0, 1, 2]
    mapping = SNPMapping(
        direct_match_source_indices=np.array([0, 1, 2]),
        direct_match_target_indices=np.array([0, 1, 2]),
        flip_match_source_indices=np.array([], dtype=int),
        flip_match_target_indices=np.array([], dtype=int),
        missing_target_indices=np.array([], dtype=int),
        n_source_snps=3,
        n_target_snps=3,
    )

    stream = _single_sample_stream(sample_id="s1", genotypes=source_genotypes)
    results = list(get_projected_snp_stream(from_stream=stream, snp_mapping=mapping))

    assert len(results) == 1
    sid, projected = results[0]
    assert sid == "s1"

    assert projected[0, 0] == 1 and projected.sum(axis=0)[0] == 1
    assert projected[1, 1] == 1 and projected.sum(axis=0)[1] == 1
    assert projected[2, 2] == 1 and projected.sum(axis=0)[2] == 1


def test_projection_reversal_swaps_ref_alt():
    mapping = SNPMapping(
        direct_match_source_indices=np.array([], dtype=int),
        direct_match_target_indices=np.array([], dtype=int),
        flip_match_source_indices=np.array([0, 1, 2]),
        flip_match_target_indices=np.array([0, 1, 2]),
        missing_target_indices=np.array([], dtype=int),
        n_source_snps=3,
        n_target_snps=3,
    )

    source_genotypes = [0, 1, 2]
    stream = _single_sample_stream(sample_id="s1", genotypes=source_genotypes)
    results = list(get_projected_snp_stream(from_stream=stream, snp_mapping=mapping))

    _, projected = results[0]

    assert projected[2, 0] == 1, "Source REF/REF (row 0) should become ALT/ALT (row 2)"
    assert projected[0, 0] == 0

    assert projected[1, 1] == 1, "Source HET (row 1) should remain HET (row 1)"

    assert projected[0, 2] == 1, "Source ALT/ALT (row 2) should become REF/REF (row 0)"
    assert projected[2, 2] == 0


def test_projection_missing_targets_get_missing_channel():
    mapping = SNPMapping(
        direct_match_source_indices=np.array([0]),
        direct_match_target_indices=np.array([0]),
        flip_match_source_indices=np.array([], dtype=int),
        flip_match_target_indices=np.array([], dtype=int),
        missing_target_indices=np.array([1, 2]),
        n_source_snps=1,
        n_target_snps=3,
    )

    stream = _single_sample_stream(sample_id="s1", genotypes=[0])
    results = list(get_projected_snp_stream(from_stream=stream, snp_mapping=mapping))

    _, projected = results[0]
    assert projected.shape == (3, 3)

    assert projected[0, 0] == 1

    for missing_idx in [1, 2]:
        assert projected[:, missing_idx].sum() == 0, (
            f"Target {missing_idx} should be missing (all-zeros)"
        )


def test_projection_mixed_direct_and_swap():
    mapping = SNPMapping(
        direct_match_source_indices=np.array([0]),
        direct_match_target_indices=np.array([1]),
        flip_match_source_indices=np.array([1]),
        flip_match_target_indices=np.array([0]),
        missing_target_indices=np.array([2]),
        n_source_snps=2,
        n_target_snps=3,
    )

    stream = _single_sample_stream(sample_id="s1", genotypes=[0, 2])
    results = list(get_projected_snp_stream(from_stream=stream, snp_mapping=mapping))

    _, projected = results[0]

    assert projected[0, 1] == 1, "Direct match: source[0]=REF/REF -> target[1]=REF/REF"

    assert projected[0, 0] == 1, (
        "Swap match: source[1]=ALT/ALT -> target[0] should become REF/REF after swap"
    )
    assert projected[2, 0] == 0

    assert projected[:, 2].sum() == 0, "Missing target should be all-zeros"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
