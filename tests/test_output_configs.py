import pytest

from eir_auto_gp.multi_task.modelling.output_configs import (
    _get_auto_output_dim,
    get_output_configs,
)


@pytest.mark.parametrize(
    "n_targets, expected_dim",
    [
        (1, 512),
        (2, 512),
        (8, 512),
        (15, 512),
        (20, 512),
        (21, 1024),
        (30, 1024),
    ],
)
def test_auto_output_dim_tiers(n_targets: int, expected_dim: int) -> None:
    assert _get_auto_output_dim(n_targets=n_targets) == expected_dim


def test_auto_output_dim_per_group_configs() -> None:
    output_groups = {
        "small_group": ["trait_a", "trait_b"],
        "large_group": [f"trait_lg_{i}" for i in range(25)],
    }

    all_columns = []
    for cols in output_groups.values():
        all_columns.extend(cols)

    configs = get_output_configs(
        output_groups=output_groups,
        output_cat_columns=[],
        output_con_columns=all_columns,
        model_size="large",
        output_head="shared_mlp_residual",
        n_output_layers=2,
        output_dim="auto",
    )

    assert len(configs) == 2

    dims_by_group = {}
    for cfg in configs:
        name = cfg["output_info"]["output_name"]
        dim = cfg["model_config"]["model_init_config"]["fc_task_dim"]
        dims_by_group[name] = dim

    assert dims_by_group["eir_auto_gp_small_group"] == 512
    assert dims_by_group["eir_auto_gp_large_group"] == 1024


def test_fixed_output_dim_uniform_across_groups() -> None:
    output_groups = {
        "small_group": ["trait_a", "trait_b"],
        "large_group": [f"trait_{i}" for i in range(25)],
    }

    all_columns = []
    for cols in output_groups.values():
        all_columns.extend(cols)

    configs = get_output_configs(
        output_groups=output_groups,
        output_cat_columns=[],
        output_con_columns=all_columns,
        model_size="large",
        output_head="shared_mlp_residual",
        n_output_layers=2,
        output_dim=512,
    )

    for cfg in configs:
        assert cfg["model_config"]["model_init_config"]["fc_task_dim"] == 512


def test_auto_output_dim_does_not_mutate_across_groups() -> None:
    output_groups = {
        "group_a": ["trait_a"],
        "group_b": [f"trait_{i}" for i in range(25)],
    }

    all_columns = []
    for cols in output_groups.values():
        all_columns.extend(cols)

    configs = get_output_configs(
        output_groups=output_groups,
        output_cat_columns=[],
        output_con_columns=all_columns,
        model_size="large",
        output_head="shared_mlp_residual",
        n_output_layers=2,
        output_dim="auto",
    )

    dims = [c["model_config"]["model_init_config"]["fc_task_dim"] for c in configs]
    assert dims[0] != dims[1], "Different group sizes should get different dims"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
