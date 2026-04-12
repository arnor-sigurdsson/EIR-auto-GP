import pytest

from eir_auto_gp.multi_task.modelling.output_configs import (
    get_output_configs,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
