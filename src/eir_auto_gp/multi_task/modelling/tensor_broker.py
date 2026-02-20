from typing import Any


def _get_staggered_cache_names(
    layer_index: int,
    total_layers: int,
    n_lcl_blocks: int,
    use_fc0_skips: bool = True,
    use_lcl_fusion_skips: bool = True,
) -> list[str]:
    cache_names = []

    if use_fc0_skips:
        cache_names.append("fc_0_output")

    if n_lcl_blocks == 0 or not use_lcl_fusion_skips:
        return cache_names

    num_sections = n_lcl_blocks + 1
    section_size = total_layers / num_sections
    section = int(layer_index / section_size)

    if section > 0:
        lcl_block_index = min(section - 1, n_lcl_blocks - 1)
        cache_names.append(f"lcl_block_{lcl_block_index}")

    return cache_names


def _get_output_head_cache_names(
    use_fc0_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    include_tabular: bool = True,
) -> list[str]:
    cache_names = []

    if use_fc0_skips:
        cache_names.append("fc_0_output")

    if use_lcl_to_output_skips == "fc_1_only":
        cache_names.append("lcl_block_0_fc_1")
    elif use_lcl_to_output_skips is True:
        cache_names.extend(["lcl_block_0_fc_1", "lcl_block_0_fc_2"])

    if include_tabular:
        cache_names.append("tabular_output")

    return cache_names


def generate_tb_base_config(
    num_layers: int,
    tb_block_frequency: int,
    output_head: str,
    target_columns: list[str],
    output_groups: dict[str, list[str]] | None,
    n_lcl_blocks: int = 0,
    use_fc0_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    use_lcl_fusion_skips: bool = True,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    # only checked for is not None, kept as int for possible per-expert routing later
    output_num_experts: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    base_cache_names = _get_staggered_cache_names(
        layer_index=0,
        total_layers=num_layers,
        n_lcl_blocks=n_lcl_blocks,
        use_fc0_skips=use_fc0_skips,
        use_lcl_fusion_skips=use_lcl_fusion_skips,
    )
    message_configs: list[dict[str, Any]] = []

    if base_cache_names:
        message_configs.append(
            {
                "name": "base_fusion_residual_block",
                "layer_path": "fusion_modules.computed.fusion_modules.fusion.0.0",
                "use_from_cache": base_cache_names,
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
                "kernel_width_divisible_by": 4,
            }
        )

    num_layers_adjusted = num_layers - 2

    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            cache_names = _get_staggered_cache_names(
                layer_index=layer + 2,
                total_layers=num_layers,
                n_lcl_blocks=n_lcl_blocks,
                use_fc0_skips=use_fc0_skips,
                use_lcl_fusion_skips=use_lcl_fusion_skips,
            )
            if cache_names:
                message_configs.append(
                    {
                        "name": f"{layer}_fusion_residual_block",
                        "layer_path": f"fusion_modules.computed.fusion_modules"
                        f".fusion.1.{layer}",
                        "use_from_cache": cache_names,
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 4,
                    }
                )

    genotype_cache_names = _get_output_head_cache_names(
        use_fc0_skips=use_fc0_skips,
        use_lcl_to_output_skips=use_lcl_to_output_skips,
        include_tabular=False,
    )

    if output_head == "linear":
        if genotype_cache_names:
            message_configs.append(
                {
                    "name": "final_layer",
                    "layer_path": "output_modules.eir_auto_gp.linear_layer",
                    "use_from_cache": genotype_cache_names,
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                    "kernel_width_divisible_by": 4,
                }
            )
        if include_tabular:
            message_configs.append(
                {
                    "name": "tabular_to_output",
                    "layer_path": "output_modules.eir_auto_gp.linear_layer",
                    "use_from_cache": ["tabular_output"],
                    "projection_type": "mlp_residual",
                    "cache_fusion_type": "additive",
                    "cache_dropout_p": tabular_cache_dropout_p,
                }
            )
    elif output_head == "mlp":
        for target_column in target_columns:
            if genotype_cache_names:
                message_configs.append(
                    {
                        "name": f"final_layer_{target_column}",
                        "layer_path": f"output_modules.eir_auto_gp.multi_task_branches."
                        f"{target_column}.0.1",
                        "use_from_cache": genotype_cache_names,
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 4,
                    }
                )
            if include_tabular:
                message_configs.append(
                    {
                        "name": f"tabular_to_{target_column}",
                        "layer_path": f"output_modules.eir_auto_gp.multi_task_branches."
                        f"{target_column}.0.1",
                        "use_from_cache": ["tabular_output"],
                        "projection_type": "mlp_residual",
                        "cache_fusion_type": "additive",
                        "cache_dropout_p": tabular_cache_dropout_p,
                    }
                )
    elif output_head == "shared_mlp_residual":
        assert output_groups is not None
        for group_name, _group_columns in output_groups.items():
            if genotype_cache_names:
                if output_num_experts is not None:
                    layer_target = "input_identity"
                else:
                    layer_target = "shared_branch"
                message_configs.append(
                    {
                        "name": f"final_layer_{group_name}",
                        "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                        f".{layer_target}",
                        "use_from_cache": genotype_cache_names,
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 4,
                    }
                )
            if include_tabular:
                message_configs.append(
                    {
                        "name": f"tabular_to_{group_name}",
                        "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                        f".output_identity",
                        "use_from_cache": ["tabular_output"],
                        "projection_type": "mlp_residual",
                        "cache_fusion_type": "additive",
                        "cache_dropout_p": tabular_cache_dropout_p,
                    }
                )

    return {"message_configs": message_configs}


def generate_tb_mgmoe_config(
    num_layers: int,
    tb_block_frequency: int,
    num_experts: int,
    output_head: str,
    target_columns: list[str],
    output_groups: dict[str, list[str]] | None,
    n_lcl_blocks: int = 0,
    use_fc0_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    use_lcl_fusion_skips: bool = True,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    output_num_experts: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    base_cache_names = _get_staggered_cache_names(
        layer_index=0,
        total_layers=num_layers,
        n_lcl_blocks=n_lcl_blocks,
        use_fc0_skips=use_fc0_skips,
        use_lcl_fusion_skips=use_lcl_fusion_skips,
    )

    if base_cache_names:
        for expert in range(num_experts):
            message_configs.append(
                {
                    "name": f"expert_{expert}_base_fusion_residual_block",
                    "layer_path": f"fusion_modules.computed.expert_branches"
                    f".expert_{expert}.0.0",
                    "use_from_cache": base_cache_names,
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                    "kernel_width_divisible_by": 4,
                }
            )

    num_layers_adjusted = num_layers - 2
    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            cache_names = _get_staggered_cache_names(
                layer_index=layer + 2,
                total_layers=num_layers,
                n_lcl_blocks=n_lcl_blocks,
                use_fc0_skips=use_fc0_skips,
                use_lcl_fusion_skips=use_lcl_fusion_skips,
            )
            if cache_names:
                for expert in range(num_experts):
                    message_configs.append(
                        {
                            "name": f"expert_{expert}_{layer}_fusion_residual_block",
                            "layer_path": f"fusion_modules.computed.expert_branches"
                            f".expert_{expert}.1.{layer}",
                            "use_from_cache": cache_names,
                            "projection_type": "lcl+mlp_residual",
                            "cache_fusion_type": "sum",
                            "kernel_width_divisible_by": 4,
                        }
                    )

    genotype_cache_names = _get_output_head_cache_names(
        use_fc0_skips=use_fc0_skips,
        use_lcl_to_output_skips=use_lcl_to_output_skips,
        include_tabular=False,
    )

    if output_head == "linear":
        if genotype_cache_names:
            message_configs.append(
                {
                    "name": "final_layer",
                    "layer_path": "output_modules.eir_auto_gp.linear_layer",
                    "use_from_cache": genotype_cache_names,
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                    "kernel_width_divisible_by": 4,
                }
            )
        if include_tabular:
            message_configs.append(
                {
                    "name": "tabular_to_output",
                    "layer_path": "output_modules.eir_auto_gp.linear_layer",
                    "use_from_cache": ["tabular_output"],
                    "projection_type": "mlp_residual",
                    "cache_fusion_type": "additive",
                    "cache_dropout_p": tabular_cache_dropout_p,
                }
            )
    elif output_head == "shared_mlp_residual":
        assert output_groups is not None
        for group_name, _group_columns in output_groups.items():
            if genotype_cache_names:
                if output_num_experts is not None:
                    layer_target = "input_identity"
                else:
                    layer_target = "shared_branch"
                message_configs.append(
                    {
                        "name": f"final_layer_{group_name}",
                        "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                        f".{layer_target}",
                        "use_from_cache": genotype_cache_names,
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 4,
                    }
                )
            if include_tabular:
                message_configs.append(
                    {
                        "name": f"tabular_to_{group_name}",
                        "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                        f".output_identity",
                        "use_from_cache": ["tabular_output"],
                        "projection_type": "mlp_residual",
                        "cache_fusion_type": "additive",
                        "cache_dropout_p": tabular_cache_dropout_p,
                    }
                )

    return {"message_configs": message_configs}
