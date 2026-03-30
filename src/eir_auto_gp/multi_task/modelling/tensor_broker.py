import random
from typing import Any


def _get_staggered_cache_names(
    use_fc0_to_fusion_skips: bool = True,
) -> list[str]:
    cache_names = []

    if use_fc0_to_fusion_skips:
        cache_names.append("fc_0_output")

    return cache_names


def _get_output_head_cache_names(
    use_fc0_to_output_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    include_tabular: bool = True,
) -> list[str]:
    cache_names = []

    if use_fc0_to_output_skips:
        cache_names.append("fc_0_output")

    if use_lcl_to_output_skips:
        cache_names.append("lcl_block_0_output")

    if include_tabular:
        cache_names.append("tabular_output")

    return cache_names


def _get_fusion_block_cache_configs(
    num_layers: int,
    tb_block_frequency: int,
    use_fc0_to_fusion_skips: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    cache_configs: list[dict[str, Any]] = []
    cache_names: list[str] = []

    base_has_skip = bool(
        _get_staggered_cache_names(use_fc0_to_fusion_skips=use_fc0_to_fusion_skips)
    )
    if base_has_skip:
        name = "fusion_block_base"
        cache_configs.append(
            {
                "name": name,
                "layer_path": "fusion_modules.computed.fusion_modules.fusion.0.0",
                "cache_tensor": True,
                "layer_cache_target": "output",
            }
        )
        cache_names.append(name)

    num_layers_adjusted = num_layers - 2
    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0 and base_has_skip:
            name = f"fusion_block_{layer}"
            cache_configs.append(
                {
                    "name": name,
                    "layer_path": f"fusion_modules.computed.fusion_modules"
                    f".fusion.1.{layer}",
                    "cache_tensor": True,
                    "layer_cache_target": "output",
                }
            )
            cache_names.append(name)

    return cache_configs, cache_names


def _build_dense_output_head_messages(
    output_head: str,
    target_columns: list[str],
    output_groups: dict[str, list[str]] | None,
    dense_cache_names: list[str],
    cache_fusion_type: str,
    include_tabular: bool,
    tabular_cache_dropout_p: float,
    output_num_experts: int | None,
) -> list[dict[str, Any]]:
    message_configs: list[dict[str, Any]] = []

    if output_head == "linear":
        if dense_cache_names:
            message_configs.append(
                {
                    "name": "final_layer",
                    "layer_path": "output_modules.eir_auto_gp.linear_layer",
                    "use_from_cache": dense_cache_names,
                    "multi_source_aggregation": "attention",
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": cache_fusion_type,
                    "kernel_width_divisible_by": 12,
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
            if dense_cache_names:
                message_configs.append(
                    {
                        "name": f"final_layer_{target_column}",
                        "layer_path": f"output_modules.eir_auto_gp.multi_task_branches."
                        f"{target_column}.0.1",
                        "use_from_cache": dense_cache_names,
                        "multi_source_aggregation": "attention",
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": cache_fusion_type,
                        "kernel_width_divisible_by": 12,
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
            if dense_cache_names:
                if output_num_experts is not None:
                    layer_target = "input_identity"
                else:
                    layer_target = "shared_branch"
                message_configs.append(
                    {
                        "name": f"final_layer_{group_name}",
                        "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                        f".{layer_target}",
                        "use_from_cache": dense_cache_names,
                        "multi_source_aggregation": "attention",
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": cache_fusion_type,
                        "kernel_width_divisible_by": 12,
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

    return message_configs


def generate_tb_base_config(
    num_layers: int,
    tb_block_frequency: int,
    output_head: str,
    target_columns: list[str],
    output_groups: dict[str, list[str]] | None,
    use_fc0_to_output_skips: bool = True,
    use_fc0_to_fusion_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    output_num_experts: int | None = None,
    dense_output_connections: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    base_cache_names = _get_staggered_cache_names(
        use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
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
                "kernel_width_divisible_by": 12,
            }
        )

    num_layers_adjusted = num_layers - 2

    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            cache_names = _get_staggered_cache_names(
                use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
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
                        "kernel_width_divisible_by": 12,
                    }
                )

    if dense_output_connections:
        fusion_cache_configs, fusion_cache_names = _get_fusion_block_cache_configs(
            num_layers=num_layers,
            tb_block_frequency=tb_block_frequency,
            use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
        )
        message_configs.extend(fusion_cache_configs)

        dense_cache_names = _get_output_head_cache_names(
            use_fc0_to_output_skips=use_fc0_to_output_skips,
            use_lcl_to_output_skips=use_lcl_to_output_skips,
            include_tabular=False,
        )
        dense_cache_names.extend(fusion_cache_names)

        message_configs.extend(
            _build_dense_output_head_messages(
                output_head=output_head,
                target_columns=target_columns,
                output_groups=output_groups,
                dense_cache_names=dense_cache_names,
                cache_fusion_type="sum",
                include_tabular=include_tabular,
                tabular_cache_dropout_p=tabular_cache_dropout_p,
                output_num_experts=output_num_experts,
            )
        )
    else:
        genotype_cache_names = _get_output_head_cache_names(
            use_fc0_to_output_skips=use_fc0_to_output_skips,
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
                        "kernel_width_divisible_by": 12,
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
                            "layer_path": f"output_modules.eir_auto_gp."
                            f"multi_task_branches.{target_column}.0.1",
                            "use_from_cache": genotype_cache_names,
                            "projection_type": "lcl+mlp_residual",
                            "cache_fusion_type": "sum",
                            "kernel_width_divisible_by": 12,
                        }
                    )
                if include_tabular:
                    message_configs.append(
                        {
                            "name": f"tabular_to_{target_column}",
                            "layer_path": f"output_modules.eir_auto_gp."
                            f"multi_task_branches.{target_column}.0.1",
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
                            "kernel_width_divisible_by": 12,
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
    use_fc0_to_output_skips: bool = True,
    use_fc0_to_fusion_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    output_num_experts: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    base_cache_names = _get_staggered_cache_names(
        use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
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
                    "kernel_width_divisible_by": 12,
                }
            )

    num_layers_adjusted = num_layers - 2
    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            cache_names = _get_staggered_cache_names(
                use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
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
                            "kernel_width_divisible_by": 12,
                        }
                    )

    genotype_cache_names = _get_output_head_cache_names(
        use_fc0_to_output_skips=use_fc0_to_output_skips,
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
                    "kernel_width_divisible_by": 12,
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
                        "kernel_width_divisible_by": 12,
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


def _get_fusion_layer_targets(
    num_fusion_layers: int,
    tb_block_frequency: int,
) -> list[str]:
    base = "fusion_modules.computed.fusion_modules.fusion"
    targets = [f"{base}.0.0"]
    num_layers_adjusted = num_fusion_layers - 2
    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            targets.append(f"{base}.1.{layer}")
    return targets


def _get_mgmoe_fusion_layer_targets(
    num_fusion_layers: int,
    tb_block_frequency: int,
    num_mgmoe_experts: int,
) -> list[str]:
    base = "fusion_modules.computed.expert_branches"
    targets = []
    for expert in range(num_mgmoe_experts):
        targets.append(f"{base}.expert_{expert}.0.0")
        num_layers_adjusted = num_fusion_layers - 2
        for layer in range(0, num_layers_adjusted + 1):
            if layer % tb_block_frequency == 0:
                targets.append(f"{base}.expert_{expert}.1.{layer}")
    return targets


def generate_tb_informed_mgmoe_config(
    expert_names: list[str],
    num_fusion_layers: int,
    tb_block_frequency: int,
    num_mgmoe_experts: int,
    fusion_factor: int = 1,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    output_num_experts: int | None = None,
    use_fc0_output_skips: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    mgmoe_targets = _get_mgmoe_fusion_layer_targets(
        num_fusion_layers=num_fusion_layers,
        tb_block_frequency=tb_block_frequency,
        num_mgmoe_experts=num_mgmoe_experts,
    )

    for name in expert_names:
        for factor_idx in range(fusion_factor):
            target = random.choice(mgmoe_targets)
            message_configs.append(
                {
                    "name": f"expert_{name}_fc_0_to_mgmoe_f{factor_idx}",
                    "layer_path": target,
                    "use_from_cache": [f"expert_{name}_fc_0"],
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                    "kernel_width_divisible_by": 12,
                }
            )

    for name in expert_names:
        if use_fc0_output_skips:
            if output_num_experts is not None:
                deep_target = "input_identity"
            else:
                deep_target = "shared_branch.0.0.0"

            message_configs.append(
                {
                    "name": f"expert_{name}_deep_to_output",
                    "layer_path": f"output_modules.eir_auto_gp_{name}.{deep_target}",
                    "use_from_cache": [f"expert_{name}_fc_0"],
                    "projection_type": "lcl+mlp_residual",
                    "projection_lcl_residual_blocks": True,
                    "cache_fusion_type": "sum",
                    "kernel_width_divisible_by": 12,
                }
            )

        if include_tabular:
            message_configs.append(
                {
                    "name": f"tabular_to_{name}",
                    "layer_path": f"output_modules.eir_auto_gp_{name}.output_identity",
                    "use_from_cache": ["tabular_output"],
                    "projection_type": "mlp_residual",
                    "cache_fusion_type": "additive",
                    "cache_dropout_p": tabular_cache_dropout_p,
                }
            )

    return {"message_configs": message_configs}


def generate_tb_informed_moe_config(
    expert_names: list[str],
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    output_num_experts: int | None = None,
    use_fc0_output_skips: bool = True,
    num_fusion_layers: int | None = None,
    tb_block_frequency: int = 1,
    dense_output_connections: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    if num_fusion_layers is not None:
        fusion_targets = _get_fusion_layer_targets(
            num_fusion_layers=num_fusion_layers,
            tb_block_frequency=tb_block_frequency,
        )
        target_to_experts: dict[str, list[str]] = {}
        for i, name in enumerate(expert_names):
            target = fusion_targets[i % len(fusion_targets)]
            target_to_experts.setdefault(target, []).append(name)

        for target, names in target_to_experts.items():
            cache_names = [f"expert_{n}_fc_0" for n in names]
            expert_label = "+".join(names)
            msg: dict[str, Any] = {
                "name": f"experts_{expert_label}_fc_0_to_fusion",
                "layer_path": target,
                "use_from_cache": cache_names,
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
                "kernel_width_divisible_by": 12,
            }
            if len(cache_names) > 1:
                msg["multi_source_aggregation"] = "attention"
            message_configs.append(msg)

    fusion_cache_names: list[str] = []
    if dense_output_connections and num_fusion_layers is not None:
        fusion_cache_configs, fusion_cache_names = _get_fusion_block_cache_configs(
            num_layers=num_fusion_layers,
            tb_block_frequency=tb_block_frequency,
            use_fc0_to_fusion_skips=True,
        )
        message_configs.extend(fusion_cache_configs)

    for name in expert_names:
        if use_fc0_output_skips:
            if output_num_experts is not None:
                deep_target = "input_identity"
            else:
                deep_target = "shared_branch.0.0.0"

            if dense_output_connections and fusion_cache_names:
                dense_cache_names = [f"expert_{name}_fc_0"] + fusion_cache_names
                message_configs.append(
                    {
                        "name": f"expert_{name}_deep_to_output",
                        "layer_path": f"output_modules.eir_auto_gp_{name}"
                        f".{deep_target}",
                        "use_from_cache": dense_cache_names,
                        "multi_source_aggregation": "attention",
                        "projection_type": "lcl+mlp_residual",
                        "projection_lcl_residual_blocks": True,
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 12,
                    }
                )
            else:
                message_configs.append(
                    {
                        "name": f"expert_{name}_deep_to_output",
                        "layer_path": f"output_modules.eir_auto_gp_{name}"
                        f".{deep_target}",
                        "use_from_cache": [f"expert_{name}_fc_0"],
                        "projection_type": "lcl+mlp_residual",
                        "projection_lcl_residual_blocks": True,
                        "cache_fusion_type": "sum",
                        "kernel_width_divisible_by": 12,
                    }
                )

        if include_tabular:
            message_configs.append(
                {
                    "name": f"tabular_to_{name}",
                    "layer_path": f"output_modules.eir_auto_gp_{name}.output_identity",
                    "use_from_cache": ["tabular_output"],
                    "projection_type": "mlp_residual",
                    "cache_fusion_type": "additive",
                    "cache_dropout_p": tabular_cache_dropout_p,
                }
            )

    return {"message_configs": message_configs}
