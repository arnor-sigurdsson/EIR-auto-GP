import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import yaml

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def get_base_global_config() -> Dict[str, Any]:
    base = {
        "basic_experiment": {
            "output_folder": "FILL",
            "batch_size": "FILL",
            "valid_size": "FILL",
            "n_epochs": 5000,
            "dataloader_workers": "FILL",
            "device": "FILL",
        },
        "evaluation_checkpoint": {
            "checkpoint_interval": "FILL",
            "sample_interval": "FILL",
            "saved_result_detail_level": 2,
        },
        "optimization": {
            "lr": "FILL",
            "gradient_clipping": 1.0,
            "optimizer": "adabelief",
        },
        "lr_schedule": {
            "lr_plateau_patience": 6,
        },
        "training_control": {
            "early_stopping_buffer": "FILL",
            "early_stopping_patience": 8,
            "mixing_alpha": "FILL",
        },
        "attribution_analysis": {
            "compute_attributions": False,
            "attribution_background_samples": 64,
            "max_attributions_per_class": 1000,
            "attributions_every_sample_factor": 5,
        },
        "visualization_logging": {
            "no_pbar": False,
        },
        "metrics": {
            "con_averaging_metrics": ["pcc", "r2"],
        },
    }
    return base


def get_base_input_genotype_config() -> Dict[str, Any]:
    base = {
        "input_info": {
            "input_source": "FILL",
            "input_name": "genotype",
            "input_type": "omics",
            "input_inner_key": "genotype",
        },
        "input_type_info": {
            "mixing_subtype": "mixup",
            "na_augment_alpha": 0.6,
            "na_augment_beta": 2.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 49.0,
            "snp_file": "FILL",
        },
        "model_config": {
            "model_type": "genome-local-net",
            "model_init_config": {
                "rb_do": 0.0,
                "channel_exp_base": 3,
                "kernel_width": "FILL",
                "first_kernel_expansion": "FILL",
                "l1": 0.0,
                "cutoff": 4096,
                "attention_inclusion_cutoff": 0,
            },
        },
        "tensor_broker_config": {
            "message_configs": [
                {
                    "name": "first_layer_tensor",
                    "layer_path": "input_modules.genotype.fc_0",
                    "cache_tensor": True,
                    "layer_cache_target": "input",
                }
            ]
        },
    }

    return base


def get_base_tabular_input_config() -> Dict[str, Any]:
    base = {
        "input_info": {
            "input_source": "FILL",
            "input_name": "eir_tabular",
            "input_type": "tabular",
        },
        "input_type_info": {
            "label_parsing_chunk_size": 20000,
            "input_cat_columns": ["FILL"],
            "input_con_columns": ["FILL"],
        },
        "model_config": {
            "model_type": "tabular",
            "model_init_config": {
                "fc_layer": True,
            },
        },
    }

    return base


def get_base_fusion_config(
    target_columns: list[str],
    output_groups: Optional[dict[str, list[str]]],
    model_type: str = "mlp-residual",
    model_size: "str" = "nano",
    output_head: str = "linear",
) -> Dict[str, Any]:

    fmsp = get_fusion_model_size_params(model_size=model_size)

    config_base = {
        "fc_do": 0.0,
        "fc_task_dim": fmsp.fc_dim,
        "layers": [fmsp.n_layers],
        "rb_do": 0.0,
        "stochastic_depth_p": 0.0,
    }

    if model_type == "mlp-residual":
        tb_base = generate_tb_base_config(
            num_layers=fmsp.n_layers,
            tb_block_frequency=fmsp.tb_block_frequency,
            output_head=output_head,
            target_columns=target_columns,
            output_groups=output_groups,
        )
        base = {
            "model_config": config_base,
            "model_type": "mlp-residual",
            "tensor_broker_config": tb_base,
        }

    elif model_type == "mgmoe":
        config_base["mg_num_experts"] = 8
        config_base["fc_task_dim"] = fmsp.fc_dim // 4
        tb_mgmoe = generate_tb_mgmoe_config(
            num_layers=fmsp.n_layers,
            tb_block_frequency=fmsp.tb_block_frequency,
            num_experts=8,
            output_head=output_head,
        )
        base = {
            "model_config": config_base,
            "model_type": "mgmoe",
            "tensor_broker_config": tb_mgmoe,
        }
    else:
        raise ValueError()

    return base


@dataclass
class FusionModelSizeParams:
    n_layers: int
    fc_dim: int
    tb_block_frequency: int


def get_fusion_model_size_params(model_size: str) -> FusionModelSizeParams:
    param_dict = {
        "nano": FusionModelSizeParams(n_layers=2, fc_dim=128, tb_block_frequency=1),
        "mini": FusionModelSizeParams(n_layers=4, fc_dim=256, tb_block_frequency=2),
        "small": FusionModelSizeParams(n_layers=8, fc_dim=512, tb_block_frequency=2),
        "medium": FusionModelSizeParams(n_layers=16, fc_dim=1024, tb_block_frequency=2),
        "large": FusionModelSizeParams(n_layers=24, fc_dim=2048, tb_block_frequency=4),
        "xlarge": FusionModelSizeParams(n_layers=32, fc_dim=4096, tb_block_frequency=4),
    }

    return param_dict[model_size]


def generate_tb_base_config(
    num_layers: int,
    tb_block_frequency: int,
    output_head: str,
    target_columns: list[str],
    output_groups: Optional[dict[str, list[str]]],
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = [
        {
            "name": "base_fusion_residual_block",
            "layer_path": "fusion_modules.computed.fusion_modules.fusion.0.0",
            "use_from_cache": ["first_layer_tensor"],
            "projection_type": "lcl+mlp_residual",
            "cache_fusion_type": "sum",
        }
    ]

    num_layers_adjusted = num_layers - 2

    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            message_configs.append(
                {
                    "name": f"{layer}_fusion_residual_block",
                    "layer_path": f"fusion_modules.computed.fusion_modules"
                    f".fusion.1.{layer}",
                    "use_from_cache": ["first_layer_tensor"],
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                }
            )

    if output_head == "linear":
        message_configs.append(
            {
                "name": "final_layer",
                "layer_path": "output_modules.eir_auto_gp.linear_layer",
                "use_from_cache": ["first_layer_tensor"],
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
            }
        )
    elif output_head == "mlp":
        for target_column in target_columns:
            message_configs.append(
                {
                    "name": f"final_layer_{target_column}",
                    "layer_path": f"output_modules.eir_auto_gp.multi_task_branches."
                    f"{target_column}.0.1",
                    "use_from_cache": ["first_layer_tensor"],
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                }
            )
    elif output_head == "shared_mlp_residual":
        assert output_groups is not None
        for group_name, group_columns in output_groups.items():
            message_configs.append(
                {
                    "name": f"final_layer_{group_name}",
                    "layer_path": f"output_modules.eir_auto_gp_{group_name}"
                    f".shared_branch",
                    "use_from_cache": ["first_layer_tensor"],
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                }
            )

    return {"message_configs": message_configs}


def generate_tb_mgmoe_config(
    num_layers: int,
    tb_block_frequency: int,
    num_experts: int,
    output_head: str,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    for expert in range(num_experts):
        message_configs.append(
            {
                "name": f"expert_{expert}_0_fusion_residual_block",
                "layer_path": f"fusion_modules.computed.expert_branches"
                f".expert_{expert}.0.0",
                "use_from_cache": ["first_layer_tensor"],
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
            }
        )

    num_layers_adjusted = num_layers - 2
    for layer in range(0, num_layers_adjusted + 1):
        if layer % tb_block_frequency == 0:
            for expert in range(num_experts):
                message_configs.append(
                    {
                        "name": f"expert_{expert}_{layer}_fusion_residual_block",
                        "layer_path": f"fusion_modules.computed.expert_branches"
                        f".expert_{expert}.1.{layer - 1}",
                        "use_from_cache": ["first_layer_tensor"],
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                    }
                )

    if output_head == "linear":
        message_configs.append(
            {
                "name": "final_layer",
                "layer_path": "output_modules.eir_auto_gp.linear_layer",
                "use_from_cache": ["first_layer_tensor"],
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
            }
        )

    return {"message_configs": message_configs}


@dataclass()
class SharedMLPResidualModelSizeParams:
    n_layers: int
    fc_dim: int


def get_shared_mlp_residual_model_size_params(
    model_size: str,
) -> SharedMLPResidualModelSizeParams:
    param_dict = {
        "nano": SharedMLPResidualModelSizeParams(n_layers=1, fc_dim=32),
        "mini": SharedMLPResidualModelSizeParams(n_layers=2, fc_dim=64),
        "small": SharedMLPResidualModelSizeParams(n_layers=2, fc_dim=128),
        "medium": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=256),
        "large": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=512),
        "xlarge": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=1024),
    }

    return param_dict[model_size]


def get_output_configs(
    output_groups: Optional[dict[str, list[str]]],
    output_cat_columns: list[str],
    output_con_columns: list[str],
    model_size: str,
    output_head: str = "mlp",
) -> list[dict[str, Any]]:

    shared_mlp_params = get_shared_mlp_residual_model_size_params(model_size=model_size)

    head_configs = {
        "mlp": {
            "model_type": "mlp_residual",
            "model_init_config": {
                "rb_do": 0.05,
                "fc_do": 0.05,
                "fc_task_dim": 128,
                "layers": [2],
                "stochastic_depth_p": 0.0,
                "final_layer_type": "linear",
            },
        },
        "linear": {
            "model_type": "linear",
        },
        "shared_mlp_residual": {
            "model_type": "shared_mlp_residual",
            "model_init_config": {
                "layers": [shared_mlp_params.n_layers],
                "fc_task_dim": shared_mlp_params.fc_dim,
                "rb_do": 0.00,
                "fc_do": 0.00,
                "stochastic_depth_p": 0.0,
            },
        },
    }

    if output_head not in head_configs:
        raise ValueError(f"Output head {output_head} not recognized.")

    head_config = head_configs[output_head]

    if output_head in ["linear", "mlp"]:
        return [
            create_base_config(
                head_config=head_config,
                output_cat_columns=output_cat_columns,
                output_con_columns=output_con_columns,
            )
        ]
    elif output_head == "shared_mlp_residual":
        return create_shared_mlp_configs(
            head_config=head_config,
            output_groups=output_groups,
            output_cat_columns=output_cat_columns,
            output_con_columns=output_con_columns,
        )
    else:
        raise ValueError(f"Output head {output_head} not recognized.")


def create_base_config(
    head_config: dict[str, Any],
    output_cat_columns: Sequence[str],
    output_con_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        "output_info": {
            "output_name": "eir_auto_gp",
            "output_source": "FILL",
            "output_type": "tabular",
        },
        "output_type_info": {
            "target_cat_columns": list(output_cat_columns),
            "target_con_columns": list(output_con_columns),
        },
        "model_config": head_config,
    }


def create_shared_mlp_configs(
    head_config: dict[str, Any],
    output_groups: Optional[dict[str, list[str]]],
    output_cat_columns: list[str],
    output_con_columns: list[str],
) -> list[dict[str, Any]]:
    if output_groups is None:
        raise ValueError("output_groups must be provided for shared_mlp_residual")

    parsed_configs = []

    for group_name, group_columns in output_groups.items():
        cur_cat_columns = [col for col in output_cat_columns if col in group_columns]
        cur_con_columns = [col for col in output_con_columns if col in group_columns]
        parsed_configs.append(
            {
                "output_info": {
                    "output_name": f"eir_auto_gp_{group_name}",
                    "output_source": "FILL",
                    "output_type": "tabular",
                },
                "output_type_info": {
                    "target_cat_columns": cur_cat_columns,
                    "target_con_columns": cur_con_columns,
                },
                "model_config": head_config,
            }
        )

    return parsed_configs


@dataclass(frozen=True)
class AggregateConfig:
    global_config: dict[str, Any]
    input_genotype_config: dict[str, Any]
    input_tabular_config: dict[str, Any]
    fusion_config: dict[str, Any]
    output_config: list[dict[str, Any]]


def get_aggregate_config(
    model_size: str,
    target_columns: list[str],
    output_cat_columns: list[str],
    output_con_columns: list[str],
    n_random_groups: int,
    output_groups: str = "random",
    output_head: str = "linear",
    fusion_type: str = "mlp-residual",
) -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config()
    input_tabular_config = get_base_tabular_input_config()

    if output_groups:
        logger.info(
            "Output groups detected. Using output groups and setting output"
            "head to shared residual MLP."
        )
        output_head = "shared_mlp_residual"
        output_groups = _build_output_groups(
            output_groups=output_groups,
            target_columns=target_columns,
            n_random_groups=n_random_groups,
            cat_columns=output_cat_columns,
            con_columns=output_con_columns,
        )
    else:
        output_groups = None

    fusion_config = get_base_fusion_config(
        model_type=fusion_type,
        model_size=model_size,
        output_head=output_head,
        target_columns=target_columns,
        output_groups=output_groups,
    )
    output_configs = get_output_configs(
        output_head=output_head,
        output_groups=output_groups,
        output_cat_columns=output_cat_columns,
        output_con_columns=output_con_columns,
        model_size=model_size,
    )

    return AggregateConfig(
        global_config=global_config,
        input_genotype_config=input_genotype_config,
        input_tabular_config=input_tabular_config,
        fusion_config=fusion_config,
        output_config=output_configs,
    )


def _build_output_groups(
    output_groups: str | int,
    target_columns: list[str],
    n_random_groups: int,
    cat_columns: Optional[list[str]],
    con_columns: Optional[list[str]],
) -> dict[str, list[str]]:
    if isinstance(output_groups, str):
        if output_groups.lower() == "random":
            return _create_random_groups(
                target_columns=target_columns,
                num_groups=n_random_groups,
            )
        elif output_groups.lower() == "semirandom":
            return _create_semirandom_groups(
                cat_columns=cat_columns or [],
                con_columns=con_columns or [],
                num_groups=n_random_groups,
            )
        else:
            with open(output_groups, "r") as file:
                return yaml.safe_load(file)
    elif isinstance(output_groups, int):
        return _create_random_groups(
            target_columns=target_columns,
            num_groups=output_groups,
        )
    else:
        raise ValueError(
            "output_groups must be either a string "
            "(file path, 'random', or 'semirandom') or an integer"
        )


def _create_random_groups(
    target_columns: Sequence[str],
    num_groups: int,
    seed: int = 42,
) -> Dict[str, list[str]]:
    target_columns = list(target_columns)

    if num_groups > len(target_columns):
        raise ValueError(
            "Number of groups must be less than or equal to the "
            "number of target columns."
        )

    random.seed(seed)

    random.shuffle(target_columns)
    groups = {f"group_{i + 1}": [] for i in range(num_groups)}
    for i, target in enumerate(target_columns):
        group_key = f"group_{(i % num_groups) + 1}"
        groups[group_key].append(target)

    random.seed()

    return groups


def _create_semirandom_groups(
    cat_columns: list[str],
    con_columns: list[str],
    num_groups: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    if not cat_columns and not con_columns:
        raise ValueError("At least one of cat_columns or con_columns must be provided")

    random.seed(seed)

    total_cols = len(cat_columns) + len(con_columns)
    if total_cols < num_groups:
        raise ValueError(
            "Number of groups must be less than or equal to the "
            "total number of target columns."
        )

    if cat_columns and con_columns:
        cat_groups = max(1, round(num_groups * len(cat_columns) / total_cols))
        con_groups = max(1, num_groups - cat_groups)
    else:
        cat_groups = num_groups if cat_columns else 0
        con_groups = num_groups if con_columns else 0

    groups = {}
    group_counter = 1

    if cat_columns:
        cat_cols = list(cat_columns)
        random.shuffle(cat_cols)
        for i in range(cat_groups):
            start_idx = i * len(cat_cols) // cat_groups
            end_idx = (i + 1) * len(cat_cols) // cat_groups
            groups[f"group_{group_counter}"] = cat_cols[start_idx:end_idx]
            group_counter += 1

    if con_columns:
        con_cols = list(con_columns)
        random.shuffle(con_cols)
        for i in range(con_groups):
            start_idx = i * len(con_cols) // con_groups
            end_idx = (i + 1) * len(con_cols) // con_groups
            groups[f"group_{group_counter}"] = con_cols[start_idx:end_idx]
            group_counter += 1

    random.seed()

    return groups
