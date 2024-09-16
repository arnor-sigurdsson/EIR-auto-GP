from dataclasses import dataclass
from typing import Any, Dict

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
    model_type: str = "mlp-residual",
    model_size: "str" = "nano",
) -> Dict[str, Any]:

    mp = get_model_size_params(model_size=model_size)

    config_base = {
        "fc_do": 0.0,
        "fc_task_dim": mp.fc_dim,
        "layers": [mp.n_layers],
        "rb_do": 0.0,
        "stochastic_depth_p": 0.0,
    }

    if model_type == "mlp-residual":
        tb_base = generate_tb_base_config(
            num_layers=mp.n_layers,
            tb_block_frequency=mp.tb_block_frequency,
        )
        base = {
            "model_config": config_base,
            "model_type": "mlp-residual",
            "tensor_broker_config": tb_base,
        }

    elif model_type == "mgmoe":
        config_base["mg_num_experts"] = 8
        config_base["fc_task_dim"] = mp.fc_dim // 4
        tb_mgmoe = generate_tb_mgmoe_config(
            num_layers=mp.n_layers,
            tb_block_frequency=mp.tb_block_frequency,
            num_experts=8,
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
class ModelSizeParams:
    n_layers: int
    fc_dim: int
    tb_block_frequency: int


def get_model_size_params(model_size: str) -> ModelSizeParams:
    param_dict = {
        "nano": ModelSizeParams(n_layers=2, fc_dim=128, tb_block_frequency=1),
        "small": ModelSizeParams(n_layers=8, fc_dim=512, tb_block_frequency=2),
        "medium": ModelSizeParams(n_layers=16, fc_dim=1024, tb_block_frequency=2),
        "large": ModelSizeParams(n_layers=24, fc_dim=2048, tb_block_frequency=4),
        "xlarge": ModelSizeParams(n_layers=32, fc_dim=4096, tb_block_frequency=4),
    }

    return param_dict[model_size]


def generate_tb_base_config(
    num_layers: int, tb_block_frequency: int
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = [
        {
            "name": "base_fusion_residual_block",
            "layer_path": "fusion_modules.computed.fusion_modules.fusion.0.0",
            "use_from_cache": ["first_layer_tensor"],
            "projection_type": "lcl_residual",
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
                    "projection_type": "lcl_residual",
                    "cache_fusion_type": "sum",
                }
            )

    message_configs.append(
        {
            "name": "final_layer",
            "layer_path": "output_modules.eir_auto_gp.linear_layer",
            "use_from_cache": ["first_layer_tensor"],
            "projection_type": "lcl_residual",
            "cache_fusion_type": "sum",
        }
    )

    return {"message_configs": message_configs}


def generate_tb_mgmoe_config(
    num_layers: int,
    tb_block_frequency: int,
    num_experts: int,
) -> dict[str, list[dict[str, Any]]]:
    message_configs: list[dict[str, Any]] = []

    for expert in range(num_experts):
        message_configs.append(
            {
                "name": f"expert_{expert}_0_fusion_residual_block",
                "layer_path": f"fusion_modules.computed.expert_branches"
                f".expert_{expert}.0.0",
                "use_from_cache": ["first_layer_tensor"],
                "projection_type": "lcl_residual",
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
                        "projection_type": "lcl_residual",
                        "cache_fusion_type": "sum",
                    }
                )

    message_configs.append(
        {
            "name": "final_layer",
            "layer_path": "output_modules.eir_auto_gp.linear_layer",
            "use_from_cache": ["first_layer_tensor"],
            "projection_type": "lcl_residual",
            "cache_fusion_type": "sum",
        }
    )

    return {"message_configs": message_configs}


def get_base_output_config(output_head: str = "mlp") -> Dict[str, Any]:
    if output_head == "mlp":
        head_config = {
            "model_type": "mlp_residual",
            "model_init_config": {
                "rb_do": 0.2,
                "fc_do": 0.2,
                "fc_task_dim": 512,
                "layers": [2],
                "stochastic_depth_p": 0.2,
                "final_layer_type": "linear",
            },
        }
    elif output_head == "linear":
        head_config = {
            "model_type": "linear",
        }
    else:
        raise ValueError(f"Output head {output_head} not recognized.")

    base = {
        "output_info": {
            "output_name": "eir_auto_gp",
            "output_source": "FILL",
            "output_type": "tabular",
        },
        "output_type_info": {
            "target_con_columns": ["FILL"],
            "target_cat_columns": ["FILL"],
        },
        "model_config": head_config,
    }
    return base


@dataclass(frozen=True)
class AggregateConfig:
    global_config: Dict[str, Any]
    input_genotype_config: Dict[str, Any]
    input_tabular_config: Dict[str, Any]
    fusion_config: Dict[str, Any]
    output_config: Dict[str, Any]


def get_aggregate_config(
    model_size: str,
    output_head: str = "linear",
    fusion_type: str = "mlp-residual",
) -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config()
    input_tabular_config = get_base_tabular_input_config()
    fusion_config = get_base_fusion_config(
        model_type=fusion_type, model_size=model_size
    )
    output_config = get_base_output_config(output_head=output_head)

    return AggregateConfig(
        global_config=global_config,
        input_genotype_config=input_genotype_config,
        input_tabular_config=input_tabular_config,
        fusion_config=fusion_config,
        output_config=output_config,
    )
