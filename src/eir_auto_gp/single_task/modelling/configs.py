from dataclasses import dataclass
from typing import Any

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


@dataclass
class STFusionModelSizeParams:
    n_layers: int
    fc_dim: int
    tb_block_frequency: int


def get_st_fusion_model_size_params(model_size: str) -> STFusionModelSizeParams:
    param_dict = {
        "nano": STFusionModelSizeParams(n_layers=1, fc_dim=64, tb_block_frequency=1),
        "mini": STFusionModelSizeParams(n_layers=2, fc_dim=128, tb_block_frequency=1),
        "small": STFusionModelSizeParams(n_layers=2, fc_dim=256, tb_block_frequency=1),
        "medium": STFusionModelSizeParams(n_layers=2, fc_dim=512, tb_block_frequency=1),
        "large": STFusionModelSizeParams(n_layers=4, fc_dim=768, tb_block_frequency=2),
        "xlarge": STFusionModelSizeParams(
            n_layers=6, fc_dim=1024, tb_block_frequency=2
        ),
    }
    return param_dict[model_size]


@dataclass
class STOutputHeadSizeParams:
    n_layers: int
    fc_dim: int


def get_st_output_head_size_params(model_size: str) -> STOutputHeadSizeParams:
    param_dict = {
        "nano": STOutputHeadSizeParams(n_layers=1, fc_dim=32),
        "mini": STOutputHeadSizeParams(n_layers=1, fc_dim=64),
        "small": STOutputHeadSizeParams(n_layers=2, fc_dim=128),
        "medium": STOutputHeadSizeParams(n_layers=2, fc_dim=512),
        "large": STOutputHeadSizeParams(n_layers=2, fc_dim=512),
        "xlarge": STOutputHeadSizeParams(n_layers=4, fc_dim=768),
    }
    return param_dict[model_size]


def get_base_global_config() -> dict[str, Any]:
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
            "cat_averaging_metrics": ["roc-auc-macro", "ap-macro"],
        },
    }
    return base


def get_base_input_genotype_config(use_tensor_broker: bool = False) -> dict[str, Any]:
    base = {
        "input_info": {
            "input_source": "FILL",
            "input_name": "genotype",
            "input_type": "omics",
        },
        "input_type_info": {
            "mixing_subtype": "cutmix-block",
            "na_augment_alpha": 0.6,
            "na_augment_beta": 2.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 49.0,
            "snp_file": "FILL",
        },
        "model_config": {
            "model_type": "genome-local-net",
            "model_init_config": {
                "rb_do": 0.1,
                "channel_exp_base": 2,
                "kernel_width": "FILL",
                "first_kernel_expansion": "FILL",
                "l1": 0.0,
                "cutoff": 4096,
                "attention_inclusion_cutoff": 0,
            },
        },
    }

    if use_tensor_broker:
        base["tensor_broker_config"] = {
            "message_configs": [
                {
                    "name": "first_layer_tensor",
                    "layer_path": "input_modules.genotype.fc_0",
                    "cache_tensor": True,
                    "layer_cache_target": "input",
                }
            ]
        }

    return base


def get_base_tabular_input_config() -> dict[str, Any]:
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
    model_size: str = "medium", use_tensor_broker: bool = False
) -> dict[str, Any]:
    fmsp = get_st_fusion_model_size_params(model_size)

    base = {
        "model_config": {
            "fc_do": 0.1,
            "fc_task_dim": fmsp.fc_dim,
            "layers": [fmsp.n_layers],
            "rb_do": 0.1,
            "stochastic_depth_p": 0.1,
        },
        "model_type": "mlp-residual",
    }

    if use_tensor_broker:
        message_configs = [
            {
                "name": "base_fusion_residual_block",
                "layer_path": "fusion_modules.computed.fusion_modules.fusion.0.0",
                "use_from_cache": ["first_layer_tensor"],
                "projection_type": "lcl+mlp_residual",
                "cache_fusion_type": "sum",
            }
        ]

        num_layers_adjusted = fmsp.n_layers - 2 if fmsp.n_layers > 1 else 0
        for layer in range(0, num_layers_adjusted + 1):
            if layer % fmsp.tb_block_frequency == 0:
                message_configs.append(
                    {
                        "name": f"{layer}_fusion_residual_block",
                        "layer_path": f"fusion_modules.computed.fusion_modules."
                        f"fusion.1.{layer}",
                        "use_from_cache": ["first_layer_tensor"],
                        "projection_type": "lcl+mlp_residual",
                        "cache_fusion_type": "sum",
                    }
                )

        base["tensor_broker_config"] = {"message_configs": message_configs}

    return base


def get_base_output_config(
    model_size: str = "medium",
    target_columns: list[str] | None = None,
    use_tensor_broker: bool = False,
) -> dict[str, Any]:
    ohsp = get_st_output_head_size_params(model_size)

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
        "model_config": {
            "model_type": "mlp_residual",
            "model_init_config": {
                "rb_do": 0.2,
                "fc_do": 0.2,
                "fc_task_dim": ohsp.fc_dim,
                "layers": [ohsp.n_layers],
                "stochastic_depth_p": 0.2,
                "final_layer_type": "linear",
            },
        },
    }

    if use_tensor_broker and target_columns:
        message_configs = []
        for target_column in target_columns:
            message_configs.append(
                {
                    "name": f"final_layer_{target_column}",
                    "layer_path": f"output_modules.eir_auto_gp."
                    f"multi_task_branches.{target_column}.2.final",
                    "use_from_cache": ["first_layer_tensor"],
                    "projection_type": "lcl+mlp_residual",
                    "cache_fusion_type": "sum",
                }
            )
        base["tensor_broker_config"] = {"message_configs": message_configs}

    return base


@dataclass(frozen=True)
class AggregateConfig:
    global_config: dict[str, Any]
    input_genotype_config: dict[str, Any]
    input_tabular_config: dict[str, Any]
    fusion_config: dict[str, Any]
    output_config: dict[str, Any]


def get_aggregate_config(
    model_size: str = "medium",
    target_columns: list[str] | None = None,
    use_tensor_broker: bool = False,
) -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config(
        use_tensor_broker=use_tensor_broker
    )
    input_tabular_config = get_base_tabular_input_config()
    fusion_config = get_base_fusion_config(
        model_size=model_size, use_tensor_broker=use_tensor_broker
    )
    output_config = get_base_output_config(
        model_size=model_size,
        target_columns=target_columns,
        use_tensor_broker=use_tensor_broker,
    )

    return AggregateConfig(
        global_config,
        input_genotype_config,
        input_tabular_config,
        fusion_config,
        output_config,
    )
