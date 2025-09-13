from dataclasses import dataclass
from typing import Any, Dict

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


@dataclass
class STFusionModelSizeParams:
    n_layers: int
    fc_dim: int


def get_st_fusion_model_size_params(model_size: str) -> STFusionModelSizeParams:
    param_dict = {
        "nano": STFusionModelSizeParams(n_layers=1, fc_dim=64),
        "mini": STFusionModelSizeParams(n_layers=2, fc_dim=128),
        "small": STFusionModelSizeParams(n_layers=2, fc_dim=256),
        "medium": STFusionModelSizeParams(n_layers=2, fc_dim=512),
        "large": STFusionModelSizeParams(n_layers=4, fc_dim=768),
        "xlarge": STFusionModelSizeParams(n_layers=6, fc_dim=1024),
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
        },
        "optimization": {
            "lr": "FILL",
            "gradient_clipping": 1.0,
            "optimizer": "adabelief",
        },
        "lr_schedule": {
            "lr_plateau_patience": 4,
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
            "mixing_subtype": "cutmix-block",
            "na_augment_alpha": 1.0,
            "na_augment_beta": 9.0,
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


def get_base_fusion_config(model_size: str = "medium") -> Dict[str, Any]:
    fmsp = get_st_fusion_model_size_params(model_size)
    base = {
        "model_config": {
            "fc_do": 0.1,
            "fc_task_dim": fmsp.fc_dim,
            "layers": [fmsp.n_layers],
            "rb_do": 0.1,
            "stochastic_depth_p": 0.1,
        },
        "model_type": "default",
    }
    return base


def get_base_output_config(model_size: str = "medium") -> Dict[str, Any]:
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
    return base


@dataclass(frozen=True)
class AggregateConfig:
    global_config: Dict[str, Any]
    input_genotype_config: Dict[str, Any]
    input_tabular_config: Dict[str, Any]
    fusion_config: Dict[str, Any]
    output_config: Dict[str, Any]


def get_aggregate_config(model_size: str = "medium") -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config()
    input_tabular_config = get_base_tabular_input_config()
    fusion_config = get_base_fusion_config(model_size=model_size)
    output_config = get_base_output_config(model_size=model_size)

    return AggregateConfig(
        global_config,
        input_genotype_config,
        input_tabular_config,
        fusion_config,
        output_config,
    )
