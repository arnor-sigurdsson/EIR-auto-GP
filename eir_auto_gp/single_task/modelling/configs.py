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
        },
        "optimization": {
            "lr": 0.0002,
            "gradient_clipping": 1.0,
            "optimizer": "adabelief",
        },
        "lr_schedule": {
            "lr_plateau_patience": 4,
        },
        "training_control": {
            "early_stopping_buffer": "FILL",
            "early_stopping_patience": 6,
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
                "kernel_width": 16,
                "first_kernel_expansion": -4,
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


def get_base_fusion_config() -> Dict[str, Any]:
    base = {
        "model_config": {
            "fc_do": 0.1,
            "fc_task_dim": 512,
            "layers": [2],
            "rb_do": 0.1,
            "stochastic_depth_p": 0.1,
        },
        "model_type": "default",
    }
    return base


def get_base_output_config() -> Dict[str, Any]:
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
                "fc_task_dim": 512,
                "layers": [2],
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


def get_aggregate_config() -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config()
    input_tabular_config = get_base_tabular_input_config()
    fusion_config = get_base_fusion_config()
    output_config = get_base_output_config()

    return AggregateConfig(
        global_config,
        input_genotype_config,
        input_tabular_config,
        fusion_config,
        output_config,
    )
