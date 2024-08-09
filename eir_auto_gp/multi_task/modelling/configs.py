from dataclasses import dataclass
from typing import Any, Dict

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


def get_base_global_config() -> Dict[str, Any]:
    base = {
        "output_folder": "FILL",
        "checkpoint_interval": "FILL",
        "batch_size": "FILL",
        "sample_interval": "FILL",
        "save_evaluation_sample_results": False,
        "lr": 0.00001,
        "lr_plateau_patience": 4,
        "gradient_clipping": 1.0,
        "valid_size": "FILL",
        "n_epochs": 5000,
        "dataloader_workers": "FILL",
        "device": "FILL",
        "early_stopping_buffer": "FILL",
        "early_stopping_patience": 6,
        "compute_attributions": False,
        "attribution_background_samples": 64,
        "max_attributions_per_class": 1000,
        "attributions_every_sample_factor": 5,
        "no_pbar": False,
        "mixing_alpha": "FILL",
        "optimizer": "adabelief",
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
            "na_augment_beta": 19.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 49.0,
            "snp_file": "FILL",
        },
        "model_config": {
            "model_type": "genome-local-net",
            "model_init_config": {
                "rb_do": 0.1,
                "channel_exp_base": 3,
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


def get_base_fusion_config(model_type: str = "mlp-residual") -> Dict[str, Any]:
    if model_type == "mlp-residual":
        base = {
            "model_config": {
                "fc_do": 0.1,
                "fc_task_dim": 512,
                "layers": [4],
                "rb_do": 0.1,
                "stochastic_depth_p": 0.1,
            },
            "model_type": "mlp-residual",
        }
    elif model_type == "mgmoe":
        base = {
            "model_config": {
                "fc_do": 0.1,
                "fc_task_dim": 512,
                "layers": [4],
                "rb_do": 0.1,
                "stochastic_depth_p": 0.1,
                "mg_num_experts": 8,
            },
            "model_type": "mgmoe",
        }
    else:
        raise ValueError()

    return base


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
    output_head: str = "mlp",
    fusion_type: str = "mlp-residual",
) -> AggregateConfig:
    global_config = get_base_global_config()
    input_genotype_config = get_base_input_genotype_config()
    input_tabular_config = get_base_tabular_input_config()
    fusion_config = get_base_fusion_config(model_type=fusion_type)
    output_config = get_base_output_config(output_head=output_head)

    return AggregateConfig(
        global_config=global_config,
        input_genotype_config=input_genotype_config,
        input_tabular_config=input_tabular_config,
        fusion_config=fusion_config,
        output_config=output_config,
    )
