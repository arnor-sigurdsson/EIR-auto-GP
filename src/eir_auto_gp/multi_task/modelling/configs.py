from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from eir.models.input.array.models_locally_connected import (
    LCLResidualBlock,
    LCParameterSpec,
    calc_value_after_expansion,
    generate_lcl_residual_blocks_auto,
)

from eir_auto_gp.multi_task.modelling.output_configs import (
    _build_output_groups,
    get_output_configs,
)
from eir_auto_gp.multi_task.modelling.tensor_broker import (
    generate_tb_base_config,
    generate_tb_informed_moe_config,
    generate_tb_mgmoe_config,
)
from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


@dataclass
class ArchitectureParams:
    model_size: str
    output_groups: str
    n_random_output_groups: int
    n_fusion_layers: int | None
    fusion_dim: int | None
    skip_to_every_n_fusion_layers: int | None
    n_output_layers: int | None
    output_dim: int | str | None
    use_fc0_to_output_skips: bool
    use_fc0_to_fusion_skips: bool
    use_lcl_to_output_skips: bool | str
    fusion_model_type: str
    mgmoe_num_experts: int
    output_num_experts: int | None
    channel_exp_base: int = 3
    expert_groups_file: str | None = None

    @classmethod
    def from_modelling_config(cls, config: dict[str, Any]) -> "ArchitectureParams":
        return cls(
            model_size=config["model_size"],
            output_groups=config["output_groups"],
            n_random_output_groups=config["n_random_output_groups"],
            n_fusion_layers=config["n_fusion_layers"],
            fusion_dim=config["fusion_dim"],
            skip_to_every_n_fusion_layers=config["skip_to_every_n_fusion_layers"],
            n_output_layers=config["n_output_layers"],
            output_dim=config["output_dim"],
            use_fc0_to_output_skips=config["use_fc0_to_output_skips"],
            use_fc0_to_fusion_skips=config["use_fc0_to_fusion_skips"],
            use_lcl_to_output_skips=config["use_lcl_to_output_skips"],
            fusion_model_type=config["fusion_model_type"],
            mgmoe_num_experts=config["mgmoe_num_experts"],
            output_num_experts=config.get("output_num_experts"),
            channel_exp_base=config.get("channel_exp_base", 3),
            expert_groups_file=config.get("expert_groups_file"),
        )


@dataclass
class TabularSkipParams:
    enabled: bool = True
    drop_prob: float = 1.00
    cache_dropout_p: float = 0.50


@dataclass
class AdversarialParams:
    enabled: bool = True
    lambda_: float = 0.5
    hidden_dim: int = 64
    layers: list[int] | None = None


def get_base_global_config(
    adversarial_configs: list[dict[str, Any]] | None = None,
    manifold_mixup_layer_groups: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
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
            "optimizer": "adamw",
        },
        "lr_schedule": {
            "lr_plateau_patience": 8,
        },
        "training_control": {
            "early_stopping_buffer": "FILL",
            "early_stopping_patience": 10,
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

    if adversarial_configs:
        base["adversarial_training"] = {"adversarial_configs": adversarial_configs}

    if manifold_mixup_layer_groups:
        base["training_control"]["manifold_mixup_layer_groups"] = (
            manifold_mixup_layer_groups
        )

    return base


def _parse_expert_groups_file(
    path: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    with open(path) as f:
        data = yaml.safe_load(f)

    snp_groups = {name: group["snps"] for name, group in data.items()}
    output_groups = {name: group["traits"] for name, group in data.items()}
    return snp_groups, output_groups


def _write_snps_only_yaml(
    snp_groups: dict[str, list[str]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(snp_groups, f)
    return output_path


def get_base_input_genotype_config(
    use_fc0_to_output_skips: bool = True,
    use_fc0_to_fusion_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    expert_names: list[str] | None = None,
    channel_exp_base: int = 3,
) -> dict[str, Any]:
    if expert_names is not None:
        return _get_informed_moe_input_genotype_config(
            expert_names=expert_names,
            use_fc0_to_output_skips=use_fc0_to_output_skips,
            use_fc0_to_fusion_skips=use_fc0_to_fusion_skips,
            channel_exp_base=channel_exp_base,
        )

    message_configs = []

    if use_fc0_to_output_skips or use_fc0_to_fusion_skips:
        message_configs.append(
            {
                "name": "fc_0_output",
                "layer_path": "input_modules.genotype.fc_0",
                "cache_tensor": True,
                "layer_cache_target": "output",
            }
        )

    if use_lcl_to_output_skips:
        message_configs.append(
            {
                "name": "lcl_block_0_output",
                "layer_path": "input_modules.genotype.lcl_blocks.0",
                "cache_tensor": True,
                "layer_cache_target": "output",
            }
        )

    base = {
        "input_info": {
            "input_source": "FILL",
            "input_name": "genotype",
            "input_type": "omics",
        },
        "input_type_info": {
            "mixing_subtype": "mixup",
            "na_augment_alpha": 0.5,
            "na_augment_beta": 1.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 49.0,
            "snp_file": "FILL",
        },
        "model_config": {
            "model_type": "genome-local-net",
            "model_init_config": {
                "rb_do": 0.10,
                "stochastic_depth_p": 0.00,
                "channel_exp_base": channel_exp_base,
                "kernel_width": "FILL",
                "first_kernel_expansion": "FILL",
                "l1": 0.0,
                "cutoff": 4096,
                "attention_inclusion_cutoff": 0,
            },
        },
        "tensor_broker_config": {
            "message_configs": message_configs,
        },
    }

    return base


def _get_informed_moe_input_genotype_config(
    expert_names: list[str],
    use_fc0_to_output_skips: bool = True,
    use_fc0_to_fusion_skips: bool = True,
    channel_exp_base: int = 3,
) -> dict[str, Any]:
    message_configs = []

    base_cutoff = 4096
    cutoff_per_expert = base_cutoff // len(expert_names)
    nearest_power_of_2 = 2 ** (cutoff_per_expert - 1).bit_length()
    adjusted_cutoff = max(256, nearest_power_of_2)

    needs_fc0_cache = use_fc0_to_output_skips or use_fc0_to_fusion_skips
    for name in expert_names:
        if needs_fc0_cache:
            message_configs.append(
                {
                    "name": f"expert_{name}_fc_0",
                    "layer_path": f"input_modules.genotype.expert_branches.{name}.fc_0",
                    "cache_tensor": True,
                    "layer_cache_target": "output",
                }
            )

    return {
        "input_info": {
            "input_source": "FILL",
            "input_name": "genotype",
            "input_type": "omics",
        },
        "input_type_info": {
            "mixing_subtype": "mixup",
            "na_augment_alpha": 0.5,
            "na_augment_beta": 1.0,
            "shuffle_augment_alpha": 1.0,
            "shuffle_augment_beta": 49.0,
            "snp_file": "FILL",
        },
        "model_config": {
            "model_type": "genome-local-net-informed-moe",
            "model_init_config": {
                "rb_do": 0.10,
                "stochastic_depth_p": 0.00,
                "channel_exp_base": channel_exp_base,
                "kernel_width": "FILL",
                "first_kernel_expansion": "FILL",
                "l1": 0.0,
                "cutoff": adjusted_cutoff,
                "attention_inclusion_cutoff": 0,
            },
        },
        "tensor_broker_config": {
            "message_configs": message_configs,
        },
    }


def get_num_lcl_blocks(
    n_snps: int,
    kernel_width: int,
    first_kernel_expansion: int,
    channel_exp_base: int = 3,
    rb_do: float = 0.10,
    stochastic_depth_p: float = 0.00,
    cutoff: int = 4096,
) -> int:
    in_features = n_snps * 3

    fc_0_kernel_size = calc_value_after_expansion(
        base=kernel_width,
        expansion=first_kernel_expansion,
    )
    fc_0_out_feature_sets = calc_value_after_expansion(
        base=2**channel_exp_base,
        expansion=1,
    )

    num_chunks = in_features // fc_0_kernel_size
    fc_0_out_features = num_chunks * fc_0_out_feature_sets

    spec = LCParameterSpec(
        in_features=fc_0_out_features,
        kernel_width=kernel_width,
        channel_exp_base=channel_exp_base,
        dropout_p=rb_do,
        stochastic_depth_p=stochastic_depth_p,
        cutoff=cutoff,
        attention_inclusion_cutoff=0,
        direction="down",
    )

    blocks = generate_lcl_residual_blocks_auto(lcl_parameter_spec=spec)

    num_blocks = sum(1 for m in blocks if isinstance(m, LCLResidualBlock))
    return num_blocks


def get_base_tabular_input_config(
    cache_for_output_heads: bool = True,
    drop_prob: float = 0.5,
) -> dict[str, Any]:
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
                "drop_prob": drop_prob,
                "layers": [2],
                "fc_dim": "auto",
            },
        },
    }

    if cache_for_output_heads:
        base["tensor_broker_config"] = {
            "message_configs": [
                {
                    "name": "tabular_output",
                    "layer_path": "input_modules.eir_tabular.mlp_blocks",
                    "cache_tensor": True,
                    "layer_cache_target": "output",
                }
            ]
        }

    return base


def get_base_fusion_config(
    target_columns: list[str],
    output_groups: dict[str, list[str]] | None,
    model_type: str = "mlp-residual-sum",
    model_size: str = "nano",
    output_head: str = "linear",
    n_fusion_layers: int | None = None,
    fusion_dim: int | None = None,
    skip_to_every_n_fusion_layers: int | None = None,
    use_fc0_to_output_skips: bool = True,
    use_fc0_to_fusion_skips: bool = True,
    use_lcl_to_output_skips: bool | str = False,
    include_tabular: bool = True,
    tabular_cache_dropout_p: float = 0.00,
    mgmoe_num_experts: int = 8,
    output_num_experts: int | None = None,
    expert_names: list[str] | None = None,
) -> dict[str, Any]:
    if n_fusion_layers is not None:
        assert fusion_dim is not None
        assert skip_to_every_n_fusion_layers is not None
        fmsp = FusionModelSizeParams(
            n_layers=n_fusion_layers,
            fc_dim=fusion_dim,
            tb_block_frequency=skip_to_every_n_fusion_layers,
        )
    else:
        fmsp = get_fusion_model_size_params(model_size=model_size)

    config_base = {
        "fc_do": 0.10,
        "fc_task_dim": fmsp.fc_dim,
        "layers": [fmsp.n_layers],
        "rb_do": 0.10,
        "stochastic_depth_p": 0.10,
    }

    # note early exit from this function if expert names are passed in
    if expert_names is not None:
        tb_config = generate_tb_informed_moe_config(
            expert_names=expert_names,
            include_tabular=include_tabular,
            tabular_cache_dropout_p=tabular_cache_dropout_p,
            output_num_experts=output_num_experts,
            use_fc0_output_skips=use_fc0_to_output_skips,
            num_fusion_layers=fmsp.n_layers if use_fc0_to_fusion_skips else None,
            tb_block_frequency=fmsp.tb_block_frequency,
        )

        if model_type == "mgmoe":
            config_base["mg_num_experts"] = mgmoe_num_experts
            config_base["fc_task_dim"] = fmsp.fc_dim // 4

        return {
            "model_config": config_base,
            "model_type": model_type,
            "tensor_broker_config": tb_config,
        }

    tb_kwargs = {
        "num_layers": fmsp.n_layers,
        "tb_block_frequency": fmsp.tb_block_frequency,
        "output_head": output_head,
        "target_columns": target_columns,
        "output_groups": output_groups,
        "use_fc0_to_output_skips": use_fc0_to_output_skips,
        "use_fc0_to_fusion_skips": use_fc0_to_fusion_skips,
        "use_lcl_to_output_skips": use_lcl_to_output_skips,
        "include_tabular": include_tabular,
        "tabular_cache_dropout_p": tabular_cache_dropout_p,
        "output_num_experts": output_num_experts,
    }

    if model_type in ("mlp-residual", "mlp-residual-sum"):
        tb_config = generate_tb_base_config(**tb_kwargs)
        base = {
            "model_config": config_base,
            "model_type": model_type,
            "tensor_broker_config": tb_config,
        }

    elif model_type == "mgmoe":
        config_base["mg_num_experts"] = mgmoe_num_experts
        config_base["fc_task_dim"] = fmsp.fc_dim // 4
        tb_config = generate_tb_mgmoe_config(
            num_experts=mgmoe_num_experts,
            **tb_kwargs,
        )
        base = {
            "model_config": config_base,
            "model_type": "mgmoe",
            "tensor_broker_config": tb_config,
        }
    else:
        raise ValueError(f"Unknown fusion model type: {model_type!r}")

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


def _get_adversarial_configs(
    output_groups: dict[str, list[str]],
    adversarial_lambda: float = 0.1,
    adversarial_hidden_dim: int = 256,
    adversarial_layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    if adversarial_layers is None:
        adversarial_layers = [2]

    adversarial_configs = []

    for group_name in output_groups.keys():
        adversarial_configs.append(
            {
                "name": f"{group_name}_vs_covariates",
                "enabled": True,
                "embedding_layer_path": f"output_modules."
                f"eir_auto_gp_{group_name}.shared_branch.1",
                "target_layer_path": "input_modules.eir_tabular.layer",
                "lambda_adv": adversarial_lambda,
                "warmup_steps": 5000,
                "fc_dim": adversarial_hidden_dim,
                "layers": adversarial_layers,
                "projection_type": "mlp_residual",
                "dropout_p": 0.1,
                "target_cache_target": "input",
            }
        )

    return adversarial_configs


def _get_manifold_mixup_layer_groups_informed_moe(
    expert_names: list[str],
) -> dict[str, list[str]]:
    input_experts = [
        f"input_modules.genotype.expert_branches.{name}.lcl_blocks"
        for name in expert_names
    ]

    output_entry = [
        f"output_modules.eir_auto_gp_{name}.shared_branch.0.0.0"
        for name in expert_names
    ]

    output_deep = [
        f"output_modules.eir_auto_gp_{name}.shared_branch.0.1" for name in expert_names
    ]

    return {
        "input_experts": input_experts,
        "output_entry": output_entry,
        "output_deep": output_deep,
    }


def _get_manifold_mixup_layer_groups_base(
    output_groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    group_names = list(output_groups.keys())

    output_entry = [
        f"output_modules.eir_auto_gp_{name}.shared_branch.0.0.0" for name in group_names
    ]

    output_deep = [
        f"output_modules.eir_auto_gp_{name}.shared_branch.0.1" for name in group_names
    ]

    return {
        "input_encoder": ["input_modules.genotype.lcl_blocks"],
        "output_entry": output_entry,
        "output_deep": output_deep,
    }


@dataclass(frozen=True)
class AggregateConfig:
    global_config: dict[str, Any]
    input_genotype_config: dict[str, Any]
    input_tabular_config: dict[str, Any]
    fusion_config: dict[str, Any]
    output_config: list[dict[str, Any]]
    expert_snp_groups_file: str | None = None


def get_aggregate_config(
    arch_params: ArchitectureParams,
    target_columns: list[str],
    output_cat_columns: list[str],
    output_con_columns: list[str],
    tabular_params: TabularSkipParams | None = None,
    adversarial_params: AdversarialParams | None = None,
    categorical_as_survival: bool = False,
) -> AggregateConfig:
    if tabular_params is None:
        tabular_params = TabularSkipParams()
    if adversarial_params is None:
        adversarial_params = AdversarialParams()

    expert_names: list[str] | None = None
    expert_snp_groups_file: str | None = None

    if arch_params.expert_groups_file:
        if arch_params.output_groups:
            raise ValueError(
                "Cannot specify both 'expert_groups_file' and 'output_groups'. "
                "The expert groups file defines both input SNP groups and output "
                "groups — passing a separate output_groups would create mismatched "
                "tensor broker wiring. Remove the 'output_groups' argument."
            )

        snp_groups, expert_output_groups = _parse_expert_groups_file(
            path=arch_params.expert_groups_file,
        )
        expert_names = list(snp_groups.keys())

        snps_only_path = (
            Path(arch_params.expert_groups_file).parent / "expert_snps_only.yaml"
        )
        _write_snps_only_yaml(
            snp_groups=snp_groups,
            output_path=snps_only_path,
        )
        expert_snp_groups_file = str(snps_only_path)

        output_head = "shared_mlp_residual"
        built_output_groups = expert_output_groups
        logger.info(
            "Expert groups file detected with %d groups: %s. "
            "Using informed MoE encoder and shared_mlp_residual output head.",
            len(expert_names),
            expert_names,
        )
    elif arch_params.output_groups:
        logger.info(
            "Output groups detected. Using output groups and setting output"
            "head to shared residual MLP."
        )
        output_head = "shared_mlp_residual"
        built_output_groups = _build_output_groups(
            output_groups=arch_params.output_groups,
            target_columns=target_columns,
            n_random_groups=arch_params.n_random_output_groups,
            cat_columns=output_cat_columns,
            con_columns=output_con_columns,
        )
    else:
        output_head = "linear"
        built_output_groups = None

    adversarial_configs = None
    if (
        tabular_params.enabled
        and adversarial_params.enabled
        and built_output_groups is not None
    ):
        adversarial_configs = _get_adversarial_configs(
            output_groups=built_output_groups,
            adversarial_lambda=adversarial_params.lambda_,
            adversarial_hidden_dim=adversarial_params.hidden_dim,
            adversarial_layers=adversarial_params.layers,
        )

    manifold_mixup_layer_groups = None
    if expert_names is not None:
        manifold_mixup_layer_groups = _get_manifold_mixup_layer_groups_informed_moe(
            expert_names=expert_names,
        )
    elif built_output_groups is not None:
        manifold_mixup_layer_groups = _get_manifold_mixup_layer_groups_base(
            output_groups=built_output_groups,
        )

    global_config = get_base_global_config(
        adversarial_configs=adversarial_configs,
        manifold_mixup_layer_groups=manifold_mixup_layer_groups,
    )
    input_genotype_config = get_base_input_genotype_config(
        use_fc0_to_output_skips=arch_params.use_fc0_to_output_skips,
        use_fc0_to_fusion_skips=arch_params.use_fc0_to_fusion_skips,
        use_lcl_to_output_skips=arch_params.use_lcl_to_output_skips,
        expert_names=expert_names,
        channel_exp_base=arch_params.channel_exp_base,
    )
    input_tabular_config = get_base_tabular_input_config(
        cache_for_output_heads=tabular_params.enabled,
        drop_prob=tabular_params.drop_prob,
    )

    fusion_config = get_base_fusion_config(
        model_type=arch_params.fusion_model_type,
        model_size=arch_params.model_size,
        output_head=output_head,
        target_columns=target_columns,
        output_groups=built_output_groups,
        n_fusion_layers=arch_params.n_fusion_layers,
        fusion_dim=arch_params.fusion_dim,
        skip_to_every_n_fusion_layers=arch_params.skip_to_every_n_fusion_layers,
        use_fc0_to_output_skips=arch_params.use_fc0_to_output_skips,
        use_fc0_to_fusion_skips=arch_params.use_fc0_to_fusion_skips,
        use_lcl_to_output_skips=arch_params.use_lcl_to_output_skips,
        include_tabular=tabular_params.enabled,
        tabular_cache_dropout_p=tabular_params.cache_dropout_p,
        mgmoe_num_experts=arch_params.mgmoe_num_experts,
        output_num_experts=arch_params.output_num_experts,
        expert_names=expert_names,
    )
    output_configs = get_output_configs(
        output_head=output_head,
        output_groups=built_output_groups,
        output_cat_columns=output_cat_columns,
        output_con_columns=output_con_columns,
        model_size=arch_params.model_size,
        n_output_layers=arch_params.n_output_layers,
        output_dim=arch_params.output_dim,
        output_num_experts=arch_params.output_num_experts,
        categorical_as_survival=categorical_as_survival,
    )

    return AggregateConfig(
        global_config=global_config,
        input_genotype_config=input_genotype_config,
        input_tabular_config=input_tabular_config,
        fusion_config=fusion_config,
        output_config=output_configs,
        expert_snp_groups_file=expert_snp_groups_file,
    )
