import copy
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass()
class SharedMLPResidualModelSizeParams:
    n_layers: int
    fc_dim: int


def get_shared_mlp_residual_model_size_params(
    model_size: str,
) -> SharedMLPResidualModelSizeParams:
    param_dict = {
        "nano": SharedMLPResidualModelSizeParams(n_layers=2, fc_dim=32),
        "mini": SharedMLPResidualModelSizeParams(n_layers=2, fc_dim=64),
        "small": SharedMLPResidualModelSizeParams(n_layers=2, fc_dim=128),
        "medium": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=256),
        "large": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=512),
        "xlarge": SharedMLPResidualModelSizeParams(n_layers=4, fc_dim=1024),
    }

    return param_dict[model_size]


def get_output_configs(
    output_groups: dict[str, list[str]] | None,
    output_cat_columns: list[str],
    output_con_columns: list[str],
    model_size: str,
    output_head: str = "mlp",
    n_output_layers: int | None = None,
    output_dim: int | None = None,
    categorical_as_survival: bool = False,
    expert_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if n_output_layers is not None:
        assert output_dim is not None
        shared_mlp_params = SharedMLPResidualModelSizeParams(
            n_layers=n_output_layers,
            fc_dim=output_dim,
        )
    else:
        shared_mlp_params = get_shared_mlp_residual_model_size_params(
            model_size=model_size
        )

    head_configs = {
        "mlp": {
            "model_type": "mlp_residual",
            "model_init_config": {
                "rb_do": 0.10,
                "fc_do": 0.10,
                "fc_task_dim": 128,
                "layers": [2],
                "stochastic_depth_p": 0.10,
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
                "rb_do": 0.10,
                "fc_do": 0.10,
                "stochastic_depth_p": 0.10,
            },
        },
    }

    if output_head not in head_configs:
        raise ValueError(f"Output head {output_head} not recognized.")

    head_config = head_configs[output_head]

    if output_head in ["linear", "mlp"]:
        return _build_base_output_configs(
            head_config=head_config,
            output_cat_columns=output_cat_columns,
            output_con_columns=output_con_columns,
            categorical_as_survival=categorical_as_survival,
        )
    elif output_head == "shared_mlp_residual":
        return create_shared_mlp_config(
            head_config=head_config,
            output_groups=output_groups,
            output_cat_columns=output_cat_columns,
            output_con_columns=output_con_columns,
            categorical_as_survival=categorical_as_survival,
            use_expert_groups=expert_names is not None,
        )
    else:
        raise ValueError(f"Output head {output_head} not recognized.")


def _build_base_output_configs(
    head_config: dict[str, Any],
    output_cat_columns: Sequence[str],
    output_con_columns: Sequence[str],
    categorical_as_survival: bool = False,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    if categorical_as_survival and output_cat_columns:
        configs.append(
            create_survival_config(
                head_config=head_config,
                event_columns=list(output_cat_columns),
            )
        )
        if output_con_columns:
            configs.append(
                create_base_config(
                    head_config=head_config,
                    output_cat_columns=[],
                    output_con_columns=output_con_columns,
                    output_name="eir_auto_gp_tabular",
                )
            )
    else:
        configs.append(
            create_base_config(
                head_config=head_config,
                output_cat_columns=output_cat_columns,
                output_con_columns=output_con_columns,
            )
        )

    return configs


def create_base_config(
    head_config: dict[str, Any],
    output_cat_columns: Sequence[str],
    output_con_columns: Sequence[str],
    output_name: str = "eir_auto_gp",
) -> dict[str, Any]:
    return {
        "output_info": {
            "output_name": output_name,
            "output_source": "FILL",
            "output_type": "tabular",
        },
        "output_type_info": {
            "target_cat_columns": list(output_cat_columns),
            "target_con_columns": list(output_con_columns),
        },
        "model_config": head_config,
    }


def create_survival_config(
    head_config: dict[str, Any],
    event_columns: list[str],
    output_name: str = "eir_auto_gp",
    loss_function: str = "CoxPHLoss",
    num_durations: int = 0,
) -> dict[str, Any]:
    time_columns = [f"{col}_Time" for col in event_columns]
    return {
        "output_info": {
            "output_name": output_name,
            "output_source": "FILL",
            "output_type": "survival",
        },
        "output_type_info": {
            "event_columns": event_columns,
            "time_columns": time_columns,
            "loss_function": loss_function,
            "num_durations": num_durations,
        },
        "model_config": head_config,
    }


def create_shared_mlp_config(
    head_config: dict[str, Any],
    output_groups: dict[str, list[str]] | None,
    output_cat_columns: list[str],
    output_con_columns: list[str],
    categorical_as_survival: bool = False,
    use_expert_groups: bool = False,
) -> list[dict[str, Any]]:
    if output_groups is None:
        raise ValueError("output_groups must be provided for shared_mlp_residual")

    if categorical_as_survival:
        raise ValueError(
            "categorical_as_survival is not supported with grouped (batched) output."
        )

    all_output_columns = set(output_cat_columns) | set(output_con_columns)
    validated_groups: dict[str, list[str]] = {}

    for group_name, group_columns in output_groups.items():
        cur_columns = [col for col in group_columns if col in all_output_columns]
        if not cur_columns:
            unmodelled = [c for c in group_columns if c not in all_output_columns]
            raise ValueError(
                f"Output group '{group_name}' has no matching output columns. "
                f"Group defines {len(group_columns)} traits but none appear in "
                f"output_cat_columns or output_con_columns. "
                f"Unmatched traits (first 5): {unmodelled[:5]}. "
                f"Either remove this group from the output groups file or "
                f"add its traits to the model outputs."
            )
        validated_groups[group_name] = cur_columns

    final_head_config = copy.deepcopy(head_config)
    if use_expert_groups:
        final_head_config["model_init_config"]["expert_groups"] = validated_groups

    return [
        create_base_config(
            head_config=final_head_config,
            output_cat_columns=output_cat_columns,
            output_con_columns=output_con_columns,
        )
    ]


def _build_output_groups(
    output_groups: str | int,
    target_columns: list[str],
    n_random_groups: int,
    cat_columns: list[str] | None,
    con_columns: list[str] | None,
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
            with open(output_groups) as file:
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
) -> dict[str, list[str]]:
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
