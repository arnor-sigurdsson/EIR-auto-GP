from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from eir_auto_gp.utils.utils import get_logger

logger = get_logger(name=__name__)


@dataclass
class CustomConfig:
    """Advanced configuration for model architecture and training.

    These parameters can be set via a YAML file passed with ``--custom_config``.
    When not specified, defaults are used.

    :param use_lcl_to_output_skips:
        Controls LCL block skip connections to output heads.
        When ``True``, fc_1 and fc_2 intermediate features are cached
        and sent to output heads alongside fc_0_output. When ``"fc_1_only"``,
        only fc_1 is used (more parameter-efficient). When ``False``, output
        heads receive only fc_0_output.

    :param use_lcl_fusion_skips:
        When enabled, fusion layers receive multi-scale LCL block outputs
        in addition to fc_0_output.

    :param weighted_sampling:
        Controls weighted sampling during training.
        ``"auto"`` enables it only when there are categorical targets but
        no continuous targets. ``"true"``/``"false"`` force it on/off.

    :param optimize_model:
        Enables model optimizations including ``torch.compile``
        and mixed precision (bf16) training when supported by hardware.

    :param modelling_data_format:
        Storage format for data during modelling.
        ``"disk"`` reads from disk (lower memory), ``"memory"`` loads all
        data into RAM (faster), ``"auto"`` decides based on dataset size.

    :param n_fusion_layers:
        Number of fusion layers. When set, all granular architecture
        parameters (``fusion_dim``, ``skip_to_every_n_fusion_layers``,
        ``n_output_layers``, ``output_dim``) must also be specified.
        Cannot be combined with ``model_size``.

    :param fusion_dim:
        Dimension of fusion layers.

    :param skip_to_every_n_fusion_layers:
        Tensor broker skip connection frequency to fusion layers.

    :param n_output_layers:
        Number of layers in shared MLP residual output heads.

    :param output_dim:
        Dimension of shared MLP residual output head layers.

    :param batch_size:
        Training batch size. When ``None``, automatically determined
        based on dataset size.

    :param use_fc0_skips:
        When ``True``, the fc_0 layer output (first LCL layer) is cached
        and sent via tensor broker to fusion layers and output heads.
        This is the primary skip connection in the architecture.
        When ``False``, no fc_0 features are sent — fusion and output
        heads rely only on the normal forward-pass features
        (plus any LCL block skips if enabled separately).

    :param fusion_model_type:
        Fusion module architecture type.
        ``"mlp-residual-sum"`` uses a standard MLP-residual fusion.
        ``"mgmoe"`` uses Multi-Gate Mixture of Experts fusion.

    :param mgmoe_num_experts:
        Number of experts when ``fusion_model_type`` is ``"mgmoe"``.
        Ignored for other fusion model types.
    """

    use_lcl_to_output_skips: bool | str = True
    use_lcl_fusion_skips: bool = True
    use_fc0_skips: bool = True
    weighted_sampling: str = "auto"
    optimize_model: bool = False
    modelling_data_format: str = "disk"
    n_fusion_layers: int | None = None
    fusion_dim: int | None = None
    skip_to_every_n_fusion_layers: int | None = None
    n_output_layers: int | None = None
    output_dim: int | None = None
    batch_size: int | None = None
    fusion_model_type: str = "mlp-residual-sum"
    mgmoe_num_experts: int = 8

    def __post_init__(self) -> None:
        valid_skip_values = (True, False, "fc_1_only")
        if self.use_lcl_to_output_skips not in valid_skip_values:
            raise ValueError(
                f"use_lcl_to_output_skips must be one of {valid_skip_values}, "
                f"got {self.use_lcl_to_output_skips!r}"
            )

        valid_sampling = ("auto", "true", "false")
        if self.weighted_sampling not in valid_sampling:
            raise ValueError(
                f"weighted_sampling must be one of {valid_sampling}, "
                f"got {self.weighted_sampling!r}"
            )

        valid_formats = ("disk", "memory", "auto")
        if self.modelling_data_format not in valid_formats:
            raise ValueError(
                f"modelling_data_format must be one of {valid_formats}, "
                f"got {self.modelling_data_format!r}"
            )

        valid_fusion_types = ("mlp-residual-sum", "mgmoe")
        if self.fusion_model_type not in valid_fusion_types:
            raise ValueError(
                f"fusion_model_type must be one of {valid_fusion_types}, "
                f"got {self.fusion_model_type!r}"
            )

        if self.mgmoe_num_experts < 1:
            raise ValueError(
                f"mgmoe_num_experts must be >= 1, got {self.mgmoe_num_experts}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CustomConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Custom config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        valid_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ValueError(
                f"Unknown keys in custom config: {unknown}. "
                f"Allowed keys: {sorted(valid_keys)}"
            )

        config = cls(**data)

        for key in data:
            logger.info("Custom config: %s = %s", key, data[key])

        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
