"""PyTorch MLP policy/value model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:  # pragma: no cover - torch is optional for non-ML usage
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for highsociety.ml.models.mlp") from exc


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for the MLP policy/value model."""

    input_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        for size in self.hidden_sizes:
            if size <= 0:
                raise ValueError("hidden layer sizes must be positive")

    def to_dict(self) -> dict[str, object]:
        """Serialize configuration to a dict."""
        return {
            "input_dim": self.input_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "MLPConfig":
        """Deserialize configuration from a dict."""
        hidden_sizes = data.get("hidden_sizes", (128, 128))
        if isinstance(hidden_sizes, (list, tuple)):
            hidden_tuple = tuple(int(size) for size in hidden_sizes)
        else:
            hidden_tuple = (128, 128)
        return MLPConfig(
            input_dim=int(data.get("input_dim", 0)),
            action_dim=int(data.get("action_dim", 0)),
            hidden_sizes=hidden_tuple,
            activation=str(data.get("activation", "relu")),
            dropout=float(data.get("dropout", 0.0)),
        )


class MLPPolicyValue(nn.Module):
    """MLP with separate policy and value heads."""

    def __init__(self, config: MLPConfig) -> None:
        """Initialize the model from a configuration."""
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        activation_cls = _ACTIVATIONS[config.activation]
        last_dim = config.input_dim
        for size in config.hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(activation_cls())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            last_dim = size
        self.body = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, config.action_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, features: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
        """Compute action logits and value estimates."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        hidden = self.body(features)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def load_state_dict_strict(self, state_dict: dict[str, "torch.Tensor"]) -> None:
        """Load model weights with strict matching."""
        self.load_state_dict(state_dict, strict=True)


def build_mlp(
    input_dim: int,
    action_dim: int,
    hidden_sizes: Iterable[int] = (128, 128),
    activation: str = "relu",
    dropout: float = 0.0,
) -> MLPPolicyValue:
    """Build an MLPPolicyValue model from explicit arguments."""
    config = MLPConfig(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(hidden_sizes),
        activation=activation,
        dropout=dropout,
    )
    return MLPPolicyValue(config)
