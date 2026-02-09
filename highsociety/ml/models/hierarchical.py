"""Hierarchical policy/value model with action decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - torch is optional for non-ML usage
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for highsociety.ml.models.hierarchical") from exc


# Money card values in sorted order (11 cards total)
MONEY_CARD_VALUES: tuple[int, ...] = (1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000)

# Number of possession values (1-10)
NUM_POSSESSIONS = 10


@dataclass(frozen=True)
class HierarchicalConfig:
    """Configuration for the hierarchical policy/value model."""

    input_dim: int = 34
    trunk_sizes: tuple[int, ...] = (256, 256)
    card_head_hidden: int = 128
    discard_head_hidden: int = 64
    activation: str = "relu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not self.trunk_sizes:
            raise ValueError("trunk_sizes cannot be empty")
        for size in self.trunk_sizes:
            if size <= 0:
                raise ValueError("trunk layer sizes must be positive")
        if self.card_head_hidden <= 0:
            raise ValueError("card_head_hidden must be positive")
        if self.discard_head_hidden <= 0:
            raise ValueError("discard_head_hidden must be positive")
        if self.activation not in ("relu", "tanh", "gelu"):
            raise ValueError(f"Unsupported activation: {self.activation}")
        if self.dropout < 0:
            raise ValueError("dropout must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Serialize configuration to a dict."""
        return {
            "input_dim": self.input_dim,
            "trunk_sizes": list(self.trunk_sizes),
            "card_head_hidden": self.card_head_hidden,
            "discard_head_hidden": self.discard_head_hidden,
            "activation": self.activation,
            "dropout": self.dropout,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "HierarchicalConfig":
        """Deserialize configuration from a dict."""
        trunk_sizes = data.get("trunk_sizes", (256, 256))
        if isinstance(trunk_sizes, (list, tuple)):
            trunk_tuple = tuple(int(size) for size in trunk_sizes)
        else:
            trunk_tuple = (256, 256)
        return HierarchicalConfig(
            input_dim=int(data.get("input_dim", 34)),
            trunk_sizes=trunk_tuple,
            card_head_hidden=int(data.get("card_head_hidden", 128)),
            discard_head_hidden=int(data.get("discard_head_hidden", 64)),
            activation=str(data.get("activation", "relu")),
            dropout=float(data.get("dropout", 0.0)),
        )


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class HierarchicalPolicyValue(nn.Module):
    """Hierarchical model with separate heads for action type, card selection, discard, and value.

    Architecture:
        Input (34 features)
            |
        Shared Trunk (256 -> 256)
            |
        +-------+-------+-------+-------+
        |       |       |       |       |
      Type    Card   Discard  Value
      Head   Select   Head    Head
      (3)    (11)    (10)     (1)

    The type head outputs logits for PASS/BID/DISCARD.
    The card selector outputs sigmoid probabilities for each of 11 money cards.
    The discard head outputs logits for each of 10 possession values.
    The value head outputs a scalar state value estimate.
    """

    def __init__(self, config: HierarchicalConfig) -> None:
        """Initialize the model from a configuration."""
        super().__init__()
        self.config = config

        # Build shared trunk
        activation_cls = _ACTIVATIONS[config.activation]
        trunk_layers: list[nn.Module] = []
        prev_dim = config.input_dim
        for size in config.trunk_sizes:
            trunk_layers.append(nn.Linear(prev_dim, size))
            trunk_layers.append(activation_cls())
            if config.dropout > 0:
                trunk_layers.append(nn.Dropout(config.dropout))
            prev_dim = size
        self.trunk = nn.Sequential(*trunk_layers)
        trunk_output_dim = config.trunk_sizes[-1]

        # Type head: outputs 3 logits (PASS, BID, DISCARD)
        self.type_head = nn.Linear(trunk_output_dim, 3)

        # Card selector head: outputs 11 values (sigmoid applied in forward)
        self.card_head = nn.Sequential(
            nn.Linear(trunk_output_dim, config.card_head_hidden),
            activation_cls(),
            nn.Linear(config.card_head_hidden, len(MONEY_CARD_VALUES)),
        )

        # Discard head: outputs 10 logits (one per possession value 1-10)
        self.discard_head = nn.Sequential(
            nn.Linear(trunk_output_dim, config.discard_head_hidden),
            activation_cls(),
            nn.Linear(config.discard_head_hidden, NUM_POSSESSIONS),
        )

        # Value head: outputs scalar value estimate
        self.value_head = nn.Linear(trunk_output_dim, 1)

    def forward(
        self, features: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Compute action type logits, card probabilities, discard logits, and value.

        Args:
            features: (batch, input_dim) encoded observations

        Returns:
            type_logits: (batch, 3) logits for PASS/BID/DISCARD
            card_probs: (batch, 11) sigmoid probabilities for each money card
            discard_logits: (batch, 10) logits for each possession value
            value: (batch, 1) state value estimate
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        trunk_out = self.trunk(features)

        type_logits = self.type_head(trunk_out)
        card_probs = torch.sigmoid(self.card_head(trunk_out))
        discard_logits = self.discard_head(trunk_out)
        value = self.value_head(trunk_out)

        return type_logits, card_probs, discard_logits, value


def build_hierarchical(
    input_dim: int = 34,
    trunk_sizes: tuple[int, ...] = (256, 256),
    card_head_hidden: int = 128,
    discard_head_hidden: int = 64,
    activation: str = "relu",
    dropout: float = 0.0,
) -> HierarchicalPolicyValue:
    """Build a HierarchicalPolicyValue model from explicit arguments."""
    config = HierarchicalConfig(
        input_dim=input_dim,
        trunk_sizes=trunk_sizes,
        card_head_hidden=card_head_hidden,
        discard_head_hidden=discard_head_hidden,
        activation=activation,
        dropout=dropout,
    )
    return HierarchicalPolicyValue(config)
