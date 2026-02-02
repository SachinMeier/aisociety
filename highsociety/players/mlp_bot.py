"""MLP policy/value bot backed by a PyTorch model."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.checkpoints import load_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder

if TYPE_CHECKING:  # pragma: no cover
    from highsociety.ml.models.mlp import MLPConfig, MLPPolicyValue


@dataclass
class MLPPolicyBot:
    """Bot that selects actions using a trained MLP policy/value model."""

    name: str = "mlp"
    checkpoint: str | None = None
    temperature: float = 0.0
    seed: int | None = None
    device: str = "cpu"
    kind: str = "mlp"
    _rng: random.Random = field(init=False)
    _player_id: int | None = None
    _encoder: BasicEncoder = field(init=False, repr=False)
    _model: "MLPPolicyValue" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the model checkpoint and initialize RNG."""
        self._rng = random.Random(self.seed)
        if self.checkpoint is None:
            raise ValueError("checkpoint is required for MLPPolicyBot")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        self._load_checkpoint(self.checkpoint)

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Reset the bot for a new game, reseeding if needed."""
        self._player_id = player_id
        seed_value = self.seed
        if seed_value is None:
            config_seed = game_config.get("seed")
            if isinstance(config_seed, int):
                seed_value = config_seed
        if seed_value is not None:
            self._rng = random.Random(seed_value + player_id)

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Select an action using the MLP policy."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        if self._player_id is None:
            raise InvalidState("Player has not been reset")
        import torch

        features = torch.tensor(
            self._encoder.encode(observation),
            dtype=torch.float32,
            device=self.device,
        )
        logits, _ = self._model(features)
        mask = torch.tensor(
            self._encoder.action_mask(legal_actions),
            dtype=torch.bool,
            device=self.device,
        )
        logits = logits.squeeze(0).masked_fill(~mask, -1e9)
        if not torch.any(mask):
            raise InvalidState("No legal actions in action mask")
        if self.temperature <= 0:
            action_idx = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / self.temperature, dim=0).cpu().tolist()
            action_idx = _sample_index(self._rng, probs)
        return self._encoder.action_space.action_at(action_idx)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler."""

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model + encoder configuration from a checkpoint."""
        bundle = load_checkpoint(checkpoint_path)
        encoder = BasicEncoder.from_config(bundle.encoder_config)
        model_config = _load_model_config(bundle.model_config)
        if model_config.input_dim != encoder.feature_size:
            raise InvalidState("Encoder feature size does not match model config")
        if model_config.action_dim != encoder.action_space.size:
            raise InvalidState("Action space size does not match model config")
        model = _load_model(model_config, bundle.model_state, self.device)
        self._encoder = encoder
        self._model = model


def _load_model_config(data: dict[str, object]) -> "MLPConfig":
    """Load the MLPConfig from a mapping."""
    from highsociety.ml.models.mlp import MLPConfig

    return MLPConfig.from_dict(data)


def _load_model(
    config: "MLPConfig",
    state_dict: dict[str, object],
    device: str,
) -> "MLPPolicyValue":
    """Load an MLPPolicyValue model from a state dict."""
    from highsociety.ml.models.mlp import MLPPolicyValue

    model = MLPPolicyValue(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _sample_index(rng: random.Random, probs: list[float]) -> int:
    """Sample an index from a probability distribution."""
    total = sum(probs)
    if total <= 0:
        return 0
    threshold = rng.random() * total
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if cumulative >= threshold:
            return idx
    return max(0, len(probs) - 1)
