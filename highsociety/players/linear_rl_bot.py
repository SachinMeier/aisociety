"""Linear reinforcement learning bot implementation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.checkpoints import load_linear_checkpoint
from highsociety.ml.encoders.linear import LinearFeatureEncoder
from highsociety.ml.models.linear import LinearModel
from highsociety.players.colors import BOT_COLORS


class LinearRLBot:
    """Bot that selects actions using a linear model over engineered features."""

    name: str
    kind: str
    color: str
    epsilon: float
    _rng: random.Random
    _model: LinearModel
    _encoder: LinearFeatureEncoder
    _player_id: int | None

    def __init__(
        self,
        name: str = "linear_rl",
        checkpoint_path: str | None = None,
        epsilon: float = 0.0,
        seed: int | None = None,
        model: LinearModel | None = None,
        encoder: LinearFeatureEncoder | None = None,
    ) -> None:
        """Initialize the linear RL bot from a checkpoint or provided model."""
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        self.name = name
        self.kind = "linear_rl"
        self.color = BOT_COLORS["linear_rl"]
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)
        self._player_id = None
        if checkpoint_path is not None:
            model_loaded, encoder_loaded, _metadata = load_linear_checkpoint(
                Path(checkpoint_path)
            )
            self._model = model_loaded
            self._encoder = encoder_loaded
        else:
            if model is None or encoder is None:
                raise ValueError("model and encoder must be provided when no checkpoint")
            self._model = model
            self._encoder = encoder

    def reset(self, game_config: Mapping[str, object], player_id: int, seat: int) -> None:
        """Reset internal state for a new game."""
        del game_config, seat
        self._player_id = player_id

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Select the next action using epsilon-greedy scoring."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        if self._player_id is None:
            raise InvalidState("Player has not been reset")
        if self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)
        best_action = legal_actions[0]
        best_features = self._encoder.encode(observation, best_action)
        best_score = self._model.predict(best_features)
        for action in legal_actions[1:]:
            features = self._encoder.encode(observation, action)
            score = self._model.predict(features)
            if score > best_score:
                best_action = action
                best_score = score
        return best_action

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler for the linear bot."""
        del result
