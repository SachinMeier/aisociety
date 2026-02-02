"""Linear RL bot implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.encoders.linear import LinearFeatureEncoder
from highsociety.ml.models.linear import LinearModel


@dataclass
class LinearRLBot:
    """Selects actions using a linear model over handcrafted features."""

    name: str
    encoder: LinearFeatureEncoder
    model: LinearModel
    epsilon: float = 0.0
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be between 0 and 1")

    def reset(self, game_config: dict, player_id: int, seat: int) -> None:
        """Reset RNG state for a new game."""
        seed = int(game_config.get("seed", 0))
        self._rng = random.Random(seed + seat)
        del player_id

    def act(self, observation: object, legal_actions: list[Action]) -> Action:
        """Select the best-scoring legal action."""
        if not legal_actions:
            raise InvalidState("No legal actions available")
        if self.epsilon > 0 and self._rng.random() < self.epsilon:
            return self._rng.choice(legal_actions)
        best_action = legal_actions[0]
        best_score = self._score_action(observation, best_action)
        for action in legal_actions[1:]:
            score = self._score_action(observation, action)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def on_game_end(self, result: GameResult) -> None:
        """Handle game end notifications (no-op)."""
        del result

    def _score_action(self, observation: object, action: Action) -> float:
        """Score a candidate action with the linear model."""
        features = self.encoder.encode(observation, action)
        return self.model.predict(features)
