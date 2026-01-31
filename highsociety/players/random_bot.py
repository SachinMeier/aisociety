"""Random baseline player implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidAction
from highsociety.domain.rules import GameResult


@dataclass
class RandomBot:
    """Random bot that selects uniformly from legal actions."""

    name: str = "random"
    seed: int | None = None
    kind: str = "random"
    _rng: random.Random = field(init=False)
    _player_id: int | None = None

    def __post_init__(self) -> None:
        """Initialize the RNG."""
        self._rng = random.Random(self.seed)

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
        """Select a random legal action."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        return self._rng.choice(legal_actions)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler."""
