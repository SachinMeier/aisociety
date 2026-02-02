"""Random-action baseline player."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult


@dataclass
class RandomBot:
    """Selects random legal actions."""

    name: str = "random"
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def reset(self, game_config: dict, player_id: int, seat: int) -> None:
        """Reset the bot RNG for a new game."""
        seed = int(game_config.get("seed", 0))
        self._rng = random.Random(seed + seat)
        del player_id

    def act(self, observation: object, legal_actions: list[Action]) -> Action:
        """Return a random legal action."""
        del observation
        if not legal_actions:
            raise InvalidState("No legal actions available")
        return self._rng.choice(legal_actions)

    def on_game_end(self, result: GameResult) -> None:
        """Handle game end (no-op for random bot)."""
        del result
