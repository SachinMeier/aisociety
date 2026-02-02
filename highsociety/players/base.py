"""Base player protocol used by the runtime."""

from __future__ import annotations

from typing import Protocol

from highsociety.domain.actions import Action
from highsociety.domain.rules import GameResult


class Player(Protocol):
    """Runtime player API used by game runners and trainers."""

    name: str

    def reset(self, game_config: dict, player_id: int, seat: int) -> None:
        """Reset player state for a new game."""

    def act(self, observation: object, legal_actions: list[Action]) -> Action:
        """Return the next action for the player."""

    def on_game_end(self, result: GameResult) -> None:
        """Receive the final game result."""
