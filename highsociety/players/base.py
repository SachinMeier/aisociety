"""Player protocol definitions for runtime use."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from highsociety.domain.actions import Action
from highsociety.domain.rules import GameResult

if TYPE_CHECKING:
    from highsociety.app.observations import Observation


class Player(Protocol):
    """Runtime player API used by the game runner."""

    name: str

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Reset internal state for a new game."""

    def act(self, observation: "Observation", legal_actions: list[Action]) -> Action:
        """Select the next action given an observation and legal actions."""

    def on_game_end(self, result: GameResult) -> None:
        """Receive the final game result."""
