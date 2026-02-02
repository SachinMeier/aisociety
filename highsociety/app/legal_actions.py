"""Helpers for computing legal actions from a game state."""

from __future__ import annotations

from highsociety.domain.actions import Action
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState


def legal_actions(state: GameState, player_id: int) -> list[Action]:
    """Return legal actions for the given player in the provided state."""
    return RulesEngine.legal_actions(state, player_id)
