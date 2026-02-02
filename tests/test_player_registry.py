"""Tests for the player registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.rules import GameResult
from highsociety.players.registry import PlayerRegistry


@dataclass
class DummyPlayer:
    """Simple player used for registry tests."""

    name: str
    created_with: Mapping[str, Any]

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """No-op reset for testing."""

    def act(self, observation: object, legal_actions: list[Action]) -> Action:
        """Return a pass action for testing."""
        return Action(ActionKind.PASS)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler for testing."""


def _dummy_factory(spec: Mapping[str, Any]) -> DummyPlayer:
    """Create a DummyPlayer from a spec mapping."""
    name = str(spec.get("name", "dummy"))
    return DummyPlayer(name=name, created_with=dict(spec))


def test_registry_creates_player() -> None:
    """Registry creates players from registered factories."""
    registry = PlayerRegistry()
    registry.register("dummy", _dummy_factory)
    player = registry.create({"type": "dummy", "name": "alpha"})
    assert isinstance(player, DummyPlayer)
    assert player.name == "alpha"


def test_registry_rejects_duplicate_types() -> None:
    """Registry rejects duplicate type registrations."""
    registry = PlayerRegistry()
    registry.register("dummy", _dummy_factory)
    with pytest.raises(ValueError):
        registry.register("dummy", _dummy_factory)


def test_registry_errors_for_unknown_type() -> None:
    """Registry errors when type is not registered."""
    registry = PlayerRegistry()
    with pytest.raises(KeyError):
        registry.create({"type": "missing"})


def test_registry_errors_for_missing_type() -> None:
    """Registry errors when spec has no type."""
    registry = PlayerRegistry()
    with pytest.raises(ValueError):
        registry.create({"name": "nameless"})
