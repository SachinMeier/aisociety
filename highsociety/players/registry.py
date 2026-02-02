"""Registry for creating player instances from specs."""

from __future__ import annotations

from typing import Callable, Mapping

from highsociety.players.base import Player
from highsociety.players.random_bot import RandomBot


class PlayerRegistry:
    """Registry of player factories keyed by type."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, Callable[[Mapping[str, object]], Player]] = {}

    def register(self, type_name: str, factory: Callable[[Mapping[str, object]], Player]) -> None:
        """Register a factory for a given player type."""
        if not type_name:
            raise ValueError("type_name must be non-empty")
        self._factories[type_name] = factory

    def create(self, spec: Mapping[str, object]) -> Player:
        """Create a player instance from a spec mapping."""
        type_value = spec.get("type")
        if not isinstance(type_value, str) or not type_value:
            raise ValueError("Player spec must include a non-empty type")
        factory = self._factories.get(type_value)
        if factory is None:
            raise ValueError(f"Unknown player type: {type_value}")
        return factory(spec)


def build_default_registry() -> PlayerRegistry:
    """Return a registry with built-in player types."""
    registry = PlayerRegistry()
    registry.register("random", _build_random_bot)
    return registry


def _build_random_bot(spec: Mapping[str, object]) -> Player:
    """Build a random bot from a spec mapping."""
    name_value = spec.get("name")
    name = name_value if isinstance(name_value, str) and name_value else "random"
    return RandomBot(name=name)
