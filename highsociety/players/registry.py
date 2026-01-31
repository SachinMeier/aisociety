"""Player registry for creating players from specs."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from highsociety.players.base import Player

PlayerFactory = Callable[[Mapping[str, Any]], Player]


class PlayerRegistry:
    """Registry of player factories keyed by type string."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, PlayerFactory] = {}

    def register(self, player_type: str, factory: PlayerFactory) -> None:
        """Register a factory for a player type."""
        if not player_type:
            raise ValueError("Player type is required")
        if player_type in self._factories:
            raise ValueError(f"Player type already registered: {player_type}")
        self._factories[player_type] = factory

    def create(self, spec: Mapping[str, Any]) -> Player:
        """Create a player instance from a spec mapping."""
        player_type = self._extract_type(spec)
        if player_type not in self._factories:
            raise KeyError(f"Unknown player type: {player_type}")
        return self._factories[player_type](spec)

    def available_types(self) -> tuple[str, ...]:
        """Return the registered player types as a sorted tuple."""
        return tuple(sorted(self._factories.keys()))

    def _extract_type(self, spec: Mapping[str, Any]) -> str:
        """Extract the player type from the spec mapping."""
        player_type = spec.get("type")
        if not isinstance(player_type, str) or not player_type:
            raise ValueError("Player spec must include a non-empty 'type'")
        return player_type
