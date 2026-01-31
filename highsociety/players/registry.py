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


def build_default_registry() -> PlayerRegistry:
    """Return a registry pre-populated with baseline players."""
    registry = PlayerRegistry()
    register_baseline_players(registry)
    return registry


def register_baseline_players(registry: PlayerRegistry) -> None:
    """Register random and heuristic player factories."""
    from highsociety.players.heuristic_bot import HeuristicBot
    from highsociety.players.random_bot import RandomBot

    registry.register("random", _random_factory(RandomBot))
    registry.register("heuristic", _heuristic_factory(HeuristicBot))


def _random_factory(bot_cls: type) -> PlayerFactory:
    """Create a factory for the random bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        seed = params.get("seed")
        name = _coerce_name(spec.get("name"), default="random")
        return bot_cls(name=name, seed=seed)

    return factory


def _heuristic_factory(bot_cls: type) -> PlayerFactory:
    """Create a factory for the heuristic bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        seed = params.get("seed")
        style = params.get("style", "balanced")
        name = _coerce_name(spec.get("name"), default="heuristic")
        return bot_cls(name=name, style=style, seed=seed)

    return factory


def _coerce_params(params: object) -> dict[str, object]:
    """Coerce params into a dict."""
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("PlayerSpec.params must be a dict")
    return dict(params)


def _coerce_name(name: object, default: str) -> str:
    """Coerce the player name into a string."""
    if isinstance(name, str) and name:
        return name
    return default
