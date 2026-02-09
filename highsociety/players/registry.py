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
    """Register baseline player factories."""
    from highsociety.players.heuristic_bot import HeuristicBot
    from highsociety.players.random_bot import RandomBot
    from highsociety.players.static_bot import StaticBot

    registry.register("random", _random_factory(RandomBot))
    registry.register("heuristic", _heuristic_factory(HeuristicBot))
    registry.register("static", _static_factory(StaticBot))

    # ML-based bots require torch/numpy; skip registration when unavailable
    # so the server can run without heavy ML dependencies.
    try:
        from highsociety.players.mlp_bot import MLPPolicyBot

        registry.register("mlp", _mlp_factory(MLPPolicyBot))
    except ImportError:
        pass
    try:
        registry.register("linear_rl", _linear_rl_factory())
    except ImportError:
        pass
    try:
        from highsociety.players.hierarchical_bot import HierarchicalBot

        registry.register("hierarchical", _hierarchical_factory(HierarchicalBot))
    except ImportError:
        pass


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


def _static_factory(bot_cls: type) -> PlayerFactory:
    """Create a factory for the static bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        name = _coerce_name(spec.get("name"), default="static")
        kwargs: dict[str, object] = {"name": name}
        if "title_budget" in params:
            kwargs["title_budget"] = params["title_budget"]
        if "value_scale" in params:
            kwargs["value_scale"] = params["value_scale"]
        if "cancel_value" in params:
            kwargs["cancel_value"] = params["cancel_value"]
        return bot_cls(**kwargs)

    return factory


def _mlp_factory(bot_cls: type) -> PlayerFactory:
    """Create a factory for the MLP bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        checkpoint = spec.get("checkpoint") or params.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("MLP bot requires a checkpoint path")
        temperature = float(params.get("temperature", 0.0))
        seed = params.get("seed")
        device = params.get("device", "cpu")
        name = _coerce_name(spec.get("name"), default="mlp")
        return bot_cls(
            name=name,
            checkpoint=str(checkpoint),
            temperature=temperature,
            seed=seed,
            device=str(device),
        )

    return factory


def _linear_rl_factory() -> PlayerFactory:
    """Create a factory for the linear RL bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        checkpoint = spec.get("checkpoint") or params.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("linear_rl requires a checkpoint path")
        seed = params.get("seed")
        epsilon = float(params.get("epsilon", 0.0))
        name = _coerce_name(spec.get("name"), default="linear_rl")
        from highsociety.players.linear_rl_bot import LinearRLBot

        return LinearRLBot(
            name=name,
            checkpoint_path=checkpoint,
            epsilon=epsilon,
            seed=seed if isinstance(seed, int) else None,
        )

    return factory


def _hierarchical_factory(bot_cls: type) -> PlayerFactory:
    """Create a factory for the hierarchical bot."""

    def factory(spec: Mapping[str, Any]) -> Player:
        params = _coerce_params(spec.get("params"))
        checkpoint = spec.get("checkpoint") or params.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("hierarchical bot requires a checkpoint path")
        temperature = float(params.get("temperature", 0.0))
        seed = params.get("seed")
        device = params.get("device", "cpu")
        name = _coerce_name(spec.get("name"), default="hierarchical")
        return bot_cls(
            name=name,
            checkpoint=str(checkpoint),
            temperature=temperature,
            seed=seed,
            device=str(device),
        )

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
