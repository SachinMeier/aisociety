"""Run specification models and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

_ALLOWED_MODES = {"play", "eval", "train"}


@dataclass(frozen=True)
class PlayerSpec:
    """Specification for creating a single player instance."""

    type: str
    name: str | None = None
    checkpoint: str | None = None
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate player spec fields."""
        if not self.type:
            raise ValueError("PlayerSpec.type is required")
        if not isinstance(self.params, dict):
            raise ValueError("PlayerSpec.params must be a dict")

    def to_mapping(self) -> dict[str, object]:
        """Return a mapping suitable for player factories."""
        return {
            "type": self.type,
            "name": self.name,
            "checkpoint": self.checkpoint,
            "params": dict(self.params),
        }

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "PlayerSpec":
        """Create a PlayerSpec from a mapping."""
        params = data.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("PlayerSpec.params must be a dict")
        params = dict(params)
        if "style" in data and "style" not in params:
            params["style"] = data["style"]
        if "seed" in data and "seed" not in params:
            params["seed"] = data["seed"]
        return PlayerSpec(
            type=str(data.get("type", "")),
            name=data.get("name"),
            checkpoint=data.get("checkpoint"),
            params=params,
        )


@dataclass(frozen=True)
class RunSpec:
    """Specification for a batch run."""

    mode: str
    seed: int
    num_games: int
    players: tuple[PlayerSpec, ...]
    rules: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate run spec fields."""
        if self.mode not in _ALLOWED_MODES:
            raise ValueError(f"Invalid mode: {self.mode}")
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if not (3 <= len(self.players) <= 5):
            raise ValueError("Player count must be 3-5")
        if not isinstance(self.rules, dict):
            raise ValueError("rules must be a dict")

    def to_mapping(self) -> dict[str, object]:
        """Return a mapping representation of the run spec."""
        return {
            "mode": self.mode,
            "seed": self.seed,
            "num_games": self.num_games,
            "players": [spec.to_mapping() for spec in self.players],
            "rules": dict(self.rules),
        }

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "RunSpec":
        """Create a RunSpec from a mapping."""
        players_data = data.get("players", [])
        if not isinstance(players_data, list):
            raise ValueError("players must be a list")
        players = tuple(PlayerSpec.from_mapping(item) for item in players_data)
        rules = data.get("rules", {})
        if rules is None:
            rules = {}
        if not isinstance(rules, dict):
            raise ValueError("rules must be a dict")
        return RunSpec(
            mode=str(data.get("mode", "")),
            seed=int(data.get("seed", 0)),
            num_games=int(data.get("num_games", 0)),
            players=players,
            rules=dict(rules),
        )


def parse_run_spec(spec: RunSpec | Mapping[str, Any]) -> RunSpec:
    """Normalize a run spec input into a RunSpec instance."""
    if isinstance(spec, RunSpec):
        return spec
    return RunSpec.from_mapping(spec)
