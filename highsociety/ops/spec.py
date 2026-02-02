"""Run and player specifications for ops workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PlayerSpec:
    """Configuration for a player instance in a run spec."""

    type: str
    name: str | None = None
    checkpoint: str | None = None
    params: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        """Serialize the spec into a mapping."""
        payload: dict[str, object] = {"type": self.type}
        if self.name:
            payload["name"] = self.name
        if self.checkpoint:
            payload["checkpoint"] = self.checkpoint
        if self.params:
            payload["params"] = dict(self.params)
        return payload

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "PlayerSpec":
        """Create a PlayerSpec from a mapping."""
        type_value = data.get("type")
        if not isinstance(type_value, str) or not type_value:
            raise ValueError("PlayerSpec.type must be a non-empty string")
        name_value = data.get("name")
        name = name_value if isinstance(name_value, str) and name_value else None
        checkpoint_value = data.get("checkpoint")
        checkpoint = (
            checkpoint_value if isinstance(checkpoint_value, str) and checkpoint_value else None
        )
        params_value = data.get("params", {}) or {}
        if not isinstance(params_value, Mapping):
            raise ValueError("PlayerSpec.params must be a mapping")
        return PlayerSpec(
            type=type_value,
            name=name,
            checkpoint=checkpoint,
            params=dict(params_value),
        )


@dataclass(frozen=True)
class RunSpec:
    """Configuration for a run (play/eval/train)."""

    mode: str
    seed: int
    num_games: int
    players: tuple[PlayerSpec, ...]
    rules: dict[str, object] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, object]:
        """Serialize the run spec into a mapping."""
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
        mode_value = data.get("mode")
        if not isinstance(mode_value, str) or not mode_value:
            raise ValueError("RunSpec.mode must be a non-empty string")
        players_value = data.get("players", [])
        if not isinstance(players_value, list):
            raise ValueError("RunSpec.players must be a list")
        players = tuple(PlayerSpec.from_mapping(item) for item in players_value)
        rules_value = data.get("rules", {}) or {}
        if not isinstance(rules_value, Mapping):
            raise ValueError("RunSpec.rules must be a mapping")
        return RunSpec(
            mode=mode_value,
            seed=int(data.get("seed", 0)),
            num_games=int(data.get("num_games", 0)),
            players=players,
            rules=dict(rules_value),
        )
