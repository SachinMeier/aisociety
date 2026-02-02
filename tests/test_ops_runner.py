"""Tests for run specs and the run manager."""

from __future__ import annotations

import pytest

from highsociety.ops.runner import RunManager
from highsociety.ops.spec import RunSpec, parse_run_spec
from highsociety.players.registry import build_default_registry


def test_runspec_rejects_invalid_mode() -> None:
    """RunSpec validation rejects unsupported modes."""
    with pytest.raises(ValueError):
        parse_run_spec(
            {
                "mode": "invalid",
                "seed": 1,
                "num_games": 1,
                "players": [{"type": "random"}] * 3,
                "rules": {},
            }
        )


def test_runspec_requires_valid_player_count() -> None:
    """RunSpec validation enforces 3-5 players."""
    with pytest.raises(ValueError):
        RunSpec.from_mapping(
            {
                "mode": "play",
                "seed": 1,
                "num_games": 1,
                "players": [{"type": "random"}] * 2,
                "rules": {},
            }
        )


def test_run_manager_runs_multiple_games() -> None:
    """RunManager executes multiple games and returns results."""
    registry = build_default_registry()
    manager = RunManager(registry=registry)
    spec = RunSpec.from_mapping(
        {
            "mode": "play",
            "seed": 5,
            "num_games": 2,
            "players": [{"type": "random"}] * 3,
            "rules": {},
        }
    )

    result = manager.run(spec)

    assert len(result.games) == 2
    assert not result.errors
