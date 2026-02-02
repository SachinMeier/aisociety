"""Tests for CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from highsociety.ops.spec import PlayerSpec, RunSpec
from scripts.eval import _ensure_eval_mode
from scripts.run import execute_spec


def test_execute_spec_dry_run_does_not_write_artifacts(tmp_path: Path) -> None:
    """Dry run validates spec without creating artifacts."""
    spec_path = _write_spec(tmp_path / "spec.json")
    output_dir = tmp_path / "runs"

    result = execute_spec(spec_path, output_dir=output_dir, dry_run=True)

    assert result is None
    assert not output_dir.exists()


def test_execute_spec_writes_artifacts(tmp_path: Path) -> None:
    """Spec execution writes summary and results artifacts."""
    spec_path = _write_spec(tmp_path / "spec.json")
    output_dir = tmp_path / "output"

    result = execute_spec(spec_path, output_dir=output_dir, dry_run=False)

    assert result is not None
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "results.csv").exists()


def test_eval_mode_validation() -> None:
    """Eval CLI rejects non-eval specs."""
    players = (
        PlayerSpec(type="random"),
        PlayerSpec(type="random"),
        PlayerSpec(type="random"),
    )
    spec = RunSpec(mode="play", seed=1, num_games=1, players=players, rules={})
    with pytest.raises(ValueError):
        _ensure_eval_mode(spec)


def _write_spec(path: Path) -> Path:
    """Write a minimal spec file to disk."""
    spec = {
        "mode": "eval",
        "seed": 1,
        "num_games": 1,
        "players": [{"type": "random"}] * 3,
        "rules": {},
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path
