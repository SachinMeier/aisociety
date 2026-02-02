"""Artifact writers for run results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from highsociety.domain.rules import RulesEngine
from highsociety.ops.metrics import SummaryMetrics, compute_summary
from highsociety.ops.runner import GameRun, RunResult
from highsociety.ops.spec import RunSpec


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem paths to generated artifacts."""

    summary_json: Path
    results_csv: Path
    spec_json: Path


def write_artifacts(
    run_result: RunResult,
    output_dir: Path,
    spec_source: Mapping[str, Any] | None = None,
) -> ArtifactPaths:
    """Write summary and results artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "results.csv"
    spec_path = output_dir / "spec.json"
    summary = compute_summary(run_result)
    write_summary(summary_path, summary)
    write_results_csv(results_path, run_result.games)
    if spec_source is None:
        write_spec(spec_path, run_result.spec)
    else:
        write_spec_source(spec_path, spec_source)
    return ArtifactPaths(summary_json=summary_path, results_csv=results_path, spec_json=spec_path)


def write_summary(path: Path, summary: SummaryMetrics) -> None:
    """Write summary metrics to JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, sort_keys=True)


def write_results_csv(path: Path, games: tuple[GameRun, ...]) -> None:
    """Write per-game results to CSV."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["game_id", "winners", "scores", "money_remaining", "rounds"],
        )
        writer.writeheader()
        for game in games:
            writer.writerow(_game_row(game))


def write_spec(path: Path, spec: RunSpec) -> None:
    """Write the run spec to JSON for reproducibility."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec.to_mapping(), handle, indent=2, sort_keys=True)


def write_spec_source(path: Path, spec: Mapping[str, Any]) -> None:
    """Write the original run spec mapping to JSON."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2, sort_keys=True)


def _game_row(game: GameRun) -> dict[str, str | int]:
    """Return a CSV row for a single game."""
    scores = {pid: float(score) for pid, score in game.result.scores.items()}
    return {
        "game_id": game.game_id,
        "winners": ",".join(str(pid) for pid in game.result.winners),
        "scores": json.dumps(scores, sort_keys=True),
        "money_remaining": json.dumps(game.result.money_remaining, sort_keys=True),
        "rounds": _round_count(game),
    }


def _round_count(game: GameRun) -> int:
    """Compute the number of rounds revealed in a game."""
    total_cards = len(RulesEngine.create_status_deck())
    remaining_cards = len(game.state.status_deck)
    return total_cards - remaining_cards
