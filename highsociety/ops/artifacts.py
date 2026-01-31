"""Artifact writers for run results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from highsociety.domain.rules import RulesEngine
from highsociety.ops.metrics import SummaryMetrics, compute_summary
from highsociety.ops.runner import GameRun, RunResult


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem paths to generated artifacts."""

    summary_json: Path
    results_csv: Path


def write_artifacts(run_result: RunResult, output_dir: Path) -> ArtifactPaths:
    """Write summary and results artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    results_path = output_dir / "results.csv"
    summary = compute_summary(run_result)
    write_summary(summary_path, summary)
    write_results_csv(results_path, run_result.games)
    return ArtifactPaths(summary_json=summary_path, results_csv=results_path)


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
