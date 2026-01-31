"""Tests for metrics and artifact writers."""

from __future__ import annotations

import csv
import json
from fractions import Fraction

from highsociety.domain.rules import GameResult, RulesEngine
from highsociety.domain.state import GameState, PlayerState
from highsociety.ops.artifacts import write_artifacts
from highsociety.ops.metrics import compute_summary
from highsociety.ops.runner import GameRun, RunResult
from highsociety.ops.spec import PlayerSpec, RunSpec


def _make_state() -> GameState:
    """Create a basic game state for testing artifacts."""
    players = [PlayerState(id=0), PlayerState(id=1), PlayerState(id=2)]
    deck = RulesEngine.create_status_deck()
    return GameState(players=players, status_deck=deck, starting_player=0)


def _make_run_result() -> RunResult:
    """Build a run result with deterministic outcomes."""
    state = _make_state()
    game1 = GameRun(
        game_id="g1",
        result=GameResult(
            winners=(0,),
            scores={0: Fraction(10), 1: Fraction(8)},
            money_remaining={0: 1000, 1: 2000, 2: 3000},
            poorest=(2,),
        ),
        state=state,
    )
    game2 = GameRun(
        game_id="g2",
        result=GameResult(
            winners=(1, 2),
            scores={1: Fraction(5), 2: Fraction(5)},
            money_remaining={0: 1000, 1: 2000, 2: 3000},
            poorest=(0,),
        ),
        state=state,
    )
    players = (
        PlayerSpec(type="random"),
        PlayerSpec(type="random"),
        PlayerSpec(type="random"),
    )
    spec = RunSpec(mode="play", seed=1, num_games=2, players=players, rules={})
    return RunResult(spec=spec, games=(game1, game2), errors=())


def test_compute_summary_metrics() -> None:
    """Summary metrics aggregate wins and scores."""
    run_result = _make_run_result()
    summary = compute_summary(run_result)

    assert summary.games_total == 2
    assert summary.games_finished == 2
    assert summary.win_counts[0] == 1
    assert summary.win_counts[1] == 1
    assert summary.win_counts[2] == 1
    assert summary.poorest_counts[0] == 1
    assert summary.poorest_counts[2] == 1
    assert summary.average_scores[0] == 10.0
    assert summary.average_scores[1] == 6.5
    assert summary.average_scores[2] == 5.0


def test_write_artifacts(tmp_path) -> None:
    """Artifact writer outputs summary.json and results.csv."""
    run_result = _make_run_result()
    paths = write_artifacts(run_result, tmp_path)

    assert paths.summary_json.exists()
    assert paths.results_csv.exists()

    summary_data = json.loads(paths.summary_json.read_text(encoding="utf-8"))
    assert "win_rates" in summary_data
    assert "average_scores" in summary_data
    assert "finish_rate" in summary_data

    with paths.results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
    assert header == ["game_id", "winners", "scores", "money_remaining", "rounds"]
