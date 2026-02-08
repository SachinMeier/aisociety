"""Inspect-trainings page behavior tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from highsociety.ops.dashboard.app import (
    _downsample_training_games,
    _load_training_players_frame,
    _page_inspect_trainings,
    _training_with_rolling_rates,
)


class _NullContext:
    """Simple context manager used by fake expander."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeStreamlit:
    """Minimal Streamlit stub for inspect-trainings page tests."""

    def __init__(self, root: Path) -> None:
        self._root = str(root)
        self.altair_chart_calls = 0
        self.line_chart_calls = 0
        self.bar_chart_calls = 0

    def subheader(self, _label: str) -> None:
        return

    def text_input(self, _label: str, value: str = "", key: str | None = None) -> str:
        if key == "inspect_trainings_root":
            return self._root
        return value

    def selectbox(self, _label: str, options, format_func=None, key: str | None = None):
        del format_func
        values = list(options)
        if key == "inspect_training_status_metric":
            return values[0]
        return values[0]

    def caption(self, _message: str) -> None:
        return

    def number_input(
        self,
        _label: str,
        min_value: int | None = None,
        max_value: int | None = None,
        value: int = 0,
        step: int = 1,
        key: str | None = None,
    ) -> int:
        del min_value, max_value, step, key
        return value

    def json(self, _payload: object) -> None:
        return

    def expander(self, _label: str):
        return _NullContext()

    def warning(self, _message: str) -> None:
        return

    def error(self, _message: str) -> None:
        return

    def markdown(self, _body: str, unsafe_allow_html: bool = False) -> None:
        del unsafe_allow_html
        return

    def altair_chart(self, _chart: object, use_container_width: bool = False) -> None:
        del use_container_width
        self.altair_chart_calls += 1

    def line_chart(self, _data: object) -> None:
        self.line_chart_calls += 1

    def bar_chart(self, _data: object) -> None:
        self.bar_chart_calls += 1


def _write_training_artifacts(tmp_path: Path) -> Path:
    """Write one training-artifacts folder and return its root."""
    history_root = tmp_path / "trainings_history"
    run_dir = history_root / "train_0001"
    run_dir.mkdir(parents=True)
    summary = {
        "bot_type": "linear",
        "games_total": 2,
        "games_logged": 2,
        "learner_seats": [0, 1],
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    players_path = run_dir / "training_players.csv"
    with players_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game_index",
                "seed",
                "player_id",
                "is_learner",
                "won",
                "poorest",
                "score",
                "money_remaining",
                "titles",
                "scandal",
                "debt",
                "theft",
                "cumulative_wins",
                "cumulative_poorest",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "game_index": 1,
                "seed": 1,
                "player_id": 0,
                "is_learner": 1,
                "won": 1,
                "poorest": 0,
                "score": 8.0,
                "money_remaining": 9000,
                "titles": 1,
                "scandal": 0,
                "debt": 0,
                "theft": 0,
                "cumulative_wins": 1,
                "cumulative_poorest": 0,
            }
        )
        writer.writerow(
            {
                "game_index": 1,
                "seed": 1,
                "player_id": 1,
                "is_learner": 1,
                "won": 0,
                "poorest": 1,
                "score": 0.0,
                "money_remaining": 1000,
                "titles": 0,
                "scandal": 1,
                "debt": 0,
                "theft": 0,
                "cumulative_wins": 0,
                "cumulative_poorest": 1,
            }
        )
        writer.writerow(
            {
                "game_index": 2,
                "seed": 2,
                "player_id": 0,
                "is_learner": 1,
                "won": 0,
                "poorest": 0,
                "score": 5.0,
                "money_remaining": 7000,
                "titles": 1,
                "scandal": 0,
                "debt": 1,
                "theft": 0,
                "cumulative_wins": 1,
                "cumulative_poorest": 0,
            }
        )
        writer.writerow(
            {
                "game_index": 2,
                "seed": 2,
                "player_id": 1,
                "is_learner": 1,
                "won": 1,
                "poorest": 0,
                "score": 7.0,
                "money_remaining": 6000,
                "titles": 0,
                "scandal": 0,
                "debt": 0,
                "theft": 1,
                "cumulative_wins": 1,
                "cumulative_poorest": 1,
            }
        )
    return history_root


def test_inspect_trainings_renders_progress_charts(tmp_path: Path) -> None:
    """Inspect page should render learner progress charts for artifact rows."""
    history_root = _write_training_artifacts(tmp_path)
    streamlit = _FakeStreamlit(history_root)

    _page_inspect_trainings(streamlit)

    total_charts = (
        streamlit.altair_chart_calls + streamlit.line_chart_calls + streamlit.bar_chart_calls
    )
    assert total_charts > 0


def test_training_with_rolling_rates_is_per_learner(tmp_path: Path) -> None:
    """Rolling rates should be computed independently for each learner."""
    history_root = _write_training_artifacts(tmp_path)
    frame = _load_training_players_frame(history_root / "train_0001" / "training_players.csv")
    learner_frame = frame.loc[
        frame["is_learner"] == 1,
        ["game_index", "player_id", "won", "poorest"],
    ].copy()
    result = _training_with_rolling_rates(learner_frame, window_size=2)
    learner0 = result.loc[result["player_id"] == 0].sort_values("game_index")
    learner1 = result.loc[result["player_id"] == 1].sort_values("game_index")

    assert learner0["rolling_win_rate"].tolist() == [1.0, 0.5]
    assert learner0["rolling_poorest_rate"].tolist() == [0.0, 0.0]
    assert learner1["rolling_win_rate"].tolist() == [0.0, 0.5]
    assert learner1["rolling_poorest_rate"].tolist() == [1.0, 0.5]


def test_downsample_training_games_limits_unique_games(tmp_path: Path) -> None:
    """Downsampling should cap the number of unique games while keeping endpoints."""
    del tmp_path
    import pandas as pd

    frame = pd.DataFrame(
        {
            "game_index": list(range(1, 101)),
            "player_id": [0] * 100,
        }
    )
    sampled = _downsample_training_games(frame, max_games=10)

    assert int(sampled["game_index"].nunique()) == 10
    assert int(sampled["game_index"].min()) == 1
    assert int(sampled["game_index"].max()) == 100
