"""Inspect-runs page behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

from highsociety.ops.dashboard.app import _page_inspect_runs


class _NullContext:
    """Simple context manager used by fake expander."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeStreamlit:
    """Minimal Streamlit stub for inspect-runs page tests."""

    def __init__(self, runs_root: Path) -> None:
        self._runs_root = str(runs_root)
        self.altair_chart_calls = 0
        self.bar_chart_calls = 0

    def subheader(self, _label: str) -> None:  # noqa: D401
        return

    def text_input(self, _label: str, value: str = "", key: str | None = None) -> str:
        if key == "inspect_runs_root":
            return self._runs_root
        return value

    def selectbox(self, _label: str, options, format_func=None, key: str | None = None):
        del format_func, key
        return list(options)[0]

    def caption(self, _message: str) -> None:
        return

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

    def bar_chart(self, _data: object) -> None:
        self.bar_chart_calls += 1


def _write_summary_run(tmp_path: Path, summary: dict[str, object]) -> Path:
    """Write one run directory with a summary.json and return its root."""
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_0001"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return runs_root


def _empty_chart_summary(games_finished: int) -> dict[str, object]:
    """Build a summary with no chartable metrics."""
    return {
        "games_total": 5,
        "games_finished": games_finished,
        "games_errored": 0,
        "finish_rate": 0.0,
        "win_counts": {},
        "win_rates": {},
        "poorest_counts": {},
        "poorest_rates": {},
        "title_counts": {},
        "scandal_counts": {},
        "debt_counts": {},
        "theft_counts": {},
    }


def test_inspect_runs_hides_charts_when_no_games_finished(tmp_path: Path) -> None:
    """No charts should render for runs that have zero completed games."""
    summary = _empty_chart_summary(games_finished=0)
    summary["win_counts"] = {"0": 0}
    summary["win_rates"] = {"0": 0.0}
    runs_root = _write_summary_run(tmp_path, summary)
    streamlit = _FakeStreamlit(runs_root)

    _page_inspect_runs(streamlit)

    assert streamlit.altair_chart_calls == 0
    assert streamlit.bar_chart_calls == 0


def test_inspect_runs_hides_charts_when_no_chart_data(tmp_path: Path) -> None:
    """No charts should render when metric sections are empty."""
    runs_root = _write_summary_run(tmp_path, _empty_chart_summary(games_finished=3))
    streamlit = _FakeStreamlit(runs_root)

    _page_inspect_runs(streamlit)

    assert streamlit.altair_chart_calls == 0
    assert streamlit.bar_chart_calls == 0
