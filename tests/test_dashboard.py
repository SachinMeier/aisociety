"""Tests for the dashboard module."""

from __future__ import annotations

import pytest

from highsociety.ops.dashboard.app import _stretch_width_kwargs, run_dashboard


def test_dashboard_requires_streamlit() -> None:
    """Dashboard requires streamlit to be installed."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            run_dashboard()
        return
    pytest.skip("streamlit available; dashboard import validated")


def test_stretch_width_kwargs_prefers_width() -> None:
    """Width-based APIs should use stretch mode instead of deprecated flags."""

    def _has_width(*, width: str = "content") -> None:
        del width

    assert _stretch_width_kwargs(_has_width) == {"width": "stretch"}


def test_stretch_width_kwargs_falls_back_to_use_container_width() -> None:
    """Older APIs should keep using use_container_width for compatibility."""

    def _legacy_chart(*, use_container_width: bool = False) -> None:
        del use_container_width

    assert _stretch_width_kwargs(_legacy_chart) == {"use_container_width": True}
