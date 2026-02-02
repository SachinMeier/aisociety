"""Tests for the dashboard module."""

from __future__ import annotations

import pytest

from highsociety.ops.dashboard.app import run_dashboard


def test_dashboard_requires_streamlit() -> None:
    """Dashboard requires streamlit to be installed."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            run_dashboard()
        return
    pytest.skip("streamlit available; dashboard import validated")
