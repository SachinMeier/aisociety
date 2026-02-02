"""Streamlit entrypoint for the High Society dashboard."""

from __future__ import annotations

from highsociety.ops.dashboard.app import run_dashboard


def main() -> None:
    """Run the dashboard entrypoint."""
    run_dashboard()


if __name__ == "__main__":
    main()
