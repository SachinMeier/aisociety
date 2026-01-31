"""Streamlit dashboard for running and inspecting games."""

from __future__ import annotations

from pathlib import Path

from highsociety.ops.cli import execute_spec, load_spec, resolve_output_dir
from highsociety.ops.metrics import compute_summary


def run_dashboard() -> None:
    """Run the Streamlit dashboard."""
    try:
        import streamlit
    except ImportError as exc:
        raise ImportError("streamlit is required to run the dashboard") from exc
    streamlit.title("High Society Run Dashboard")
    spec_path = streamlit.text_input("Spec path", value="")
    output_root = streamlit.text_input("Output directory (optional)", value="")
    dry_run = streamlit.checkbox("Dry run", value=False)
    run_clicked = streamlit.button("Run")
    if not run_clicked:
        streamlit.info("Provide a spec path and click Run to start.")
        return
    if not spec_path:
        streamlit.error("Spec path is required.")
        return
    path = Path(spec_path)
    if not path.exists():
        streamlit.error(f"Spec file not found: {path}")
        return
    spec = load_spec(path)
    if dry_run:
        streamlit.success(f"Spec validated: {spec.mode} ({spec.num_games} games)")
        return
    output_dir = resolve_output_dir(Path(output_root) if output_root else None)
    result = execute_spec(path, output_dir=output_dir, dry_run=False, write_outputs=True)
    if result is None:
        streamlit.error("Run failed to execute.")
        return
    summary = compute_summary(result)
    streamlit.success(f"Run complete: {output_dir}")
    streamlit.json(summary.to_dict())
