"""Run evaluation specs with artifact output."""

from __future__ import annotations

import argparse
from pathlib import Path

from highsociety.ops.cli import execute_spec, load_spec, resolve_output_dir
from highsociety.ops.spec import RunSpec


def main() -> None:
    """Run the evaluation CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate High Society specs.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to eval spec file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for artifacts (default: runs/<timestamp>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the spec without running games.",
    )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    _ensure_eval_mode(spec)
    if args.dry_run:
        resolve_output_dir(args.output)
        return
    execute_spec(args.spec, output_dir=args.output, dry_run=False, write_outputs=True)


def _ensure_eval_mode(spec: RunSpec) -> None:
    """Ensure the spec is in eval mode."""
    if spec.mode != "eval":
        raise ValueError("Eval CLI requires mode='eval'")


if __name__ == "__main__":
    main()
