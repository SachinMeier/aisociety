"""Run a High Society batch spec from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from highsociety.ops.cli import execute_spec, load_spec, resolve_output_dir


def main() -> None:
    """Run the CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run High Society specs.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to spec file.")
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
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Run games without writing artifacts.",
    )
    args = parser.parse_args()
    if args.dry_run:
        load_spec(args.spec)
        resolve_output_dir(args.output)
        return
    execute_spec(
        args.spec,
        output_dir=args.output,
        dry_run=False,
        write_outputs=not args.no_artifacts,
    )


if __name__ == "__main__":
    main()
