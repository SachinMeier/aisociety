"""Run a High Society batch spec from the command line."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from highsociety.ops.artifacts import write_artifacts
from highsociety.ops.runner import RunManager, RunResult
from highsociety.ops.spec import RunSpec


def load_spec(path: Path) -> RunSpec:
    """Load a run spec from JSON or YAML."""
    data = _load_spec_data(path)
    if not isinstance(data, Mapping):
        raise ValueError("Spec file must contain a mapping")
    return RunSpec.from_mapping(data)


def resolve_output_dir(output_dir: Path | None, base_dir: Path | None = None) -> Path:
    """Resolve the output directory for a run."""
    if output_dir is not None:
        return output_dir
    root = base_dir or Path("runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / timestamp


def execute_spec(
    spec_path: Path,
    output_dir: Path | None = None,
    dry_run: bool = False,
    write_outputs: bool = True,
) -> RunResult | None:
    """Execute a run spec and optionally write artifacts."""
    spec = load_spec(spec_path)
    if dry_run:
        return None
    manager = RunManager()
    result = manager.run(spec)
    if write_outputs:
        target = resolve_output_dir(output_dir)
        write_artifacts(result, target)
    return result


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
    execute_spec(
        args.spec,
        output_dir=args.output,
        dry_run=args.dry_run,
        write_outputs=not args.no_artifacts,
    )


def _load_spec_data(path: Path) -> Mapping[str, Any]:
    """Load spec data from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return _load_json(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    raise ValueError("Spec file must be .json or .yaml")


def _load_json(path: Path) -> Mapping[str, Any]:
    """Load spec data from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Load spec data from a YAML file, if available."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("PyYAML is required for YAML specs") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("YAML spec must be a mapping")
    return data


if __name__ == "__main__":
    main()
