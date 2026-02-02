"""CLI helpers for loading specs and executing runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from highsociety.ops.artifacts import write_artifacts
from highsociety.ops.runner import RunManager, RunResult
from highsociety.ops.spec import RunSpec


def load_spec(path: Path) -> RunSpec:
    """Load a run spec from JSON or YAML."""
    spec, _ = load_spec_with_data(path)
    return spec


def load_spec_with_data(path: Path) -> tuple[RunSpec, Mapping[str, Any]]:
    """Load a run spec and return both the parsed spec and raw mapping."""
    data = _load_spec_data(path)
    if not isinstance(data, Mapping):
        raise ValueError("Spec file must contain a mapping")
    return RunSpec.from_mapping(data), data


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
    spec, spec_data = load_spec_with_data(spec_path)
    if dry_run:
        return None
    manager = RunManager()
    result = manager.run(spec)
    if write_outputs:
        target = resolve_output_dir(output_dir)
        write_artifacts(result, target, spec_source=spec_data)
    return result


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
