"""Checkpoint helpers for ML models."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from highsociety.ml.encoders.linear import LinearFeatureEncoder
    from highsociety.ml.models.linear import LinearModel

_DEFAULT_LINEAR_FILENAME = "linear.pkl"


@dataclass(frozen=True)
class CheckpointBundle:
    """Container for checkpoint contents."""

    model_state: dict[str, Any]
    model_config: dict[str, object]
    encoder_config: dict[str, object]
    metrics: dict[str, object]
    metadata: dict[str, object]


def save_checkpoint(
    path: Path | str,
    model_state: dict[str, Any],
    model_config: dict[str, object],
    encoder_config: dict[str, object],
    metrics: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Save a checkpoint to a file or directory."""
    target = Path(path)
    metrics = metrics or {}
    metadata = metadata or {}
    if target.suffix:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model_state,
            "model_config": model_config,
            "encoder_config": encoder_config,
            "metrics": metrics,
            "metadata": metadata,
        }
        torch = _require_torch()
        torch.save(payload, target)
        return target
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / "model.pt"
    torch = _require_torch()
    torch.save({"model_state": model_state}, model_path)
    _write_json(target / "config.json", model_config)
    _write_json(target / "encoder.json", encoder_config)
    if metrics:
        _write_json(target / "metrics.json", metrics)
    if metadata:
        _write_json(target / "metadata.json", metadata)
    return model_path


def load_checkpoint(path: Path | str) -> CheckpointBundle:
    """Load a checkpoint from a file or directory."""
    target = Path(path)
    if target.is_dir():
        model_path = target / "model.pt"
        model_state = _load_torch(model_path).get("model_state", {})
        model_config = _read_json(target / "config.json")
        encoder_config = _read_json(target / "encoder.json")
        metrics = _read_json(target / "metrics.json") if (target / "metrics.json").exists() else {}
        metadata = (
            _read_json(target / "metadata.json")
            if (target / "metadata.json").exists()
            else {}
        )
        return CheckpointBundle(
            model_state=model_state,
            model_config=model_config,
            encoder_config=encoder_config,
            metrics=metrics,
            metadata=metadata,
        )
    payload = _load_torch(target)
    return CheckpointBundle(
        model_state=payload.get("model_state", {}),
        model_config=payload.get("model_config", {}),
        encoder_config=payload.get("encoder_config", {}),
        metrics=payload.get("metrics", {}),
        metadata=payload.get("metadata", {}),
    )


def save_linear_checkpoint(
    path: Path | str,
    model: "LinearModel",
    encoder: "LinearFeatureEncoder",
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Save a linear model checkpoint to a pickle file."""
    target = _resolve_linear_path(Path(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.to_state(),
        "encoder": encoder.to_config(),
        "metadata": dict(metadata or {}),
    }
    with target.open("wb") as handle:
        pickle.dump(payload, handle)
    return target


def load_linear_checkpoint(
    path: Path | str,
) -> tuple["LinearModel", "LinearFeatureEncoder", dict[str, object]]:
    """Load a linear checkpoint and return model, encoder, and metadata."""
    from highsociety.ml.encoders.linear import LinearFeatureEncoder
    from highsociety.ml.models.linear import LinearModel

    target = _resolve_linear_path(Path(path))
    with target.open("rb") as handle:
        payload = pickle.load(handle)
    model = LinearModel.from_state(payload.get("model", {}))
    encoder = LinearFeatureEncoder.from_config(payload.get("encoder", {}))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dict")
    return model, encoder, dict(metadata)


def _resolve_linear_path(path: Path) -> Path:
    """Resolve the checkpoint file path, allowing directory inputs."""
    if path.suffix:
        return path
    return path / _DEFAULT_LINEAR_FILENAME


def _load_torch(path: Path) -> dict[str, Any]:
    """Load a torch checkpoint payload."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    torch = _require_torch()
    return torch.load(path, map_location="cpu")


def _write_json(path: Path, data: dict[str, object]) -> None:
    """Write a JSON file to disk."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid checkpoint JSON: {path}")
    return data


def _require_torch() -> Any:
    """Import torch or raise a helpful error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required to load/save ML checkpoints") from exc
    return torch
