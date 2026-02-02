"""Train a linear reinforcement learning bot via self-play."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Mapping

from highsociety.ml.checkpoints import load_linear_checkpoint, save_linear_checkpoint
from highsociety.ml.training.linear_train import (
    LinearTrainSpec,
    LinearTrainingConfig,
    train_linear_self_play,
)


def main() -> None:
    """Run the linear training CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a linear RL bot.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=None,
        help="Optional JSON/YAML training spec to load.",
    )
    parser.add_argument("--games", type=int, default=500, help="Number of self-play games.")
    parser.add_argument(
        "--player-count",
        type=int,
        default=3,
        help="Number of players per game (3-5).",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base RNG seed.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for updates.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log progress every N games (0 to disable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Checkpoint path (default: checkpoints/linear_<timestamp>.pkl).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from a linear checkpoint.",
    )
    args = parser.parse_args()

    spec = _load_spec(args.spec) if args.spec else None
    config = _resolve_config(args, spec)
    if config.log_every:
        logging.basicConfig(level=logging.INFO)
    output_path = _resolve_output_path(args.output, spec)
    resume_path = _resolve_resume_path(args.resume, spec)

    if resume_path is not None:
        model, encoder, _metadata = load_linear_checkpoint(resume_path)
        result = train_linear_self_play(config, model=model, encoder=encoder)
    else:
        result = train_linear_self_play(config)

    metadata = {
        "config": {
            "num_games": config.num_games,
            "player_count": config.player_count,
            "seed": config.seed,
            "epsilon": config.epsilon,
            "learning_rate": config.learning_rate,
            "log_every": config.log_every,
            "opponents": [spec.to_mapping() for spec in config.opponents],
            "opponent_weights": list(config.opponent_weights),
            "learner_seats": list(config.learner_seats) if config.learner_seats else None,
        },
        "metrics": {
            "average_reward": result.average_reward,
            "average_winners": result.average_winners,
        },
        "resume": str(resume_path) if resume_path else None,
    }
    if spec is not None:
        metadata["spec"] = spec.to_mapping()
    save_linear_checkpoint(output_path, result.model, result.encoder, metadata)
    print(f"Saved checkpoint to {output_path}")


def _default_output_path() -> Path:
    """Return the default checkpoint path for a training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("checkpoints") / f"linear_{timestamp}.pkl"


def _resolve_config(args: argparse.Namespace, spec: LinearTrainSpec | None) -> LinearTrainingConfig:
    """Resolve the training config from args/spec."""
    if spec is not None:
        return spec.to_config()
    return LinearTrainingConfig(
        num_games=args.games,
        player_count=args.player_count,
        seed=args.seed,
        epsilon=args.epsilon,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
    )


def _resolve_output_path(output: Path | None, spec: LinearTrainSpec | None) -> Path:
    """Resolve the output path from args/spec."""
    if output is not None:
        return output
    if spec is not None and spec.output:
        return Path(spec.output)
    return _default_output_path()


def _resolve_resume_path(resume: Path | None, spec: LinearTrainSpec | None) -> Path | None:
    """Resolve the resume path from args/spec."""
    if resume is not None:
        return resume
    if spec is not None and spec.resume:
        return Path(spec.resume)
    return None


def _load_spec(path: Path) -> LinearTrainSpec:
    """Load a LinearTrainSpec from a JSON/YAML file."""
    data = _load_spec_data(path)
    if not isinstance(data, Mapping):
        raise ValueError("Spec file must contain a mapping")
    return LinearTrainSpec.from_mapping(data)


def _load_spec_data(path: Path) -> Mapping[str, object]:
    """Load spec data from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return _load_json(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    raise ValueError("Spec file must be .json or .yaml")


def _load_json(path: Path) -> Mapping[str, object]:
    """Load JSON spec data from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Spec file must contain a mapping")
    return data


def _load_yaml(path: Path) -> Mapping[str, object]:
    """Load YAML spec data from disk."""
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
