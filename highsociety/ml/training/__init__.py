"""Training loops for ML bots."""

from highsociety.ml.training.linear_train import (
    LinearTrainingConfig,
    LinearTrainingResult,
    LinearTrainSpec,
    train_linear_self_play,
)

__all__ = [
    "LinearTrainSpec",
    "LinearTrainingConfig",
    "LinearTrainingResult",
    "train_linear_self_play",
]
