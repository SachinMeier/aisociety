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

try:  # Optional dependency on torch.
    from highsociety.ml.training.hierarchical_train import (
        HierarchicalTrainSpec,
        train_hierarchical,
    )
    from highsociety.ml.training.hierarchical_train import (
        TrainMetrics as HierarchicalTrainMetrics,
    )
except ImportError:  # pragma: no cover - torch is optional
    HierarchicalTrainSpec = None  # type: ignore[assignment]
    HierarchicalTrainMetrics = None  # type: ignore[assignment]
    train_hierarchical = None  # type: ignore[assignment]
else:
    __all__.extend([
        "HierarchicalTrainSpec",
        "HierarchicalTrainMetrics",
        "train_hierarchical",
    ])
