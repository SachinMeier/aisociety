"""Model implementations for ML policies."""

from highsociety.ml.models.linear import LinearModel

__all__ = ["LinearModel"]

try:  # Optional dependency on torch.
    from highsociety.ml.models.hierarchical import HierarchicalConfig, HierarchicalPolicyValue
    from highsociety.ml.models.mlp import MLPConfig, MLPPolicyValue
except ImportError:  # pragma: no cover - torch is optional for linear-only use
    MLPConfig = None  # type: ignore[assignment]
    MLPPolicyValue = None  # type: ignore[assignment]
    HierarchicalConfig = None  # type: ignore[assignment]
    HierarchicalPolicyValue = None  # type: ignore[assignment]
else:
    __all__ = [
        "LinearModel",
        "MLPConfig",
        "MLPPolicyValue",
        "HierarchicalConfig",
        "HierarchicalPolicyValue",
    ]
