"""Linear model for action scoring and value prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass
class LinearModel:
    """Linear model with a weight vector and bias term."""

    weights: np.ndarray
    bias: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize model parameters."""
        if self.weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if not isinstance(self.bias, (float, int)):
            raise ValueError("bias must be a numeric type")
        self.bias = float(self.bias)

    def predict(self, features: np.ndarray) -> float:
        """Return the linear score for a single feature vector."""
        self._validate_features(features)
        return float(np.dot(self.weights, features) + self.bias)

    def batch_predict(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Return scores for a batch of feature vectors."""
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] != self.weights.shape[0]:
            raise ValueError("feature matrix shape does not match weights")
        return feature_matrix @ self.weights + self.bias

    def update(self, features: np.ndarray, target: float, learning_rate: float) -> float:
        """Apply a gradient step toward a target value and return the error."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        prediction = self.predict(features)
        error = float(target - prediction)
        self.weights += learning_rate * error * features
        self.bias += learning_rate * error
        return error

    def to_state(self) -> dict[str, object]:
        """Return a serializable state representation for checkpoints."""
        return {"weights": self.weights.tolist(), "bias": self.bias}

    @staticmethod
    def from_state(state: Mapping[str, object]) -> "LinearModel":
        """Instantiate a linear model from a serialized state mapping."""
        weights = np.asarray(state.get("weights", []), dtype=np.float32)
        bias = float(state.get("bias", 0.0))
        return LinearModel(weights=weights, bias=bias)

    @staticmethod
    def initialize(
        feature_size: int, seed: int | None = None, scale: float = 0.01
    ) -> "LinearModel":
        """Create a new model with small random weights."""
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        rng = np.random.default_rng(seed)
        weights = rng.normal(0.0, scale, size=feature_size).astype(np.float32)
        return LinearModel(weights=weights, bias=0.0)

    def _validate_features(self, features: np.ndarray) -> None:
        """Validate that features match the model's expected shape."""
        if features.ndim != 1 or features.shape[0] != self.weights.shape[0]:
            raise ValueError("feature vector shape does not match weights")
