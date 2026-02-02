"""Linear model for simple RL baselines."""

from __future__ import annotations

from typing import Mapping

import numpy as np


class LinearModel:
    """Simple linear model with incremental updates."""

    def __init__(self, weights: np.ndarray, bias: float = 0.0) -> None:
        """Initialize the linear model with weights and bias."""
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array")
        self.weights = weights.astype(np.float32, copy=False)
        self.bias = float(bias)

    @classmethod
    def initialize(cls, feature_size: int, seed: int | None = None) -> "LinearModel":
        """Initialize weights with a small random normal distribution."""
        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        rng = np.random.default_rng(seed)
        weights = rng.normal(loc=0.0, scale=0.01, size=feature_size).astype(np.float32)
        return cls(weights=weights, bias=0.0)

    def predict(self, features: np.ndarray) -> float:
        """Return the linear prediction for the given features."""
        if features.shape != self.weights.shape:
            raise ValueError("feature vector has wrong shape")
        return float(np.dot(self.weights, features) + self.bias)

    def update(self, features: np.ndarray, reward: float, learning_rate: float) -> None:
        """Apply a simple delta rule update for the given reward."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        prediction = self.predict(features)
        error = float(reward) - prediction
        self.weights += learning_rate * error * features
        self.bias += learning_rate * error

    def to_state(self) -> dict[str, object]:
        """Serialize model state for checkpointing."""
        return {"weights": self.weights.tolist(), "bias": self.bias}

    @staticmethod
    def from_state(state: Mapping[str, object]) -> "LinearModel":
        """Deserialize model state from a checkpoint mapping."""
        weights = state.get("weights", [])
        bias = state.get("bias", 0.0)
        if not isinstance(weights, list):
            raise ValueError("weights must be a list")
        return LinearModel(weights=np.array(weights, dtype=np.float32), bias=float(bias))
