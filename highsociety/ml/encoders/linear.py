"""Linear feature encoder for the linear RL bot."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import StatusKind


class LinearFeatureEncoder:
    """Encode observations/actions into linear feature vectors."""

    _FEATURE_NAMES = (
        "bias",
        "action_is_pass",
        "action_bid_sum",
        "action_bid_count",
        "player_hand_sum",
        "player_open_bid_sum",
        "player_possession_sum",
        "player_titles",
        "player_scandal",
        "player_debt",
        "player_theft_pending",
        "round_highest_bid",
        "round_any_bid",
        "round_card_value",
        "round_is_possession",
        "round_is_title",
        "round_is_misfortune",
        "pending_discard",
    )

    def __init__(self, version: int = 1) -> None:
        """Initialize the encoder with a version identifier."""
        self.version = version

    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names."""
        return list(self._FEATURE_NAMES)

    def feature_size(self) -> int:
        """Return the size of the feature vector."""
        return len(self._FEATURE_NAMES)

    def encode(self, observation: Observation, action: Action) -> np.ndarray:
        """Encode an observation/action pair into a feature vector."""
        action_bid_sum = float(sum(action.cards)) if action.kind == ActionKind.BID else 0.0
        action_bid_count = float(len(action.cards)) if action.kind == ActionKind.BID else 0.0
        round_kind = observation.round_kind
        features = np.array(
            [
                1.0,
                1.0 if action.kind == ActionKind.PASS else 0.0,
                action_bid_sum,
                action_bid_count,
                float(sum(observation.player_hand)),
                float(sum(observation.player_open_bid)),
                float(sum(observation.player_possessions)),
                float(observation.player_titles),
                float(observation.player_scandal),
                float(observation.player_debt),
                float(observation.player_theft_pending),
                float(observation.round_highest_bid),
                1.0 if observation.round_any_bid else 0.0,
                float(observation.round_value or 0),
                1.0 if round_kind == StatusKind.POSSESSION else 0.0,
                1.0 if round_kind == StatusKind.TITLE else 0.0,
                1.0 if round_kind == StatusKind.MISFORTUNE else 0.0,
                1.0 if observation.pending_discard_options else 0.0,
            ],
            dtype=np.float32,
        )
        return features

    def to_config(self) -> dict[str, object]:
        """Serialize the encoder configuration."""
        return {"version": self.version}

    @staticmethod
    def from_config(config: Mapping[str, object]) -> "LinearFeatureEncoder":
        """Deserialize the encoder configuration."""
        version = config.get("version", 1)
        return LinearFeatureEncoder(version=int(version))
