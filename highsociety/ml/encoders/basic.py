"""Basic feature encoder for MLP training."""

from __future__ import annotations

from typing import Iterable

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.cards import ALLOWED_MONEY_VALUES, StatusKind
from highsociety.ml.action_space import ActionSpace, default_action_space


class BasicEncoder:
    """Encode observations into fixed-length numeric features."""

    def __init__(self, action_space: ActionSpace | None = None) -> None:
        """Initialize the encoder with an optional action space."""
        self.action_space = action_space or default_action_space()
        self._feature_names = self._build_feature_names()
        self.feature_size = len(self._feature_names)

    def encode(self, observation: Observation) -> list[float]:
        """Encode an observation into a numeric feature vector."""
        features: list[float] = []
        features.extend(
            self._one_hot_values(observation.player_hand, ALLOWED_MONEY_VALUES)
        )
        features.extend(
            self._one_hot_values(observation.player_open_bid, ALLOWED_MONEY_VALUES)
        )
        features.extend(self._one_hot_values(observation.player_possessions, range(1, 11)))
        features.append(float(observation.player_titles))
        features.append(float(observation.player_scandal))
        features.append(float(observation.player_debt))
        features.append(float(observation.player_theft_pending))
        features.append(1.0 if observation.round_kind == StatusKind.POSSESSION else 0.0)
        features.append(1.0 if observation.round_kind == StatusKind.TITLE else 0.0)
        features.append(1.0 if observation.round_kind == StatusKind.MISFORTUNE else 0.0)
        features.append(float(observation.round_value or 0))
        features.append(float(observation.round_highest_bid))
        features.append(1.0 if observation.round_any_bid else 0.0)
        features.append(1.0 if observation.pending_discard_options else 0.0)
        features.append(1.0 if observation.player_passed else 0.0)
        features.append(float(observation.status_deck_size))
        features.append(float(observation.red_revealed))
        return features

    def action_mask(self, legal_actions: list[Action]) -> list[int]:
        """Return a 0/1 mask aligned to the action space."""
        return self.action_space.mask(legal_actions)

    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names."""
        return list(self._feature_names)

    def _one_hot_values(
        self, values: Iterable[int], allowed: Iterable[int]
    ) -> list[float]:
        """Return 0/1 indicators for allowed values present in values."""
        value_set = set(values)
        return [1.0 if value in value_set else 0.0 for value in allowed]

    def _build_feature_names(self) -> list[str]:
        """Build the ordered list of feature names."""
        names: list[str] = []
        names.extend([f"hand_{value}" for value in ALLOWED_MONEY_VALUES])
        names.extend([f"open_bid_{value}" for value in ALLOWED_MONEY_VALUES])
        names.extend([f"possession_{value}" for value in range(1, 11)])
        names.extend(
            [
                "player_titles",
                "player_scandal",
                "player_debt",
                "player_theft_pending",
                "round_is_possession",
                "round_is_title",
                "round_is_misfortune",
                "round_card_value",
                "round_highest_bid",
                "round_any_bid",
                "pending_discard",
                "player_passed",
                "status_deck_size",
                "red_revealed",
            ]
        )
        return names
