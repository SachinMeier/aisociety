"""Basic feature encoder for High Society observations."""

from __future__ import annotations

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.cards import ALLOWED_MONEY_VALUES, MisfortuneKind, StatusKind
from highsociety.ml.action_space import ActionSpace, default_action_space

_MAX_MONEY_TOTAL = sum(ALLOWED_MONEY_VALUES)
_MAX_POSSESSION_SUM = sum(range(1, 11))
_MAX_TITLES = 3
_MAX_SCANDAL = 1
_MAX_DEBT = 1
_MAX_THEFT_PENDING = 1
_MAX_RED_REVEALED = 4
_MAX_STATUS_COUNTS = {"possession": 10, "title": 3, "misfortune": 3}


class BasicEncoder:
    """Encode observations and legal actions into fixed-size vectors."""

    def __init__(self, action_space: ActionSpace | None = None) -> None:
        """Initialize the encoder with a fixed action space."""
        self.action_space: ActionSpace = action_space or default_action_space()

    @property
    def feature_size(self) -> int:
        """Return the fixed feature vector size."""
        return 34

    def encode(self, observation: Observation) -> list[float]:
        """Encode an observation into a numeric feature vector."""
        features: list[float] = []
        hand_values = set(observation.self_view.hand)
        money_values = sorted(ALLOWED_MONEY_VALUES)
        for value in money_values:
            features.append(1.0 if value in hand_values else 0.0)
        round_view = observation.round
        current_bid = 0
        highest_bid = 0
        highest_bidder = None
        active_players = 0
        player_count = 0
        if round_view is not None:
            current_bid = sum(round_view.open_bids.get(observation.player_id, ()))
            highest_bid = round_view.highest_bid
            highest_bidder = round_view.highest_bidder
            active_players = sum(1 for passed in round_view.passed.values() if not passed)
            player_count = len(round_view.passed)
        total_money = sum(observation.self_view.hand) + current_bid
        features.append(_normalize(total_money, _MAX_MONEY_TOTAL))
        features.append(_normalize(current_bid, _MAX_MONEY_TOTAL))
        possession_sum = sum(observation.self_view.possessions)
        features.append(_normalize(possession_sum, _MAX_POSSESSION_SUM))
        features.append(_normalize(observation.self_view.titles, _MAX_TITLES))
        features.append(_normalize(observation.self_view.scandal, _MAX_SCANDAL))
        features.append(_normalize(observation.self_view.debt, _MAX_DEBT))
        features.append(_normalize(observation.self_view.theft_pending, _MAX_THEFT_PENDING))
        status = observation.status_card
        kind_features = [0.0, 0.0, 0.0]
        misfortune_features = [0.0, 0.0, 0.0]
        status_value = 0.0
        if status is not None:
            if status.kind == StatusKind.POSSESSION:
                kind_features[0] = 1.0
                status_value = _normalize(status.value or 0, 10)
            elif status.kind == StatusKind.TITLE:
                kind_features[1] = 1.0
            else:
                kind_features[2] = 1.0
                if status.misfortune == MisfortuneKind.SCANDAL:
                    misfortune_features[0] = 1.0
                elif status.misfortune == MisfortuneKind.DEBT:
                    misfortune_features[1] = 1.0
                elif status.misfortune == MisfortuneKind.THEFT:
                    misfortune_features[2] = 1.0
        features.extend(kind_features)
        features.append(status_value)
        features.extend(misfortune_features)
        features.append(_normalize(highest_bid, _MAX_MONEY_TOTAL))
        features.append(_normalize(current_bid, _MAX_MONEY_TOTAL))
        gap = max(0, highest_bid - current_bid)
        features.append(_normalize(gap, _MAX_MONEY_TOTAL))
        active_ratio = (active_players / player_count) if player_count else 0.0
        features.append(active_ratio)
        features.append(1.0 if highest_bidder == observation.player_id else 0.0)
        remaining = observation.public.remaining_counts
        features.append(_normalize(remaining.get("possession", 0), _MAX_STATUS_COUNTS["possession"]))
        features.append(_normalize(remaining.get("title", 0), _MAX_STATUS_COUNTS["title"]))
        features.append(_normalize(remaining.get("misfortune", 0), _MAX_STATUS_COUNTS["misfortune"]))
        features.append(_normalize(observation.public.red_revealed, _MAX_RED_REVEALED))
        return features

    def action_mask(self, legal_actions: list[Action]) -> list[int]:
        """Return a 0/1 mask for legal actions."""
        return self.action_space.mask(legal_actions)

    def config(self) -> dict[str, object]:
        """Return encoder configuration for checkpointing."""
        return {
            "name": "basic",
            "feature_size": self.feature_size,
            "action_space": self.action_space.config(),
        }

    @staticmethod
    def from_config(config: dict[str, object]) -> "BasicEncoder":
        """Construct a BasicEncoder from a config mapping."""
        action_space_config = config.get("action_space")
        if isinstance(action_space_config, dict):
            action_space = ActionSpace.from_config(action_space_config)
        else:
            action_space = default_action_space()
        return BasicEncoder(action_space=action_space)


def _normalize(value: int | float, maximum: int | float) -> float:
    """Normalize a numeric value to the 0-1 range."""
    if maximum <= 0:
        return 0.0
    clamped = max(0.0, float(value))
    return min(clamped / float(maximum), 1.0)
