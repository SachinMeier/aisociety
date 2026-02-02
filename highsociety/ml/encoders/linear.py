"""Linear feature encoder for High Society observations and actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import ALLOWED_MONEY_VALUES, MisfortuneKind, StatusKind

_MONEY_VALUES = tuple(ALLOWED_MONEY_VALUES)
_MAX_MONEY = float(sum(_MONEY_VALUES))
_MAX_MONEY_CARD = float(max(_MONEY_VALUES))
_MAX_POSSESSION_SUM = 55.0
_MAX_POSSESSION_COUNT = 10.0
_MAX_TITLES = 3.0
_MAX_RED = 4.0


@dataclass(frozen=True)
class LinearFeatureEncoder:
    """Encode observations and actions into linear feature vectors."""

    version: int = 1

    def feature_names(self) -> tuple[str, ...]:
        """Return the ordered feature names for this encoder."""
        return _build_feature_names()

    def feature_size(self) -> int:
        """Return the dimensionality of encoded feature vectors."""
        return len(self.feature_names())

    def encode(self, observation: Observation, action: Action) -> np.ndarray:
        """Encode an observation-action pair into a numeric feature vector."""
        features: list[float] = []
        features.extend(_encode_hand(observation))
        features.extend(_encode_private_state(observation))
        features.extend(_encode_round_state(observation))
        features.extend(_encode_status_card(observation))
        features.extend(_encode_public_state(observation))
        features.extend(_encode_action(observation, action))
        return np.asarray(features, dtype=np.float32)

    def to_config(self) -> dict[str, object]:
        """Return a serializable configuration mapping for checkpoints."""
        return {"version": self.version}

    @staticmethod
    def from_config(config: Mapping[str, object]) -> "LinearFeatureEncoder":
        """Instantiate an encoder from a configuration mapping."""
        version = int(config.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported encoder version: {version}")
        return LinearFeatureEncoder(version=version)


def _build_feature_names() -> tuple[str, ...]:
    """Return the canonical feature ordering for the linear encoder."""
    names: list[str] = []
    for value in _MONEY_VALUES:
        names.append(f"hand_has_{value}")
    names.extend(
        [
            "hand_total",
            "possessions_total",
            "possessions_count",
            "titles",
            "scandal",
            "debt",
            "theft_pending",
            "round_highest_bid",
            "round_highest_bidder_is_self",
            "round_open_bid_total",
            "round_active_players",
            "status_is_possession",
            "status_is_title",
            "status_is_misfortune",
            "status_value",
            "misfortune_is_scandal",
            "misfortune_is_debt",
            "misfortune_is_theft",
            "public_remaining_possession",
            "public_remaining_title",
            "public_remaining_misfortune",
            "public_red_revealed",
            "public_money_discarded_self",
            "public_money_discarded_total",
            "action_is_pass",
            "action_is_bid",
            "action_is_discard",
            "action_bid_sum",
            "action_bid_count",
            "action_bid_max",
            "action_total_after",
            "action_discard_value",
        ]
    )
    return tuple(names)


def _encode_hand(observation: Observation) -> list[float]:
    """Encode the player's current hand as binary indicators and totals."""
    hand_values = set(observation.self_view.hand)
    features = [1.0 if value in hand_values else 0.0 for value in _MONEY_VALUES]
    total = float(sum(observation.self_view.hand))
    features.append(total / _MAX_MONEY if _MAX_MONEY else 0.0)
    return features


def _encode_private_state(observation: Observation) -> list[float]:
    """Encode private holdings (possessions, titles, misfortunes)."""
    possessions_total = float(sum(observation.self_view.possessions))
    possessions_count = float(len(observation.self_view.possessions))
    titles = float(observation.self_view.titles)
    scandal = float(observation.self_view.scandal)
    debt = float(observation.self_view.debt)
    theft_pending = float(observation.self_view.theft_pending)
    return [
        possessions_total / _MAX_POSSESSION_SUM,
        possessions_count / _MAX_POSSESSION_COUNT,
        titles / _MAX_TITLES,
        scandal,
        debt,
        theft_pending,
    ]


def _encode_round_state(observation: Observation) -> list[float]:
    """Encode current round information for the player."""
    round_view = observation.round
    player_id = observation.player_id
    player_count = _player_count(observation)
    if round_view is None:
        return [
            0.0,
            0.0,
            0.0,
            float(player_count) / float(player_count),
        ]
    highest_bid = float(round_view.highest_bid)
    highest_bidder_is_self = 1.0 if round_view.highest_bidder == player_id else 0.0
    open_bid_total = float(sum(round_view.open_bids.get(player_id, ())))
    active_players = float(sum(1 for passed in round_view.passed.values() if not passed))
    return [
        highest_bid / _MAX_MONEY,
        highest_bidder_is_self,
        open_bid_total / _MAX_MONEY,
        active_players / float(player_count),
    ]


def _encode_status_card(observation: Observation) -> list[float]:
    """Encode the revealed status card into categorical features."""
    card = observation.status_card
    if card is None:
        return [0.0] * 7
    is_possession = 1.0 if card.kind == StatusKind.POSSESSION else 0.0
    is_title = 1.0 if card.kind == StatusKind.TITLE else 0.0
    is_misfortune = 1.0 if card.kind == StatusKind.MISFORTUNE else 0.0
    status_value = float(card.value or 0) / 10.0
    misfortune = card.misfortune
    is_scandal = 1.0 if misfortune == MisfortuneKind.SCANDAL else 0.0
    is_debt = 1.0 if misfortune == MisfortuneKind.DEBT else 0.0
    is_theft = 1.0 if misfortune == MisfortuneKind.THEFT else 0.0
    return [
        is_possession,
        is_title,
        is_misfortune,
        status_value,
        is_scandal,
        is_debt,
        is_theft,
    ]


def _encode_public_state(observation: Observation) -> list[float]:
    """Encode public information such as remaining cards and discards."""
    remaining = observation.public.remaining_counts
    remaining_possession = float(remaining.get("possession", 0)) / 10.0
    remaining_title = float(remaining.get("title", 0)) / 3.0
    remaining_misfortune = float(remaining.get("misfortune", 0)) / 3.0
    red_revealed = float(observation.public.red_revealed) / _MAX_RED
    player_id = observation.player_id
    discarded_self = float(sum(observation.public.money_discarded.get(player_id, ())))
    discarded_total = float(
        sum(sum(values) for values in observation.public.money_discarded.values())
    )
    player_count = _player_count(observation)
    discarded_total_scale = _MAX_MONEY * float(player_count)
    discarded_total_norm = discarded_total / discarded_total_scale if discarded_total_scale else 0.0
    return [
        remaining_possession,
        remaining_title,
        remaining_misfortune,
        red_revealed,
        discarded_self / _MAX_MONEY,
        discarded_total_norm,
    ]


def _encode_action(observation: Observation, action: Action) -> list[float]:
    """Encode the candidate action into action-specific features."""
    is_pass = 1.0 if action.kind == ActionKind.PASS else 0.0
    is_bid = 1.0 if action.kind == ActionKind.BID else 0.0
    is_discard = 1.0 if action.kind == ActionKind.DISCARD_POSSESSION else 0.0
    bid_sum = float(sum(action.cards))
    bid_count = float(len(action.cards))
    bid_max = float(max(action.cards)) if action.cards else 0.0
    round_view = observation.round
    if round_view is None:
        open_bid_total = 0.0
    else:
        open_bid_total = float(sum(round_view.open_bids.get(observation.player_id, ())))
    if action.kind == ActionKind.BID:
        total_after = open_bid_total + bid_sum
    else:
        total_after = open_bid_total
    discard_value = float(action.possession_value or 0)
    return [
        is_pass,
        is_bid,
        is_discard,
        bid_sum / _MAX_MONEY,
        bid_count / float(len(_MONEY_VALUES)),
        bid_max / _MAX_MONEY_CARD if _MAX_MONEY_CARD else 0.0,
        total_after / _MAX_MONEY,
        discard_value / 10.0,
    ]


def _player_count(observation: Observation) -> int:
    """Return the total player count inferred from public data."""
    count = len(observation.public.money_discarded)
    return count if count > 0 else 1
