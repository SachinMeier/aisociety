"""Heuristic baseline player implementation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import StatusKind
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult

_STYLE_THRESHOLDS: dict[str, float] = {
    "cautious": 0.4,
    "balanced": 0.6,
    "aggressive": 0.8,
}


@dataclass
class HeuristicBot:
    """Rule-based bot with configurable risk thresholds."""

    name: str = "heuristic"
    style: str = "balanced"
    seed: int | None = None
    kind: str = "heuristic"
    _rng: random.Random = field(init=False)
    _player_id: int | None = None

    def __post_init__(self) -> None:
        """Initialize the RNG and validate style."""
        if self.style not in _STYLE_THRESHOLDS:
            raise ValueError(f"Unknown style: {self.style}")
        self._rng = random.Random(self.seed)

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Reset the bot for a new game, reseeding if needed."""
        self._player_id = player_id
        seed_value = self.seed
        if seed_value is None:
            config_seed = game_config.get("seed")
            if isinstance(config_seed, int):
                seed_value = config_seed
        if seed_value is not None:
            self._rng = random.Random(seed_value + player_id)

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Select an action using simple bid thresholds."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        if self._player_id is None:
            raise InvalidState("Player has not been reset")
        discard_actions = [
            action
            for action in legal_actions
            if action.kind == ActionKind.DISCARD_POSSESSION
        ]
        if discard_actions:
            return _choose_lowest_discard(discard_actions)
        if observation.round is None or observation.status_card is None:
            return _fallback_action(legal_actions)
        open_bid = observation.round.open_bids.get(self._player_id, ())
        current_bid = sum(open_bid)
        total_money = sum(observation.self_view.hand) + current_bid
        ratio = _STYLE_THRESHOLDS[self.style]
        if observation.status_card.kind == StatusKind.MISFORTUNE:
            ratio = min(1.0, ratio + 0.2)
        max_spend = total_money * ratio
        bid_actions = [action for action in legal_actions if action.kind == ActionKind.BID]
        scored: list[tuple[int, Action]] = []
        for action in bid_actions:
            new_total = current_bid + sum(action.cards)
            if new_total <= max_spend:
                scored.append((new_total, action))
        if scored:
            min_total = min(total for total, _ in scored)
            candidates = [action for total, action in scored if total == min_total]
            candidates.sort(key=lambda item: item.cards)
            return self._rng.choice(candidates)
        return _fallback_action(legal_actions)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler."""


def _choose_lowest_discard(actions: list[Action]) -> Action:
    """Choose the discard action with the lowest possession value."""
    return min(actions, key=lambda action: action.possession_value or 0)


def _fallback_action(actions: list[Action]) -> Action:
    """Return PASS if available; otherwise return the first action."""
    for action in actions:
        if action.kind == ActionKind.PASS:
            return action
    return actions[0]
