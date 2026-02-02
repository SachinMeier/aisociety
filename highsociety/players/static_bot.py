"""Static budget player implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import MisfortuneKind, StatusCard, StatusKind
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.players.colors import BOT_COLORS
from highsociety.players.utils import choose_lowest_discard, fallback_action

_DEFAULT_TITLE_BUDGET = 40000
_DEFAULT_VALUE_SCALE = 4000
_DEFAULT_CANCEL_VALUE = 7
_DEBT_VALUE = 5

BudgetRule = Callable[["StaticBot", StatusCard, int], int]


def _title_budget(bot: "StaticBot", _card: StatusCard, _player_count: int) -> int:
    """Return the static title budget."""
    return bot.title_budget


def _possession_budget(bot: "StaticBot", card: StatusCard, player_count: int) -> int:
    """Return the budget for a possession card."""
    if card.value is None:
        raise InvalidState("Possession card missing value")
    return player_count * card.value * bot.value_scale


def _misfortune_budget(bot: "StaticBot", card: StatusCard, player_count: int) -> int:
    """Return the budget for a misfortune card."""
    if card.misfortune is None:
        raise InvalidState("Misfortune card missing kind")
    if card.misfortune == MisfortuneKind.SCANDAL:
        return bot.title_budget
    if card.misfortune == MisfortuneKind.DEBT:
        return player_count * _DEBT_VALUE * bot.value_scale
    return bot.cancel_value * bot.value_scale


_BUDGET_RULES: dict[StatusKind, BudgetRule] = {
    StatusKind.TITLE: _title_budget,
    StatusKind.POSSESSION: _possession_budget,
    StatusKind.MISFORTUNE: _misfortune_budget,
}


@dataclass
class StaticBot:
    """Bot that bids up to a static budget derived from card type values."""

    name: str = "static"
    title_budget: int = _DEFAULT_TITLE_BUDGET
    value_scale: int = _DEFAULT_VALUE_SCALE
    cancel_value: int = _DEFAULT_CANCEL_VALUE
    kind: str = "static"
    color: str = BOT_COLORS["static"]
    _player_id: int | None = None
    _player_count: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        _validate_non_negative_int("title_budget", self.title_budget)
        _validate_non_negative_int("cancel_value", self.cancel_value)
        if not isinstance(self.value_scale, int) or self.value_scale <= 0:
            raise ValueError("value_scale must be a positive integer")

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Reset the bot for a new game."""
        self._player_id = player_id
        player_count = game_config.get("player_count")
        if isinstance(player_count, int):
            self._player_count = player_count
        else:
            self._player_count = None

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Select an action using a static budget per status card type."""
        if not legal_actions:
            raise InvalidAction("No legal actions available")
        if self._player_id is None or self._player_count is None:
            raise InvalidState("Player has not been reset")
        discard_actions = [
            action
            for action in legal_actions
            if action.kind == ActionKind.DISCARD_POSSESSION
        ]
        if discard_actions:
            return choose_lowest_discard(discard_actions)
        if observation.round is None or observation.status_card is None:
            return fallback_action(legal_actions)
        budget = self._budget_for_card(observation.status_card)
        open_bid = observation.round.open_bids.get(self._player_id, ())
        current_bid = sum(open_bid)
        bid_actions = [action for action in legal_actions if action.kind == ActionKind.BID]
        scored: list[tuple[int, Action]] = []
        for action in bid_actions:
            new_total = current_bid + sum(action.cards)
            if new_total <= budget:
                scored.append((new_total, action))
        if scored:
            min_total = min(total for total, _ in scored)
            candidates = [action for total, action in scored if total == min_total]
            candidates.sort(key=lambda item: item.cards)
            return candidates[0]
        return fallback_action(legal_actions)

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler."""

    def _budget_for_card(self, card: StatusCard) -> int:
        """Return the budget for a given status card."""
        if self._player_count is None:
            raise InvalidState("Player has not been reset")
        rule = _BUDGET_RULES.get(card.kind)
        if rule is None:
            raise InvalidState(f"Unknown status card kind: {card.kind}")
        return rule(self, card, self._player_count)


def _validate_non_negative_int(name: str, value: object) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
