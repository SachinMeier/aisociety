"""Observation builders for info-set-safe ML features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from highsociety.domain.cards import MisfortuneKind, MoneyCard, StatusCard, StatusKind
from highsociety.domain.errors import InvalidState
from highsociety.domain.state import GameState, PlayerState


@dataclass(frozen=True)
class Observation:
    """Info-set-safe observation for a single player."""

    player_id: int
    player_hand: tuple[int, ...]
    player_open_bid: tuple[int, ...]
    player_possessions: tuple[int, ...]
    player_titles: int
    player_scandal: int
    player_debt: int
    player_theft_pending: int
    player_passed: bool
    round_kind: StatusKind | None
    round_value: int | None
    round_misfortune: MisfortuneKind | None
    round_highest_bid: int
    round_highest_bidder: int | None
    round_any_bid: bool
    round_turn_player: int | None
    round_starting_player: int | None
    pending_discard_options: tuple[int, ...] | None
    status_deck_size: int
    red_revealed: int
    game_over: bool


def build_observation(state: GameState, player_id: int) -> Observation:
    """Build an info-set-safe observation for the requested player."""
    player = _get_player(state, player_id)
    round_state = state.round
    if round_state is None:
        round_kind = None
        round_value = None
        round_misfortune = None
        round_highest_bid = 0
        round_highest_bidder = None
        round_any_bid = False
        round_turn_player = None
        round_starting_player = None
    else:
        card = round_state.card
        round_kind = card.kind
        round_value = card.value
        round_misfortune = card.misfortune
        round_highest_bid = round_state.highest_bid
        round_highest_bidder = round_state.highest_bidder
        round_any_bid = round_state.any_bid
        round_turn_player = round_state.turn_player
        round_starting_player = round_state.starting_player
    pending_discard = state.pending_discard
    pending_options = (
        pending_discard.options
        if pending_discard is not None and pending_discard.player_id == player_id
        else None
    )
    return Observation(
        player_id=player_id,
        player_hand=_money_values(player.hand),
        player_open_bid=_money_values(player.open_bid),
        player_possessions=_status_values(player.possessions),
        player_titles=player.titles,
        player_scandal=player.scandal,
        player_debt=player.debt,
        player_theft_pending=player.theft_pending,
        player_passed=player.passed,
        round_kind=round_kind,
        round_value=round_value,
        round_misfortune=round_misfortune,
        round_highest_bid=round_highest_bid,
        round_highest_bidder=round_highest_bidder,
        round_any_bid=round_any_bid,
        round_turn_player=round_turn_player,
        round_starting_player=round_starting_player,
        pending_discard_options=pending_options,
        status_deck_size=len(state.status_deck),
        red_revealed=state.red_revealed,
        game_over=state.game_over,
    )


def _get_player(state: GameState, player_id: int) -> PlayerState:
    """Return the PlayerState for a given id or raise if missing."""
    for player in state.players:
        if player.id == player_id:
            return player
    raise InvalidState("Unknown player id")


def _money_values(cards: Iterable[MoneyCard]) -> tuple[int, ...]:
    """Return sorted money card values for the given cards."""
    return tuple(sorted(card.value for card in cards))


def _status_values(cards: Iterable[StatusCard]) -> tuple[int, ...]:
    """Return sorted possession values for the given status cards."""
    values = [card.value for card in cards if card.value is not None]
    return tuple(sorted(values))
