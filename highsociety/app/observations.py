"""Observation builders for High Society."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from highsociety.domain.cards import MoneyCard, StatusCard, StatusKind
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState, RoundState


# Cache the full status deck - it never changes
@lru_cache(maxsize=1)
def _cached_status_deck() -> tuple[StatusCard, ...]:
    """Return a cached copy of the full status deck."""
    return tuple(RulesEngine.create_status_deck())


@dataclass(frozen=True)
class RoundView:
    """Public round information for a player observation."""

    turn_player: int
    highest_bid: int
    highest_bidder: int | None
    open_bids: dict[int, tuple[int, ...]]
    passed: dict[int, bool]


@dataclass(frozen=True)
class SelfView:
    """Private player information for a player observation."""

    hand: tuple[int, ...]
    possessions: tuple[int, ...]
    titles: int
    scandal: int
    debt: int
    theft_pending: int


@dataclass(frozen=True)
class PublicView:
    """Public game information for a player observation."""

    revealed_status: tuple[StatusCard, ...]
    remaining_counts: dict[str, int]
    red_revealed: int
    money_discarded: dict[int, tuple[int, ...]]


@dataclass(frozen=True)
class Observation:
    """Player-centric view of the game state (info-set safe)."""

    player_id: int
    status_card: StatusCard | None
    round: RoundView | None
    self_view: SelfView
    public: PublicView


def build_observation(state: GameState, player_id: int) -> Observation:
    """Build an info-set-safe observation for the requested player."""
    player = _get_player(state, player_id)
    round_view = _build_round_view(state.round, state.players)
    self_view = SelfView(
        hand=_sorted_values(player.hand),
        possessions=_sorted_possessions(player),
        titles=player.titles,
        scandal=player.scandal,
        debt=player.debt,
        theft_pending=player.theft_pending,
    )
    public = PublicView(
        revealed_status=_revealed_status_cards(state),
        remaining_counts=_remaining_counts(state),
        red_revealed=state.red_revealed,
        money_discarded=_money_discarded(state.players),
    )
    status_card = state.round.card if state.round is not None else None
    return Observation(
        player_id=player_id,
        status_card=status_card,
        round=round_view,
        self_view=self_view,
        public=public,
    )


def _build_round_view(
    round_state: RoundState | None, players: list[PlayerState]
) -> RoundView | None:
    """Build round info for an observation."""
    if round_state is None:
        return None
    open_bids = {player.id: _sorted_values(player.open_bid) for player in players}
    passed = {player.id: player.passed for player in players}
    return RoundView(
        turn_player=round_state.turn_player,
        highest_bid=round_state.highest_bid,
        highest_bidder=round_state.highest_bidder,
        open_bids=open_bids,
        passed=passed,
    )


def _get_player(state: GameState, player_id: int) -> PlayerState:
    """Return the player with the given id."""
    for player in state.players:
        if player.id == player_id:
            return player
    raise InvalidState("Unknown player id")


def _sorted_values(cards: list[MoneyCard]) -> tuple[int, ...]:
    """Return a sorted tuple of money card values."""
    return tuple(sorted(card.value for card in cards))


def _sorted_possessions(player: PlayerState) -> tuple[int, ...]:
    """Return a sorted tuple of possession values for a player."""
    return tuple(sorted(card.value for card in player.possessions))


def _money_discarded(players: list[PlayerState]) -> dict[int, tuple[int, ...]]:
    """Return each player's discarded money card values."""
    return {player.id: _sorted_values(player.money_discarded) for player in players}


def _remaining_counts(state: GameState) -> dict[str, int]:
    """Return remaining status card counts by kind."""
    counts = {"possession": 0, "title": 0, "misfortune": 0}
    for card in state.status_deck:
        if card.kind == StatusKind.POSSESSION:
            counts["possession"] += 1
        elif card.kind == StatusKind.TITLE:
            counts["title"] += 1
        else:
            counts["misfortune"] += 1
    return counts


def _revealed_status_cards(state: GameState) -> tuple[StatusCard, ...]:
    """Return a stable ordering of revealed status cards."""
    full_deck = _cached_status_deck()  # Use cached deck instead of creating new one
    remaining_counts = Counter(state.status_deck)
    revealed: list[StatusCard] = []
    for card in full_deck:
        if remaining_counts[card] > 0:
            remaining_counts[card] -= 1
            continue
        revealed.append(card)
    return tuple(revealed)
