"""Tests for observation and legal action helpers."""

from __future__ import annotations

from highsociety.app.legal_actions import legal_actions
from highsociety.app.observations import Observation, build_observation
from highsociety.domain.actions import ActionKind
from highsociety.domain.cards import MoneyCard, StatusCard, StatusKind
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState, RoundState


def _make_players(hands: list[list[int]]) -> list[PlayerState]:
    """Build player states from explicit money card values."""
    players: list[PlayerState] = []
    for idx, values in enumerate(hands):
        hand = [MoneyCard(value) for value in values]
        players.append(PlayerState(id=idx, hand=hand))
    return players


def _collect_observed_values(obs: Observation) -> set[int]:
    """Collect all numeric values visible in the observation."""
    values = set(obs.self_view.hand)
    values.update(obs.self_view.possessions)
    if obs.status_card is not None and obs.status_card.value is not None:
        values.add(obs.status_card.value)
    if obs.round is not None:
        for cards in obs.round.open_bids.values():
            values.update(cards)
    for cards in obs.public.money_discarded.values():
        values.update(cards)
    for card in obs.public.revealed_status:
        if card.value is not None:
            values.add(card.value)
    return values


def _make_round_state(card: StatusCard, turn_player: int) -> RoundState:
    """Create a round state with defaults for testing."""
    return RoundState(
        card=card,
        starting_player=turn_player,
        turn_player=turn_player,
    )


def test_observation_is_info_set_safe() -> None:
    """Observation excludes opponent private hands."""
    players = _make_players([[1000, 2000], [3000], [4000]])
    deck = RulesEngine.create_status_deck()
    card = deck.pop(0)
    round_state = _make_round_state(card, turn_player=0)
    state = GameState(players=players, status_deck=deck, round=round_state, starting_player=0)

    obs = build_observation(state, player_id=0)
    observed_values = _collect_observed_values(obs)

    assert obs.self_view.hand == (1000, 2000)
    assert 3000 not in observed_values
    assert 4000 not in observed_values


def test_legal_actions_include_explicit_bid_sets() -> None:
    """Legal actions enumerate explicit card subsets."""
    players = _make_players([[1000, 2000], [3000], [4000]])
    deck = RulesEngine.create_status_deck()
    card = StatusCard(kind=StatusKind.POSSESSION, value=5)
    round_state = _make_round_state(card, turn_player=0)
    state = GameState(players=players, status_deck=deck, round=round_state, starting_player=0)

    actions = legal_actions(state, player_id=0)
    bid_cards = {action.cards for action in actions if action.kind == ActionKind.BID}

    assert (1000,) in bid_cards
    assert (2000,) in bid_cards
    assert (1000, 2000) in bid_cards
