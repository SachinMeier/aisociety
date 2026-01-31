import random

import pytest

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import MisfortuneKind, MoneyCard, StatusCard, StatusKind
from highsociety.domain.errors import InvalidAction
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState


def make_players(n: int = 3):
    players = []
    for idx in range(n):
        players.append(PlayerState(id=idx, hand=RulesEngine.create_money_hand()))
    return players


def make_game_with_deck(deck, starting_player=0):
    return GameState(players=make_players(3), status_deck=list(deck), starting_player=starting_player)


def test_bid_must_exceed_highest():
    deck = [StatusCard(kind=StatusKind.POSSESSION, value=5)]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.BID, cards=(1000,)))
    with pytest.raises(InvalidAction):
        RulesEngine.apply_action(state, 1, Action(ActionKind.BID, cards=(1000,)))


def test_pass_returns_open_bid_possession():
    deck = [StatusCard(kind=StatusKind.POSSESSION, value=5)]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.BID, cards=(1000,)))
    RulesEngine.apply_action(state, 1, Action(ActionKind.PASS))
    player1 = state.players[1]
    assert player1.passed is True
    assert not player1.open_bid
    assert 1000 in [card.value for card in state.players[0].open_bid]


def test_free_win_when_no_bids():
    deck = [StatusCard(kind=StatusKind.POSSESSION, value=3)]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.PASS))
    RulesEngine.apply_action(state, 1, Action(ActionKind.PASS))
    assert state.round is None
    winner = state.players[2]
    assert [card.value for card in winner.possessions] == [3]
    assert not state.money_discard


def test_misfortune_first_pass_gets_card():
    deck = [StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.DEBT)]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.BID, cards=(1000,)))
    RulesEngine.apply_action(state, 1, Action(ActionKind.BID, cards=(2000,)))
    RulesEngine.apply_action(state, 2, Action(ActionKind.PASS))
    assert state.round is None
    assert state.players[2].debt == 1
    discarded = [card.value for card in state.money_discard]
    assert sorted(discarded) == [1000, 2000]


def test_theft_requires_discard_when_possessions_exist():
    deck = [StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.THEFT)]
    state = make_game_with_deck(deck)
    player = state.players[0]
    player.possessions.append(StatusCard(kind=StatusKind.POSSESSION, value=4))
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.PASS))
    assert state.pending_discard is not None
    RulesEngine.apply_action(state, 0, Action(ActionKind.DISCARD_POSSESSION, possession_value=4))
    assert state.pending_discard is None
    assert not player.possessions


def test_theft_pending_discards_next_possession():
    deck = [
        StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.THEFT),
        StatusCard(kind=StatusKind.POSSESSION, value=9),
    ]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.PASS))
    assert state.players[0].theft_pending == 1
    RulesEngine.start_round(state, rng)
    RulesEngine.apply_action(state, 0, Action(ActionKind.BID, cards=(1000,)))
    RulesEngine.apply_action(state, 1, Action(ActionKind.PASS))
    RulesEngine.apply_action(state, 2, Action(ActionKind.PASS))
    assert not state.players[0].possessions
    assert state.players[0].theft_pending == 0


def test_fourth_red_ends_game():
    deck = [
        StatusCard(kind=StatusKind.TITLE),
        StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.SCANDAL),
        StatusCard(kind=StatusKind.TITLE),
        StatusCard(kind=StatusKind.TITLE),
    ]
    state = make_game_with_deck(deck)
    rng = random.Random(0)
    for _ in range(3):
        RulesEngine.start_round(state, rng)
        while state.round is not None:
            current = state.round.turn_player
            RulesEngine.apply_action(state, current, Action(ActionKind.PASS))
    RulesEngine.start_round(state, rng)
    assert state.game_over is True


def test_scoring_with_titles_and_scandal():
    deck = []
    state = GameState(players=make_players(3), status_deck=deck, starting_player=0)
    p0, p1, _p2 = state.players
    p0.possessions = [StatusCard(kind=StatusKind.POSSESSION, value=3)]
    p0.titles = 1
    p0.scandal = 1
    p1.possessions = [StatusCard(kind=StatusKind.POSSESSION, value=4)]
    p1.titles = 0
    p1.debt = 1
    p0.hand = [MoneyCard(1000)]
    p1.hand = [MoneyCard(2000)]
    state.players[2].hand = [MoneyCard(1000)]
    result = RulesEngine.score_game(state)
    assert 0 in result.poorest
    assert result.winners == (1,)
