import pytest

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import (
    ALLOWED_MONEY_VALUES,
    MisfortuneKind,
    MoneyCard,
    StatusCard,
    StatusKind,
)
from highsociety.domain.errors import InvalidState
from highsociety.domain.state import GameState, PlayerState


def test_money_card_valid_values():
    for value in ALLOWED_MONEY_VALUES:
        assert MoneyCard(value).value == value


def test_money_card_invalid_value():
    with pytest.raises(ValueError):
        MoneyCard(999)


def test_status_card_valid_possession():
    card = StatusCard(kind=StatusKind.POSSESSION, value=7)
    assert card.red is False


def test_status_card_invalid_possession():
    with pytest.raises(ValueError):
        StatusCard(kind=StatusKind.POSSESSION, value=0)


def test_status_card_valid_title():
    card = StatusCard(kind=StatusKind.TITLE)
    assert card.red is True


def test_status_card_invalid_title():
    with pytest.raises(ValueError):
        StatusCard(kind=StatusKind.TITLE, value=1)


def test_status_card_valid_misfortune():
    card = StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.DEBT)
    assert card.red is False


def test_status_card_invalid_misfortune():
    with pytest.raises(ValueError):
        StatusCard(kind=StatusKind.MISFORTUNE)


def test_action_validation():
    Action(ActionKind.PASS)
    Action(ActionKind.BID, cards=(1000,))
    Action(ActionKind.DISCARD_POSSESSION, possession_value=3)
    with pytest.raises(ValueError):
        Action(ActionKind.PASS, cards=(1000,))
    with pytest.raises(ValueError):
        Action(ActionKind.BID, cards=())
    with pytest.raises(ValueError):
        Action(ActionKind.DISCARD_POSSESSION)


def test_player_state_validation_duplicate_hand():
    hand = [MoneyCard(1000), MoneyCard(1000)]
    with pytest.raises(InvalidState):
        PlayerState(id=0, hand=hand)


def test_game_state_validation_player_count():
    players = [PlayerState(id=0, hand=[MoneyCard(1000)])]
    with pytest.raises(InvalidState):
        GameState(players=players, status_deck=[], starting_player=0)
