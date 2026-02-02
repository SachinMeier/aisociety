"""Tests for baseline players."""

from __future__ import annotations

from highsociety.app.legal_actions import legal_actions
from highsociety.app.observations import build_observation
from highsociety.domain.actions import ActionKind
from highsociety.domain.cards import MisfortuneKind, MoneyCard, StatusCard, StatusKind
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState, RoundState
from highsociety.players.heuristic_bot import HeuristicBot
from highsociety.players.random_bot import RandomBot
from highsociety.players.static_bot import StaticBot


def _make_state(highest_bid: int = 0, possession_value: int = 5) -> GameState:
    """Create a simple game state with an active round."""
    players = [
        PlayerState(id=0, hand=[MoneyCard(1000), MoneyCard(2000)]),
        PlayerState(id=1, hand=[MoneyCard(3000)]),
        PlayerState(id=2, hand=[MoneyCard(4000)]),
    ]
    deck = RulesEngine.create_status_deck()
    card = StatusCard(kind=StatusKind.POSSESSION, value=possession_value)
    round_state = RoundState(
        card=card,
        starting_player=0,
        turn_player=0,
        highest_bid=highest_bid,
    )
    return GameState(players=players, status_deck=deck, round=round_state, starting_player=0)


def _make_state_with_card(
    card: StatusCard,
    highest_bid: int = 0,
    hand: list[MoneyCard] | None = None,
) -> GameState:
    """Create a simple game state with a specified status card."""
    if hand is None:
        hand = [MoneyCard(1000), MoneyCard(2000)]
    players = [
        PlayerState(id=0, hand=list(hand)),
        PlayerState(id=1, hand=[MoneyCard(3000)]),
        PlayerState(id=2, hand=[MoneyCard(4000)]),
    ]
    deck = RulesEngine.create_status_deck()
    round_state = RoundState(
        card=card,
        starting_player=0,
        turn_player=0,
        highest_bid=highest_bid,
    )
    return GameState(players=players, status_deck=deck, round=round_state, starting_player=0)


def test_random_bot_is_deterministic_and_legal() -> None:
    """Random bot selects legal actions deterministically under a seed."""
    state = _make_state()
    obs = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    bot_a = RandomBot(seed=42)
    bot_b = RandomBot(seed=42)
    bot_a.reset({"seed": 1}, player_id=0, seat=0)
    bot_b.reset({"seed": 1}, player_id=0, seat=0)

    action_a = bot_a.act(obs, actions)
    action_b = bot_b.act(obs, actions)

    assert action_a == action_b
    assert action_a in actions


def test_heuristic_bot_returns_legal_action() -> None:
    """Heuristic bot returns a legal action."""
    state = _make_state(highest_bid=100000)
    obs = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    bot = HeuristicBot(style="cautious", seed=7)
    bot.reset({"seed": 1}, player_id=0, seat=0)

    action = bot.act(obs, actions)

    assert action in actions
    assert action.kind == ActionKind.PASS


def test_heuristic_bot_scales_bids_by_value() -> None:
    """Heuristic bot bids higher for higher-valued possessions."""
    low_state = _make_state(possession_value=1)
    high_state = _make_state(possession_value=10)

    bot = HeuristicBot(style="cautious", seed=3)
    bot.reset({"seed": 1}, player_id=0, seat=0)

    low_obs = build_observation(low_state, player_id=0)
    low_actions = legal_actions(low_state, player_id=0)
    low_action = bot.act(low_obs, low_actions)

    bot.reset({"seed": 1}, player_id=0, seat=0)
    high_obs = build_observation(high_state, player_id=0)
    high_actions = legal_actions(high_state, player_id=0)
    high_action = bot.act(high_obs, high_actions)

    assert low_action.kind == ActionKind.PASS
    assert high_action.kind == ActionKind.BID


def test_static_bot_bids_within_possession_budget() -> None:
    """Static bot bids when the possession budget supports it."""
    card = StatusCard(kind=StatusKind.POSSESSION, value=2)
    state = _make_state_with_card(card)
    obs = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    bot = StaticBot()
    bot.reset({"player_count": 3}, player_id=0, seat=0)

    action = bot.act(obs, actions)

    assert action.kind == ActionKind.BID
    assert action.cards == (1000,)


def test_static_bot_passes_when_title_budget_too_low() -> None:
    """Static bot passes if the title budget is below the minimum bid."""
    card = StatusCard(kind=StatusKind.TITLE)
    state = _make_state_with_card(card)
    obs = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    bot = StaticBot(title_budget=500)
    bot.reset({"player_count": 3}, player_id=0, seat=0)

    action = bot.act(obs, actions)

    assert action.kind == ActionKind.PASS


def test_static_bot_uses_debt_budget() -> None:
    """Static bot uses the debt budget to cap bids."""
    card = StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.DEBT)
    hand = [MoneyCard(15000), MoneyCard(1000)]
    state = _make_state_with_card(card, highest_bid=14000, hand=hand)
    obs = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    bot = StaticBot()
    bot.reset({"player_count": 3}, player_id=0, seat=0)

    action = bot.act(obs, actions)

    assert action.kind == ActionKind.BID
    assert action.cards == (15000,)
