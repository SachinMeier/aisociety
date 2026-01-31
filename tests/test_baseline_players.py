"""Tests for baseline players."""

from __future__ import annotations

from highsociety.app.legal_actions import legal_actions
from highsociety.app.observations import build_observation
from highsociety.domain.actions import ActionKind
from highsociety.domain.cards import MoneyCard, StatusCard, StatusKind
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState, RoundState
from highsociety.players.heuristic_bot import HeuristicBot
from highsociety.players.random_bot import RandomBot


def _make_state(highest_bid: int = 0) -> GameState:
    """Create a simple game state with an active round."""
    players = [
        PlayerState(id=0, hand=[MoneyCard(1000), MoneyCard(2000)]),
        PlayerState(id=1, hand=[MoneyCard(3000)]),
        PlayerState(id=2, hand=[MoneyCard(4000)]),
    ]
    deck = RulesEngine.create_status_deck()
    card = StatusCard(kind=StatusKind.POSSESSION, value=5)
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
