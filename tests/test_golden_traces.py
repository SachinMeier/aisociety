"""Golden trace tests to validate core rule flows."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import MisfortuneKind, StatusCard, StatusKind
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState


def _make_deck(entries: list[dict]) -> list[StatusCard]:
    """Build a status deck from a JSON fixture."""
    deck: list[StatusCard] = []
    for entry in entries:
        kind = entry["kind"]
        if kind == "possession":
            deck.append(StatusCard(kind=StatusKind.POSSESSION, value=entry["value"]))
        elif kind == "title":
            deck.append(StatusCard(kind=StatusKind.TITLE))
        elif kind == "misfortune":
            misfortune = MisfortuneKind(entry["misfortune"])
            deck.append(StatusCard(kind=StatusKind.MISFORTUNE, misfortune=misfortune))
        else:
            raise ValueError(f"Unknown card kind: {kind}")
    return deck


def _make_players(count: int) -> list[PlayerState]:
    """Create player states with full starting hands."""
    return [PlayerState(id=idx, hand=RulesEngine.create_money_hand()) for idx in range(count)]


def _assert_expectations(state: GameState, expect: dict) -> None:
    """Validate expectations defined in a golden trace."""
    if "round_active" in expect:
        assert (state.round is not None) == expect["round_active"]
    if "starting_player" in expect:
        assert state.starting_player == expect["starting_player"]
    if "money_discard" in expect:
        values = sorted(card.value for card in state.money_discard)
        assert values == sorted(expect["money_discard"])
    if "player_possessions" in expect:
        for player_id, values in expect["player_possessions"].items():
            pid = int(player_id)
            possessions = [card.value for card in state.players[pid].possessions]
            assert sorted(possessions) == sorted(values)
    if "player_flags" in expect:
        for player_id, flags in expect["player_flags"].items():
            pid = int(player_id)
            player = state.players[pid]
            for flag, value in flags.items():
                assert getattr(player, flag) == value


def _apply_action(state: GameState, action_spec: dict) -> None:
    """Apply an action specified in the golden trace."""
    player_id = action_spec["player"]
    kind = action_spec["kind"]
    if kind == "pass":
        action = Action(ActionKind.PASS)
    elif kind == "bid":
        cards = tuple(action_spec.get("cards", []))
        action = Action(ActionKind.BID, cards=cards)
    elif kind == "discard_possession":
        action = Action(
            ActionKind.DISCARD_POSSESSION,
            possession_value=action_spec["possession_value"],
        )
    else:
        raise ValueError(f"Unknown action kind: {kind}")
    RulesEngine.apply_action(state, player_id, action)


@pytest.mark.parametrize("trace", json.loads(Path("tests/fixtures/golden_traces.json").read_text())["traces"])
def test_golden_traces(trace: dict) -> None:
    """Run golden traces against the rules engine."""
    players = _make_players(trace["players"])
    deck = _make_deck(trace["deck"])
    state = GameState(
        players=players,
        status_deck=deck,
        starting_player=trace["starting_player"],
    )
    rng = random.Random(0)
    for step in trace["steps"]:
        if step.get("start_round"):
            RulesEngine.start_round(state, rng)
            continue
        if "action" in step:
            _apply_action(state, step["action"])
            continue
        if "expect" in step:
            _assert_expectations(state, step["expect"])
            continue
        raise ValueError("Unknown step in golden trace")
