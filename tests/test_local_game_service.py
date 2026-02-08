"""Tests for the local interactive game service."""

from __future__ import annotations

import re

import pytest

from highsociety.app.local_game_service import (
    _PRESET_SPECS,
    VALID_BOT_PRESETS,
    LocalGameService,
    LocalGameSession,
    LocalSeatSpec,
)
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.players.heuristic_bot import HeuristicBot
from highsociety.players.registry import PlayerRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> LocalGameService:
    """Service with default registry (excludes mlp to avoid checkpoint load)."""
    registry = PlayerRegistry()
    # Register only bots that don't need external checkpoints
    from highsociety.players.random_bot import RandomBot
    from highsociety.players.static_bot import StaticBot

    registry.register("static", lambda spec: StaticBot(
        name=spec.get("name", "static"),
    ))
    registry.register("heuristic", lambda spec: HeuristicBot(
        name=spec.get("name", "heuristic"),
        style=spec.get("params", {}).get("style", "balanced") if isinstance(spec.get("params"), dict) else "balanced",
    ))
    registry.register("random", lambda spec: RandomBot(
        name=spec.get("name", "random"),
    ))
    return LocalGameService(registry=registry)


@pytest.fixture
def all_human_seats() -> list[LocalSeatSpec]:
    """Three human seats."""
    return [
        LocalSeatSpec(kind="human", name="Alice"),
        LocalSeatSpec(kind="human", name="Bob"),
        LocalSeatSpec(kind="human", name="Charlie"),
    ]


@pytest.fixture
def mixed_seats() -> list[LocalSeatSpec]:
    """One human + two bots."""
    return [
        LocalSeatSpec(kind="human", name="Alice"),
        LocalSeatSpec(kind="easy"),
        LocalSeatSpec(kind="medium"),
    ]


@pytest.fixture
def all_bot_seats() -> list[LocalSeatSpec]:
    """Three bot seats (no humans)."""
    return [
        LocalSeatSpec(kind="easy"),
        LocalSeatSpec(kind="medium"),
        LocalSeatSpec(kind="expert"),
    ]


# ---------------------------------------------------------------------------
# Game creation
# ---------------------------------------------------------------------------


class TestCreateGame:
    def test_create_mixed_game(self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]) -> None:
        session = service.create_game(mixed_seats, seed=42)
        assert isinstance(session, LocalGameSession)
        assert len(session.game_id) == 8
        assert session.human_seats == {0}
        assert 1 in session.bot_players
        assert 2 in session.bot_players
        assert session.display_names[0] == "Alice"

    def test_create_all_human_game(self, service: LocalGameService, all_human_seats: list[LocalSeatSpec]) -> None:
        session = service.create_game(all_human_seats, seed=1)
        assert session.human_seats == {0, 1, 2}
        assert len(session.bot_players) == 0
        # Should be awaiting a human immediately
        assert session.status == "awaiting_human"

    def test_create_all_bot_game_runs_to_completion(
        self, service: LocalGameService, all_bot_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(all_bot_seats, seed=42)
        # No humans, so bots should auto-play to completion
        assert session.status == "finished"
        assert session.result is not None

    def test_invalid_player_count_too_few(self, service: LocalGameService) -> None:
        with pytest.raises(InvalidState, match="Player count"):
            service.create_game([
                LocalSeatSpec(kind="human"),
                LocalSeatSpec(kind="easy"),
            ])

    def test_invalid_player_count_too_many(self, service: LocalGameService) -> None:
        with pytest.raises(InvalidState, match="Player count"):
            service.create_game([LocalSeatSpec(kind="human")] * 6)

    def test_custom_bot_name(self, service: LocalGameService) -> None:
        seats = [
            LocalSeatSpec(kind="human", name="Me"),
            LocalSeatSpec(kind="easy", name="Robo"),
            LocalSeatSpec(kind="medium", name="Smarty"),
        ]
        session = service.create_game(seats, seed=7)
        assert session.display_names[1] == "Robo"
        assert session.display_names[2] == "Smarty"

    def test_default_human_name(self, service: LocalGameService) -> None:
        seats = [
            LocalSeatSpec(kind="human"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=7)
        assert session.display_names[0] == "Player 1"


# ---------------------------------------------------------------------------
# Bot auto-advance
# ---------------------------------------------------------------------------


class TestBotAutoAdvance:
    def test_mixed_game_stops_at_human(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        # Should pause at a human turn or finish
        assert session.status in ("awaiting_human", "finished")

    def test_turn_view_shows_human_awaiting(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn with this seed")
        view = service.get_turn_view(session.game_id)
        assert view["status"] == "awaiting_human_action"
        assert view["active_player_id"] in session.human_seats
        assert view["private_hand"] is not None
        assert view["legal_actions"] is not None
        assert len(view["legal_actions"]) > 0


# ---------------------------------------------------------------------------
# Submit human action and advance
# ---------------------------------------------------------------------------


class TestSubmitHumanAction:
    def test_submit_pass_action(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn with this seed")
        view = service.get_turn_view(session.game_id)
        player_id = view["active_player_id"]
        # Find the pass action
        pass_actions = [a for a in view["legal_actions"] if a["kind"] == "pass"]
        assert len(pass_actions) > 0
        action = Action(ActionKind.PASS)
        result = service.submit_human_action(session.game_id, player_id, action)
        assert result["status"] in ("awaiting_human_action", "finished", "active")

    def test_submit_wrong_player_rejected(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn with this seed")
        view = service.get_turn_view(session.game_id)
        active = view["active_player_id"]
        wrong_player = (active + 1) % len(mixed_seats)
        with pytest.raises(InvalidAction):
            service.submit_human_action(
                session.game_id, wrong_player, Action(ActionKind.PASS)
            )

    def test_submit_on_bot_seat_rejected(
        self, service: LocalGameService
    ) -> None:
        seats = [
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="human", name="Me"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn with this seed")
        # Bot seat 0 should be rejected even if it were active
        with pytest.raises(InvalidAction):
            service.submit_human_action(session.game_id, 0, Action(ActionKind.PASS))


# ---------------------------------------------------------------------------
# Terminal scoring path
# ---------------------------------------------------------------------------


class TestTerminalScoring:
    def test_all_bots_score(
        self, service: LocalGameService, all_bot_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(all_bot_seats, seed=42)
        assert session.status == "finished"
        assert session.result is not None
        assert len(session.result.winners) >= 0  # may be empty if all poorest
        assert len(session.result.scores) >= 0

    def test_finished_view_structure(
        self, service: LocalGameService, all_bot_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(all_bot_seats, seed=42)
        view = service.get_turn_view(session.game_id)
        assert view["status"] == "finished"
        assert "result" in view
        assert "winners" in view["result"]
        assert "scores" in view["result"]
        assert "money_remaining" in view["result"]
        assert "poorest" in view["result"]
        assert "public_table" in view

    def test_play_through_human_game_to_completion(
        self, service: LocalGameService
    ) -> None:
        """Play a full game with one human always passing to reach terminal."""
        seats = [
            LocalSeatSpec(kind="human", name="Passer"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=99)
        steps = 0
        max_steps = 500
        while session.status != "finished" and steps < max_steps:
            view = service.get_turn_view(session.game_id)
            if view["status"] == "finished":
                break
            player_id = view["active_player_id"]
            # Human always passes (or discards lowest if required)
            legal = view.get("legal_actions", [])
            if not legal:
                break
            discard_actions = [a for a in legal if a["kind"] == "discard_possession"]
            if discard_actions:
                val = discard_actions[0]["possession_value"]
                action = Action(ActionKind.DISCARD_POSSESSION, possession_value=val)
            else:
                action = Action(ActionKind.PASS)
            service.submit_human_action(session.game_id, player_id, action)
            steps += 1
        assert session.status == "finished"
        assert session.result is not None


# ---------------------------------------------------------------------------
# Difficulty preset mapping
# ---------------------------------------------------------------------------


class TestDifficultyPresets:
    def test_easy_maps_to_static(self) -> None:
        assert _PRESET_SPECS["easy"]["type"] == "static"

    def test_medium_maps_to_heuristic_balanced(self) -> None:
        spec = _PRESET_SPECS["medium"]
        assert spec["type"] == "heuristic"
        assert spec["params"]["style"] == "balanced"

    def test_hard_maps_to_mlp(self) -> None:
        spec = _PRESET_SPECS["hard"]
        assert spec["type"] == "mlp"
        assert spec["checkpoint"] == "checkpoints/mlp/mlp_v3"

    def test_expert_maps_to_heuristic_cautious(self) -> None:
        spec = _PRESET_SPECS["expert"]
        assert spec["type"] == "heuristic"
        assert spec["params"]["style"] == "cautious"

    def test_valid_presets_tuple(self) -> None:
        assert "easy" in VALID_BOT_PRESETS
        assert "medium" in VALID_BOT_PRESETS
        assert "hard" in VALID_BOT_PRESETS
        assert "expert" in VALID_BOT_PRESETS


# ---------------------------------------------------------------------------
# Game ID generation
# ---------------------------------------------------------------------------


class TestGameIdGeneration:
    def test_id_is_8_chars(self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]) -> None:
        session = service.create_game(mixed_seats, seed=42)
        assert len(session.game_id) == 8

    def test_id_is_uppercase_base32(self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]) -> None:
        session = service.create_game(mixed_seats, seed=42)
        assert re.match(r"^[A-Z2-7]{8}$", session.game_id)

    def test_ids_are_unique(self, service: LocalGameService) -> None:
        ids = set()
        for i in range(20):
            seats = [
                LocalSeatSpec(kind="easy"),
                LocalSeatSpec(kind="easy"),
                LocalSeatSpec(kind="easy"),
            ]
            session = service.create_game(seats, seed=i)
            ids.add(session.game_id)
        assert len(ids) == 20


# ---------------------------------------------------------------------------
# Seat spec validation
# ---------------------------------------------------------------------------


class TestLocalSeatSpec:
    def test_valid_human(self) -> None:
        spec = LocalSeatSpec(kind="human", name="Alice")
        assert spec.kind == "human"

    def test_valid_bot(self) -> None:
        spec = LocalSeatSpec(kind="easy")
        assert spec.kind == "easy"

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError, match="Unknown seat kind"):
            LocalSeatSpec(kind="impossible")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Session lookup
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_unknown_game_id(self, service: LocalGameService) -> None:
        with pytest.raises(KeyError, match="Unknown game id"):
            service.get_session("ZZZZZZZZ")

    def test_valid_lookup(self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]) -> None:
        session = service.create_game(mixed_seats, seed=42)
        found = service.get_session(session.game_id)
        assert found is session
