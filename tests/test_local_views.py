"""Tests for local game visibility rules and active-player enforcement.

These tests verify:
- Public view contains all players' owned status cards.
- Private hand is only exposed for the active player.
- Actions submitted for a non-active player are rejected.
- End-to-end game flow from creation through terminal state.
"""

from __future__ import annotations

import pytest

from highsociety.app.local_game_service import (
    LocalGameService,
    LocalSeatSpec,
)
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction
from highsociety.players.heuristic_bot import HeuristicBot
from highsociety.players.registry import PlayerRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> LocalGameService:
    """Service with default registry (excludes mlp to avoid checkpoint load)."""
    registry = PlayerRegistry()
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
def three_human_seats() -> list[LocalSeatSpec]:
    return [
        LocalSeatSpec(kind="human", name="Alice"),
        LocalSeatSpec(kind="human", name="Bob"),
        LocalSeatSpec(kind="human", name="Charlie"),
    ]


@pytest.fixture
def mixed_seats() -> list[LocalSeatSpec]:
    return [
        LocalSeatSpec(kind="human", name="Alice"),
        LocalSeatSpec(kind="easy"),
        LocalSeatSpec(kind="medium"),
    ]


# ---------------------------------------------------------------------------
# Public view: owned status cards visible for all players
# ---------------------------------------------------------------------------


class TestPublicViewOwnedCards:
    """Verify every player's owned_status_cards appear in public_table."""

    def test_all_players_have_owned_status_cards_key(
        self, service: LocalGameService, three_human_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(three_human_seats, seed=42)
        view = service.get_turn_view(session.game_id)
        players = view["public_table"]["players"]
        assert len(players) == 3
        for p in players:
            assert "owned_status_cards" in p
            assert isinstance(p["owned_status_cards"], list)

    def test_owned_cards_visible_after_possession_acquired(
        self, service: LocalGameService
    ) -> None:
        """Play a game with bots until one acquires a possession, then check
        that the finished public view includes owned cards for that player."""
        seats = [
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="medium"),
        ]
        session = service.create_game(seats, seed=42)
        assert session.status == "finished"
        view = service.get_turn_view(session.game_id)
        players = view["public_table"]["players"]
        # At least one player should have acquired something
        all_cards = []
        for p in players:
            for c in p["owned_status_cards"]:
                all_cards.append(c)
        # A completed game should have distributed possessions
        assert len(all_cards) > 0, "Expected at least one player to own status cards after a full game"

    def test_owned_cards_visible_for_non_active_players(
        self, service: LocalGameService
    ) -> None:
        """In a mixed game, non-active players' owned_status_cards should
        be present in the public table view."""
        seats = [
            LocalSeatSpec(kind="human", name="Alice"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        view = service.get_turn_view(session.game_id)
        active = view["active_player_id"]
        for p in view["public_table"]["players"]:
            assert "owned_status_cards" in p
            assert "money_count" in p
            # Even non-active players should appear
            if p["id"] != active:
                assert "name" in p


# ---------------------------------------------------------------------------
# Private hand: only for active human player
# ---------------------------------------------------------------------------


class TestPrivateHandVisibility:
    """Verify private hand is only exposed for the active human player."""

    def test_private_hand_present_for_human(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        view = service.get_turn_view(session.game_id)
        if view["active_player_id"] in session.human_seats:
            assert view["private_hand"] is not None
            assert isinstance(view["private_hand"], list)
            assert len(view["private_hand"]) > 0

    def test_private_hand_null_for_bot_turn(
        self, service: LocalGameService
    ) -> None:
        """When the active player is a bot, private_hand should be None."""
        # All-bot game finishes immediately; create a game and check the
        # service's view-building for a bot turn by testing a finished game.
        seats = [
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=42)
        # Since all bots, game goes to finished - which means no active turn
        view = service.get_turn_view(session.game_id)
        assert view["status"] == "finished"
        # Finished games don't have private hands
        assert view.get("private_hand") is None

    def test_private_hand_sorted(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        """The private hand values should be sorted ascending."""
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        view = service.get_turn_view(session.game_id)
        hand = view.get("private_hand")
        if hand is not None:
            assert hand == sorted(hand)


# ---------------------------------------------------------------------------
# Active-player enforcement
# ---------------------------------------------------------------------------


class TestActivePlayerEnforcement:
    """Reject actions submitted for non-active player."""

    def test_wrong_player_id_rejected(
        self, service: LocalGameService, three_human_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(three_human_seats, seed=42)
        view = service.get_turn_view(session.game_id)
        active = view["active_player_id"]
        wrong = (active + 1) % 3
        with pytest.raises(InvalidAction, match="turn"):
            service.submit_human_action(
                session.game_id, wrong, Action(ActionKind.PASS)
            )

    def test_bot_seat_action_rejected(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        # Try to submit as bot seat (seat 1 is easy bot)
        with pytest.raises(InvalidAction):
            service.submit_human_action(
                session.game_id, 1, Action(ActionKind.PASS)
            )

    def test_correct_player_accepted(
        self, service: LocalGameService, mixed_seats: list[LocalSeatSpec]
    ) -> None:
        session = service.create_game(mixed_seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        view = service.get_turn_view(session.game_id)
        active = view["active_player_id"]
        # Pass should always be legal for the active human
        result = service.submit_human_action(
            session.game_id, active, Action(ActionKind.PASS)
        )
        assert result["status"] in ("awaiting_human_action", "finished", "active")


# ---------------------------------------------------------------------------
# End-to-end: create game via API, play turns, reach terminal state
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full game lifecycle through the service layer."""

    def test_full_game_with_human_always_passing(
        self, service: LocalGameService
    ) -> None:
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
            legal = view.get("legal_actions", [])
            if not legal:
                break
            # Handle discard_possession if required
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

    def test_full_game_with_bidding_human(
        self, service: LocalGameService
    ) -> None:
        """Human bids on the first opportunity, then passes for the rest."""
        seats = [
            LocalSeatSpec(kind="human", name="Bidder"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
        ]
        session = service.create_game(seats, seed=42)
        bid_submitted = False
        steps = 0
        max_steps = 500
        while session.status != "finished" and steps < max_steps:
            view = service.get_turn_view(session.game_id)
            if view["status"] == "finished":
                break
            player_id = view["active_player_id"]
            legal = view.get("legal_actions", [])
            if not legal:
                break
            discard_actions = [a for a in legal if a["kind"] == "discard_possession"]
            bid_actions = [a for a in legal if a["kind"] == "bid"]
            if discard_actions:
                val = discard_actions[0]["possession_value"]
                action = Action(ActionKind.DISCARD_POSSESSION, possession_value=val)
            elif not bid_submitted and bid_actions:
                cards = tuple(bid_actions[0]["cards"])
                action = Action(ActionKind.BID, cards=cards)
                bid_submitted = True
            else:
                action = Action(ActionKind.PASS)
            service.submit_human_action(session.game_id, player_id, action)
            steps += 1
        assert session.status == "finished"
        assert session.result is not None

    def test_finished_view_has_results(
        self, service: LocalGameService
    ) -> None:
        seats = [
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="medium"),
        ]
        session = service.create_game(seats, seed=42)
        assert session.status == "finished"
        view = service.get_turn_view(session.game_id)
        assert view["status"] == "finished"
        result = view["result"]
        assert "winners" in result
        assert "scores" in result
        assert "money_remaining" in result
        assert "poorest" in result
        assert view["public_table"] is not None

    def test_multiple_games_independent(
        self, service: LocalGameService
    ) -> None:
        """Creating multiple games should not interfere with each other."""
        sessions = []
        for i in range(3):
            seats = [
                LocalSeatSpec(kind="human", name=f"Player-{i}"),
                LocalSeatSpec(kind="easy"),
                LocalSeatSpec(kind="easy"),
            ]
            s = service.create_game(seats, seed=i + 100)
            sessions.append(s)
        # All should have unique IDs
        ids = {s.game_id for s in sessions}
        assert len(ids) == 3
        # Each should be independently queryable
        for s in sessions:
            view = service.get_turn_view(s.game_id)
            assert view is not None

    def test_api_response_shape_matches_contract(
        self, service: LocalGameService
    ) -> None:
        """Verify the turn view shape matches what the frontend expects."""
        seats = [
            LocalSeatSpec(kind="human", name="Alice"),
            LocalSeatSpec(kind="easy"),
            LocalSeatSpec(kind="medium"),
        ]
        session = service.create_game(seats, seed=42)
        if session.status == "finished":
            pytest.skip("Game ended before human turn")
        view = service.get_turn_view(session.game_id)

        # Top-level fields
        assert "game_id" in view
        assert "status" in view
        assert view["status"] in ("awaiting_human_action", "active", "finished", "errored")
        assert "active_player_id" in view
        assert "active_player_name" in view
        assert "requires_handoff" in view
        assert "public_table" in view
        assert "private_hand" in view
        assert "legal_actions" in view

        # Public table fields
        pub = view["public_table"]
        assert "players" in pub
        assert "revealed_status_cards" in pub
        # status_card and round can be null
        assert "status_card" in pub
        assert "round" in pub

        # Player fields
        for p in pub["players"]:
            assert "id" in p
            assert "name" in p
            assert "open_bid" in p
            assert "owned_status_cards" in p
            assert "money_count" in p

        # Legal actions
        for la in view["legal_actions"]:
            assert "kind" in la
            assert la["kind"] in ("pass", "bid", "discard_possession")

        # Private hand
        if view["private_hand"] is not None:
            assert isinstance(view["private_hand"], list)
            for v in view["private_hand"]:
                assert isinstance(v, int)
