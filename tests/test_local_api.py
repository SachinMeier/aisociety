"""Tests for the local pass-and-play FastAPI endpoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction
from highsociety.server.local_api import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _stub_turn_view(
    *,
    status: str = "awaiting_human_action",
    active_player_id: int | None = 0,
    active_player_name: str | None = "Alice",
    requires_handoff: bool = False,
    private_hand: list[int] | None = None,
    legal_actions: list[dict] | None = None,
) -> dict:
    """Return a minimal turn-view dict matching the service contract."""
    return {
        "status": status,
        "active_player_id": active_player_id,
        "active_player_name": active_player_name,
        "requires_handoff": requires_handoff,
        "public_table": {
            "status_card": {"kind": "possession", "value": 8},
            "round": {
                "highest_bid": 0,
                "highest_bidder": None,
                "turn_player": active_player_id if active_player_id is not None else 0,
            },
            "players": [
                {
                    "id": 0,
                    "name": "Alice",
                    "open_bid": [],
                    "owned_status_cards": [],
                    "money_count": 11,
                },
                {
                    "id": 1,
                    "name": "Bot-Easy",
                    "open_bid": [],
                    "owned_status_cards": [],
                    "money_count": 11,
                },
                {
                    "id": 2,
                    "name": "Bot-Hard",
                    "open_bid": [],
                    "owned_status_cards": [],
                    "money_count": 11,
                },
            ],
            "revealed_status_cards": [],
        },
        "private_hand": private_hand if private_hand is not None else [25000, 20000, 15000, 12000, 10000, 8000, 6000, 4000, 3000, 2000, 1000],
        "legal_actions": legal_actions if legal_actions is not None else [
            {"kind": "pass"},
            {"kind": "bid", "cards": [1000]},
        ],
    }


@pytest.fixture
def mock_service() -> MagicMock:
    return MagicMock(spec=["create_game", "get_turn_view", "submit_human_action"])


@pytest.fixture
def client(mock_service: MagicMock) -> TestClient:
    app = create_app(service=mock_service)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Create game
# ---------------------------------------------------------------------------


def test_create_game_success(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.create_game.return_value = SimpleNamespace(game_id="K7Q2M8P4")
    resp = client.post(
        "/api/local-games",
        json={
            "seats": [
                {"type": "human", "name": "Alice"},
                {"type": "easy"},
                {"type": "hard"},
            ]
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["game_id"] == "K7Q2M8P4"
    mock_service.create_game.assert_called_once()
    call_args = mock_service.create_game.call_args
    seats = call_args[0][0]
    assert len(seats) == 3
    assert seats[0].kind == "human"
    assert seats[0].name == "Alice"


def test_create_game_too_few_seats(client: TestClient) -> None:
    resp = client.post(
        "/api/local-games",
        json={"seats": [{"type": "human"}, {"type": "easy"}]},
    )
    assert resp.status_code == 422


def test_create_game_too_many_seats(client: TestClient) -> None:
    resp = client.post(
        "/api/local-games",
        json={
            "seats": [{"type": "human"}] * 6,
        },
    )
    assert resp.status_code == 422


def test_create_game_service_error(
    client: TestClient, mock_service: MagicMock
) -> None:
    resp = client.post(
        "/api/local-games",
        json={
            "seats": [
                {"type": "bogus"},
                {"type": "human"},
                {"type": "easy"},
            ]
        },
    )
    assert resp.status_code == 400
    assert "seat kind" in resp.json()["detail"].lower() or "bogus" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Get game state / turn
# ---------------------------------------------------------------------------


def test_get_game_state(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.get_turn_view.return_value = _stub_turn_view()
    resp = client.get("/api/local-games/K7Q2M8P4")
    assert resp.status_code == 200
    body = resp.json()
    assert body["game_id"] == "K7Q2M8P4"
    assert body["status"] == "awaiting_human_action"
    assert body["active_player_id"] == 0
    assert body["public_table"]["status_card"]["kind"] == "possession"
    assert body["public_table"]["status_card"]["value"] == 8
    assert len(body["public_table"]["players"]) == 3
    assert len(body["private_hand"]) == 11
    assert len(body["legal_actions"]) == 2


def test_get_turn(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.get_turn_view.return_value = _stub_turn_view(
        requires_handoff=True,
        active_player_id=1,
        active_player_name="Bot-Easy",
    )
    resp = client.get("/api/local-games/ABCD1234/turn")
    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_handoff"] is True
    assert body["active_player_id"] == 1


def test_get_game_not_found(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.get_turn_view.side_effect = KeyError("Unknown game id")
    resp = client.get("/api/local-games/NOTFOUND")
    assert resp.status_code == 404


def test_get_turn_not_found(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.get_turn_view.side_effect = KeyError("Unknown game id")
    resp = client.get("/api/local-games/NOTFOUND/turn")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Submit action
# ---------------------------------------------------------------------------


def test_submit_pass_action(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.return_value = _stub_turn_view(
        active_player_id=1, active_player_name="Bot-Easy"
    )
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 0, "action": {"kind": "pass"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["game_id"] == "K7Q2M8P4"
    mock_service.submit_human_action.assert_called_once()
    call_args = mock_service.submit_human_action.call_args
    assert call_args[0][0] == "K7Q2M8P4"
    assert call_args[0][1] == 0
    action = call_args[0][2]
    assert isinstance(action, Action)
    assert action.kind == ActionKind.PASS


def test_submit_bid_action(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.return_value = _stub_turn_view()
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 0, "action": {"kind": "bid", "cards": [2000, 1000]}},
    )
    assert resp.status_code == 200
    call_args = mock_service.submit_human_action.call_args
    action = call_args[0][2]
    assert action.kind == ActionKind.BID
    assert action.cards == (1000, 2000)


def test_submit_discard_action(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.return_value = _stub_turn_view()
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={
            "player_id": 0,
            "action": {"kind": "discard_possession", "possession_value": 5},
        },
    )
    assert resp.status_code == 200
    call_args = mock_service.submit_human_action.call_args
    action = call_args[0][2]
    assert action.kind == ActionKind.DISCARD_POSSESSION
    assert action.possession_value == 5


def test_submit_action_game_not_found(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.side_effect = KeyError("Unknown game id")
    resp = client.post(
        "/api/local-games/NOTFOUND/actions",
        json={"player_id": 0, "action": {"kind": "pass"}},
    )
    assert resp.status_code == 404


def test_submit_action_invalid_action(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.side_effect = InvalidAction("Bid must exceed current highest")
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 0, "action": {"kind": "pass"}},
    )
    assert resp.status_code == 400
    assert "Bid must exceed" in resp.json()["detail"]


def test_submit_action_wrong_player(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.submit_human_action.side_effect = InvalidAction(
        "Not player 2's turn (current: 0)"
    )
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 2, "action": {"kind": "pass"}},
    )
    assert resp.status_code == 409
    assert "turn" in resp.json()["detail"].lower()


def test_submit_bid_without_cards(client: TestClient) -> None:
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 0, "action": {"kind": "bid"}},
    )
    assert resp.status_code == 400


def test_submit_unknown_action_kind(client: TestClient) -> None:
    resp = client.post(
        "/api/local-games/K7Q2M8P4/actions",
        json={"player_id": 0, "action": {"kind": "fly_away"}},
    )
    assert resp.status_code == 400
    assert "Unknown action kind" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Turn response shape: finished game with results
# ---------------------------------------------------------------------------


def test_finished_game_response(
    client: TestClient, mock_service: MagicMock
) -> None:
    view = _stub_turn_view(
        status="finished",
        active_player_id=None,
        active_player_name=None,
        private_hand=None,
        legal_actions=[],
    )
    # Service uses "result" key for finished games
    view["result"] = {
        "winners": [0],
        "scores": {"0": 12, "1": 8, "2": 4},
        "money_remaining": {"0": 45000, "1": 30000, "2": 20000},
        "poorest": [2],
    }
    mock_service.get_turn_view.return_value = view
    resp = client.get("/api/local-games/K7Q2M8P4")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "finished"
    assert body["results"] is not None
    assert body["results"]["winners"] == [0]
    assert body["legal_actions"] == []


# ---------------------------------------------------------------------------
# Response shape: owned status cards, round info
# ---------------------------------------------------------------------------


def test_player_with_owned_cards(
    client: TestClient, mock_service: MagicMock
) -> None:
    view = _stub_turn_view()
    view["public_table"]["players"][0]["owned_status_cards"] = [
        {"kind": "possession", "value": 5},
        {"kind": "title"},
    ]
    mock_service.get_turn_view.return_value = view
    resp = client.get("/api/local-games/K7Q2M8P4")
    assert resp.status_code == 200
    body = resp.json()
    cards = body["public_table"]["players"][0]["owned_status_cards"]
    assert len(cards) == 2
    assert cards[0]["kind"] == "possession"
    assert cards[0]["value"] == 5
    assert cards[1]["kind"] == "title"
    assert cards[1].get("value") is None


def test_create_game_with_seed(
    client: TestClient, mock_service: MagicMock
) -> None:
    mock_service.create_game.return_value = SimpleNamespace(game_id="SEED1234")
    resp = client.post(
        "/api/local-games",
        json={
            "seats": [
                {"type": "human", "name": "Alice"},
                {"type": "easy"},
                {"type": "medium"},
            ],
            "seed": 42,
        },
    )
    assert resp.status_code == 201
    call_args = mock_service.create_game.call_args
    assert call_args[1]["seed"] == 42
