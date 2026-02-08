"""FastAPI endpoints for local pass-and-play games."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from highsociety.app.local_game_service import LocalGameService, LocalSeatSpec
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction, InvalidState

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class SeatSpec(BaseModel):
    type: str = Field(
        ...,
        description="Seat type: 'human', 'easy', 'medium', 'hard', or 'expert'",
    )
    name: str | None = Field(
        default=None,
        description="Optional display name for the player",
    )


class CreateGameRequest(BaseModel):
    seats: list[SeatSpec] = Field(
        ..., min_length=3, max_length=5, description="3-5 seat definitions"
    )
    seed: int | None = Field(
        default=None, description="Optional RNG seed for reproducible games"
    )


class StatusCardView(BaseModel):
    kind: str
    value: int | None = None
    misfortune: str | None = None


class RoundView(BaseModel):
    highest_bid: int
    highest_bidder: int | None
    turn_player: int


class PlayerPublicView(BaseModel):
    id: int
    name: str
    open_bid: list[int]
    owned_status_cards: list[StatusCardView]
    money_count: int


class PublicTableView(BaseModel):
    status_card: StatusCardView | None
    round: RoundView | None
    players: list[PlayerPublicView]
    revealed_status_cards: list[StatusCardView]


class ActionView(BaseModel):
    kind: str
    cards: list[int] | None = None
    possession_value: int | None = None


class TurnResponse(BaseModel):
    game_id: str
    status: str
    active_player_id: int | None = None
    active_player_name: str | None = None
    requires_handoff: bool = False
    public_table: PublicTableView
    private_hand: list[int] | None = None
    legal_actions: list[ActionView] = []
    results: dict[str, Any] | None = None


class ActionRequest(BaseModel):
    kind: str
    cards: list[int] | None = None
    possession_value: int | None = None


class SubmitActionRequest(BaseModel):
    player_id: int
    action: ActionRequest


class CreateGameResponse(BaseModel):
    game_id: str


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(service: LocalGameService | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="High Society Local Play", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    svc = service or LocalGameService()
    app.state.service = svc

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/local-games", response_model=CreateGameResponse, status_code=201)
    def create_game(req: CreateGameRequest) -> CreateGameResponse:
        try:
            seat_specs = [
                LocalSeatSpec(kind=s.type, name=s.name) for s in req.seats
            ]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            session = svc.create_game(seat_specs, seed=req.seed)
        except (ValueError, InvalidState) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CreateGameResponse(game_id=session.game_id)

    @app.get("/api/local-games/{game_id}", response_model=TurnResponse)
    def get_game_state(game_id: str) -> TurnResponse:
        try:
            view = svc.get_turn_view(game_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _turn_view_to_response(game_id, view)

    @app.get("/api/local-games/{game_id}/turn", response_model=TurnResponse)
    def get_turn(game_id: str) -> TurnResponse:
        try:
            view = svc.get_turn_view(game_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _turn_view_to_response(game_id, view)

    @app.post("/api/local-games/{game_id}/actions", response_model=TurnResponse)
    def submit_action(game_id: str, req: SubmitActionRequest) -> TurnResponse:
        action = _parse_action(req.action)
        try:
            view = svc.submit_human_action(game_id, req.player_id, action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidAction as exc:
            msg = str(exc)
            # Wrong-player errors get 409 Conflict
            if "turn" in msg.lower() or "not a human seat" in msg.lower():
                raise HTTPException(status_code=409, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc
        except InvalidState as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _turn_view_to_response(game_id, view)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_action(req: ActionRequest) -> Action:
    """Convert an ActionRequest pydantic model into a domain Action."""
    try:
        kind = ActionKind(req.kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Unknown action kind: {req.kind}"
        ) from exc
    if kind == ActionKind.PASS:
        return Action(kind=ActionKind.PASS)
    if kind == ActionKind.BID:
        if not req.cards:
            raise HTTPException(status_code=400, detail="Bid action requires cards")
        return Action(kind=ActionKind.BID, cards=tuple(req.cards))
    if kind == ActionKind.DISCARD_POSSESSION:
        if req.possession_value is None:
            raise HTTPException(
                status_code=400, detail="Discard action requires possession_value"
            )
        return Action(
            kind=ActionKind.DISCARD_POSSESSION,
            possession_value=req.possession_value,
        )
    raise HTTPException(status_code=400, detail=f"Unknown action kind: {req.kind}")


def _status_card_view(card_dict: dict[str, Any]) -> StatusCardView:
    """Convert a status card dict from the service into a view model."""
    return StatusCardView(
        kind=card_dict["kind"],
        value=card_dict.get("value"),
        misfortune=card_dict.get("misfortune"),
    )


def _turn_view_to_response(game_id: str, view: dict[str, Any]) -> TurnResponse:
    """Convert the service turn view dict into a TurnResponse."""
    pub = view["public_table"]

    status_card = None
    if pub.get("status_card") is not None:
        status_card = _status_card_view(pub["status_card"])

    round_view = None
    if pub.get("round") is not None:
        r = pub["round"]
        round_view = RoundView(
            highest_bid=r["highest_bid"],
            highest_bidder=r.get("highest_bidder"),
            turn_player=r["turn_player"],
        )

    players = []
    for p in pub["players"]:
        owned = [_status_card_view(c) for c in p.get("owned_status_cards", [])]
        players.append(
            PlayerPublicView(
                id=p["id"],
                name=p["name"],
                open_bid=p.get("open_bid", []),
                owned_status_cards=owned,
                money_count=p.get("money_count", 0),
            )
        )

    revealed = [
        _status_card_view(c) for c in pub.get("revealed_status_cards", [])
    ]

    public_table = PublicTableView(
        status_card=status_card,
        round=round_view,
        players=players,
        revealed_status_cards=revealed,
    )

    legal_actions = []
    for la in view.get("legal_actions", []):
        legal_actions.append(
            ActionView(
                kind=la["kind"],
                cards=la.get("cards"),
                possession_value=la.get("possession_value"),
            )
        )

    # Service uses "result" key; plan contract uses "results"
    results = view.get("results") or view.get("result")

    return TurnResponse(
        game_id=game_id,
        status=view["status"],
        active_player_id=view.get("active_player_id"),
        active_player_name=view.get("active_player_name"),
        requires_handoff=view.get("requires_handoff", False),
        public_table=public_table,
        private_hand=view.get("private_hand"),
        legal_actions=legal_actions,
        results=results,
    )
