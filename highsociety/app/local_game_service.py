"""Interactive local game service for pass-and-play sessions.

Unlike GameRunner which executes a full game in one blocking call,
this service pauses when a human decision is needed and resumes
when the human submits an action via HTTP.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from highsociety.app.observations import build_observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.domain.state import GameState
from highsociety.players.base import Player
from highsociety.players.registry import PlayerRegistry, build_default_registry

if TYPE_CHECKING:
    from highsociety.server.game_server import GameServer


def _import_game_server():  # noqa: ANN202
    """Lazy import to avoid circular dependency with server.__init__."""
    from highsociety.server.game_server import GameServer, PlayerManifestEntry
    return GameServer, PlayerManifestEntry


# ---------------------------------------------------------------------------
# Difficulty preset mapping
# ---------------------------------------------------------------------------

_PRESET_SPECS: dict[str, dict[str, Any]] = {
    "easy": {"type": "static", "name": "Easy Bot"},
    "medium": {"type": "heuristic", "name": "Medium Bot", "params": {"style": "balanced"}},
    "hard": {
        "type": "mlp",
        "name": "Hard Bot",
        "checkpoint": "checkpoints/mlp/mlp_v3",
    },
    "expert": {
        "type": "heuristic",
        "name": "Expert Bot",
        "params": {"style": "cautious"},
    },
}

VALID_BOT_PRESETS = tuple(_PRESET_SPECS.keys())


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalSeatSpec:
    """Specification for a single seat at the table."""

    kind: Literal["human", "easy", "medium", "hard", "expert"]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("human", *VALID_BOT_PRESETS):
            raise ValueError(f"Unknown seat kind: {self.kind}")


@dataclass
class LocalGameSession:
    """In-memory state for one interactive local game."""

    game_id: str
    server_game_id: str  # internal UUID used by GameServer
    seats: list[LocalSeatSpec]
    display_names: list[str]
    human_seats: set[int]
    bot_players: dict[int, Player]  # seat_id -> Player instance (bots only)
    server: GameServer
    result: GameResult | None = None
    status: str = "active"  # active | awaiting_human | finished | errored
    error: str | None = None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class LocalGameService:
    """Orchestrate interactive local games with mixed human/bot seats."""

    def __init__(self, registry: PlayerRegistry | None = None) -> None:
        self._registry = registry or build_default_registry()
        self._sessions: dict[str, LocalGameSession] = {}
        GameServerCls, _ = _import_game_server()
        self._server: GameServer = GameServerCls()

    # -- public API ----------------------------------------------------------

    def create_game(
        self,
        seats: list[LocalSeatSpec],
        seed: int | None = None,
    ) -> LocalGameSession:
        """Create a new game session with the given seat configuration.

        Returns the session after auto-advancing bots to the first human
        turn (or to game end if all seats are bots).
        """
        if not (3 <= len(seats) <= 5):
            raise InvalidState("Player count must be 3-5")

        game_id = self._new_game_id()
        display_names: list[str] = []
        human_seats: set[int] = set()
        bot_players: dict[int, Player] = {}

        for idx, seat in enumerate(seats):
            if seat.kind == "human":
                human_seats.add(idx)
                display_names.append(seat.name or f"Player {idx + 1}")
            else:
                spec = dict(_PRESET_SPECS[seat.kind])
                if seat.name:
                    spec["name"] = seat.name
                player = self._registry.create(spec)
                bot_players[idx] = player
                display_names.append(seat.name or spec.get("name", f"Bot {idx + 1}"))

        _, ManifestEntry = _import_game_server()
        manifest = [
            ManifestEntry(name=display_names[i], kind=seats[i].kind)
            for i in range(len(seats))
        ]
        server_game_id = self._server.new_game(manifest=manifest, seed=seed)

        # Reset bot players
        game_config: dict[str, object] = {
            "seed": seed,
            "game_id": server_game_id,
            "player_count": len(seats),
        }
        for seat_id, player in bot_players.items():
            player.reset(game_config, player_id=seat_id, seat=seat_id)

        session = LocalGameSession(
            game_id=game_id,
            server_game_id=server_game_id,
            seats=list(seats),
            display_names=display_names,
            human_seats=human_seats,
            bot_players=bot_players,
            server=self._server,
        )
        self._sessions[game_id] = session

        self._advance_until_human_or_terminal(session)
        return session

    def get_session(self, game_id: str) -> LocalGameSession:
        """Look up a game session by its short ID."""
        if game_id not in self._sessions:
            raise KeyError(f"Unknown game id: {game_id}")
        return self._sessions[game_id]

    def get_turn_view(self, game_id: str) -> dict[str, Any]:
        """Build the current turn payload for the UI.

        Returns a dict matching the Turn Response contract from the plan.
        """
        session = self.get_session(game_id)
        state = self._server.get_state(session.server_game_id)

        if session.status == "finished":
            return self._build_finished_view(session, state)
        if session.status == "errored":
            return {
                "game_id": game_id,
                "status": "errored",
                "error": session.error,
            }

        active_player_id = _current_player_id(state)
        is_human = active_player_id in session.human_seats
        legal = self._server.legal_actions(session.server_game_id, active_player_id)
        # Re-read state after legal_actions (may start a round)
        state = self._server.get_state(session.server_game_id)
        if state.game_over:
            self._finalize(session)
            return self._build_finished_view(session, state)

        return {
            "game_id": game_id,
            "status": "awaiting_human_action" if is_human else "active",
            "active_player_id": active_player_id,
            "active_player_name": session.display_names[active_player_id],
            "requires_handoff": is_human and len(session.human_seats) > 1,
            "public_table": self._build_public_table(state, session),
            "private_hand": _sorted_hand(state.players[active_player_id].hand)
            if is_human
            else None,
            "legal_actions": _serialize_actions(legal) if is_human else None,
            "round_history": [
                {
                    "card": _serialize_card(rec.card),
                    "winner_id": rec.winner_id,
                    "winner_name": session.display_names[rec.winner_id],
                    "coins_spent": list(rec.coins_spent),
                }
                for rec in state.round_history
            ],
        }

    def submit_human_action(
        self,
        game_id: str,
        player_id: int,
        action: Action,
    ) -> dict[str, Any]:
        """Submit a human action and advance the game.

        Returns the updated turn view after bot auto-advance.
        """
        session = self.get_session(game_id)
        if session.status == "finished":
            state = self._server.get_state(session.server_game_id)
            return self._build_finished_view(session, state)

        state = self._server.get_state(session.server_game_id)
        current = _current_player_id(state)
        if current != player_id:
            raise InvalidAction(
                f"Not player {player_id}'s turn (current: {current})"
            )
        if player_id not in session.human_seats:
            raise InvalidAction(f"Seat {player_id} is not a human seat")

        step = self._server.step(session.server_game_id, player_id, action)
        if step.fatal:
            session.status = "errored"
            session.error = step.error
            raise InvalidState(step.error or "Fatal game error")
        if step.error:
            raise InvalidAction(step.error)

        self._advance_until_human_or_terminal(session)
        return self.get_turn_view(game_id)

    # -- internal helpers ----------------------------------------------------

    def _advance_until_human_or_terminal(self, session: LocalGameSession) -> None:
        """Auto-play bot turns until a human must act or the game ends."""
        max_steps = 5000  # safety limit
        for _ in range(max_steps):
            state = self._server.get_state(session.server_game_id)
            if state.game_over:
                self._finalize(session)
                return

            current = _current_player_id(state)

            if current in session.human_seats:
                session.status = "awaiting_human"
                return

            # Current player is a bot — auto-play
            bot = session.bot_players.get(current)
            if bot is None:
                session.status = "errored"
                session.error = f"No bot registered for seat {current}"
                return

            legal = self._server.legal_actions(session.server_game_id, current)
            state = self._server.get_state(session.server_game_id)
            if state.game_over:
                self._finalize(session)
                return
            if not legal:
                session.status = "errored"
                session.error = f"No legal actions for bot at seat {current}"
                return

            observation = build_observation(state, current)
            action = bot.act(observation, legal)
            step = self._server.step(session.server_game_id, current, action)
            if step.fatal:
                session.status = "errored"
                session.error = step.error
                return
            if step.error:
                session.status = "errored"
                session.error = f"Bot action error: {step.error}"
                return

        session.status = "errored"
        session.error = "Game exceeded maximum step limit"

    def _finalize(self, session: LocalGameSession) -> None:
        """Mark session as finished and compute final scores."""
        if session.result is not None:
            return
        session.result = self._server.score_game(session.server_game_id)
        session.status = "finished"
        for bot in session.bot_players.values():
            bot.on_game_end(session.result)

    def _new_game_id(self) -> str:
        """Generate a short uppercase Base32 game ID (8 chars), unique in-memory."""
        for _ in range(100):
            raw = os.urandom(5)  # 5 bytes -> 8 Base32 chars
            token = base64.b32encode(raw).decode("ascii").rstrip("=")[:8].upper()
            if token not in self._sessions:
                return token
        raise RuntimeError("Failed to generate unique game ID")

    def _build_public_table(
        self,
        state: GameState,
        session: LocalGameSession,
    ) -> dict[str, Any]:
        """Build the public table view payload."""
        players_view = []
        for ps in state.players:
            players_view.append({
                "id": ps.id,
                "name": session.display_names[ps.id],
                "open_bid": [c.value for c in ps.open_bid],
                "owned_status_cards": _serialize_status_cards(ps),
                "money_count": len(ps.hand),
            })

        status_card = None
        round_view = None
        if state.round is not None:
            status_card = _serialize_card(state.round.card)
            round_view = {
                "highest_bid": state.round.highest_bid,
                "highest_bidder": state.round.highest_bidder,
                "turn_player": state.round.turn_player,
            }

        return {
            "status_card": status_card,
            "round": round_view,
            "players": players_view,
            "revealed_status_cards": [
                _serialize_card(c) for c in state.status_discard
            ],
        }

    def _build_finished_view(
        self,
        session: LocalGameSession,
        state: GameState,
    ) -> dict[str, Any]:
        """Build the terminal game-over payload."""
        result = session.result
        if result is None:
            result = self._server.score_game(session.server_game_id)
            session.result = result
        return {
            "game_id": session.game_id,
            "status": "finished",
            "result": {
                "winners": list(result.winners),
                "scores": {
                    str(pid): float(score)
                    for pid, score in result.scores.items()
                },
                "money_remaining": result.money_remaining,
                "poorest": list(result.poorest),
            },
            "public_table": self._build_public_table(state, session),
            "round_history": [
                {
                    "card": _serialize_card(rec.card),
                    "winner_id": rec.winner_id,
                    "winner_name": session.display_names[rec.winner_id],
                    "coins_spent": list(rec.coins_spent),
                }
                for rec in state.round_history
            ],
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _current_player_id(state: GameState) -> int:
    """Return the player id expected to act next."""
    if state.pending_discard is not None:
        return state.pending_discard.player_id
    if state.round is None:
        return state.starting_player
    return state.round.turn_player


def _sorted_hand(hand: list) -> list[int]:
    """Return sorted money card values from a hand."""
    return sorted(c.value for c in hand)


def _serialize_card(card) -> dict[str, Any]:
    """Serialize a StatusCard to a dict."""
    result: dict[str, Any] = {"kind": card.kind.value}
    if card.value is not None:
        result["value"] = card.value
    if card.misfortune is not None:
        result["misfortune"] = card.misfortune.value
    return result


def _serialize_status_cards(ps) -> list[dict[str, Any]]:
    """Serialize a player's owned status cards (possessions + markers)."""
    cards: list[dict[str, Any]] = []
    for c in ps.possessions:
        cards.append(_serialize_card(c))
    for _ in range(ps.titles):
        cards.append({"kind": "title"})
    if ps.scandal:
        cards.append({"kind": "misfortune", "misfortune": "scandal"})
    if ps.debt:
        cards.append({"kind": "misfortune", "misfortune": "debt"})
    if ps.theft:
        cards.append({"kind": "misfortune", "misfortune": "theft"})
    return cards


def _serialize_actions(actions: list[Action]) -> list[dict[str, Any]]:
    """Serialize legal actions for the UI."""
    result = []
    for a in actions:
        entry: dict[str, Any] = {"kind": a.kind.value}
        if a.kind == ActionKind.BID:
            entry["cards"] = list(a.cards)
        elif a.kind == ActionKind.DISCARD_POSSESSION:
            entry["possession_value"] = a.possession_value
        result.append(entry)
    return result
