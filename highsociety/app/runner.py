"""Game runner that orchestrates player turns."""

from __future__ import annotations

from dataclasses import dataclass

from highsociety.app.observations import build_observation
from highsociety.domain.errors import InvalidAction, InvalidState
from highsociety.domain.rules import GameResult
from highsociety.domain.state import GameState
from highsociety.players.base import Player
from highsociety.server.game_server import GameServer, PlayerManifestEntry


@dataclass(frozen=True)
class GameRunnerResult:
    """Result of a completed game run."""

    game_id: str
    state: GameState
    result: GameResult


class GameRunner:
    """Run a single game by driving player actions."""

    def __init__(self, server: GameServer | None = None) -> None:
        """Initialize the runner with an optional game server."""
        self._server = server or GameServer()

    def run_game(
        self,
        players: list[Player],
        seed: int | None = None,
        starting_player: int | None = None,
    ) -> GameRunnerResult:
        """Run a game to completion and return the final result."""
        if not (3 <= len(players) <= 5):
            raise InvalidState("Player count must be 3-5")
        manifest = [
            PlayerManifestEntry(name=player.name, kind=_player_kind(player))
            for player in players
        ]
        game_id = self._server.new_game(
            manifest=manifest,
            seed=seed,
            starting_player=starting_player,
        )
        game_config: dict[str, object] = {
            "seed": seed,
            "game_id": game_id,
            "player_count": len(players),
        }
        for seat, player in enumerate(players):
            player.reset(game_config, player_id=seat, seat=seat)
        while True:
            state = self._server.get_state(game_id)
            if state.game_over:
                break
            current_player_id = _current_player_id(state)
            legal = self._server.legal_actions(game_id, current_player_id)
            state = self._server.get_state(game_id)
            if state.game_over:
                break
            if not legal:
                raise InvalidState("No legal actions available for current player")
            observation = build_observation(state, current_player_id)
            action = players[current_player_id].act(observation, legal)
            step_result = self._server.step(game_id, current_player_id, action)
            if step_result.fatal:
                raise InvalidState(step_result.error or "Fatal game error")
            if step_result.error:
                raise InvalidAction(step_result.error)
        result = self._server.score_game(game_id)
        for player in players:
            player.on_game_end(result)
        final_state = self._server.get_state(game_id)
        return GameRunnerResult(game_id=game_id, state=final_state, result=result)


def _current_player_id(state: GameState) -> int:
    """Return the player id expected to act next."""
    if state.pending_discard is not None:
        return state.pending_discard.player_id
    if state.round is None:
        return state.starting_player
    return state.round.turn_player


def _player_kind(player: Player) -> str:
    """Return a descriptive kind label for a player."""
    kind = getattr(player, "kind", None)
    if isinstance(kind, str) and kind:
        return kind
    return player.__class__.__name__
