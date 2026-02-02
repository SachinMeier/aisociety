"""RL-style environment adapter over the game server."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from highsociety.app.observations import Observation, build_observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult
from highsociety.domain.state import GameState
from highsociety.server.game_server import GameServer, PlayerManifestEntry


@dataclass(frozen=True)
class StepInfo:
    """Additional info returned from an environment step."""

    error: str | None
    fatal: bool
    result: GameResult | None


class EnvAdapter:
    """Environment wrapper exposing observe/legal/step APIs."""

    def __init__(self, player_count: int = 3, server: GameServer | None = None) -> None:
        """Initialize the adapter for a given player count."""
        if not (3 <= player_count <= 5):
            raise InvalidState("Player count must be 3-5")
        self._player_count = player_count
        self._server = server or GameServer()
        self._game_id: str | None = None

    def reset(self, seed: int | None = None, starting_player: int | None = None) -> GameState:
        """Start a new game and return its initial state."""
        manifest = tuple(
            PlayerManifestEntry(name=f"p{idx}", kind="env")
            for idx in range(self._player_count)
        )
        self._game_id = self._server.new_game(
            manifest=manifest,
            seed=seed,
            starting_player=starting_player,
        )
        return self._server.get_state(self._game_id)

    def get_state(self) -> GameState:
        """Return the current game state."""
        return self._server.get_state(self._require_game_id())

    def observe(self, player_id: int) -> Observation:
        """Return an info-set-safe observation for a player."""
        self._ensure_round_started()
        return build_observation(self.get_state(), player_id)

    def legal_actions(self, player_id: int) -> list[Action]:
        """Return legal actions for the given player."""
        return self._server.legal_actions(self._require_game_id(), player_id)

    def step(
        self, player_id: int, action: Action
    ) -> tuple[GameState, float, bool, StepInfo]:
        """Apply an action and return (state, reward, done, info)."""
        result = self._server.step(self._require_game_id(), player_id, action)
        state = self.get_state()
        done = state.game_over or result.fatal
        game_result: GameResult | None = None
        reward = 0.0
        if done:
            game_result = self._server.score_game(self._require_game_id())
            reward = 1.0 if player_id in game_result.winners else 0.0
        info = StepInfo(error=result.error, fatal=result.fatal, result=game_result)
        return state, reward, done, info

    def clone(self) -> "EnvAdapter":
        """Return a deep copy of the environment."""
        return copy.deepcopy(self)

    def serialize(self) -> dict[str, object]:
        """Serialize the current environment state for debugging."""
        return {"game_id": self._require_game_id(), "state": self.get_state()}

    def _require_game_id(self) -> str:
        """Return the active game id or raise if missing."""
        if self._game_id is None:
            raise InvalidState("Environment has not been reset")
        return self._game_id

    def _ensure_round_started(self) -> None:
        """Ensure a round is active before observing."""
        state = self.get_state()
        if state.game_over or state.pending_discard is not None:
            return
        if state.round is None:
            self._server.legal_actions(self._require_game_id(), state.starting_player)
