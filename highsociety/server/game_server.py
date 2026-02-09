from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Iterable

from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidAction, InvalidState, RuleViolation
from highsociety.domain.rules import GameResult, RulesEngine
from highsociety.domain.state import GameState, PlayerState


@dataclass(frozen=True)
class PlayerManifestEntry:
    name: str
    kind: str


@dataclass
class GameRecord:
    game_id: str
    state: GameState
    manifest: tuple[PlayerManifestEntry, ...]
    rng: random.Random
    status: str = "active"
    error: str | None = None


@dataclass(frozen=True)
class StepResult:
    state: GameState | None
    error: str | None = None
    fatal: bool = False


class GameServer:
    def __init__(self) -> None:
        self._games: dict[str, GameRecord] = {}

    def new_game(
        self,
        manifest: Iterable[PlayerManifestEntry],
        seed: int | None = None,
        starting_player: int | None = None,
    ) -> str:
        # TODO(api): validate manifest against external player registry.
        manifest_list = list(manifest)
        if not (3 <= len(manifest_list) <= 5):
            raise ValueError("Player count must be 3-5")
        players = []
        for idx, _entry in enumerate(manifest_list):
            players.append(PlayerState(id=idx, hand=RulesEngine.create_money_hand()))
        deck = RulesEngine.create_status_deck()
        rng = random.Random(seed)
        rng.shuffle(deck)
        if starting_player is None:
            starting_player = players[0].id
        state = GameState(
            players=players,
            status_deck=deck,
            starting_player=starting_player,
        )
        game_id = str(uuid.uuid4())
        record = GameRecord(
            game_id=game_id,
            state=state,
            manifest=tuple(manifest_list),
            rng=rng,
        )
        self._games[game_id] = record
        return game_id

    def list_games(self) -> list[str]:
        return list(self._games.keys())

    def get_state(self, game_id: str) -> GameState:
        return self._get_record(game_id).state

    def get_status(self, game_id: str) -> str:
        return self._get_record(game_id).status

    def legal_actions(self, game_id: str, player_id: int) -> list[Action]:
        record = self._get_record(game_id)
        if record.status != "active":
            return []
        self._ensure_round_started(record)
        return RulesEngine.legal_actions(record.state, player_id)

    def step(self, game_id: str, player_id: int, action: Action) -> StepResult:
        record = self._get_record(game_id)
        if record.status == "errored":
            return StepResult(None, record.error, True)
        if record.state.game_over:
            record.status = "finished"
            return StepResult(record.state, "Game is over", False)
        try:
            self._ensure_round_started(record)
            if record.state.game_over:
                record.status = "finished"
                return StepResult(record.state, "Game is over", False)
            RulesEngine.apply_action(record.state, player_id, action)
            if record.state.game_over:
                record.status = "finished"
            return StepResult(record.state)
        except InvalidAction as exc:
            return StepResult(record.state, str(exc), False)
        except (InvalidState, RuleViolation) as exc:
            record.status = "errored"
            record.error = str(exc)
            return StepResult(record.state, str(exc), True)
        except Exception as exc:  # noqa: BLE001
            record.status = "errored"
            record.error = f"Unexpected error: {exc}"
            return StepResult(record.state, record.error, True)

    def score_game(self, game_id: str) -> GameResult:
        record = self._get_record(game_id)
        return RulesEngine.score_game(record.state)

    def _ensure_round_started(self, record: GameRecord) -> None:
        if record.state.game_over:
            return
        if record.state.pending_discard is not None:
            return
        if record.state.round is None:
            RulesEngine.start_round(record.state, record.rng)

    def _get_record(self, game_id: str) -> GameRecord:
        if game_id not in self._games:
            raise KeyError("Unknown game id")
        return self._games[game_id]

    def remove_game(self, game_id: str) -> None:
        """Remove a completed game from memory to prevent memory leaks."""
        self._games.pop(game_id, None)

    def clear_all_games(self) -> None:
        """Remove all games from memory."""
        self._games.clear()
