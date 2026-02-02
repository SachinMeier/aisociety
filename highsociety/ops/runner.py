"""Run manager for executing batches of games."""

from __future__ import annotations

from dataclasses import dataclass

from highsociety.app.runner import GameRunner, GameRunnerResult
from highsociety.domain.rules import GameResult
from highsociety.domain.state import GameState
from highsociety.ops.spec import RunSpec, parse_run_spec
from highsociety.players.base import Player
from highsociety.players.registry import PlayerRegistry, build_default_registry


@dataclass(frozen=True)
class GameRun:
    """Result of a single game in a run."""

    game_id: str
    result: GameResult
    state: GameState


@dataclass(frozen=True)
class RunError:
    """Error information for a failed game run."""

    game_index: int
    message: str


@dataclass(frozen=True)
class RunResult:
    """Aggregate result of a batch run."""

    spec: RunSpec
    games: tuple[GameRun, ...]
    errors: tuple[RunError, ...]


class RunManager:
    """Run multiple games based on a run specification."""

    def __init__(
        self,
        registry: PlayerRegistry | None = None,
        runner: GameRunner | None = None,
    ) -> None:
        """Initialize the run manager with a registry and runner."""
        self._registry = registry or build_default_registry()
        self._runner = runner or GameRunner()

    def run(self, spec: RunSpec | dict) -> RunResult:
        """Execute a batch run and return aggregated results."""
        run_spec = parse_run_spec(spec)
        games: list[GameRun] = []
        errors: list[RunError] = []
        for index in range(run_spec.num_games):
            try:
                players = self._create_players(run_spec)
                seed = run_spec.seed + index
                result = self._runner.run_game(players, seed=seed)
                games.append(_to_game_run(result))
            except Exception as exc:  # noqa: BLE001
                errors.append(RunError(game_index=index, message=str(exc)))
        return RunResult(spec=run_spec, games=tuple(games), errors=tuple(errors))

    def _create_players(self, spec: RunSpec) -> list[Player]:
        """Instantiate players for a run spec."""
        return [self._registry.create(player.to_mapping()) for player in spec.players]


def _to_game_run(result: GameRunnerResult) -> GameRun:
    """Convert a GameRunnerResult to a GameRun record."""
    return GameRun(game_id=result.game_id, result=result.result, state=result.state)
