"""Self-play training loop for the linear RL bot."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from highsociety.app.env_adapter import EnvAdapter
from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult, RulesEngine
from highsociety.domain.state import GameState
from highsociety.ml.encoders.linear import LinearFeatureEncoder
from highsociety.ml.models.linear import LinearModel
from highsociety.ops.spec import PlayerSpec
from highsociety.players.base import Player
from highsociety.players.registry import PlayerRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearTrainingConfig:
    """Configuration for linear self-play training."""

    num_games: int
    player_count: int = 3
    seed: int = 1
    epsilon: float = 0.1
    learning_rate: float = 0.05
    log_every: int = 0
    opponents: tuple[PlayerSpec, ...] = ()
    opponent_weights: tuple[float, ...] = ()
    learner_seats: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if not (3 <= self.player_count <= 5):
            raise ValueError("player_count must be between 3 and 5")
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError("epsilon must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")
        if self.opponent_weights and len(self.opponent_weights) != len(self.opponents):
            raise ValueError("opponent_weights must match opponents length")
        for weight in self.opponent_weights:
            if weight <= 0:
                raise ValueError("opponent_weights must be positive")
        if self.learner_seats is not None:
            for seat in self.learner_seats:
                if seat < 0 or seat >= self.player_count:
                    raise ValueError("learner_seats must be valid player ids")


@dataclass(frozen=True)
class LinearTrainSpec:
    """Specification for a linear training run (file-friendly)."""

    num_games: int
    player_count: int = 3
    seed: int = 1
    epsilon: float = 0.1
    learning_rate: float = 0.05
    log_every: int = 0
    output: str | None = None
    resume: str | None = None
    opponents: tuple[PlayerSpec, ...] = ()
    opponent_weights: tuple[float, ...] = ()
    learner_seats: tuple[int, ...] | None = None

    def to_config(self) -> LinearTrainingConfig:
        """Convert the spec to a training config."""
        return LinearTrainingConfig(
            num_games=self.num_games,
            player_count=self.player_count,
            seed=self.seed,
            epsilon=self.epsilon,
            learning_rate=self.learning_rate,
            log_every=self.log_every,
            opponents=self.opponents,
            opponent_weights=self.opponent_weights,
            learner_seats=self.learner_seats,
        )

    def to_mapping(self) -> dict[str, object]:
        """Serialize the spec into a mapping."""
        return {
            "num_games": self.num_games,
            "player_count": self.player_count,
            "seed": self.seed,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "log_every": self.log_every,
            "output": self.output,
            "resume": self.resume,
            "opponents": [spec.to_mapping() for spec in self.opponents],
            "opponent_weights": list(self.opponent_weights),
            "learner_seats": list(self.learner_seats) if self.learner_seats else None,
        }

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "LinearTrainSpec":
        """Create a spec from a mapping."""
        opponents_data = data.get("opponents", []) or []
        if not isinstance(opponents_data, list):
            raise ValueError("opponents must be a list")
        opponents = tuple(PlayerSpec.from_mapping(item) for item in opponents_data)
        weights_data = data.get("opponent_weights", []) or []
        if weights_data and not isinstance(weights_data, list):
            raise ValueError("opponent_weights must be a list")
        weights = tuple(float(value) for value in weights_data)
        learner_seats_data = data.get("learner_seats")
        if learner_seats_data is None:
            learner_seats = None
        else:
            if not isinstance(learner_seats_data, list):
                raise ValueError("learner_seats must be a list")
            learner_seats = tuple(int(value) for value in learner_seats_data)
        return LinearTrainSpec(
            num_games=int(data.get("num_games", 0)),
            player_count=int(data.get("player_count", 3)),
            seed=int(data.get("seed", 1)),
            epsilon=float(data.get("epsilon", 0.1)),
            learning_rate=float(data.get("learning_rate", 0.05)),
            log_every=int(data.get("log_every", 0)),
            output=_coerce_optional_str(data.get("output")),
            resume=_coerce_optional_str(data.get("resume")),
            opponents=opponents,
            opponent_weights=weights,
            learner_seats=learner_seats,
        )


@dataclass(frozen=True)
class LinearTrainingResult:
    """Result of a linear training run."""

    model: LinearModel
    encoder: LinearFeatureEncoder
    games: int
    average_reward: float
    average_winners: float


@dataclass
class _EpisodeTracker:
    """Track per-episode data for each player."""

    trajectories: dict[int, list[np.ndarray]]
    winners: tuple[int, ...] | None = None


def train_linear_self_play(
    config: LinearTrainingConfig,
    encoder: LinearFeatureEncoder | None = None,
    model: LinearModel | None = None,
    registry: PlayerRegistry | None = None,
) -> LinearTrainingResult:
    """Train a linear model via self-play and return the trained artifacts."""
    encoder = encoder or LinearFeatureEncoder()
    model = model or LinearModel.initialize(encoder.feature_size(), seed=config.seed)
    rng = random.Random(config.seed)
    registry = registry or build_default_registry()
    learner_seats = _resolve_learner_seats(config)

    total_winner_slots = 0
    total_reward = 0.0
    for game_index in range(config.num_games):
        opponents = _sample_opponents(config, rng, registry, learner_seats)
        tracker = _run_episode(
            config=config,
            encoder=encoder,
            model=model,
            rng=rng,
            seed=config.seed + game_index,
            opponents=opponents,
            learner_seats=learner_seats,
        )
        winners = tracker.winners or ()
        total_winner_slots += len(winners)
        for player_id, features_list in tracker.trajectories.items():
            reward = 1.0 if player_id in winners else 0.0
            total_reward += reward
            for features in features_list:
                model.update(features, reward, config.learning_rate)
        if config.log_every and (game_index + 1) % config.log_every == 0:
            logger.info("Completed %s games", game_index + 1)

    denom = float(config.num_games * max(1, len(learner_seats)))
    average_reward = total_reward / denom
    average_winners = total_winner_slots / float(config.num_games)
    return LinearTrainingResult(
        model=model,
        encoder=encoder,
        games=config.num_games,
        average_reward=average_reward,
        average_winners=average_winners,
    )


def _run_episode(
    config: LinearTrainingConfig,
    encoder: LinearFeatureEncoder,
    model: LinearModel,
    rng: random.Random,
    seed: int,
    opponents: dict[int, Player],
    learner_seats: tuple[int, ...],
) -> _EpisodeTracker:
    """Run a single self-play episode and capture trajectories."""
    env = EnvAdapter(player_count=config.player_count)
    env.reset(seed=seed)
    _reset_opponents(opponents, seed, config.player_count)
    trajectories: dict[int, list[np.ndarray]] = {pid: [] for pid in learner_seats}
    winners: tuple[int, ...] | None = None
    while True:
        state = env.get_state()
        if state.game_over:
            winners = _score_fallback(state).winners
            break
        current_player = _current_player_id(state)
        observation = env.observe(current_player)
        legal_actions = env.legal_actions(current_player)
        if not legal_actions:
            state_after = env.get_state()
            if state_after.game_over:
                winners = _score_fallback(state_after).winners
                break
            raise InvalidState("No legal actions available during training")
        if current_player in opponents:
            action = opponents[current_player].act(observation, legal_actions)
        else:
            action, features = _select_action(
                observation=observation,
                legal_actions=legal_actions,
                encoder=encoder,
                model=model,
                rng=rng,
                epsilon=config.epsilon,
            )
            trajectories[current_player].append(features)
        _state, _reward, done, info = env.step(current_player, action)
        if done:
            if info.result is None:
                winners = _score_fallback(env.get_state()).winners
            else:
                winners = info.result.winners
            break
    return _EpisodeTracker(trajectories=trajectories, winners=winners)


def _select_action(
    observation: Observation,
    legal_actions: list[Action],
    encoder: LinearFeatureEncoder,
    model: LinearModel,
    rng: random.Random,
    epsilon: float,
) -> tuple[Action, np.ndarray]:
    """Select an action via epsilon-greedy scoring and return (action, features)."""
    if rng.random() < epsilon:
        action = rng.choice(legal_actions)
        features = encoder.encode(observation, action)
        return action, features
    best_action = legal_actions[0]
    best_features = encoder.encode(observation, best_action)
    best_score = model.predict(best_features)
    for action in legal_actions[1:]:
        features = encoder.encode(observation, action)
        score = model.predict(features)
        if score > best_score:
            best_action = action
            best_features = features
            best_score = score
    return best_action, best_features


def _current_player_id(state: GameState) -> int:
    """Return the player expected to act next in the game state."""
    if state.pending_discard is not None:
        return state.pending_discard.player_id
    if state.round is None:
        return state.starting_player
    return state.round.turn_player


def _score_fallback(state: GameState) -> GameResult:
    """Score the game state directly if the env does not provide a result."""
    return RulesEngine.score_game(state)


def _resolve_learner_seats(config: LinearTrainingConfig) -> tuple[int, ...]:
    """Determine which seats are controlled by the learning model."""
    if config.learner_seats is not None:
        return config.learner_seats
    if config.opponents:
        return (0,)
    return tuple(range(config.player_count))


def _sample_opponents(
    config: LinearTrainingConfig,
    rng: random.Random,
    registry: PlayerRegistry,
    learner_seats: tuple[int, ...],
) -> dict[int, Player]:
    """Sample opponent bots for the non-learning seats."""
    if not config.opponents:
        return {}
    opponent_seats = [seat for seat in range(config.player_count) if seat not in learner_seats]
    if not opponent_seats:
        return {}
    weights = config.opponent_weights or None
    specs = list(config.opponents)
    opponents: dict[int, Player] = {}
    for seat in opponent_seats:
        spec = rng.choices(specs, weights=weights, k=1)[0]
        opponents[seat] = registry.create(spec.to_mapping())
    return opponents


def _reset_opponents(opponents: dict[int, Player], seed: int, player_count: int) -> None:
    """Reset opponent bots for a new game."""
    game_config: dict[str, object] = {"seed": seed, "player_count": player_count}
    for seat, opponent in opponents.items():
        opponent.reset(game_config, player_id=seat, seat=seat)


def _coerce_optional_str(value: object) -> str | None:
    """Coerce a value to an optional string."""
    if isinstance(value, str) and value:
        return value
    return None
