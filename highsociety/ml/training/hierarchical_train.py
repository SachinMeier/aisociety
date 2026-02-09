"""Training loop for the hierarchical policy/value model with GAE.

Optimized with:
- TRUE MULTIPROCESSING for parallel episode collection (bypasses GIL)
- Fast mask building (O(n) instead of O(2^n))
- Memory-efficient game cleanup
- Batched artifact logging
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Use spawn for macOS compatibility (fork is unsafe with PyTorch)
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

try:  # pragma: no cover - torch is optional for non-ML usage
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for highsociety.ml.training.hierarchical_train") from exc

try:  # pragma: no cover - tqdm is optional in minimal envs
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: range, *args: object, **kwargs: object) -> range:
        """Fallback progress wrapper when tqdm is unavailable."""
        del args, kwargs
        return iterable

from highsociety.app.env_adapter import EnvAdapter
from highsociety.app.observations import Observation
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import StatusKind
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.checkpoints import load_checkpoint, save_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder
from highsociety.ml.models.hierarchical import (
    MONEY_CARD_VALUES,
    HierarchicalConfig,
    HierarchicalPolicyValue,
)
from highsociety.ml.training.artifacts import TrainingArtifactLogger
from highsociety.ops.spec import PlayerSpec
from highsociety.players.base import Player
from highsociety.players.registry import PlayerRegistry, build_default_registry

# Enable MPS fallback for unsupported operations
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


@dataclass(frozen=True)
class HierarchicalTrainSpec:
    """Configuration for training a hierarchical policy/value model."""

    episodes: int = 50000
    seed: int = 42
    player_count: int = 3
    learning_rate: float = 1e-3
    learning_rate_final: float | None = None
    trunk_sizes: tuple[int, ...] = (256, 256)
    activation: str = "relu"
    dropout: float = 0.0
    temperature: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    discount: float = 0.99
    batch_size: int = 32  # Increased from 4 for better GPU utilization
    max_grad_norm: float = 1.0
    checkpoint_path: str | None = None
    checkpoint_every: int = 1000
    artifacts_path: str | None = None
    resume: str | None = None
    opponents: tuple[PlayerSpec, ...] = ()
    opponent_weights: tuple[float, ...] = ()
    device: str = "cpu"  # CPU is faster for small models with single-sample inference
    num_workers: int = 8  # Number of parallel workers for episode collection
    artifact_log_interval: int = 100  # Log artifacts every N episodes (batching)

    def __post_init__(self) -> None:
        """Validate training configuration values."""
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not (3 <= self.player_count <= 5):
            raise ValueError("player_count must be 3-5")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.learning_rate_final is not None:
            if self.learning_rate_final <= 0:
                raise ValueError("learning_rate_final must be positive")
            if self.learning_rate_final > self.learning_rate:
                raise ValueError("learning_rate_final must be <= learning_rate")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        if self.value_coef < 0:
            raise ValueError("value_coef must be non-negative")
        if self.gae_lambda < 0 or self.gae_lambda > 1:
            raise ValueError("gae_lambda must be between 0 and 1")
        if self.discount < 0 or self.discount > 1:
            raise ValueError("discount must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if not self.trunk_sizes:
            raise ValueError("trunk_sizes cannot be empty")
        if self.opponent_weights and len(self.opponent_weights) != len(self.opponents):
            raise ValueError("opponent_weights must match opponents length")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.artifact_log_interval <= 0:
            raise ValueError("artifact_log_interval must be positive")

    def to_mapping(self) -> dict[str, object]:
        """Serialize the training spec to a mapping."""
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "player_count": self.player_count,
            "learning_rate": self.learning_rate,
            "learning_rate_final": self.learning_rate_final,
            "trunk_sizes": list(self.trunk_sizes),
            "activation": self.activation,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "gae_lambda": self.gae_lambda,
            "discount": self.discount,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_every": self.checkpoint_every,
            "artifacts_path": self.artifacts_path,
            "resume": self.resume,
            "opponents": [spec.to_mapping() for spec in self.opponents],
            "opponent_weights": list(self.opponent_weights),
            "device": self.device,
            "num_workers": self.num_workers,
            "artifact_log_interval": self.artifact_log_interval,
        }

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "HierarchicalTrainSpec":
        """Create a training spec from a mapping."""
        trunk_sizes = data.get("trunk_sizes", (256, 256))
        if isinstance(trunk_sizes, (list, tuple)):
            trunk_tuple = tuple(int(size) for size in trunk_sizes)
        else:
            trunk_tuple = (256, 256)
        opponents_data = data.get("opponents", []) or []
        if not isinstance(opponents_data, list):
            raise ValueError("opponents must be a list")
        opponents = tuple(PlayerSpec.from_mapping(item) for item in opponents_data)
        weights_data = data.get("opponent_weights", []) or []
        if weights_data and not isinstance(weights_data, list):
            raise ValueError("opponent_weights must be a list")
        weights = tuple(float(value) for value in weights_data)
        return HierarchicalTrainSpec(
            episodes=int(data.get("episodes", 50000)),
            seed=int(data.get("seed", 42)),
            player_count=int(data.get("player_count", 3)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            learning_rate_final=_coerce_optional_float(data.get("learning_rate_final")),
            trunk_sizes=trunk_tuple,
            activation=str(data.get("activation", "relu")),
            dropout=float(data.get("dropout", 0.0)),
            temperature=float(data.get("temperature", 1.0)),
            entropy_coef=float(data.get("entropy_coef", 0.01)),
            value_coef=float(data.get("value_coef", 0.5)),
            gae_lambda=float(data.get("gae_lambda", 0.95)),
            discount=float(data.get("discount", 0.99)),
            batch_size=int(data.get("batch_size", 32)),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            checkpoint_path=_coerce_optional_str(data.get("checkpoint_path")),
            checkpoint_every=int(data.get("checkpoint_every", 1000)),
            artifacts_path=_coerce_optional_str(data.get("artifacts_path")),
            resume=_coerce_optional_str(data.get("resume")),
            opponents=opponents,
            opponent_weights=weights,
            device=str(data.get("device", "cpu")),
            num_workers=int(data.get("num_workers", 8)),
            artifact_log_interval=int(data.get("artifact_log_interval", 100)),
        )


@dataclass(frozen=True)
class TrainMetrics:
    """Summary metrics from a training run."""

    episodes: int
    wins: int
    win_rate: float
    poorest_eliminations: int
    poorest_rate: float
    average_loss: float
    average_policy_loss: float
    average_value_loss: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metrics."""
        return {
            "episodes": self.episodes,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "poorest_eliminations": self.poorest_eliminations,
            "poorest_rate": self.poorest_rate,
            "average_loss": self.average_loss,
            "average_policy_loss": self.average_policy_loss,
            "average_value_loss": self.average_value_loss,
        }


@dataclass
class EpisodeData:
    """Trajectory data from a single episode."""

    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    type_logits: list[torch.Tensor]
    won: bool
    eliminated: bool


@dataclass
class EpisodeResult:
    """Result from running an episode (for parallel collection)."""

    episode_data: EpisodeData
    episode_seed: int
    game_result: GameResult | None = None


@dataclass
class EpisodeResultSerializable:
    """Serializable episode result for multiprocessing (no torch tensors).

    Contains trajectory data needed to recompute log_probs in main process.
    """

    # Trajectory data for recomputing gradients
    observations: list[list[float]]  # Encoded observations
    action_types: list[int]  # 0=PASS, 1=BID, 2=DISCARD
    selected_cards: list[list[int]]  # For BID actions: which card indices were selected
    discard_indices: list[int]  # For DISCARD actions: which possession index

    # Values and rewards (don't need gradients)
    values: list[float]
    rewards: list[float]

    # Metadata
    won: bool
    eliminated: bool
    episode_seed: int


def _worker_run_episodes(
    args: tuple[list[int], dict, dict, dict, "HierarchicalTrainSpec"],
) -> list[EpisodeResultSerializable]:
    """Worker function that runs episodes and collects trajectory data.

    Returns trajectory data (observations, actions) that the main process
    uses to recompute log_probs with proper gradients.
    """
    episode_seeds, model_state, model_config_dict, encoder_config, spec = args

    # Recreate model in this process
    encoder = BasicEncoder.from_config(encoder_config)
    config = HierarchicalConfig.from_dict(model_config_dict)
    model = HierarchicalPolicyValue(config)
    model.load_state_dict(model_state)
    model.eval()

    device = torch.device("cpu")

    # Create environment and opponents
    rng = random.Random(episode_seeds[0] if episode_seeds else 42)
    env = EnvAdapter(player_count=spec.player_count)

    opponents: dict[int, Player] = {}
    if spec.opponents:
        registry = build_default_registry()
        opponents = _sample_opponents(spec, rng, registry)

    results: list[EpisodeResultSerializable] = []

    for seed in episode_seeds:
        env.reset(seed=seed)
        _reset_opponents(opponents, seed, spec.player_count)

        # Collect trajectory
        observations: list[list[float]] = []
        action_types: list[int] = []
        selected_cards: list[list[int]] = []
        discard_indices: list[int] = []
        values: list[float] = []

        done = False
        while not done:
            player_id, legal_actions = _next_turn(env, spec.player_count)
            if player_id is None:
                done = True
                break

            observation = env.observe(player_id)

            if player_id == 0:  # Trainee
                is_discard = legal_actions and legal_actions[0].kind == ActionKind.DISCARD_POSSESSION

                # Encode observation for later replay
                obs_encoded = encoder.encode(observation)
                observations.append(obs_encoded)

                # Select action and record what was chosen
                action, _, value, _ = _select_hierarchical_action_fast(
                    model=model,
                    encoder=encoder,
                    env=env,
                    observation=observation,
                    is_discard=is_discard,
                    temperature=spec.temperature,
                    device=device,
                    rng=rng,
                )

                values.append(float(value.item()))

                # Record action details for log_prob recomputation
                if action.kind == ActionKind.PASS:
                    action_types.append(0)
                    selected_cards.append([])
                    discard_indices.append(-1)
                elif action.kind == ActionKind.BID:
                    action_types.append(1)
                    # Record which card indices were selected
                    card_indices = []
                    for card_val in action.cards:
                        try:
                            idx = MONEY_CARD_VALUES.index(card_val)
                            card_indices.append(idx)
                        except ValueError:
                            pass
                    selected_cards.append(card_indices)
                    discard_indices.append(-1)
                else:  # DISCARD
                    action_types.append(2)
                    selected_cards.append([])
                    discard_indices.append((action.possession_value or 1) - 1)
            else:
                opponent = opponents.get(player_id)
                if opponent is not None:
                    action = opponent.act(observation, legal_actions)
                else:
                    action = rng.choice(legal_actions)

            _, _, done, _ = env.step(player_id, action)

        # Compute reward
        result = env.game_result()
        won = 0 in result.winners
        eliminated = 0 in result.poorest

        if eliminated:
            reward = -1.0
        elif won:
            reward = 1.0
        else:
            scores = result.scores
            trainee_score = scores[0]
            rank = sum(1 for s in scores if s > trainee_score) + 1
            reward = 1.0 - 2.0 * (rank - 1) / (len(scores) - 1) if len(scores) > 1 else 0.0

        rewards = [0.0] * len(observations)
        if rewards:
            rewards[-1] = reward

        results.append(EpisodeResultSerializable(
            observations=observations,
            action_types=action_types,
            selected_cards=selected_cards,
            discard_indices=discard_indices,
            values=values,
            rewards=rewards,
            won=won,
            eliminated=eliminated,
            episode_seed=seed,
        ))

    env.cleanup()
    return results


@dataclass
class WorkerContext:
    """Reusable context for a worker thread."""

    env: EnvAdapter
    opponents: dict[int, Player]
    rng: random.Random

    @classmethod
    def create(
        cls, player_count: int, spec: HierarchicalTrainSpec,
        registry: PlayerRegistry | None, seed: int
    ) -> "WorkerContext":
        """Create a fresh worker context."""
        rng = random.Random(seed)
        env = EnvAdapter(player_count=player_count)
        opponents = _sample_opponents(spec, rng, registry) if spec.opponents else {}
        return cls(env=env, opponents=opponents, rng=rng)


def train_hierarchical(
    spec: HierarchicalTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train a hierarchical policy/value model and return summary metrics.

    Uses TRUE MULTIPROCESSING for parallel episode collection when num_workers > 1.
    """
    # Use parallel training if multiple workers requested
    if spec.num_workers > 1:
        return _train_hierarchical_parallel(spec, registry)
    return _train_hierarchical_single(spec, registry)


def _train_hierarchical_parallel(
    spec: HierarchicalTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train with TRUE MULTIPROCESSING - runs episodes in parallel across CPU cores."""
    torch.manual_seed(spec.seed)

    device = _select_device(spec.device)
    print(f"Using device: {device}")
    print(f"Parallel workers: {spec.num_workers}")

    encoder, model = _initialize_model(spec, device)
    _make_model_contiguous(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)

    wins = 0
    poorest_eliminations = 0
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    artifact_logger = _training_artifact_logger(spec)

    # Prepare model config for workers
    model_config_dict = model.config.to_dict()
    encoder_config = encoder.config()

    try:
        total_batches = spec.episodes // spec.batch_size
        episode_count = 0

        # Create process pool
        with mp.Pool(processes=spec.num_workers) as pool:
            for _batch_idx in tqdm(
                range(total_batches),
                total=total_batches,
                desc="Training (parallel)",
                unit="batch",
                bar_format=_PROGRESS_BAR_FORMAT,
                dynamic_ncols=True,
            ):
                # Distribute episodes across workers
                episodes_per_worker = spec.batch_size // spec.num_workers
                remainder = spec.batch_size % spec.num_workers

                worker_args = []
                seed_offset = 0
                for worker_idx in range(spec.num_workers):
                    num_eps = episodes_per_worker + (1 if worker_idx < remainder else 0)
                    if num_eps == 0:
                        continue
                    seeds = [
                        spec.seed + episode_count + seed_offset + i + 1
                        for i in range(num_eps)
                    ]
                    seed_offset += num_eps
                    worker_args.append((
                        seeds,
                        model.state_dict(),  # Send current weights
                        model_config_dict,
                        encoder_config,
                        spec,
                    ))

                # Run episodes in parallel
                all_results: list[list[EpisodeResultSerializable]] = pool.map(
                    _worker_run_episodes, worker_args
                )

                # Flatten results
                batch_results: list[EpisodeResultSerializable] = []
                for worker_results in all_results:
                    batch_results.extend(worker_results)

                episode_count += len(batch_results)

                # Recompute log_probs with gradients using current model
                all_log_probs: list[torch.Tensor] = []
                all_advantages: list[torch.Tensor] = []
                all_returns: list[torch.Tensor] = []
                all_values: list[torch.Tensor] = []
                all_type_logits: list[torch.Tensor] = []

                for ep in batch_results:
                    if ep.won:
                        wins += 1
                    if ep.eliminated:
                        poorest_eliminations += 1

                    if not ep.observations:
                        continue

                    # Recompute log_probs by replaying through model
                    for i, obs_encoded in enumerate(ep.observations):
                        features = torch.tensor(obs_encoded, dtype=torch.float32, device=device)
                        type_logits, card_probs, discard_logits, value = model(features)
                        type_logits = type_logits.squeeze(0)
                        card_probs = card_probs.squeeze(0)
                        discard_logits = discard_logits.squeeze(0)

                        action_type = ep.action_types[i]

                        # Compute log_prob for the action that was taken
                        type_log_prob = torch.log_softmax(type_logits, dim=0)[action_type]

                        if action_type == 0:  # PASS
                            log_prob = type_log_prob
                        elif action_type == 1:  # BID
                            # Compute log_prob of the card selection
                            card_indices = ep.selected_cards[i]
                            selected = torch.zeros(len(MONEY_CARD_VALUES), device=device)
                            for idx in card_indices:
                                selected[idx] = 1.0
                            # Binary log prob
                            mask = torch.ones(len(MONEY_CARD_VALUES), dtype=torch.bool, device=device)
                            card_log_prob = _compute_binary_log_prob(card_probs, selected, mask)
                            log_prob = type_log_prob + card_log_prob
                        else:  # DISCARD
                            discard_idx = ep.discard_indices[i]
                            discard_log_prob = torch.log_softmax(discard_logits, dim=0)[discard_idx]
                            log_prob = type_log_prob + discard_log_prob

                        all_log_probs.append(log_prob)
                        all_type_logits.append(type_logits)

                    # Compute GAE
                    values_t = [torch.tensor(v, device=device) for v in ep.values]
                    returns, advantages = _compute_gae_vectorized(
                        ep.rewards,
                        values_t,
                        spec.discount,
                        spec.gae_lambda,
                        device,
                    )
                    all_advantages.append(advantages)
                    all_returns.append(returns)
                    all_values.extend(values_t)

                if not all_log_probs:
                    continue

                # Stack tensors
                log_probs_t = torch.stack(all_log_probs)
                advantages_t = torch.cat(all_advantages)
                returns_t = torch.cat(all_returns)
                values_t = torch.stack(all_values).squeeze(-1)
                type_logits_t = torch.stack(all_type_logits)

                # Compute loss
                loss, policy_loss, value_loss = _compute_loss(
                    log_probs_t,
                    advantages_t,
                    returns_t,
                    values_t,
                    type_logits_t,
                    spec.entropy_coef,
                    spec.value_coef,
                )

                # Gradient update
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec.max_grad_norm)
                _set_optimizer_lr(optimizer, _learning_rate_for_episode(spec, episode_count))
                optimizer.step()

                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))

                # Checkpoint
                if spec.checkpoint_path and _should_checkpoint(episode_count, spec):
                    _write_checkpoint(spec, model, encoder, wins, episode_count)

        # Final checkpoint
        if spec.checkpoint_path:
            _write_checkpoint(spec, model, encoder, wins, spec.episodes)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_policy = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        avg_value = sum(value_losses) / len(value_losses) if value_losses else 0.0

        metrics = TrainMetrics(
            episodes=episode_count,
            wins=wins,
            win_rate=wins / episode_count if episode_count else 0.0,
            poorest_eliminations=poorest_eliminations,
            poorest_rate=poorest_eliminations / episode_count if episode_count else 0.0,
            average_loss=avg_loss,
            average_policy_loss=avg_policy,
            average_value_loss=avg_value,
        )

        if artifact_logger is not None:
            artifact_logger.finalize(total_games=episode_count, training_metrics=metrics.to_dict())

        return metrics

    finally:
        if artifact_logger is not None:
            artifact_logger.close()


def _train_hierarchical_single(
    spec: HierarchicalTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train with single-threaded execution (original optimized version)."""
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)

    # Select device with auto-detection
    device = _select_device(spec.device)
    print(f"Using device: {device}")

    encoder, model = _initialize_model(spec, device)
    # Ensure model parameters are contiguous for MPS efficiency
    _make_model_contiguous(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    registry = registry if registry is not None else _default_registry(spec)

    wins = 0
    poorest_eliminations = 0
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []

    artifact_logger = _training_artifact_logger(spec)
    pending_artifacts: list[tuple[int, int, GameResult, tuple]] = []

    # Create single reusable worker context (threading doesn't help due to GIL)
    ctx = WorkerContext.create(spec.player_count, spec, registry, spec.seed)

    try:
        total_batches = spec.episodes // spec.batch_size
        episode_count = 0

        for _batch_idx in tqdm(
            range(total_batches),
            total=total_batches,
            desc="Training",
            unit="batch",
            bar_format=_PROGRESS_BAR_FORMAT,
            dynamic_ncols=True,
        ):
            # Collect batch of episodes (single-threaded, reuses context)
            batch_episodes: list[EpisodeData] = []
            batch_results: list[EpisodeResult] = []

            for i in range(spec.batch_size):
                episode_seed = spec.seed + episode_count + i + 1
                result = _run_single_episode(
                    model=model,
                    encoder=encoder,
                    ctx=ctx,
                    episode_seed=episode_seed,
                    spec=spec,
                    device=device,
                )
                batch_results.append(result)

            episode_count += len(batch_results)

            # Process results
            for result in batch_results:
                batch_episodes.append(result.episode_data)
                if result.episode_data.won:
                    wins += 1
                if result.episode_data.eliminated:
                    poorest_eliminations += 1

                # Queue artifact for batched logging
                if artifact_logger is not None and result.game_result is not None:
                    pending_artifacts.append((
                        episode_count,
                        result.episode_seed,
                        result.game_result,
                        (),  # Players tuple - omit for performance
                    ))

            # Batched artifact logging
            if artifact_logger is not None and len(pending_artifacts) >= spec.artifact_log_interval:
                _flush_artifacts(artifact_logger, pending_artifacts)
                pending_artifacts.clear()

            # Compute GAE and update model (vectorized)
            all_log_probs: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_values: list[torch.Tensor] = []
            all_type_logits: list[torch.Tensor] = []

            for ep in batch_episodes:
                if not ep.log_probs:
                    continue

                # Vectorized GAE computation
                returns, advantages = _compute_gae_vectorized(
                    ep.rewards,
                    ep.values,
                    spec.discount,
                    spec.gae_lambda,
                    device,
                )

                all_log_probs.extend(ep.log_probs)
                all_advantages.append(advantages)
                all_returns.append(returns)
                all_values.extend(ep.values)
                all_type_logits.extend(ep.type_logits)

            if not all_log_probs:
                continue

            # Stack tensors efficiently
            log_probs_t = torch.stack(all_log_probs)
            advantages_t = torch.cat(all_advantages)
            returns_t = torch.cat(all_returns)
            values_t = torch.stack(all_values).squeeze(-1)
            type_logits_t = torch.stack(all_type_logits)

            # Ensure tensors are contiguous for MPS
            if not log_probs_t.is_contiguous():
                log_probs_t = log_probs_t.contiguous()

            # Compute loss
            loss, policy_loss, value_loss = _compute_loss(
                log_probs_t,
                advantages_t,
                returns_t,
                values_t,
                type_logits_t,
                spec.entropy_coef,
                spec.value_coef,
            )

            # Gradient update
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec.max_grad_norm)
            _set_optimizer_lr(optimizer, _learning_rate_for_episode(spec, episode_count))
            optimizer.step()

            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))

            # Checkpoint
            if spec.checkpoint_path and _should_checkpoint(episode_count, spec):
                _write_checkpoint(spec, model, encoder, wins, episode_count)

        # Final checkpoint
        if spec.checkpoint_path:
            _write_checkpoint(spec, model, encoder, wins, spec.episodes)

        # Flush remaining artifacts
        if artifact_logger is not None and pending_artifacts:
            _flush_artifacts(artifact_logger, pending_artifacts)

        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0.0

        metrics = TrainMetrics(
            episodes=episode_count,
            wins=wins,
            win_rate=wins / episode_count if episode_count else 0.0,
            poorest_eliminations=poorest_eliminations,
            poorest_rate=poorest_eliminations / episode_count if episode_count else 0.0,
            average_loss=avg_loss,
            average_policy_loss=avg_policy_loss,
            average_value_loss=avg_value_loss,
        )

        if artifact_logger is not None:
            artifact_logger.finalize(total_games=episode_count, training_metrics=metrics.to_dict())

        return metrics

    finally:
        # Clean up worker context
        ctx.env.cleanup()
        if artifact_logger is not None:
            artifact_logger.close()


def _collect_episodes_parallel(
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    worker_contexts: list[WorkerContext],
    episode_seeds: list[int],
    spec: HierarchicalTrainSpec,
    device: torch.device,
) -> list[EpisodeResult]:
    """Collect multiple episodes in parallel using ThreadPoolExecutor."""
    results: list[EpisodeResult] = []
    num_workers = len(worker_contexts)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, seed in enumerate(episode_seeds):
            ctx = worker_contexts[i % num_workers]
            future = executor.submit(
                _run_single_episode,
                model=model,
                encoder=encoder,
                ctx=ctx,
                episode_seed=seed,
                spec=spec,
                device=device,
            )
            futures.append(future)

        for future in futures:
            results.append(future.result())

    return results


def _run_single_episode(
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    ctx: WorkerContext,
    episode_seed: int,
    spec: HierarchicalTrainSpec,
    device: torch.device,
) -> EpisodeResult:
    """Run a single episode using a reusable worker context."""
    # Reset environment (reuses existing server, cleans up old game)
    ctx.env.reset(seed=episode_seed)
    _reset_opponents(ctx.opponents, episode_seed, spec.player_count)

    episode_data = _run_episode(
        model=model,
        encoder=encoder,
        env=ctx.env,
        opponents=ctx.opponents,
        player_count=spec.player_count,
        temperature=spec.temperature,
        device=device,
        rng=ctx.rng,
    )

    # Get game result for artifact logging
    game_result = ctx.env.game_result()

    return EpisodeResult(
        episode_data=episode_data,
        episode_seed=episode_seed,
        game_result=game_result,
    )


def _flush_artifacts(
    logger: TrainingArtifactLogger,
    artifacts: list[tuple[int, int, GameResult, tuple]],
) -> None:
    """Flush queued artifacts to the logger."""
    for game_index, seed, result, players in artifacts:
        logger.record_game(
            game_index=game_index,
            seed=seed,
            result=result,
            players=players,
        )


def _make_model_contiguous(model: nn.Module) -> None:
    """Ensure all model parameters are contiguous for MPS efficiency."""
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()


def _run_episode(
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    env: EnvAdapter,
    opponents: dict[int, Player],
    player_count: int,
    temperature: float,
    device: torch.device,
    rng: random.Random,
) -> EpisodeData:
    """Run a single episode and collect trajectory data.

    Optimized: For the trainee, constructs actions directly from card selection
    without enumerating all 2^n bid combinations.
    """
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    type_logits_list: list[torch.Tensor] = []

    done = False
    while not done:
        # Get next player - we still need this check
        player_id, legal_actions = _next_turn(env, player_count)
        if player_id is None:
            done = True
            break

        observation = env.observe(player_id)

        if player_id == 0:  # Trainee
            # Use fast action selection that constructs actions directly
            is_discard = legal_actions and legal_actions[0].kind == ActionKind.DISCARD_POSSESSION
            action, log_prob, value, type_logits = _select_hierarchical_action_fast(
                model=model,
                encoder=encoder,
                env=env,
                observation=observation,
                is_discard=is_discard,
                temperature=temperature,
                device=device,
                rng=rng,
            )
            log_probs.append(log_prob)
            values.append(value)
            type_logits_list.append(type_logits)
        else:  # Opponent
            opponent = opponents.get(player_id)
            if opponent is not None:
                action = opponent.act(observation, legal_actions)
            else:
                action = rng.choice(legal_actions)

        _, _, done, _ = env.step(player_id, action)

    # Compute final reward
    state = env.get_state()
    if not state.game_over:
        raise InvalidState("Training episode ended before terminal game state")

    result = env.game_result()
    _notify_opponents(opponents, result)

    won = 0 in result.winners
    eliminated = 0 in result.poorest

    # Reward shaping: +1 for win, -1 for loss, with intermediate based on ranking
    if eliminated:
        reward = -1.0
    elif won:
        reward = 1.0
    else:
        # Rank-based reward for non-winners, non-eliminated
        scores = result.scores
        trainee_score = scores[0]
        rank = sum(1 for s in scores if s > trainee_score) + 1
        reward = 1.0 - 2.0 * (rank - 1) / (len(scores) - 1) if len(scores) > 1 else 0.0

    # Create reward list (reward only at final step)
    rewards = [0.0] * len(log_probs)
    if rewards:
        rewards[-1] = reward

    return EpisodeData(
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        type_logits=type_logits_list,
        won=won,
        eliminated=eliminated,
    )


def _select_hierarchical_action_fast(
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    env: EnvAdapter,
    observation: Observation,
    is_discard: bool,
    temperature: float,
    device: torch.device,
    rng: random.Random,
) -> tuple[Action, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select action using hierarchical sampling WITHOUT enumerating all legal actions.

    This is O(n) where n = number of cards, instead of O(2^n) for all bid combinations.
    We only call legal_actions() when absolutely necessary (for bid matching).
    """
    # Encode observation
    features = torch.tensor(encoder.encode(observation), dtype=torch.float32, device=device)

    # Forward pass
    type_logits, card_probs, discard_logits, value = model(features)
    type_logits = type_logits.squeeze(0)
    card_probs = card_probs.squeeze(0)
    discard_logits = discard_logits.squeeze(0)

    # Handle discard case directly (no need for type selection)
    if is_discard:
        # Build discard mask from observation
        discard_mask = torch.zeros(10, dtype=torch.bool, device=device)
        for poss_value in observation.self_view.possessions:
            if 1 <= poss_value <= 10:
                discard_mask[poss_value - 1] = True

        masked_discard_logits = discard_logits.masked_fill(~discard_mask, float("-inf"))

        if temperature <= 0:
            possession_idx = int(torch.argmax(masked_discard_logits).item())
            discard_log_prob = torch.tensor(0.0, device=device)
        else:
            discard_dist = torch.distributions.Categorical(logits=masked_discard_logits / temperature)
            possession_sample = discard_dist.sample()
            possession_idx = int(possession_sample.item())
            discard_log_prob = discard_dist.log_prob(possession_sample)

        action = Action(ActionKind.DISCARD_POSSESSION, possession_value=possession_idx + 1)
        return action, discard_log_prob, value.squeeze(-1), type_logits

    # Normal bidding - build type and card masks from observation
    type_mask = torch.tensor([True, bool(observation.self_view.hand), False], dtype=torch.bool, device=device)
    card_mask = torch.zeros(len(MONEY_CARD_VALUES), dtype=torch.bool, device=device)
    for card_value in observation.self_view.hand:
        try:
            idx = MONEY_CARD_VALUES.index(card_value)
            card_mask[idx] = True
        except ValueError:
            pass

    # Sample action type
    masked_type_logits = type_logits.masked_fill(~type_mask, float("-inf"))

    if temperature <= 0:
        action_type_idx = int(torch.argmax(masked_type_logits).item())
        type_log_prob = torch.tensor(0.0, device=device)
    else:
        type_dist = torch.distributions.Categorical(logits=masked_type_logits / temperature)
        action_type_sample = type_dist.sample()
        action_type_idx = int(action_type_sample.item())
        type_log_prob = type_dist.log_prob(action_type_sample)

    # PASS - construct directly
    if action_type_idx == 0:
        action = Action(ActionKind.PASS)
        return action, type_log_prob, value.squeeze(-1), type_logits

    # BID - sample cards and construct action
    masked_probs = card_probs * card_mask.float()
    selected = torch.bernoulli(masked_probs)

    # Ensure at least one card selected
    if selected.sum() == 0:
        valid_probs = masked_probs.clone()
        valid_probs[~card_mask] = 0
        if valid_probs.sum() > 0:
            selected[valid_probs.argmax()] = 1
        else:
            for i in range(len(card_mask)):
                if card_mask[i]:
                    selected[i] = 1
                    break

    # Compute log probability
    log_prob = _compute_binary_log_prob(card_probs, selected, card_mask)

    # Convert selected indices to card values
    selected_cards = tuple(
        MONEY_CARD_VALUES[i] for i in range(len(MONEY_CARD_VALUES)) if selected[i] == 1
    )

    # Construct bid action directly
    action = Action(ActionKind.BID, cards=selected_cards)
    total_log_prob = type_log_prob + log_prob

    return action, total_log_prob, value.squeeze(-1), type_logits


def _select_hierarchical_action(
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    observation: Observation,
    legal_actions: list[Action],
    temperature: float,
    device: torch.device,
    rng: random.Random,
) -> tuple[Action, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select an action using hierarchical sampling with masking (legacy)."""
    # Encode observation
    features = torch.tensor(encoder.encode(observation), dtype=torch.float32, device=device)

    # Forward pass
    type_logits, card_probs, discard_logits, value = model(features)
    type_logits = type_logits.squeeze(0)
    card_probs = card_probs.squeeze(0)
    discard_logits = discard_logits.squeeze(0)

    # Build masks - use fast O(n) method instead of iterating through O(2^n) legal actions
    type_mask, card_mask, discard_mask = _build_masks_fast(observation, device)

    # Sample action type - use masked_fill instead of clone (more efficient)
    masked_type_logits = type_logits.masked_fill(~type_mask, float("-inf"))

    if temperature <= 0:
        action_type_idx = int(torch.argmax(masked_type_logits).item())
        type_log_prob = torch.tensor(0.0, device=device)
    else:
        type_dist = torch.distributions.Categorical(logits=masked_type_logits / temperature)
        action_type_sample = type_dist.sample()
        action_type_idx = int(action_type_sample.item())
        type_log_prob = type_dist.log_prob(action_type_sample)

    # Based on action type, sample parameters
    if action_type_idx == 0:  # PASS
        action = _find_pass_action(legal_actions)
        total_log_prob = type_log_prob

    elif action_type_idx == 1:  # BID
        action, card_log_prob = _sample_bid_action(
            card_probs, card_mask, legal_actions, rng, device
        )
        total_log_prob = type_log_prob + card_log_prob

    else:  # DISCARD (action_type_idx == 2)
        # Use masked_fill instead of clone (more efficient)
        masked_discard_logits = discard_logits.masked_fill(~discard_mask, float("-inf"))

        if temperature <= 0:
            possession_idx = int(torch.argmax(masked_discard_logits).item())
            discard_log_prob = torch.tensor(0.0, device=device)
        else:
            discard_dist = torch.distributions.Categorical(
                logits=masked_discard_logits / temperature
            )
            possession_sample = discard_dist.sample()
            possession_idx = int(possession_sample.item())
            discard_log_prob = discard_dist.log_prob(possession_sample)

        action = _find_discard_action(legal_actions, possession_idx + 1)
        total_log_prob = type_log_prob + discard_log_prob

    return action, total_log_prob, value.squeeze(-1), type_logits


def _build_masks_fast(
    observation: Observation, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build action masks directly from observation (O(1) instead of O(2^n)).

    This avoids iterating through all legal bid combinations by inferring
    available actions from the observation's self_view.
    """
    type_mask = torch.zeros(3, dtype=torch.bool, device=device)
    card_mask = torch.zeros(len(MONEY_CARD_VALUES), dtype=torch.bool, device=device)
    discard_mask = torch.zeros(10, dtype=torch.bool, device=device)

    # Check if it's a discard situation (possession discard after misfortune)
    if observation.status_card is not None and observation.status_card.kind == StatusKind.MISFORTUNE:
        if observation.status_card.name == "Theft" and observation.self_view.possessions:
            type_mask[2] = True
            for poss_value in observation.self_view.possessions:
                if 1 <= poss_value <= 10:
                    discard_mask[poss_value - 1] = True
            return type_mask, card_mask, discard_mask

    # Normal bidding round
    # PASS is always available during bidding
    type_mask[0] = True

    # BID is available if player has cards in hand
    if observation.self_view.hand:
        type_mask[1] = True
        for card_value in observation.self_view.hand:
            try:
                idx = MONEY_CARD_VALUES.index(card_value)
                card_mask[idx] = True
            except ValueError:
                pass

    return type_mask, card_mask, discard_mask


def _build_masks(
    legal_actions: list[Action], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build action masks from legal actions (legacy, slower method)."""
    type_mask = torch.zeros(3, dtype=torch.bool, device=device)
    card_mask = torch.zeros(len(MONEY_CARD_VALUES), dtype=torch.bool, device=device)
    discard_mask = torch.zeros(10, dtype=torch.bool, device=device)

    for action in legal_actions:
        if action.kind == ActionKind.PASS:
            type_mask[0] = True
        elif action.kind == ActionKind.BID:
            type_mask[1] = True
            for card_value in action.cards:
                try:
                    idx = MONEY_CARD_VALUES.index(card_value)
                    card_mask[idx] = True
                except ValueError:
                    pass
        elif action.kind == ActionKind.DISCARD_POSSESSION:
            type_mask[2] = True
            if action.possession_value is not None:
                idx = action.possession_value - 1
                if 0 <= idx < 10:
                    discard_mask[idx] = True

    return type_mask, card_mask, discard_mask


def _sample_bid_action(
    card_probs: torch.Tensor,
    card_mask: torch.Tensor,
    legal_actions: list[Action],
    rng: random.Random,
    device: torch.device,
) -> tuple[Action, torch.Tensor]:
    """Sample a bid action from card probabilities."""
    # Apply mask to probabilities
    masked_probs = card_probs * card_mask.float()

    # Sample binary decisions for each card
    selected = torch.bernoulli(masked_probs)

    # Ensure at least one card is selected
    if selected.sum() == 0:
        # Select the card with highest probability among legal ones
        valid_probs = masked_probs.clone()
        valid_probs[~card_mask] = 0
        if valid_probs.sum() > 0:
            selected[valid_probs.argmax()] = 1
        else:
            # Fallback: select first available
            for i in range(len(card_mask)):
                if card_mask[i]:
                    selected[i] = 1
                    break

    # Compute log probability of selection
    log_prob = _compute_binary_log_prob(card_probs, selected, card_mask)

    # Convert selected cards to values
    selected_values = set()
    for i in range(len(MONEY_CARD_VALUES)):
        if selected[i] == 1:
            selected_values.add(MONEY_CARD_VALUES[i])

    # Find matching bid action
    action = _find_best_bid_action(selected_values, legal_actions)

    return action, log_prob


def _compute_binary_log_prob(
    probs: torch.Tensor, selected: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute log probability of binary selections (vectorized)."""
    eps = 1e-8

    # Clamp probabilities
    p = probs.clamp(eps, 1.0 - eps)

    # Compute log probs for selection and non-selection
    log_p_select = torch.log(p)
    log_p_not_select = torch.log(1.0 - p)

    # Use torch.where to select appropriate log prob based on selection
    log_probs = torch.where(selected == 1, log_p_select, log_p_not_select)

    # Mask out invalid positions and sum
    log_prob = (log_probs * mask.float()).sum()

    return log_prob


def _find_pass_action(legal_actions: list[Action]) -> Action:
    """Find the PASS action in legal actions."""
    for action in legal_actions:
        if action.kind == ActionKind.PASS:
            return action
    # Fallback (shouldn't happen if masks are correct)
    return legal_actions[0]


def _find_discard_action(legal_actions: list[Action], possession_value: int) -> Action:
    """Find a discard action with the given possession value."""
    for action in legal_actions:
        if action.kind == ActionKind.DISCARD_POSSESSION:
            if action.possession_value == possession_value:
                return action
    # Fallback to first discard action
    for action in legal_actions:
        if action.kind == ActionKind.DISCARD_POSSESSION:
            return action
    return legal_actions[0]


def _find_best_bid_action(
    selected_values: set[int], legal_actions: list[Action]
) -> Action:
    """Find the legal bid action that best matches selected card values."""
    bid_actions = [a for a in legal_actions if a.kind == ActionKind.BID]

    if not bid_actions:
        return legal_actions[0]

    # Try exact match first
    for action in bid_actions:
        action_values = set(action.cards)
        if action_values == selected_values:
            return action

    # Find best match by Jaccard similarity
    best_action = bid_actions[0]
    best_score = -1.0

    for action in bid_actions:
        action_values = set(action.cards)
        intersection = len(selected_values & action_values)
        union = len(selected_values | action_values)
        score = intersection / union if union > 0 else 0.0

        # Tie-break: prefer smaller total value (more conservative)
        if score > best_score or (
            score == best_score
            and sum(action.cards) < sum(best_action.cards)
        ):
            best_score = score
            best_action = action

    return best_action


def _compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (legacy, scalar version)."""
    T = len(rewards)
    if T == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    advantages = torch.zeros(T, device=device)
    returns = torch.zeros(T, device=device)

    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    # Normalize advantages
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def _compute_gae_vectorized(
    rewards: list[float],
    values: list[torch.Tensor],
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with vectorized tensor operations (faster on GPU)."""
    T = len(rewards)
    if T == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    # Convert to tensors on device
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t = torch.stack([v.squeeze() for v in values]).to(device)

    # Append zero for terminal value
    values_ext = torch.cat([values_t, torch.zeros(1, device=device)])

    # Compute deltas vectorized
    deltas = rewards_t + gamma * values_ext[1:] - values_ext[:-1]

    # GAE computation (still needs loop due to recurrence, but with tensors)
    advantages = torch.zeros(T, device=device)
    gae = torch.tensor(0.0, device=device)

    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values_t

    # Normalize advantages
    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def _compute_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    type_logits: torch.Tensor,
    entropy_coef: float,
    value_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined policy gradient loss."""
    # Policy loss (negative because we want to maximize)
    policy_loss = -(log_probs * advantages.detach()).mean()

    # Value loss
    value_loss = F.mse_loss(values, returns.detach())

    # Entropy bonus (from type logits only for simplicity)
    type_probs = F.softmax(type_logits, dim=-1)
    entropy = -(type_probs * (type_probs + 1e-8).log()).sum(dim=-1).mean()

    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    return loss, policy_loss, value_loss


def _next_turn(env: EnvAdapter, player_count: int) -> tuple[int | None, list[Action]]:
    """Return the next player id with legal actions."""
    for player_id in range(player_count):
        legal_actions = env.legal_actions(player_id)
        if legal_actions:
            return player_id, legal_actions
    if env.get_state().game_over:
        return None, []
    raise InvalidState("No legal actions available")


def _next_turn_fast(env: EnvAdapter, player_count: int) -> tuple[int | None, bool]:
    """Fast turn check - returns (player_id, is_discard) without enumerating legal actions.

    Falls back to regular legal_actions check since the game state logic is complex.
    """
    # The game state has complex round-robin logic that's hard to replicate.
    # Just check if there are any legal actions efficiently.
    state = env.get_state()
    if state.game_over:
        return None, False

    # Check for pending discard first (this is fast)
    if state.pending_discard is not None:
        return state.pending_discard.player_id, True

    # For normal bidding, we need to find who has legal actions
    # This still calls legal_actions but only for the check, not for mask building
    for player_id in range(player_count):
        legal = env.legal_actions(player_id)
        if legal:
            # Check if it's a discard action
            is_discard = legal[0].kind == ActionKind.DISCARD_POSSESSION
            return player_id, is_discard

    return None, False


def _get_minimal_legal_actions(env: EnvAdapter, player_id: int, is_discard: bool) -> list[Action]:
    """Get minimal legal actions needed for opponent bots (not hierarchical model).

    For discards, we need the actual options. For bids, opponents still need full enumeration
    but this is called less frequently than for the learning agent.
    """
    return env.legal_actions(player_id)


def _select_device(device_str: str) -> torch.device:
    """Select the appropriate device, with auto-detection support.

    Args:
        device_str: One of "auto", "mps", "cuda", or "cpu".
            - "auto": Use CPU (fastest for small models with single-sample inference)
            - "mps": Use Apple Silicon GPU (slower for small models due to kernel overhead)
            - "cuda": Use NVIDIA GPU
            - "cpu": Use CPU only

    Returns:
        The selected torch device.
    """
    if device_str == "auto":
        # For hierarchical training with small models and single-sample inference,
        # CPU is actually faster than MPS due to GPU kernel launch overhead.
        # MPS/CUDA only helps with large batch sizes or large models.
        return torch.device("cpu")

    if device_str == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        print("Warning: MPS not available, falling back to CPU")
        return torch.device("cpu")

    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device(device_str)


def _initialize_model(
    spec: HierarchicalTrainSpec, device: torch.device
) -> tuple[BasicEncoder, HierarchicalPolicyValue]:
    """Build a new model or restore one from a checkpoint."""
    if spec.resume:
        bundle = load_checkpoint(spec.resume)
        encoder = BasicEncoder.from_config(bundle.encoder_config)
        config = HierarchicalConfig.from_dict(bundle.model_config)
        if config.input_dim != encoder.feature_size:
            raise InvalidState("Encoder feature size does not match model config")
        model = HierarchicalPolicyValue(config).to(device)
        model.load_state_dict(bundle.model_state, strict=True)
        return encoder, model

    encoder = BasicEncoder()
    config = HierarchicalConfig(
        input_dim=encoder.feature_size,
        trunk_sizes=spec.trunk_sizes,
        activation=spec.activation,
        dropout=spec.dropout,
    )
    model = HierarchicalPolicyValue(config).to(device)
    return encoder, model


def _default_registry(spec: HierarchicalTrainSpec) -> PlayerRegistry | None:
    """Create a player registry only when opponent specs are configured."""
    if not spec.opponents:
        return None
    return build_default_registry()


def _training_artifact_logger(spec: HierarchicalTrainSpec) -> TrainingArtifactLogger | None:
    """Create a training artifact logger when enabled."""
    if not spec.artifacts_path:
        return None
    return TrainingArtifactLogger(
        spec.artifacts_path,
        bot_type="hierarchical",
        learner_seats=(0,),
        player_count=spec.player_count,
        spec=spec.to_mapping(),
    )


def _sample_opponents(
    spec: HierarchicalTrainSpec,
    rng: random.Random,
    registry: PlayerRegistry | None,
) -> dict[int, Player]:
    """Sample opponents for all non-learning seats."""
    if not spec.opponents:
        return {}
    if registry is None:
        raise InvalidState("Opponent specs require a player registry")
    weights = spec.opponent_weights or None
    specs = list(spec.opponents)
    opponents: dict[int, Player] = {}
    for seat in range(1, spec.player_count):
        opponent_spec = rng.choices(specs, weights=weights, k=1)[0]
        opponents[seat] = registry.create(opponent_spec.to_mapping())
    return opponents


def _reset_opponents(opponents: dict[int, Player], seed: int, player_count: int) -> None:
    """Reset opponent bots for a fresh game."""
    game_config: dict[str, object] = {"seed": seed, "player_count": player_count}
    for seat, opponent in opponents.items():
        opponent.reset(game_config, player_id=seat, seat=seat)


def _notify_opponents(opponents: dict[int, Player], result: GameResult) -> None:
    """Notify opponents of game completion."""
    for opponent in opponents.values():
        opponent.on_game_end(result)


def _should_checkpoint(episode: int, spec: HierarchicalTrainSpec) -> bool:
    """Return True if a checkpoint should be saved at this episode."""
    return episode % spec.checkpoint_every == 0 or episode == spec.episodes


def _write_checkpoint(
    spec: HierarchicalTrainSpec,
    model: HierarchicalPolicyValue,
    encoder: BasicEncoder,
    wins: int,
    episode: int,
) -> None:
    """Write a checkpoint snapshot to disk."""
    if spec.checkpoint_path is None:
        return
    metrics = {
        "episode": episode,
        "wins": wins,
        "win_rate": wins / episode if episode else 0.0,
        "learning_rate": _learning_rate_for_episode(spec, episode),
    }
    save_checkpoint(
        spec.checkpoint_path,
        model_state=model.state_dict(),
        model_config=model.config.to_dict(),
        encoder_config=encoder.config(),
        metrics=metrics,
        metadata={
            "type": "hierarchical",
            "episodes": spec.episodes,
            "seed": spec.seed,
            "resume": spec.resume,
            "artifacts_path": spec.artifacts_path,
            "opponents": [opponent.to_mapping() for opponent in spec.opponents],
            "opponent_weights": list(spec.opponent_weights),
        },
    )


def _learning_rate_for_episode(spec: HierarchicalTrainSpec, episode: int) -> float:
    """Return the scheduled learning rate for the given 1-based episode."""
    if spec.learning_rate_final is None:
        return spec.learning_rate
    if spec.episodes <= 1:
        return spec.learning_rate_final
    clamped_episode = min(max(1, int(episode)), int(spec.episodes))
    progress = (clamped_episode - 1) / (spec.episodes - 1)
    return spec.learning_rate + (spec.learning_rate_final - spec.learning_rate) * progress


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Update optimizer learning rate in-place."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _coerce_optional_str(value: object) -> str | None:
    """Return a stripped string or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_optional_float(value: object) -> float | None:
    """Return a float or None for missing/blank values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_train_spec(path: Path) -> HierarchicalTrainSpec:
    """Load a training spec from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Training spec must be a JSON object")
    return HierarchicalTrainSpec.from_mapping(data)


def main() -> None:
    """Run the training CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a hierarchical policy/value model.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to training spec.")
    args = parser.parse_args()
    spec = load_train_spec(args.spec)

    print("=" * 60)
    print("Hierarchical Model Training (Optimized)")
    print("=" * 60)
    print(f"  Episodes:       {spec.episodes:,}")
    print(f"  Batch size:     {spec.batch_size}")
    print(f"  Workers:        {spec.num_workers}")
    print(f"  Device:         {spec.device}")

    # Show detected device
    device = _select_device(spec.device)
    if spec.device == "auto":
        print(f"  Detected:       {device}")

    if spec.checkpoint_path:
        print(f"  Checkpoint:     {spec.checkpoint_path}")

    print()
    print("Optimizations enabled:")
    print("  - Reusable environment (no object recreation per game)")
    print("  - Memory-efficient game cleanup (no memory leaks)")
    print("  - Batched artifact logging")
    print("  - Vectorized tensor operations")
    print("  - Cached status deck (no allocation in hot path)")
    print()

    metrics = train_hierarchical(spec)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
