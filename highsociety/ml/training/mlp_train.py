"""Training loop for the MLP policy/value model.

Optimized with:
- TRUE MULTIPROCESSING for parallel episode collection (bypasses GIL)
- Reusable worker contexts (no object recreation per game)
- Batched gradient updates
- Memory-efficient game cleanup
- Batched artifact logging
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
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
    raise ImportError("PyTorch is required for highsociety.ml.training.mlp_train") from exc

try:  # pragma: no cover - tqdm is optional in minimal envs
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: range, *args: object, **kwargs: object) -> range:
        """Fallback progress wrapper when tqdm is unavailable."""
        del args, kwargs
        return iterable

from highsociety.app.env_adapter import EnvAdapter
from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.domain.rules import GameResult
from highsociety.ml.checkpoints import load_checkpoint, save_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder
from highsociety.ml.models.mlp import MLPConfig, MLPPolicyValue
from highsociety.ml.training.artifacts import TrainingArtifactLogger
from highsociety.ops.spec import PlayerSpec
from highsociety.players.base import Player
from highsociety.players.registry import PlayerRegistry, build_default_registry

# Enable MPS fallback for unsupported operations
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_PROGRESS_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


@dataclass(frozen=True)
class MLPTrainSpec:
    """Configuration for training an MLP policy/value model."""

    episodes: int = 1000
    seed: int = 0
    player_count: int = 3
    learning_rate: float = 1e-3
    learning_rate_final: float | None = None
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    dropout: float = 0.0
    temperature: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    discount: float = 0.99
    batch_size: int = 32
    max_grad_norm: float = 1.0
    checkpoint_path: str | None = None
    checkpoint_every: int = 1000
    artifacts_path: str | None = None
    resume: str | None = None
    opponents: tuple[PlayerSpec, ...] = ()
    opponent_weights: tuple[float, ...] = ()
    device: str = "cpu"
    num_workers: int = 1
    artifact_log_interval: int = 100

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
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
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
            "hidden_sizes": list(self.hidden_sizes),
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
    def from_mapping(data: Mapping[str, Any]) -> "MLPTrainSpec":
        """Create a training spec from a mapping."""
        hidden_sizes = data.get("hidden_sizes", (128, 128))
        if isinstance(hidden_sizes, (list, tuple)):
            hidden_tuple = tuple(int(size) for size in hidden_sizes)
        else:
            hidden_tuple = (128, 128)
        opponents_data = data.get("opponents", []) or []
        if not isinstance(opponents_data, list):
            raise ValueError("opponents must be a list")
        opponents = tuple(PlayerSpec.from_mapping(item) for item in opponents_data)
        weights_data = data.get("opponent_weights", []) or []
        if weights_data and not isinstance(weights_data, list):
            raise ValueError("opponent_weights must be a list")
        weights = tuple(float(value) for value in weights_data)
        return MLPTrainSpec(
            episodes=int(data.get("episodes", 1000)),
            seed=int(data.get("seed", 0)),
            player_count=int(data.get("player_count", 3)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            learning_rate_final=_coerce_optional_float(data.get("learning_rate_final")),
            hidden_sizes=hidden_tuple,
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
            num_workers=int(data.get("num_workers", 1)),
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
    entropies: list[torch.Tensor]
    won: bool
    eliminated: bool


@dataclass
class EpisodeResultSerializable:
    """Serializable episode result for multiprocessing (no torch tensors).

    Contains trajectory data needed to recompute log_probs in main process.
    """

    observations: list[list[float]]  # Encoded observations
    action_indices: list[int]  # Which action index was selected
    action_masks: list[list[bool]]  # Legal action masks
    values: list[float]
    rewards: list[float]
    won: bool
    eliminated: bool
    episode_seed: int


@dataclass
class WorkerContext:
    """Reusable context for a worker thread."""

    env: EnvAdapter
    opponents: dict[int, Player]
    rng: random.Random

    @classmethod
    def create(
        cls, player_count: int, spec: MLPTrainSpec,
        registry: PlayerRegistry | None, seed: int
    ) -> "WorkerContext":
        """Create a fresh worker context."""
        rng = random.Random(seed)
        env = EnvAdapter(player_count=player_count)
        opponents = _sample_opponents(spec, rng, registry) if spec.opponents else {}
        return cls(env=env, opponents=opponents, rng=rng)


def _worker_run_episodes(
    args: tuple[list[int], dict, dict, dict, "MLPTrainSpec"],
) -> list[EpisodeResultSerializable]:
    """Worker function that runs episodes and collects trajectory data.

    Returns trajectory data (observations, actions) that the main process
    uses to recompute log_probs with proper gradients.
    """
    episode_seeds, model_state, model_config_dict, encoder_config, spec = args

    # Recreate model in this process
    encoder = BasicEncoder.from_config(encoder_config)
    config = MLPConfig.from_dict(model_config_dict)
    model = MLPPolicyValue(config)
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
        action_indices: list[int] = []
        action_masks: list[list[bool]] = []
        values: list[float] = []

        done = False
        while not done:
            player_id, legal_actions = _next_turn(env, spec.player_count)
            if player_id is None:
                done = True
                break

            observation = env.observe(player_id)

            if player_id == 0:  # Trainee
                obs_encoded = encoder.encode(observation)
                observations.append(obs_encoded)

                # Get action mask
                mask = encoder.action_mask(legal_actions)
                action_masks.append(mask)

                # Select action
                features = torch.tensor(obs_encoded, dtype=torch.float32, device=device)
                logits, value = model(features)
                logits = logits.squeeze(0)
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
                logits = logits.masked_fill(~mask_t, -1e9)

                values.append(float(value.squeeze(-1).item()))

                if spec.temperature <= 0:
                    action_idx = int(torch.argmax(logits).item())
                else:
                    dist = torch.distributions.Categorical(logits=logits / spec.temperature)
                    action_idx = int(dist.sample().item())

                action_indices.append(action_idx)
                action = encoder.action_space.action_at(action_idx)
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
            action_indices=action_indices,
            action_masks=action_masks,
            values=values,
            rewards=rewards,
            won=won,
            eliminated=eliminated,
            episode_seed=seed,
        ))

    env.cleanup()
    return results


def train_mlp(
    spec: MLPTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train an MLP policy/value model and return summary metrics.

    Uses TRUE MULTIPROCESSING for parallel episode collection when num_workers > 1.
    """
    if spec.num_workers > 1:
        return _train_mlp_parallel(spec, registry)
    return _train_mlp_single(spec, registry)


def _train_mlp_parallel(
    spec: MLPTrainSpec,
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

    model_config_dict = model.config.to_dict()
    encoder_config = encoder.config()

    try:
        total_batches = spec.episodes // spec.batch_size
        episode_count = 0

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
                        model.state_dict(),
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
                all_entropies: list[torch.Tensor] = []

                for ep in batch_results:
                    if ep.won:
                        wins += 1
                    if ep.eliminated:
                        poorest_eliminations += 1

                    if not ep.observations:
                        continue

                    # Recompute log_probs by replaying through model
                    ep_log_probs: list[torch.Tensor] = []
                    ep_values: list[torch.Tensor] = []
                    ep_entropies: list[torch.Tensor] = []

                    for i, obs_encoded in enumerate(ep.observations):
                        features = torch.tensor(obs_encoded, dtype=torch.float32, device=device)
                        logits, value = model(features)
                        logits = logits.squeeze(0)

                        mask = torch.tensor(ep.action_masks[i], dtype=torch.bool, device=device)
                        logits = logits.masked_fill(~mask, -1e9)

                        if spec.temperature <= 0:
                            log_prob = torch.tensor(0.0, device=device)
                            entropy = torch.tensor(0.0, device=device)
                        else:
                            dist = torch.distributions.Categorical(logits=logits / spec.temperature)
                            action_t = torch.tensor(ep.action_indices[i], device=device)
                            log_prob = dist.log_prob(action_t)
                            entropy = dist.entropy()

                        ep_log_probs.append(log_prob)
                        ep_values.append(value.squeeze(-1))
                        ep_entropies.append(entropy)

                    all_log_probs.extend(ep_log_probs)
                    all_entropies.extend(ep_entropies)

                    # Compute GAE
                    returns, advantages = _compute_gae(
                        ep.rewards,
                        ep_values,
                        spec.discount,
                        spec.gae_lambda,
                        device,
                    )
                    all_advantages.append(advantages)
                    all_returns.append(returns)
                    all_values.extend(ep_values)

                if not all_log_probs:
                    continue

                # Stack tensors
                log_probs_t = torch.stack(all_log_probs)
                advantages_t = torch.cat(all_advantages)
                returns_t = torch.cat(all_returns)
                values_t = torch.stack(all_values)
                entropy_t = torch.stack(all_entropies).mean()

                # Compute loss
                policy_loss = -(log_probs_t * advantages_t.detach()).mean()
                value_loss = F.mse_loss(values_t, returns_t.detach())
                loss = policy_loss + spec.value_coef * value_loss - spec.entropy_coef * entropy_t

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

        return _build_metrics(
            episode_count, wins, poorest_eliminations, losses, policy_losses, value_losses,
            artifact_logger,
        )

    finally:
        if artifact_logger is not None:
            artifact_logger.close()


def _train_mlp_single(
    spec: MLPTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train with single-threaded execution (optimized with batching and context reuse)."""
    random.seed(spec.seed)
    torch.manual_seed(spec.seed)

    device = _select_device(spec.device)
    print(f"Using device: {device}")

    encoder, model = _initialize_model(spec, device)
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

    # Create single reusable worker context
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
            # Collect batch of episodes
            batch_episodes: list[EpisodeData] = []

            for i in range(spec.batch_size):
                episode_seed = spec.seed + episode_count + i + 1
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

                batch_episodes.append(episode_data)

                if episode_data.won:
                    wins += 1
                if episode_data.eliminated:
                    poorest_eliminations += 1

                # Queue artifact for batched logging
                if artifact_logger is not None:
                    result = ctx.env.game_result()
                    pending_artifacts.append((
                        episode_count + i + 1,
                        episode_seed,
                        result,
                        (),
                    ))

            episode_count += len(batch_episodes)

            # Batched artifact logging
            if artifact_logger is not None and len(pending_artifacts) >= spec.artifact_log_interval:
                _flush_artifacts(artifact_logger, pending_artifacts)
                pending_artifacts.clear()

            # Compute GAE and update model
            all_log_probs: list[torch.Tensor] = []
            all_advantages: list[torch.Tensor] = []
            all_returns: list[torch.Tensor] = []
            all_values: list[torch.Tensor] = []
            all_entropies: list[torch.Tensor] = []

            for ep in batch_episodes:
                if not ep.log_probs:
                    continue

                returns, advantages = _compute_gae(
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
                all_entropies.extend(ep.entropies)

            if not all_log_probs:
                continue

            # Stack tensors
            log_probs_t = torch.stack(all_log_probs)
            advantages_t = torch.cat(all_advantages)
            returns_t = torch.cat(all_returns)
            values_t = torch.stack(all_values).squeeze(-1)
            entropy_t = torch.stack(all_entropies).mean()

            # Compute loss
            policy_loss = -(log_probs_t * advantages_t.detach()).mean()
            value_loss = F.mse_loss(values_t, returns_t.detach())
            loss = policy_loss + spec.value_coef * value_loss - spec.entropy_coef * entropy_t

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

        return _build_metrics(
            episode_count, wins, poorest_eliminations, losses, policy_losses, value_losses,
            artifact_logger,
        )

    finally:
        ctx.env.cleanup()
        if artifact_logger is not None:
            artifact_logger.close()


def _run_episode(
    model: MLPPolicyValue,
    encoder: BasicEncoder,
    env: EnvAdapter,
    opponents: dict[int, Player],
    player_count: int,
    temperature: float,
    device: torch.device,
    rng: random.Random,
) -> EpisodeData:
    """Run a single episode and collect trajectory data."""
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    done = False
    while not done:
        player_id, legal_actions = _next_turn(env, player_count)
        if player_id is None:
            done = True
            break

        observation = env.observe(player_id)

        if player_id == 0:  # Trainee
            action, log_prob, value, entropy = _select_action(
                model, encoder, observation, legal_actions, temperature, device
            )
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
        else:
            opponent = opponents.get(player_id)
            if opponent is not None:
                action = opponent.act(observation, legal_actions)
            else:
                action = rng.choice(legal_actions)

        _, _, done, _ = env.step(player_id, action)

    # Compute final reward
    result = env.game_result()
    _notify_opponents(opponents, result)

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

    rewards = [0.0] * len(log_probs)
    if rewards:
        rewards[-1] = reward

    return EpisodeData(
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        entropies=entropies,
        won=won,
        eliminated=eliminated,
    )


def _next_turn(env: EnvAdapter, player_count: int) -> tuple[int | None, list[Action]]:
    """Return the next player id with legal actions."""
    for player_id in range(player_count):
        legal_actions = env.legal_actions(player_id)
        if legal_actions:
            return player_id, legal_actions
    if env.get_state().game_over:
        return None, []
    raise InvalidState("No legal actions available")


def _select_action(
    model: MLPPolicyValue,
    encoder: BasicEncoder,
    observation: Observation,
    legal_actions: list[Action],
    temperature: float,
    device: torch.device,
) -> tuple[Action, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select an action from the policy with masking."""
    features = torch.tensor(encoder.encode(observation), dtype=torch.float32, device=device)
    logits, value = model(features)
    mask = torch.tensor(encoder.action_mask(legal_actions), dtype=torch.bool, device=device)
    logits = logits.squeeze(0).masked_fill(~mask, -1e9)

    if temperature <= 0:
        action_idx = int(torch.argmax(logits).item())
        log_prob = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
    else:
        dist = torch.distributions.Categorical(logits=logits / temperature)
        sample = dist.sample()
        action_idx = int(sample.item())
        log_prob = dist.log_prob(sample)
        entropy = dist.entropy()

    action = encoder.action_space.action_at(action_idx)
    return action, log_prob, value.squeeze(-1), entropy


def _compute_gae(
    rewards: list[float],
    values: list[torch.Tensor],
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    if T == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t = torch.stack([v.squeeze() for v in values]).to(device)
    values_ext = torch.cat([values_t, torch.zeros(1, device=device)])

    deltas = rewards_t + gamma * values_ext[1:] - values_ext[:-1]

    advantages = torch.zeros(T, device=device)
    gae = torch.tensor(0.0, device=device)

    for t in range(T - 1, -1, -1):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    returns = advantages + values_t

    if T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def _build_metrics(
    episode_count: int,
    wins: int,
    poorest_eliminations: int,
    losses: list[float],
    policy_losses: list[float],
    value_losses: list[float],
    artifact_logger: TrainingArtifactLogger | None,
) -> TrainMetrics:
    """Build final training metrics."""
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


def _select_device(device_str: str) -> torch.device:
    """Select the appropriate device, with auto-detection support."""
    if device_str == "auto":
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
    spec: MLPTrainSpec, device: torch.device
) -> tuple[BasicEncoder, MLPPolicyValue]:
    """Build a new model or restore one from a checkpoint."""
    if spec.resume:
        bundle = load_checkpoint(spec.resume)
        encoder = BasicEncoder.from_config(bundle.encoder_config)
        config = MLPConfig.from_dict(bundle.model_config)
        if config.input_dim != encoder.feature_size:
            raise InvalidState("Encoder feature size does not match model config")
        if config.action_dim != encoder.action_space.size:
            raise InvalidState("Action space size does not match model config")
        model = MLPPolicyValue(config).to(device)
        model.load_state_dict(bundle.model_state, strict=True)
        return encoder, model

    encoder = BasicEncoder()
    config = MLPConfig(
        input_dim=encoder.feature_size,
        action_dim=encoder.action_space.size,
        hidden_sizes=spec.hidden_sizes,
        activation=spec.activation,
        dropout=spec.dropout,
    )
    model = MLPPolicyValue(config).to(device)
    return encoder, model


def _default_registry(spec: MLPTrainSpec) -> PlayerRegistry | None:
    """Create a player registry only when opponent specs are configured."""
    if not spec.opponents:
        return None
    return build_default_registry()


def _training_artifact_logger(spec: MLPTrainSpec) -> TrainingArtifactLogger | None:
    """Create a training artifact logger when enabled."""
    if not spec.artifacts_path:
        return None
    return TrainingArtifactLogger(
        spec.artifacts_path,
        bot_type="mlp",
        learner_seats=(0,),
        player_count=spec.player_count,
        spec=spec.to_mapping(),
    )


def _sample_opponents(
    spec: MLPTrainSpec,
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


def _should_checkpoint(episode: int, spec: MLPTrainSpec) -> bool:
    """Return True if a checkpoint should be saved at this episode."""
    return episode % spec.checkpoint_every == 0 or episode == spec.episodes


def _write_checkpoint(
    spec: MLPTrainSpec,
    model: MLPPolicyValue,
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
            "type": "mlp",
            "episodes": spec.episodes,
            "seed": spec.seed,
            "resume": spec.resume,
            "artifacts_path": spec.artifacts_path,
            "opponents": [opponent.to_mapping() for opponent in spec.opponents],
            "opponent_weights": list(spec.opponent_weights),
        },
    )


def _learning_rate_for_episode(spec: MLPTrainSpec, episode: int) -> float:
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


def load_train_spec(path: Path) -> MLPTrainSpec:
    """Load a training spec from JSON or YAML."""
    data = _load_spec_data(path)
    if not isinstance(data, Mapping):
        raise ValueError("Training spec must be a mapping")
    return MLPTrainSpec.from_mapping(data)


def _load_spec_data(path: Path) -> Mapping[str, Any]:
    """Load spec data from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return _load_json(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    raise ValueError("Spec file must be .json or .yaml")


def _load_json(path: Path) -> Mapping[str, Any]:
    """Load spec data from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Load spec data from a YAML file."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("PyYAML is required for YAML specs") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("YAML spec must be a mapping")
    return data


def main() -> None:
    """Run the training CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train an MLP policy/value model.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to training spec.")
    args = parser.parse_args()
    spec = load_train_spec(args.spec)

    print("=" * 60)
    print("MLP Model Training (Optimized)")
    print("=" * 60)
    print(f"  Episodes:       {spec.episodes:,}")
    print(f"  Batch size:     {spec.batch_size}")
    print(f"  Workers:        {spec.num_workers}")
    print(f"  Device:         {spec.device}")

    if spec.checkpoint_path:
        print(f"  Checkpoint:     {spec.checkpoint_path}")

    print()
    print("Optimizations enabled:")
    print("  - Reusable environment (no object recreation per game)")
    print("  - Memory-efficient game cleanup")
    print("  - Batched gradient updates")
    print("  - GAE advantage estimation")
    if spec.num_workers > 1:
        print(f"  - TRUE MULTIPROCESSING ({spec.num_workers} workers)")
    print()

    metrics = train_mlp(spec)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
