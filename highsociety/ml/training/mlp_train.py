"""Training loop for the MLP policy/value model."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - torch is optional for non-ML usage
    import torch
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
    max_grad_norm: float = 1.0
    checkpoint_path: str | None = None
    checkpoint_every: int = 250
    artifacts_path: str | None = None
    resume: str | None = None
    opponents: tuple[PlayerSpec, ...] = ()
    opponent_weights: tuple[float, ...] = ()
    device: str = "cpu"

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
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        if self.opponent_weights and len(self.opponent_weights) != len(self.opponents):
            raise ValueError("opponent_weights must match opponents length")
        for weight in self.opponent_weights:
            if weight <= 0:
                raise ValueError("opponent_weights must be positive")

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
            "max_grad_norm": self.max_grad_norm,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_every": self.checkpoint_every,
            "artifacts_path": self.artifacts_path,
            "resume": self.resume,
            "opponents": [spec.to_mapping() for spec in self.opponents],
            "opponent_weights": list(self.opponent_weights),
            "device": self.device,
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
            learning_rate_final=_coerce_optional_float(
                data.get("learning_rate_final"),
                "learning_rate_final",
            ),
            hidden_sizes=hidden_tuple,
            activation=str(data.get("activation", "relu")),
            dropout=float(data.get("dropout", 0.0)),
            temperature=float(data.get("temperature", 1.0)),
            entropy_coef=float(data.get("entropy_coef", 0.01)),
            value_coef=float(data.get("value_coef", 0.5)),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            checkpoint_path=_coerce_optional_str(data.get("checkpoint_path")),
            checkpoint_every=int(data.get("checkpoint_every", 250)),
            artifacts_path=_coerce_optional_str(data.get("artifacts_path")),
            resume=_coerce_optional_str(data.get("resume")),
            opponents=opponents,
            opponent_weights=weights,
            device=str(data.get("device", "cpu")),
        )


@dataclass(frozen=True)
class TrainMetrics:
    """Summary metrics from a training run."""

    episodes: int
    wins: int
    win_rate: float
    average_loss: float
    average_policy_loss: float
    average_value_loss: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metrics."""
        return {
            "episodes": self.episodes,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "average_loss": self.average_loss,
            "average_policy_loss": self.average_policy_loss,
            "average_value_loss": self.average_value_loss,
        }


def train_mlp(
    spec: MLPTrainSpec,
    registry: PlayerRegistry | None = None,
) -> TrainMetrics:
    """Train an MLP policy/value model and return summary metrics."""
    rng = random.Random(spec.seed)
    torch.manual_seed(spec.seed)
    encoder, model = _initialize_model(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    registry = registry if registry is not None else _default_registry(spec)
    wins = 0
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    artifact_logger = _training_artifact_logger(spec)
    try:
        for episode in tqdm(
            range(spec.episodes),
            total=spec.episodes,
            desc="MLP training",
            unit="run",
            bar_format=_PROGRESS_BAR_FORMAT,
            dynamic_ncols=True,
        ):
            env = EnvAdapter(player_count=spec.player_count)
            episode_seed = spec.seed + episode
            env.reset(seed=episode_seed)
            opponents = _sample_opponents(spec, rng, registry)
            _reset_opponents(opponents, episode_seed, spec.player_count)
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            done = False
            while not done:
                player_id, legal_actions = _next_turn(env, spec.player_count)
                if player_id is None:
                    done = True
                    break
                observation = env.observe(player_id)
                if player_id == 0:
                    action, log_prob, value, entropy = _select_action(
                        model,
                        encoder,
                        observation,
                        legal_actions,
                        temperature=spec.temperature,
                        device=spec.device,
                    )
                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)
                else:
                    opponent = opponents.get(player_id)
                    if opponent is not None:
                        action = opponent.act(observation, legal_actions)
                    else:
                        action = _random_action(legal_actions, rng)
                _, _, done, _ = env.step(player_id, action)
            state = env.get_state()
            if not state.game_over:
                raise InvalidState("Training episode ended before terminal game state")
            result = env.game_result()
            _notify_opponents(opponents, result)
            if artifact_logger is not None:
                artifact_logger.record_game(
                    game_index=episode + 1,
                    seed=episode_seed,
                    result=result,
                    players=tuple(state.players),
                )
            won = 1 if 0 in result.winners else 0
            reward = 1.0 if won else -1.0
            wins += won
            if not log_probs:
                continue
            returns = torch.full((len(log_probs),), reward, device=spec.device)
            log_probs_t = torch.stack(log_probs)
            values_t = torch.stack(values).squeeze(-1)
            entropy_t = torch.stack(entropies).mean()
            advantage = returns - values_t.detach()
            policy_loss = -(log_probs_t * advantage).mean()
            value_loss = (returns - values_t).pow(2).mean()
            loss = policy_loss + spec.value_coef * value_loss - spec.entropy_coef * entropy_t
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=spec.max_grad_norm)
            _set_optimizer_lr(optimizer, _learning_rate_for_episode(spec, episode + 1))
            optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            if spec.checkpoint_path and _should_checkpoint(episode + 1, spec):
                _write_checkpoint(spec, model, encoder, wins, episode + 1)
        if spec.checkpoint_path:
            _write_checkpoint(spec, model, encoder, wins, spec.episodes)
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        avg_policy_loss = sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        avg_value_loss = sum(value_losses) / len(value_losses) if value_losses else 0.0
        win_rate = wins / spec.episodes if spec.episodes else 0.0
        metrics = TrainMetrics(
            episodes=spec.episodes,
            wins=wins,
            win_rate=win_rate,
            average_loss=avg_loss,
            average_policy_loss=avg_policy_loss,
            average_value_loss=avg_value_loss,
        )
        if artifact_logger is not None:
            artifact_logger.finalize(total_games=spec.episodes, training_metrics=metrics.to_dict())
        return metrics
    finally:
        if artifact_logger is not None:
            artifact_logger.close()


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
    device: str,
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


def _random_action(legal_actions: list[Action], rng: random.Random) -> Action:
    """Select a random legal action."""
    return rng.choice(legal_actions)


def _initialize_model(spec: MLPTrainSpec) -> tuple[BasicEncoder, MLPPolicyValue]:
    """Build a new model or restore one from a checkpoint."""
    if spec.resume:
        bundle = load_checkpoint(spec.resume)
        encoder = BasicEncoder.from_config(bundle.encoder_config)
        config = MLPConfig.from_dict(bundle.model_config)
        if config.input_dim != encoder.feature_size:
            raise InvalidState("Encoder feature size does not match model config")
        if config.action_dim != encoder.action_space.size:
            raise InvalidState("Action space size does not match model config")
        model = MLPPolicyValue(config).to(spec.device)
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
    model = MLPPolicyValue(config).to(spec.device)
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
            "episodes": spec.episodes,
            "seed": spec.seed,
            "resume": spec.resume,
            "artifacts_path": spec.artifacts_path,
            "opponents": [opponent.to_mapping() for opponent in spec.opponents],
            "opponent_weights": list(spec.opponent_weights),
        },
    )


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


def _coerce_optional_str(value: object) -> str | None:
    """Return a stripped string or None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_optional_float(value: object, name: str) -> float | None:
    """Return a float or None for missing/blank values."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        value = stripped
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


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


def main() -> None:
    """Run the training CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train an MLP policy/value model.")
    parser.add_argument("--spec", required=True, type=Path, help="Path to training spec.")
    args = parser.parse_args()
    spec = load_train_spec(args.spec)
    metrics = train_mlp(spec)
    print(json.dumps(metrics.to_dict(), indent=2))


if __name__ == "__main__":
    main()
