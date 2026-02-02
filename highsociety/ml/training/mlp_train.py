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

from highsociety.app.env_adapter import EnvAdapter
from highsociety.app.observations import Observation
from highsociety.domain.actions import Action
from highsociety.domain.errors import InvalidState
from highsociety.ml.checkpoints import save_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder
from highsociety.ml.models.mlp import MLPConfig, MLPPolicyValue


@dataclass(frozen=True)
class MLPTrainSpec:
    """Configuration for training an MLP policy/value model."""

    episodes: int = 1000
    seed: int = 0
    player_count: int = 3
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "relu"
    dropout: float = 0.0
    temperature: float = 1.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    checkpoint_path: str | None = None
    checkpoint_every: int = 250
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate training configuration values."""
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not (3 <= self.player_count <= 5):
            raise ValueError("player_count must be 3-5")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
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

    def to_mapping(self) -> dict[str, object]:
        """Serialize the training spec to a mapping."""
        return {
            "episodes": self.episodes,
            "seed": self.seed,
            "player_count": self.player_count,
            "learning_rate": self.learning_rate,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "dropout": self.dropout,
            "temperature": self.temperature,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "max_grad_norm": self.max_grad_norm,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_every": self.checkpoint_every,
            "device": self.device,
        }

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "MLPTrainSpec":
        """Create a training spec from a mapping."""
        hidden_sizes = data.get("hidden_sizes", (128, 128))
        if isinstance(hidden_sizes, list | tuple):
            hidden_tuple = tuple(int(size) for size in hidden_sizes)
        else:
            hidden_tuple = (128, 128)
        return MLPTrainSpec(
            episodes=int(data.get("episodes", 1000)),
            seed=int(data.get("seed", 0)),
            player_count=int(data.get("player_count", 3)),
            learning_rate=float(data.get("learning_rate", 1e-3)),
            hidden_sizes=hidden_tuple,
            activation=str(data.get("activation", "relu")),
            dropout=float(data.get("dropout", 0.0)),
            temperature=float(data.get("temperature", 1.0)),
            entropy_coef=float(data.get("entropy_coef", 0.01)),
            value_coef=float(data.get("value_coef", 0.5)),
            max_grad_norm=float(data.get("max_grad_norm", 1.0)),
            checkpoint_path=_coerce_optional_str(data.get("checkpoint_path")),
            checkpoint_every=int(data.get("checkpoint_every", 250)),
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


def train_mlp(spec: MLPTrainSpec) -> TrainMetrics:
    """Train an MLP policy/value model and return summary metrics."""
    rng = random.Random(spec.seed)
    torch.manual_seed(spec.seed)
    encoder = BasicEncoder()
    config = MLPConfig(
        input_dim=encoder.feature_size,
        action_dim=encoder.action_space.size,
        hidden_sizes=spec.hidden_sizes,
        activation=spec.activation,
        dropout=spec.dropout,
    )
    model = MLPPolicyValue(config).to(spec.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    wins = 0
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    for episode in range(spec.episodes):
        env = EnvAdapter(player_count=spec.player_count)
        env.reset(seed=spec.seed + episode)
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        done = False
        info = None
        while not done:
            player_id, legal_actions = _next_turn(env, spec.player_count)
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
                action = _random_action(legal_actions, rng)
            _, _, done, info = env.step(player_id, action)
        if info is None or info.result is None:
            continue
        reward = 1.0 if 0 in info.result.winners else 0.0
        wins += int(reward)
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
    return TrainMetrics(
        episodes=spec.episodes,
        wins=wins,
        win_rate=win_rate,
        average_loss=avg_loss,
        average_policy_loss=avg_policy_loss,
        average_value_loss=avg_value_loss,
    )


def _next_turn(env: EnvAdapter, player_count: int) -> tuple[int, list[Action]]:
    """Return the next player id with legal actions."""
    for player_id in range(player_count):
        legal_actions = env.legal_actions(player_id)
        if legal_actions:
            return player_id, legal_actions
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
    }
    save_checkpoint(
        spec.checkpoint_path,
        model_state=model.state_dict(),
        model_config=model.config.to_dict(),
        encoder_config=encoder.config(),
        metrics=metrics,
        metadata={"episodes": spec.episodes, "seed": spec.seed},
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
