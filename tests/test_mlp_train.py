"""Tests for MLP training spec parsing and training loop wiring."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState
from highsociety.ml.checkpoints import save_checkpoint
from highsociety.ml.encoders.basic import BasicEncoder
from highsociety.ml.models.mlp import MLPConfig, MLPPolicyValue
from highsociety.ml.training import mlp_train
from highsociety.ops.spec import PlayerSpec
from highsociety.players.base import Player


class _TerminalEnv:
    """Stub environment that becomes terminal during legal action lookup."""

    def __init__(self, player_count: int = 3) -> None:
        players = [PlayerState(id=idx) for idx in range(player_count)]
        self._state = GameState(
            players=players,
            status_deck=RulesEngine.create_status_deck(),
            starting_player=0,
        )

    def reset(self, seed: int | None = None) -> GameState:
        del seed
        return self._state

    def get_state(self) -> GameState:
        return self._state

    def observe(self, player_id: int) -> object:
        del player_id
        raise AssertionError("observe should not be called for terminal stub env")

    def legal_actions(self, player_id: int) -> list[object]:
        del player_id
        self._state.game_over = True
        return []

    def step(self, player_id: int, action: object) -> None:
        del player_id, action
        raise AssertionError("step should not be called for terminal stub env")

    def game_result(self) -> object:
        return RulesEngine.score_game(self._state)


class _RecordingOpponent:
    """Simple opponent that tracks reset and end notifications."""

    def __init__(self, name: str = "opponent") -> None:
        self.name = name
        self.reset_calls = 0
        self.end_calls = 0

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        del game_config, player_id, seat
        self.reset_calls += 1

    def act(self, observation: object, legal_actions: list[object]) -> object:
        del observation, legal_actions
        raise AssertionError("act should not be called for terminal stub env")

    def on_game_end(self, result: object) -> None:
        del result
        self.end_calls += 1


class _StubRegistry:
    """Registry stub that returns recording opponents."""

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.players: list[_RecordingOpponent] = []

    def create(self, spec: Mapping[str, object]) -> Player:
        self.created.append(dict(spec))
        player = _RecordingOpponent(name=str(spec.get("type", "opponent")))
        self.players.append(player)
        return player


def test_mlp_train_spec_parses_opponents_and_resume() -> None:
    """MLP spec supports resume and opponent mix fields."""
    spec = mlp_train.MLPTrainSpec.from_mapping(
        {
            "episodes": 10,
            "player_count": 4,
            "resume": "checkpoints/mlp_prev",
            "opponents": [
                {"type": "heuristic", "params": {"style": "balanced"}},
                {"type": "static"},
            ],
            "opponent_weights": [0.75, 0.25],
        }
    )

    assert spec.resume == "checkpoints/mlp_prev"
    assert spec.opponents == (
        PlayerSpec(type="heuristic", params={"style": "balanced"}),
        PlayerSpec(type="static"),
    )
    assert spec.opponent_weights == (0.75, 0.25)
    serialized = spec.to_mapping()
    assert serialized["resume"] == "checkpoints/mlp_prev"
    assert serialized["opponents"] == [
        {"type": "heuristic", "params": {"style": "balanced"}},
        {"type": "static"},
    ]
    assert serialized["opponent_weights"] == [0.75, 0.25]


def test_mlp_training_samples_configured_opponents(monkeypatch: object) -> None:
    """MLP training instantiates and resets opponents from the spec."""
    monkeypatch.setattr(mlp_train, "EnvAdapter", _TerminalEnv)
    registry = _StubRegistry()
    spec = mlp_train.MLPTrainSpec(
        episodes=3,
        player_count=4,
        opponents=(
            PlayerSpec(type="random"),
            PlayerSpec(type="heuristic", params={"style": "balanced"}),
        ),
        opponent_weights=(0.5, 0.5),
    )

    result = mlp_train.train_mlp(spec, registry=registry)

    assert result.episodes == 3
    assert len(registry.created) == spec.episodes * (spec.player_count - 1)
    assert all(player.reset_calls == 1 for player in registry.players)
    assert all(player.end_calls == 1 for player in registry.players)


def test_mlp_training_progress_bar_uses_episode_total(monkeypatch: object) -> None:
    """Training progress bar uses episode count and ETA-friendly formatting."""
    monkeypatch.setattr(mlp_train, "EnvAdapter", _TerminalEnv)
    progress_kwargs: dict[str, object] = {}

    def _fake_tqdm(iterable: range, *args: object, **kwargs: object) -> range:
        del args
        progress_kwargs.update(kwargs)
        return iterable

    monkeypatch.setattr(mlp_train, "tqdm", _fake_tqdm)
    spec = mlp_train.MLPTrainSpec(episodes=4, player_count=3)

    result = mlp_train.train_mlp(spec)

    assert result.episodes == 4
    assert progress_kwargs["total"] == 4
    assert progress_kwargs["unit"] == "run"
    bar_format = str(progress_kwargs["bar_format"])
    assert "{elapsed}" in bar_format
    assert "{remaining}" in bar_format


def test_mlp_training_can_resume_from_checkpoint(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Resuming uses checkpoint architecture instead of spec architecture fields."""
    monkeypatch.setattr(mlp_train, "EnvAdapter", _TerminalEnv)
    resume_path = tmp_path / "mlp_resume"
    encoder = BasicEncoder()
    config = MLPConfig(
        input_dim=encoder.feature_size,
        action_dim=encoder.action_space.size,
        hidden_sizes=(32,),
        activation="tanh",
        dropout=0.0,
    )
    model = MLPPolicyValue(config)
    save_checkpoint(
        resume_path,
        model_state=model.state_dict(),
        model_config=config.to_dict(),
        encoder_config=encoder.config(),
    )
    spec = mlp_train.MLPTrainSpec(
        episodes=1,
        player_count=3,
        hidden_sizes=(128, 128),
        activation="relu",
        resume=str(resume_path),
    )

    result = mlp_train.train_mlp(spec)

    assert result.episodes == 1


def test_mlp_training_writes_per_game_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    """Training writes per-game artifacts when artifacts_path is provided."""
    monkeypatch.setattr(mlp_train, "EnvAdapter", _TerminalEnv)
    artifacts_path = tmp_path / "mlp_history"
    spec = mlp_train.MLPTrainSpec(
        episodes=2,
        player_count=3,
        artifacts_path=str(artifacts_path),
    )

    result = mlp_train.train_mlp(spec)

    assert result.episodes == 2
    summary_path = artifacts_path / "training_summary.json"
    games_path = artifacts_path / "training_games.csv"
    players_path = artifacts_path / "training_players.csv"
    assert summary_path.exists()
    assert games_path.exists()
    assert players_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["games_total"] == 2
    assert summary["games_logged"] == 2
    assert summary["learner_seats"] == [0]
    with players_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == spec.episodes * spec.player_count
    assert all("cumulative_wins" in row for row in rows)
