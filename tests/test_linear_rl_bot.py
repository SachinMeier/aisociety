"""Tests for the linear RL bot and checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from highsociety.app.legal_actions import legal_actions
from highsociety.app.observations import build_observation
from highsociety.domain.actions import ActionKind
from highsociety.domain.cards import MoneyCard, StatusCard, StatusKind
from highsociety.domain.rules import RulesEngine
from highsociety.domain.state import GameState, PlayerState, RoundState
from highsociety.ml.checkpoints import load_linear_checkpoint, save_linear_checkpoint
from highsociety.ml.encoders.linear import LinearFeatureEncoder
from highsociety.ml.models.linear import LinearModel
from highsociety.players.linear_rl_bot import LinearRLBot
from highsociety.ops.spec import PlayerSpec


class _TerminalEnv:
    """Stub environment that ends the game when legal actions are queried."""

    def __init__(self, player_count: int = 3) -> None:
        """Initialize a minimal terminal game state."""
        players = [PlayerState(id=idx) for idx in range(player_count)]
        self._state = GameState(
            players=players,
            status_deck=RulesEngine.create_status_deck(),
            starting_player=0,
        )

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the environment (no-op for stub)."""
        del seed
        return self._state

    def get_state(self) -> GameState:
        """Return the current game state."""
        return self._state

    def observe(self, player_id: int) -> object:
        """Return a valid observation for the player."""
        return build_observation(self._state, player_id)

    def legal_actions(self, player_id: int) -> list[object]:
        """End the game and return no legal actions."""
        del player_id
        self._state.game_over = True
        return []

    def step(self, player_id: int, action: object) -> None:
        """Steps should never occur for this stub."""
        del player_id, action
        raise AssertionError("step should not be called")


def _make_state() -> GameState:
    """Create a simple game state with an active round."""
    players = [
        PlayerState(id=0, hand=[MoneyCard(1000), MoneyCard(2000)]),
        PlayerState(id=1, hand=[MoneyCard(3000)]),
        PlayerState(id=2, hand=[MoneyCard(4000)]),
    ]
    deck = RulesEngine.create_status_deck()
    card = StatusCard(kind=StatusKind.POSSESSION, value=5)
    round_state = RoundState(
        card=card,
        starting_player=0,
        turn_player=0,
        highest_bid=0,
    )
    return GameState(players=players, status_deck=deck, round=round_state, starting_player=0)


def test_linear_rl_bot_prefers_highest_bid() -> None:
    """Linear RL bot selects the highest bid when action_bid_sum is favored."""
    state = _make_state()
    observation = build_observation(state, player_id=0)
    actions = legal_actions(state, player_id=0)

    encoder = LinearFeatureEncoder()
    names = encoder.feature_names()
    bid_sum_idx = names.index("action_bid_sum")
    weights = np.zeros(len(names), dtype=np.float32)
    weights[bid_sum_idx] = 1.0
    model = LinearModel(weights=weights, bias=0.0)
    bot = LinearRLBot(name="linear", encoder=encoder, model=model)
    bot.reset({"seed": 1}, player_id=0, seat=0)

    action = bot.act(observation, actions)

    assert action.kind == ActionKind.BID
    assert action.cards == (1000, 2000)


def test_linear_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Linear checkpoints round-trip model weights and metadata."""
    encoder = LinearFeatureEncoder()
    weights = np.arange(encoder.feature_size(), dtype=np.float32)
    model = LinearModel(weights=weights, bias=0.5)
    metadata = {"games": 5, "note": "test"}
    path = tmp_path / "linear.pkl"

    saved_path = save_linear_checkpoint(path, model, encoder, metadata)
    loaded_model, loaded_encoder, loaded_metadata = load_linear_checkpoint(saved_path)

    assert np.allclose(loaded_model.weights, model.weights)
    assert loaded_model.bias == model.bias
    assert loaded_encoder.version == encoder.version
    assert loaded_metadata == metadata


def test_linear_training_handles_terminal_without_actions(monkeypatch: object) -> None:
    """Training loop exits cleanly when a round ends during legal action lookup."""
    from highsociety.ml.training import linear_train

    monkeypatch.setattr(linear_train, "EnvAdapter", _TerminalEnv)
    config = linear_train.LinearTrainingConfig(
        num_games=1,
        player_count=3,
        seed=1,
        epsilon=0.2,
        learning_rate=0.05,
    )

    result = linear_train.train_linear_self_play(config)

    assert result.games == 1


def test_linear_training_with_opponents() -> None:
    """Training supports sampling opponent bots."""
    from highsociety.ml.training.linear_train import LinearTrainingConfig, train_linear_self_play

    config = LinearTrainingConfig(
        num_games=1,
        player_count=3,
        seed=2,
        epsilon=0.2,
        learning_rate=0.05,
        opponents=(PlayerSpec(type="random"),),
    )

    result = train_linear_self_play(config)

    assert result.games == 1
