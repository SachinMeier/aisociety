"""Tests for the game runner and environment adapter."""

from __future__ import annotations

from dataclasses import dataclass, field

from highsociety.app.env_adapter import EnvAdapter
from highsociety.app.legal_actions import legal_actions
from highsociety.app.observations import Observation, build_observation
from highsociety.app.runner import GameRunner
from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.rules import GameResult


@dataclass
class RecordingPlayer:
    """Player that records calls and asserts turn ownership."""

    name: str
    calls: list[int] = field(default_factory=list)
    player_id: int | None = None

    def reset(self, game_config: dict[str, object], player_id: int, seat: int) -> None:
        """Capture the assigned player id for later assertions."""
        self.player_id = player_id

    def act(self, observation: Observation, legal_actions: list[Action]) -> Action:
        """Record the call and return a valid action."""
        if self.player_id is None:
            raise AssertionError("Player id not set")
        assert observation.player_id == self.player_id
        if observation.round is not None:
            assert observation.round.turn_player == self.player_id
        assert legal_actions
        self.calls.append(self.player_id)
        for action in legal_actions:
            if action.kind == ActionKind.PASS:
                return action
        return legal_actions[0]

    def on_game_end(self, result: GameResult) -> None:
        """No-op end handler for tests."""


def test_game_runner_uses_turn_player_order() -> None:
    """Game runner calls players matching the current turn."""
    calls: list[int] = []
    players = [
        RecordingPlayer(name="p0", calls=calls),
        RecordingPlayer(name="p1", calls=calls),
        RecordingPlayer(name="p2", calls=calls),
    ]
    runner = GameRunner()
    runner.run_game(players, seed=1)
    assert calls


def test_env_adapter_exposes_legal_actions() -> None:
    """Env adapter exposes legal actions and observations."""
    env = EnvAdapter(player_count=3)
    state = env.reset(seed=1)
    current = state.starting_player
    env_actions = env.legal_actions(current)
    expected_actions = legal_actions(env.get_state(), current)
    assert env_actions == expected_actions
    obs = env.observe(current)
    expected_obs = build_observation(env.get_state(), current)
    assert obs == expected_obs
