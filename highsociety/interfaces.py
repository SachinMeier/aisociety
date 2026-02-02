"""Shared interface definitions to keep APIs consistent across the codebase."""

from __future__ import annotations

from typing import Protocol, TypedDict

from highsociety.domain.actions import Action
from highsociety.domain.rules import GameResult


class Player(Protocol):
    """Runtime player API used by the game runner."""

    name: str
    color: str

    def reset(self, game_config: dict, player_id: int, seat: int) -> None:
        """Reset player state for a new game."""

    def act(self, observation: dict, legal_actions: list[Action]) -> Action:
        """Return the next action for the player."""

    def on_game_end(self, result: GameResult) -> None:
        """Receive the final game result."""


class Policy(Protocol):
    """Policy interface for selecting actions."""

    def select_action(self, observation: dict, legal_actions: list[Action]) -> Action:
        """Select an action given an observation and legal actions."""


class Encoder(Protocol):
    """Encoder interface for ML features."""

    def encode(self, observation: dict) -> "list[float]":
        """Encode observation into a numeric feature vector."""

    def action_mask(self, legal_actions: list[Action]) -> "list[int]":
        """Return a mask indicating which actions are legal."""


class Model(Protocol):
    """Model interface for policies that output logits/value."""

    def forward(self, encoded_obs: "list[float]") -> tuple["list[float]", float]:
        """Compute action logits and a value estimate."""


class PlayerSpec(TypedDict, total=False):
    """Configuration for a player instance in a run spec."""

    type: str
    name: str
    checkpoint: str
    params: dict


class RunSpec(TypedDict):
    """Configuration for a run (play/eval/train)."""

    mode: str
    seed: int
    num_games: int
    players: list[PlayerSpec]
    rules: dict
