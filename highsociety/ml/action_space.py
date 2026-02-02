"""Fixed action space helpers for ML policies."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

from highsociety.domain.actions import Action, ActionKind
from highsociety.domain.cards import ALLOWED_MONEY_VALUES
from highsociety.domain.errors import InvalidState

_DEFAULT_POSSESSION_VALUES = tuple(range(1, 11))
_DEFAULT_ACTION_SPACE: "ActionSpace | None" = None


class ActionSpace:
    """Fixed action space mapping between indices and actions."""

    def __init__(
        self,
        money_values: Sequence[int] | None = None,
        possession_values: Sequence[int] | None = None,
    ) -> None:
        """Initialize the action space for the given card value ranges."""
        self._money_values = tuple(sorted(money_values or ALLOWED_MONEY_VALUES))
        self._possession_values = tuple(sorted(possession_values or _DEFAULT_POSSESSION_VALUES))
        actions: list[Action] = [Action(ActionKind.PASS)]
        for size in range(1, len(self._money_values) + 1):
            for combo in itertools.combinations(self._money_values, size):
                actions.append(Action(ActionKind.BID, cards=combo))
        for value in self._possession_values:
            actions.append(Action(ActionKind.DISCARD_POSSESSION, possession_value=value))
        self._actions = tuple(actions)
        self._index = {action: idx for idx, action in enumerate(self._actions)}

    @property
    def size(self) -> int:
        """Return the total number of actions in the space."""
        return len(self._actions)

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return all actions in the action space."""
        return self._actions

    def index_of(self, action: Action) -> int:
        """Return the index for a given action."""
        try:
            return self._index[action]
        except KeyError as exc:
            raise InvalidState(f"Action not in action space: {action}") from exc

    def action_at(self, index: int) -> Action:
        """Return the action at the given index."""
        if not (0 <= index < len(self._actions)):
            raise InvalidState(f"Action index out of range: {index}")
        return self._actions[index]

    def mask(self, legal_actions: Iterable[Action]) -> list[int]:
        """Return a 0/1 mask for the provided legal actions."""
        mask = [0] * len(self._actions)
        for action in legal_actions:
            idx = self.index_of(action)
            mask[idx] = 1
        return mask

    def config(self) -> dict[str, object]:
        """Return a config mapping for serialization."""
        return {
            "money_values": list(self._money_values),
            "possession_values": list(self._possession_values),
        }

    @staticmethod
    def from_config(config: dict[str, object]) -> "ActionSpace":
        """Create an action space from a config mapping."""
        money_values = config.get("money_values", ALLOWED_MONEY_VALUES)
        possession_values = config.get("possession_values", _DEFAULT_POSSESSION_VALUES)
        if not isinstance(money_values, (list, tuple)):
            raise ValueError("money_values must be a list")
        if not isinstance(possession_values, (list, tuple)):
            raise ValueError("possession_values must be a list")
        return ActionSpace(
            money_values=tuple(int(value) for value in money_values),
            possession_values=tuple(int(value) for value in possession_values),
        )


def default_action_space() -> ActionSpace:
    """Return a cached default action space."""
    global _DEFAULT_ACTION_SPACE
    if _DEFAULT_ACTION_SPACE is None:
        _DEFAULT_ACTION_SPACE = ActionSpace()
    return _DEFAULT_ACTION_SPACE
