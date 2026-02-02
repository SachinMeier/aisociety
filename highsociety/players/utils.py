"""Shared helper functions for player implementations."""

from __future__ import annotations

from highsociety.domain.actions import Action, ActionKind


def choose_lowest_discard(actions: list[Action]) -> Action:
    """Choose the discard action with the lowest possession value."""
    return min(actions, key=lambda action: action.possession_value or 0)


def fallback_action(actions: list[Action]) -> Action:
    """Return PASS if available; otherwise return the first action."""
    for action in actions:
        if action.kind == ActionKind.PASS:
            return action
    return actions[0]
