from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionKind(str, Enum):
    PASS = "pass"
    BID = "bid"
    DISCARD_POSSESSION = "discard_possession"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    cards: tuple[int, ...] = ()
    possession_value: int | None = None

    def __post_init__(self) -> None:
        if self.kind == ActionKind.PASS:
            if self.cards or self.possession_value is not None:
                raise ValueError("Pass action cannot include cards or possession")
        elif self.kind == ActionKind.BID:
            if not self.cards or self.possession_value is not None:
                raise ValueError("Bid action requires cards only")
            unique = set(self.cards)
            if len(unique) != len(self.cards):
                raise ValueError("Bid cards must be unique")
            object.__setattr__(self, "cards", tuple(sorted(self.cards)))
        elif self.kind == ActionKind.DISCARD_POSSESSION:
            if self.cards or self.possession_value is None:
                raise ValueError("Discard action requires possession value only")
        else:
            raise ValueError("Unknown action kind")
