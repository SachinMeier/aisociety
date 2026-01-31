"""Card definitions and default builders for High Society."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ALLOWED_MONEY_VALUES = (
    25000,
    20000,
    15000,
    12000,
    10000,
    8000,
    6000,
    4000,
    3000,
    2000,
    1000,
)


class StatusKind(str, Enum):
    POSSESSION = "possession"
    TITLE = "title"
    MISFORTUNE = "misfortune"


class MisfortuneKind(str, Enum):
    SCANDAL = "scandal"
    DEBT = "debt"
    THEFT = "theft"


@dataclass(frozen=True)
class MoneyCard:
    value: int

    def __post_init__(self) -> None:
        if self.value not in ALLOWED_MONEY_VALUES:
            raise ValueError(f"Invalid money value: {self.value}")


@dataclass(frozen=True)
class StatusCard:
    kind: StatusKind
    value: int | None = None
    misfortune: MisfortuneKind | None = None
    red: bool = False

    def __post_init__(self) -> None:
        if self.kind == StatusKind.POSSESSION:
            if self.value not in range(1, 11) or self.misfortune is not None:
                raise ValueError("Invalid possession card")
            object.__setattr__(self, "red", False)
        elif self.kind == StatusKind.TITLE:
            if self.value is not None or self.misfortune is not None:
                raise ValueError("Invalid title card")
            object.__setattr__(self, "red", True)
        elif self.kind == StatusKind.MISFORTUNE:
            if self.value is not None or self.misfortune is None:
                raise ValueError("Invalid misfortune card")
            object.__setattr__(self, "red", self.misfortune == MisfortuneKind.SCANDAL)
        else:
            raise ValueError("Unknown status card kind")


def default_money_hand() -> list[MoneyCard]:
    """Return a fresh default money hand with one of each allowed value."""
    return [MoneyCard(value) for value in ALLOWED_MONEY_VALUES]


def default_status_deck() -> list[StatusCard]:
    """Return the default 16-card status deck."""
    deck: list[StatusCard] = []
    for value in range(1, 11):
        deck.append(StatusCard(kind=StatusKind.POSSESSION, value=value))
    for _ in range(3):
        deck.append(StatusCard(kind=StatusKind.TITLE))
    deck.append(
        StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.SCANDAL)
    )
    deck.append(StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.DEBT))
    deck.append(StatusCard(kind=StatusKind.MISFORTUNE, misfortune=MisfortuneKind.THEFT))
    return deck
