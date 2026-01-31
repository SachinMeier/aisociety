from __future__ import annotations

from dataclasses import dataclass, field

from .cards import MoneyCard, StatusCard, StatusKind, default_money_hand
from .errors import InvalidState


@dataclass
class PlayerState:
    id: int
    hand: list[MoneyCard] = field(default_factory=default_money_hand)
    open_bid: list[MoneyCard] = field(default_factory=list)
    passed: bool = False
    possessions: list[StatusCard] = field(default_factory=list)
    titles: int = 0
    scandal: int = 0
    debt: int = 0
    theft_pending: int = 0
    money_discarded: list[MoneyCard] = field(default_factory=list)

    def __post_init__(self) -> None:
        hand_values = [card.value for card in self.hand]
        open_values = [card.value for card in self.open_bid]
        if len(hand_values) != len(set(hand_values)):
            raise InvalidState("Duplicate money card in hand")
        if len(open_values) != len(set(open_values)):
            raise InvalidState("Duplicate money card in open bid")
        if set(hand_values) & set(open_values):
            raise InvalidState("Open bid cards cannot be in hand")
        for card in self.possessions:
            if card.kind != StatusKind.POSSESSION:
                raise InvalidState("Possessions must be possession cards")
        if self.titles < 0:
            raise InvalidState("Titles cannot be negative")
        for name, value in (
            ("scandal", self.scandal),
            ("debt", self.debt),
            ("theft_pending", self.theft_pending),
        ):
            if value < 0:
                raise InvalidState(f"{name} cannot be negative")


@dataclass
class RoundState:
    card: StatusCard
    starting_player: int
    turn_player: int
    highest_bidder: int | None = None
    highest_bid: int = 0
    any_bid: bool = False
    first_passer: int | None = None


@dataclass
class PendingDiscard:
    player_id: int
    options: tuple[int, ...]


@dataclass
class GameState:
    players: list[PlayerState]
    status_deck: list[StatusCard]
    status_discard: list[StatusCard] = field(default_factory=list)
    money_discard: list[MoneyCard] = field(default_factory=list)
    red_revealed: int = 0
    round: RoundState | None = None
    game_over: bool = False
    starting_player: int = 0
    pending_discard: PendingDiscard | None = None

    def __post_init__(self) -> None:
        if not (3 <= len(self.players) <= 5):
            raise InvalidState("Player count must be 3-5")
        ids = [player.id for player in self.players]
        if len(ids) != len(set(ids)):
            raise InvalidState("Duplicate player id")
        if not (0 <= self.red_revealed <= 4):
            raise InvalidState("Invalid red_revealed")
        if self.starting_player not in ids:
            raise InvalidState("Invalid starting player id")
