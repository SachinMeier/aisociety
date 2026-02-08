from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .actions import Action, ActionKind
from .cards import (
    MisfortuneKind,
    MoneyCard,
    StatusCard,
    StatusKind,
    default_money_hand,
    default_status_deck,
)
from .errors import InvalidAction, InvalidState
from .state import GameState, PendingDiscard, PlayerState, RoundRecord, RoundState


@dataclass(frozen=True)
class GameResult:
    winners: tuple[int, ...]
    scores: dict[int, Fraction]
    money_remaining: dict[int, int]
    poorest: tuple[int, ...]


class RulesEngine:
    @staticmethod
    def create_money_hand() -> list[MoneyCard]:
        """Return a fresh default money hand for a player."""
        return default_money_hand()

    @staticmethod
    def create_status_deck() -> list[StatusCard]:
        """Return the default status deck."""
        return default_status_deck()

    @staticmethod
    def start_round(state: GameState, rng: random.Random) -> None:
        if state.game_over:
            raise InvalidState("Game is over")
        if state.pending_discard is not None:
            raise InvalidState("Pending discard must be resolved first")
        if state.round is not None:
            raise InvalidState("Round already in progress")
        if not state.status_deck:
            state.game_over = True
            return
        for player in state.players:
            player.open_bid.clear()
            player.passed = False
        card = state.status_deck.pop(0)
        if card.red:
            state.red_revealed += 1
            if state.red_revealed >= 4:
                state.status_discard.append(card)
                state.game_over = True
                return
        state.round = RoundState(
            card=card,
            starting_player=state.starting_player,
            turn_player=state.starting_player,
        )

    @staticmethod
    def legal_actions(state: GameState, player_id: int) -> list[Action]:
        if state.game_over:
            return []
        if state.pending_discard is not None:
            if state.pending_discard.player_id != player_id:
                return []
            return [
                Action(ActionKind.DISCARD_POSSESSION, possession_value=value)
                for value in state.pending_discard.options
            ]
        if state.round is None:
            return []
        if state.round.turn_player != player_id:
            return []
        player = RulesEngine._get_player(state, player_id)
        if player.passed:
            return []
        actions = [Action(ActionKind.PASS)]
        current_total = RulesEngine._sum_cards(player.open_bid)
        for subset in RulesEngine._card_subsets(player.hand):
            new_total = current_total + RulesEngine._sum_cards(subset)
            if new_total > state.round.highest_bid:
                actions.append(
                    Action(ActionKind.BID, cards=tuple(card.value for card in subset))
                )
        return actions

    @staticmethod
    def apply_action(state: GameState, player_id: int, action: Action) -> None:
        if state.game_over:
            raise InvalidAction("Game is over")
        if state.pending_discard is not None:
            RulesEngine._apply_discard(state, player_id, action)
            return
        if state.round is None:
            raise InvalidState("No active round")
        if state.round.turn_player != player_id:
            raise InvalidAction("Not this player's turn")
        player = RulesEngine._get_player(state, player_id)
        if player.passed:
            raise InvalidAction("Player has already passed")
        if action.kind == ActionKind.PASS:
            RulesEngine._apply_pass(state, player)
            return
        if action.kind == ActionKind.BID:
            RulesEngine._apply_bid(state, player, action)
            return
        raise InvalidAction("Invalid action for current state")

    @staticmethod
    def score_game(state: GameState) -> GameResult:
        money_remaining = {
            player.id: RulesEngine._sum_cards(player.hand) for player in state.players
        }
        poorest_value = min(money_remaining.values())
        poorest = tuple(
            sorted(pid for pid, value in money_remaining.items() if value == poorest_value)
        )
        scores: dict[int, Fraction] = {}
        for player in state.players:
            if player.id in poorest:
                continue
            base = sum(card.value for card in player.possessions)
            if player.debt:
                base -= 5 * player.debt
            total: Fraction = Fraction(base) * (2 ** player.titles)
            if player.scandal:
                total *= Fraction(1, 2 ** player.scandal)
            scores[player.id] = total
        if not scores:
            return GameResult((), scores, money_remaining, poorest)
        max_score = max(scores.values())
        top = [pid for pid, score in scores.items() if score == max_score]
        if len(top) == 1:
            winners = top
        else:
            max_money = max(money_remaining[pid] for pid in top)
            winners = [pid for pid in top if money_remaining[pid] == max_money]
        return GameResult(tuple(sorted(winners)), scores, money_remaining, poorest)

    @staticmethod
    def _apply_pass(state: GameState, player: PlayerState) -> None:
        round_state = state.round
        if round_state is None:
            raise InvalidState("No active round")
        if round_state.card.kind == StatusKind.MISFORTUNE:
            RulesEngine._resolve_misfortune_on_pass(state, player)
            return
        player.passed = True
        player.hand.extend(player.open_bid)
        player.open_bid.clear()
        remaining = [p for p in state.players if not p.passed]
        if len(remaining) == 1:
            RulesEngine._resolve_possession_title_round(state, remaining[0])
            return
        round_state.turn_player = RulesEngine._next_player(state, round_state.turn_player)

    @staticmethod
    def _apply_bid(state: GameState, player: PlayerState, action: Action) -> None:
        round_state = state.round
        if round_state is None:
            raise InvalidState("No active round")
        card_values = list(action.cards)
        if not card_values:
            raise InvalidAction("Bid requires cards")
        hand_map = {card.value: card for card in player.hand}
        try:
            bid_cards = [hand_map[value] for value in card_values]
        except KeyError as exc:
            raise InvalidAction("Bid contains card not in hand") from exc
        total = RulesEngine._sum_cards(player.open_bid) + RulesEngine._sum_cards(bid_cards)
        if total <= round_state.highest_bid:
            raise InvalidAction("Bid must exceed current highest")
        for card in bid_cards:
            player.hand.remove(card)
        player.open_bid.extend(bid_cards)
        round_state.highest_bid = total
        round_state.highest_bidder = player.id
        round_state.any_bid = True
        if round_state.card.kind != StatusKind.MISFORTUNE:
            remaining = [p for p in state.players if not p.passed]
            if len(remaining) == 1:
                RulesEngine._resolve_possession_title_round(state, remaining[0])
                return
        round_state.turn_player = RulesEngine._next_player(state, round_state.turn_player)

    @staticmethod
    def _resolve_possession_title_round(state: GameState, winner: PlayerState) -> None:
        round_state = state.round
        if round_state is None:
            raise InvalidState("No active round")
        coins_spent = tuple(c.value for c in winner.open_bid) if round_state.any_bid else ()
        state.round_history.append(
            RoundRecord(card=round_state.card, winner_id=winner.id, coins_spent=coins_spent)
        )
        if round_state.any_bid:
            state.money_discard.extend(winner.open_bid)
            winner.money_discarded.extend(winner.open_bid)
        winner.open_bid.clear()
        RulesEngine._award_status_card(state, winner, round_state.card)
        state.starting_player = winner.id
        state.round = None
        RulesEngine._reset_round_flags(state)

    @staticmethod
    def _resolve_misfortune_on_pass(state: GameState, passer: PlayerState) -> None:
        round_state = state.round
        if round_state is None:
            raise InvalidState("No active round")
        state.round_history.append(
            RoundRecord(card=round_state.card, winner_id=passer.id, coins_spent=())
        )
        round_state.first_passer = passer.id
        passer.hand.extend(passer.open_bid)
        passer.open_bid.clear()
        for player in state.players:
            if player.id == passer.id:
                continue
            state.money_discard.extend(player.open_bid)
            player.money_discarded.extend(player.open_bid)
            player.open_bid.clear()
        RulesEngine._award_status_card(state, passer, round_state.card)
        state.starting_player = passer.id
        state.round = None
        RulesEngine._reset_round_flags(state)

    @staticmethod
    def _award_status_card(state: GameState, player: PlayerState, card: StatusCard) -> None:
        if card.kind == StatusKind.POSSESSION:
            if player.theft_pending > 0:
                state.status_discard.append(card)
                player.theft_pending -= 1
            else:
                player.possessions.append(card)
            return
        if card.kind == StatusKind.TITLE:
            player.titles += 1
            return
        if card.kind == StatusKind.MISFORTUNE:
            if card.misfortune == MisfortuneKind.SCANDAL:
                player.scandal += 1
            elif card.misfortune == MisfortuneKind.DEBT:
                player.debt += 1
            elif card.misfortune == MisfortuneKind.THEFT:
                player.theft += 1
                if player.possessions:
                    options = tuple(sorted(card.value for card in player.possessions))
                    state.pending_discard = PendingDiscard(player.id, options)
                else:
                    player.theft_pending += 1
            else:
                raise InvalidState("Unknown misfortune")
            return
        raise InvalidState("Unknown status card")

    @staticmethod
    def _apply_discard(state: GameState, player_id: int, action: Action) -> None:
        pending = state.pending_discard
        if pending is None:
            raise InvalidState("No pending discard")
        if player_id != pending.player_id:
            raise InvalidAction("Player cannot discard now")
        if action.kind != ActionKind.DISCARD_POSSESSION:
            raise InvalidAction("Must discard a possession")
        player = RulesEngine._get_player(state, player_id)
        value = action.possession_value
        if value is None or value not in pending.options:
            raise InvalidAction("Invalid possession to discard")
        for idx, card in enumerate(player.possessions):
            if card.value == value:
                state.status_discard.append(card)
                del player.possessions[idx]
                break
        state.pending_discard = None

    @staticmethod
    def _reset_round_flags(state: GameState) -> None:
        for player in state.players:
            player.passed = False
            player.open_bid.clear()

    @staticmethod
    def _get_player(state: GameState, player_id: int) -> PlayerState:
        for player in state.players:
            if player.id == player_id:
                return player
        raise InvalidState("Unknown player id")

    @staticmethod
    def _player_index(state: GameState, player_id: int) -> int:
        for idx, player in enumerate(state.players):
            if player.id == player_id:
                return idx
        raise InvalidState("Unknown player id")

    @staticmethod
    def _next_player(state: GameState, current_player_id: int) -> int:
        current_idx = RulesEngine._player_index(state, current_player_id)
        count = len(state.players)
        for offset in range(1, count + 1):
            idx = (current_idx + offset) % count
            if not state.players[idx].passed:
                return state.players[idx].id
        raise InvalidState("No active players")

    @staticmethod
    def _sum_cards(cards: Iterable[MoneyCard]) -> int:
        return sum(card.value for card in cards)

    @staticmethod
    def _card_subsets(cards: list[MoneyCard]) -> Iterable[tuple[MoneyCard, ...]]:
        for size in range(1, len(cards) + 1):
            for combo in itertools.combinations(cards, size):
                yield combo
