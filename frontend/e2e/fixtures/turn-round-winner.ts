/**
 * 3-player scenario designed to test the round winner overlay.
 * Includes round_history so detectRoundWinnerFromHistory can trigger.
 */
import { makePlayer, makePublicTable, makeTurnResponse, CARDS } from "./base-players";

const players = [
  makePlayer({
    id: 0,
    name: "Alice",
    open_bid: [],
    owned_status_cards: [CARDS.possession5, CARDS.possession7],
    money_count: 7,
  }),
  makePlayer({
    id: 1,
    name: "Bob",
    open_bid: [],
    owned_status_cards: [CARDS.possession3],
    money_count: 9,
  }),
  makePlayer({
    id: 2,
    name: "Charlie",
    open_bid: [],
    owned_status_cards: [CARDS.possession1],
    money_count: 10,
  }),
];

export const turnRoundWinner = makeTurnResponse({
  active_player_id: 1,
  active_player_name: "Bob",
  public_table: makePublicTable({
    players,
    status_card: CARDS.possession10,
    round: null,
  }),
  round_history: [
    {
      card: CARDS.possession5,
      winner_id: 0,
      winner_name: "Alice",
      coins_spent: [3, 4, 6],
    },
  ],
  private_hand: [1, 2, 3, 4, 6, 8, 10, 15, 20],
  legal_actions: [
    { kind: "pass" },
    { kind: "bid", cards: [1] },
  ],
});
