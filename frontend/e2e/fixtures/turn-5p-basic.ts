/**
 * 5-player scenario - full table with moderate state.
 */
import { makePlayer, makePublicTable, makeRound, makeTurnResponse, CARDS } from "./base-players";

const players = [
  makePlayer({
    id: 0,
    name: "Alice",
    open_bid: [3, 4],
    owned_status_cards: [CARDS.possession5],
    money_count: 9,
  }),
  makePlayer({
    id: 1,
    name: "Bob",
    open_bid: [2, 6],
    owned_status_cards: [CARDS.possession1, CARDS.title],
    money_count: 9,
  }),
  makePlayer({
    id: 2,
    name: "Charlie",
    open_bid: [],
    owned_status_cards: [CARDS.possession3],
    money_count: -1, // passed
  }),
  makePlayer({
    id: 3,
    name: "Diana",
    open_bid: [1, 2, 3],
    owned_status_cards: [],
    money_count: 8,
  }),
  makePlayer({
    id: 4,
    name: "Eve",
    open_bid: [4, 8],
    owned_status_cards: [CARDS.scandal],
    money_count: 7,
  }),
];

export const turn5pBasic = makeTurnResponse({
  active_player_id: 3,
  active_player_name: "Diana",
  public_table: makePublicTable({
    players,
    status_card: CARDS.possession7,
    round: makeRound({
      highest_bid: 12,
      highest_bidder: 4,
      turn_player: 3,
    }),
  }),
  private_hand: [4, 6, 10, 12, 15, 20, 25],
  legal_actions: [
    { kind: "pass" },
    { kind: "bid", cards: [4] },
    { kind: "bid", cards: [6] },
  ],
});
