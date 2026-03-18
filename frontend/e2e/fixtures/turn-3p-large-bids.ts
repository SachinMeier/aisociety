/**
 * 3-player scenario with large bids to stress-test seat layout.
 * Player 0 has a 7-coin bid (max visual width), Player 1 has 5 coins.
 */
import { makePlayer, makePublicTable, makeRound, makeTurnResponse, CARDS } from "./base-players";

const players = [
  makePlayer({
    id: 0,
    name: "Alice",
    open_bid: [1, 2, 3, 4, 6, 8, 10],
    owned_status_cards: [CARDS.possession3, CARDS.possession5, CARDS.title],
    money_count: 4,
  }),
  makePlayer({
    id: 1,
    name: "Bob",
    open_bid: [2, 4, 6, 8, 10],
    owned_status_cards: [CARDS.possession1, CARDS.scandal],
    money_count: 6,
  }),
  makePlayer({
    id: 2,
    name: "Charlie",
    open_bid: [],
    owned_status_cards: [CARDS.possession7, CARDS.possession10, CARDS.debt],
    money_count: -1, // passed
  }),
];

export const turn3pLargeBids = makeTurnResponse({
  active_player_id: 0,
  active_player_name: "Alice",
  public_table: makePublicTable({
    players,
    status_card: CARDS.possession10,
    round: makeRound({
      highest_bid: 34,
      highest_bidder: 0,
      turn_player: 1,
    }),
    revealed_status_cards: [CARDS.theft],
  }),
  private_hand: [12, 15, 20, 25],
  legal_actions: [
    { kind: "pass" },
    { kind: "bid", cards: [12] },
    { kind: "bid", cards: [15] },
  ],
});
