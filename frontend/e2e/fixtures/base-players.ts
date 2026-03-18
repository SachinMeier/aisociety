import type { PlayerData, StatusCardData, TurnResponse, PublicTable, RoundData } from "../../src/types";

/** Build a PlayerData with sensible defaults. */
export function makePlayer(overrides: Partial<PlayerData> & { id: number; name: string }): PlayerData {
  return {
    open_bid: [],
    owned_status_cards: [],
    money_count: 11,
    ...overrides,
  };
}

/** Build a PublicTable with sensible defaults. */
export function makePublicTable(overrides: Partial<PublicTable> & { players: PlayerData[] }): PublicTable {
  return {
    status_card: { kind: "possession", value: 5 },
    round: null,
    revealed_status_cards: [],
    ...overrides,
  };
}

/** Build a RoundData. */
export function makeRound(overrides: Partial<RoundData>): RoundData {
  return {
    highest_bid: 0,
    highest_bidder: null,
    turn_player: 0,
    ...overrides,
  };
}

/** Build a full TurnResponse for mocking. */
export function makeTurnResponse(overrides: Partial<TurnResponse> & { public_table: PublicTable }): TurnResponse {
  return {
    game_id: "test-game-001",
    status: "awaiting_human_action",
    active_player_id: 0,
    active_player_name: "Player 1",
    requires_handoff: false,
    private_hand: [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25],
    legal_actions: [
      { kind: "pass" },
      { kind: "bid", cards: [1] },
      { kind: "bid", cards: [2] },
    ],
    round_history: [],
    ...overrides,
  };
}

/** Common status cards for reuse. */
export const CARDS: Record<string, StatusCardData> = {
  possession1: { kind: "possession", value: 1 },
  possession3: { kind: "possession", value: 3 },
  possession5: { kind: "possession", value: 5 },
  possession7: { kind: "possession", value: 7 },
  possession10: { kind: "possession", value: 10 },
  title: { kind: "title" },
  scandal: { kind: "misfortune", misfortune: "scandal" },
  debt: { kind: "misfortune", misfortune: "debt" },
  theft: { kind: "misfortune", misfortune: "theft" },
};
