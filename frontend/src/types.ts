// High Society — shared type definitions for the pass-and-play frontend

export interface StatusCardData {
  kind: "possession" | "title" | "misfortune";
  value?: number; // only present for possession cards
  misfortune?: string; // e.g. "scandal", "debt", "theft"
}

export interface PlayerData {
  id: number;
  name: string;
  open_bid: number[];
  owned_status_cards: StatusCardData[];
  money_count: number;
}

export interface RoundData {
  highest_bid: number;
  highest_bidder: number | null;
  turn_player: number;
}

export interface PublicTable {
  status_card: StatusCardData | null;
  round: RoundData | null;
  players: PlayerData[];
  revealed_status_cards: StatusCardData[];
}

export type ActionKind = "pass" | "bid" | "discard_possession";

export interface LegalAction {
  kind: ActionKind;
  cards?: number[];
  possession_value?: number;
}

export type GameStatus =
  | "awaiting_human_action"
  | "active"
  | "finished"
  | "errored";

export interface GameResults {
  winners: number[];
  scores: Record<string, number>;
  money_remaining: Record<string, number>;
  poorest: number[];
}

export interface RoundRecord {
  card: StatusCardData;
  winner_id: number;
  winner_name: string;
  coins_spent: number[];
}

export interface TurnResponse {
  game_id: string;
  status: GameStatus;
  active_player_id: number | null;
  active_player_name: string | null;
  requires_handoff: boolean;
  public_table: PublicTable;
  private_hand: number[] | null;
  legal_actions: LegalAction[];
  results?: GameResults | null;
  round_history?: RoundRecord[] | null;
  error?: string | null;
}

export interface SubmitActionRequest {
  player_id: number;
  action: LegalAction;
}
