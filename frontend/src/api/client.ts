// ---- API Client for High Society local play ----

import type {
  TurnResponse,
  LegalAction,
} from "../types";

export type { TurnResponse, LegalAction };
export type { GameResults, GameStatus, StatusCardData, PublicTable } from "../types";

export interface SeatSpec {
  type: "human" | "easy" | "medium" | "hard" | "expert";
  name?: string;
}

export interface CreateGameRequest {
  seats: SeatSpec[];
  seed?: number | null;
}

export interface CreateGameResponse {
  game_id: string;
}

// ---- API Client Functions ----

async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function createGame(seats: SeatSpec[]): Promise<CreateGameResponse> {
  return apiFetch<CreateGameResponse>("/api/local-games", {
    method: "POST",
    body: JSON.stringify({ seats } satisfies CreateGameRequest),
  });
}

export async function getGameState(gameId: string): Promise<TurnResponse> {
  return apiFetch<TurnResponse>(`/api/local-games/${gameId}`);
}

export async function getTurn(gameId: string): Promise<TurnResponse> {
  return apiFetch<TurnResponse>(`/api/local-games/${gameId}/turn`);
}

export async function submitAction(
  gameId: string,
  playerId: number,
  action: LegalAction
): Promise<TurnResponse> {
  return apiFetch<TurnResponse>(`/api/local-games/${gameId}/actions`, {
    method: "POST",
    body: JSON.stringify({ player_id: playerId, action }),
  });
}
