import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  advanceBot,
  getTurn,
  isApiError,
  submitAction,
  type TurnResponse,
  type LegalAction,
} from "../api/client";
import type { StatusCardData } from "../types";
import TableLayout from "../components/TableLayout";
import HandoffOverlay from "../components/HandoffOverlay";
import ActionPanel from "../components/ActionPanel";
import { HelpButton, HelpOverlay } from "../components/HelpOverlay";
import ConfirmDialog from "../components/ConfirmDialog";
import ChevronIcon from "../components/ChevronIcon";
import styles from "./GameTablePage.module.css";

const POLL_INTERVAL_MS = 1000;
const ROUND_WINNER_DISPLAY_MS = 3000;
const BOT_ACTION_DELAY_MS = 1500;

export interface RoundWinnerInfo {
  winnerName: string;
  coins: number[];
  cardLabel: string;
  card: StatusCardData;
}

function cardIdentity(card: StatusCardData | null | undefined): string {
  if (!card) return "none";
  return `${card.kind}:${card.value ?? ""}:${card.misfortune ?? ""}`;
}

function ownedCardsIdentity(cards: StatusCardData[]): string {
  return cards
    .map((card) => cardIdentity(card))
    .sort()
    .join("|");
}

function formatCardLabel(card: { kind: string; value?: number; misfortune?: string } | null): string {
  if (!card) return "the card";
  if (card.kind === "possession" && card.value != null) return `${card.value}`;
  if (card.kind === "title") return "Title";
  if (card.kind === "misfortune" && card.misfortune) {
    switch (card.misfortune) {
      case "scandal": return "Scandal";
      case "debt": return "Debt";
      case "theft": return "Theft";
    }
  }
  return "the card";
}

function isDiscardAction(
  action: LegalAction
): action is LegalAction & { kind: "discard_possession"; possession_value: number } {
  return action.kind === "discard_possession" && action.possession_value != null;
}

function detectRoundWinnerFromHistory(
  prevTurn: TurnResponse,
  newTurn: TurnResponse
): RoundWinnerInfo | null {
  const prevLen = prevTurn.round_history?.length ?? 0;
  const newHistory = newTurn.round_history ?? [];
  if (newHistory.length <= prevLen) return null;

  const lastRound = newHistory[newHistory.length - 1];
  if (!lastRound?.card) return null;

  return {
    winnerName: lastRound.winner_name || `Player ${lastRound.winner_id}`,
    coins: lastRound.coins_spent,
    cardLabel: formatCardLabel(lastRound.card),
    card: lastRound.card,
  };
}

function detectRoundWinnerFromTableDiff(
  prevTurn: TurnResponse,
  newTurn: TurnResponse
): RoundWinnerInfo | null {
  const prevCard = prevTurn.public_table.status_card;
  if (!prevCard) return null;
  if (cardIdentity(prevCard) === cardIdentity(newTurn.public_table.status_card)) {
    return null;
  }

  const prevPlayersById = new Map(prevTurn.public_table.players.map((p) => [p.id, p]));
  const changedOwners = newTurn.public_table.players.filter((p) => {
    const prev = prevPlayersById.get(p.id);
    if (!prev) return false;
    return ownedCardsIdentity(prev.owned_status_cards) !== ownedCardsIdentity(p.owned_status_cards);
  });

  const winner =
    changedOwners.length === 1
      ? changedOwners[0]
      : (
          prevTurn.public_table.round?.highest_bidder != null
            ? newTurn.public_table.players.find(
                (p) => p.id === prevTurn.public_table.round?.highest_bidder
              ) ?? null
            : null
        );
  if (!winner) return null;

  const prevWinner = prevPlayersById.get(winner.id);
  const coins = prevCard.kind === "misfortune" ? [] : (prevWinner?.open_bid ?? []);

  return {
    winnerName: winner.name || `Player ${winner.id}`,
    coins,
    cardLabel: formatCardLabel(prevCard),
    card: prevCard,
  };
}

export default function GameTablePage() {
  const { id: gameId } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [turn, setTurn] = useState<TurnResponse | null>(null);
  const [handoffRevealed, setHandoffRevealed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roundWinner, setRoundWinner] = useState<RoundWinnerInfo | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [showQuitConfirm, setShowQuitConfirm] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Track the previous turn to detect round transitions
  const prevTurnRef = useRef<TurnResponse | null>(null);
  const roundWinnerTimeoutRef = useRef<number | null>(null);
  // Suppress polling while the bot-animation loop is running
  const animatingRef = useRef(false);

  function detectRoundWinner(prevTurn: TurnResponse | null, newTurn: TurnResponse) {
    if (!prevTurn) return;
    const winnerInfo =
      detectRoundWinnerFromHistory(prevTurn, newTurn) ??
      detectRoundWinnerFromTableDiff(prevTurn, newTurn);
    if (!winnerInfo) return;

    setRoundWinner(winnerInfo);

    if (roundWinnerTimeoutRef.current != null) {
      window.clearTimeout(roundWinnerTimeoutRef.current);
    }
    roundWinnerTimeoutRef.current = window.setTimeout(() => {
      setRoundWinner(null);
      roundWinnerTimeoutRef.current = null;
    }, ROUND_WINNER_DISPLAY_MS);
  }

  const applyTurn = useCallback(
    (data: TurnResponse) => {
      detectRoundWinner(prevTurnRef.current, data);
      prevTurnRef.current = data;
      setTurn(data);
    },
    []
  );

  const fetchTurn = useCallback(async () => {
    if (!gameId) return;
    try {
      const data = await getTurn(gameId);
      applyTurn(data);
      setError(null);

      // Don't navigate immediately on finish - let the roundWinner overlay show first
      // Navigation is handled by a separate useEffect

      // Reset handoff reveal when active player changes
      if (data.requires_handoff) {
        setHandoffRevealed(false);
      }
    } catch (e) {
      if (isApiError(e) && e.status === 404) {
        navigate("/", { replace: true });
        return;
      }
      setError(e instanceof Error ? e.message : "Failed to fetch game state");
    }
  }, [gameId, applyTurn, navigate]);

  // Initial fetch + poll when bots are thinking
  useEffect(() => {
    fetchTurn();

    const interval = setInterval(() => {
      if (turn?.status === "active" && !animatingRef.current) {
        fetchTurn();
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [fetchTurn, turn?.status]);

  // Navigate to results page when game is finished AND no roundWinner overlay is showing
  useEffect(() => {
    if (turn?.status === "finished" && !roundWinner && gameId) {
      navigate(`/game/${gameId}/result`, { state: { turn } });
    }
  }, [turn, roundWinner, gameId, navigate]);

  const handleAction = useCallback(async (action: LegalAction) => {
    if (!gameId || !turn || turn.active_player_id == null || submitting) return;
    setSubmitting(true);
    animatingRef.current = true;
    setError(null);
    try {
      let data = await submitAction(gameId, turn.active_player_id, action);
      applyTurn(data);
      setHandoffRevealed(false);

      // Animate bot turns one at a time
      while (data.status === "active") {
        await new Promise((r) => setTimeout(r, BOT_ACTION_DELAY_MS));
        data = await advanceBot(gameId);
        applyTurn(data);
      }
    } catch (e) {
      if (isApiError(e) && e.status === 404) {
        navigate("/", { replace: true });
        return;
      }
      setError(e instanceof Error ? e.message : "Failed to submit action");
    } finally {
      animatingRef.current = false;
      setSubmitting(false);
    }
  }, [gameId, turn, submitting, applyTurn, navigate]);

  useEffect(() => {
    if (!turn || turn.status !== "awaiting_human_action" || submitting) return;
    const discardActions = turn.legal_actions.filter(isDiscardAction);
    const discardOnly =
      discardActions.length > 0 && discardActions.length === turn.legal_actions.length;
    if (!discardOnly) return;
    const lowestDiscard = discardActions.reduce((lowest, action) =>
      action.possession_value < lowest.possession_value ? action : lowest
    );
    void handleAction(lowestDiscard);
  }, [turn, submitting, handleAction]);

  useEffect(() => {
    return () => {
      if (roundWinnerTimeoutRef.current != null) {
        window.clearTimeout(roundWinnerTimeoutRef.current);
      }
    };
  }, []);

  if (!turn) {
    return (
      <div className={styles.container}>
        <p className={styles.loading}>Loading game...</p>
        {error && <p className={styles.error}>{error}</p>}
      </div>
    );
  }

  const handoffPending = turn.requires_handoff && !handoffRevealed;
  // Handoff gate: hide private info until player confirms
  if (handoffPending && !roundWinner) {
    return (
      <HandoffOverlay
        playerName={turn.active_player_name ?? "Unknown"}
        onReveal={() => setHandoffRevealed(true)}
      />
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.gameArea} data-testid="game-area">
        {error && <p className={styles.error}>{error}</p>}

        <TableLayout
          publicTable={turn.public_table}
          activePlayerId={turn.active_player_id}
          roundWinner={roundWinner}
        />
      </div>

      {!handoffPending && (
        <div data-testid="side-panel" className={`${styles.sidePanel} ${sidebarCollapsed ? styles.sidePanelCollapsed : ""}`}>
          <button
            className={styles.collapseToggle}
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            <ChevronIcon direction={sidebarCollapsed ? "down" : "up"} size={18} />
            {sidebarCollapsed && <span className={styles.collapseLabel}>Actions</span>}
          </button>
          {!sidebarCollapsed && (
            <>
              <div className={styles.quitRow}>
                <HelpButton onClick={() => setShowHelp(true)} />
                <button
                  className={styles.quitBtn}
                  onClick={() => setShowQuitConfirm(true)}
                >
                  Quit
                </button>
              </div>
              <ActionPanel
                legalActions={turn.legal_actions}
                privateHand={turn.private_hand}
                currentBid={
                  turn.active_player_id != null
                    ? (turn.public_table.players.find(
                        (p) => p.id === turn.active_player_id
                      )?.open_bid ?? [])
                    : []
                }
                highestBid={turn.public_table.round?.highest_bid ?? 0}
                onSubmit={handleAction}
              />
            </>
          )}
        </div>
      )}

      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}
      {showQuitConfirm && (
        <ConfirmDialog
          title="Leave this game?"
          message="Your progress will be lost and you'll return to the home screen."
          confirmLabel="Quit"
          cancelLabel="Stay"
          onConfirm={() => navigate("/")}
          onCancel={() => setShowQuitConfirm(false)}
        />
      )}
    </div>
  );
}
