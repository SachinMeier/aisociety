import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  getTurn,
  submitAction,
  type TurnResponse,
  type LegalAction,
} from "../api/client";
import TableLayout from "../components/TableLayout";
import HandoffOverlay from "../components/HandoffOverlay";
import RoundWinnerOverlay from "../components/RoundWinnerOverlay";
import ActionPanel from "../components/ActionPanel";
import { HelpButton, HelpOverlay } from "../components/HelpOverlay";
import ConfirmDialog from "../components/ConfirmDialog";
import ChevronIcon from "../components/ChevronIcon";
import styles from "./GameTablePage.module.css";

const POLL_INTERVAL_MS = 1000;
const ROUND_WINNER_DISPLAY_MS = 4000;

export interface RoundWinnerInfo {
  winnerName: string;
  coins: number[];
  cardLabel: string;
}

function formatCardLabel(card: { kind: string; value?: number; misfortune?: string } | null): string {
  if (!card) return "the card";
  if (card.kind === "possession" && card.value != null) return `Possession ${card.value}`;
  if (card.kind === "title") return "the Title (2\u00d7)";
  if (card.kind === "misfortune" && card.misfortune) {
    switch (card.misfortune) {
      case "scandal": return "Scandal (\u00bd\u00d7)";
      case "debt": return "Debt (\u22125)";
      case "theft": return "Theft (Cancel)";
    }
  }
  return "the card";
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

  function detectRoundWinner(prevTurn: TurnResponse | null, newTurn: TurnResponse) {
    if (!prevTurn) return;

    const prevLen = prevTurn.round_history?.length ?? 0;
    const newHistory = newTurn.round_history ?? [];
    if (newHistory.length <= prevLen) return;

    // A new round was completed — show the most recent one
    const lastRound = newHistory[newHistory.length - 1];

    setRoundWinner({
      winnerName: lastRound.winner_name,
      coins: lastRound.coins_spent,
      cardLabel: formatCardLabel(lastRound.card),
    });

    setTimeout(() => setRoundWinner(null), ROUND_WINNER_DISPLAY_MS);
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
      setError(e instanceof Error ? e.message : "Failed to fetch game state");
    }
  }, [gameId, applyTurn]);

  // Initial fetch + poll when bots are thinking
  useEffect(() => {
    fetchTurn();

    const interval = setInterval(() => {
      if (turn?.status === "active") {
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

  const handleAction = async (action: LegalAction) => {
    if (!gameId || !turn || turn.active_player_id == null || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const data = await submitAction(gameId, turn.active_player_id, action);
      applyTurn(data);
      setHandoffRevealed(false);
      // Don't navigate immediately on finish - let the roundWinner overlay show first
      // Navigation is handled by a separate useEffect
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to submit action");
    } finally {
      setSubmitting(false);
    }
  };

  if (!turn) {
    return (
      <div className={styles.container}>
        <p className={styles.loading}>Loading game...</p>
        {error && <p className={styles.error}>{error}</p>}
      </div>
    );
  }

  // Round winner notification gate: show auction result for ALL completed auctions.
  // This ensures the winner is displayed for ~3 seconds regardless of game mode.
  if (roundWinner) {
    return <RoundWinnerOverlay roundWinner={roundWinner} />;
  }

  // Handoff gate: hide private info until player confirms
  if (turn.requires_handoff && !handoffRevealed) {
    return (
      <HandoffOverlay
        playerName={turn.active_player_name ?? "Unknown"}
        onReveal={() => setHandoffRevealed(true)}
      />
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.gameArea}>
        {error && <p className={styles.error}>{error}</p>}

        <TableLayout
          publicTable={turn.public_table}
          activePlayerId={turn.active_player_id}
          roundWinner={roundWinner}
        />
      </div>

      <div className={`${styles.sidePanel} ${sidebarCollapsed ? styles.sidePanelCollapsed : ""}`}>
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
