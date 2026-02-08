import { useLocation, useNavigate, useParams } from "react-router-dom";
import { useEffect, useState } from "react";
import { getGameState, type TurnResponse } from "../api/client";
import { formatMoney } from "../components/MoneyCard";
import PaperCard from "../components/PaperCard";
import styles from "./ResultPage.module.css";

export default function ResultPage() {
  const { id: gameId } = useParams<{ id: string }>();
  const location = useLocation();
  const navigate = useNavigate();

  const passedTurn = (location.state as { turn?: TurnResponse })?.turn;
  const [turn, setTurn] = useState<TurnResponse | null>(passedTurn ?? null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (turn) return;
    if (!gameId) return;

    getGameState(gameId)
      .then(setTurn)
      .catch((e) =>
        setError(e instanceof Error ? e.message : "Failed to load results")
      );
  }, [gameId, turn]);

  if (error) {
    return (
      <div className={styles.container}>
        <p className={styles.error}>{error}</p>
        <button className={styles.secondaryBtn} onClick={() => navigate("/")}>
          Back to Home
        </button>
      </div>
    );
  }

  if (!turn || !turn.results) {
    return (
      <div className={styles.container}>
        <p className={styles.loading}>Loading results...</p>
      </div>
    );
  }

  const { results, public_table } = turn;
  const players = public_table.players;

  // Sort by score descending, but eliminated players go last
  const scoreEntries = players
    .map((p) => ({
      id: p.id,
      name: p.name,
      score: results.scores[String(p.id)] ?? 0,
      moneyRemaining: results.money_remaining[String(p.id)] ?? 0,
      eliminated: results.poorest.includes(p.id),
      isWinner: results.winners.includes(p.id),
    }))
    .sort((a, b) => {
      // Eliminated players always go last
      if (a.eliminated && !b.eliminated) return 1;
      if (!a.eliminated && b.eliminated) return -1;
      // Otherwise sort by score
      return b.score - a.score;
    });

  const winnerNames = scoreEntries
    .filter((e) => e.isWinner)
    .map((e) => e.name)
    .join(" & ");

  // Pre-compute placements (winners excluded, they get stars)
  let rank = 1;
  const placements = scoreEntries.map((entry) => {
    if (entry.eliminated) return null;
    if (entry.isWinner) {
      rank++; // Winner takes rank 1, increment for next
      return null; // Star instead of number
    }
    return rank++;
  });

  return (
    <div className={styles.container}>
      {/* Winner announcement */}
      <div className={styles.winnerSection}>
        <div className={styles.crownIcon}>&#9813;</div>
        <h1 className={styles.winnerName}>{winnerNames || "No winner"} wins!</h1>
      </div>

      {/* Scoreboard on paper card */}
      <PaperCard className={styles.paperCard}>
        <div className={styles.cardHeader}>Final Standings</div>
        <div className={styles.standingsList}>
          {scoreEntries.map((p, i) => (
            <div
              key={p.id}
              className={`${styles.standingRow} ${p.eliminated ? styles.eliminated : ""} ${p.isWinner ? styles.winner : ""}`}
            >
              <div className={styles.rankBadge}>
                {p.isWinner ? "\u2605" : p.eliminated ? "\u2717" : placements[i]}
              </div>
              <div className={styles.playerName}>{p.name}</div>
              <div className={styles.statsCol}>
                {p.eliminated ? (
                  <div className={styles.eliminatedLabel}>Poorest</div>
                ) : (
                  <div className={styles.scoreValue}>{p.score} pts</div>
                )}
                <div className={styles.moneyValue}>{formatMoney(p.moneyRemaining)}</div>
              </div>
            </div>
          ))}
        </div>
      </PaperCard>

      {/* Actions */}
      <div className={styles.actions}>
        <button className={styles.secondaryBtn} onClick={() => navigate("/")}>
          Home
        </button>
        <button className={styles.primaryBtn} onClick={() => navigate("/new")}>
          Play Again
        </button>
      </div>
    </div>
  );
}
