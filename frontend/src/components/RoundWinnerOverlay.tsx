import type { RoundWinnerInfo } from "../routes/GameTablePage";
import Coin from "./MoneyCard";

interface RoundWinnerOverlayProps {
  roundWinner: RoundWinnerInfo;
}

/**
 * Full-screen overlay shown when a round ends and a handoff is pending.
 * Displays who won the auction without exposing any private player info,
 * then auto-clears (timer managed by GameTablePage) to reveal the handoff screen.
 */
export default function RoundWinnerOverlay({ roundWinner }: RoundWinnerOverlayProps) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(10, 20, 10, 0.95)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        fontFamily: "'Georgia', serif",
        animation: "fadeInCenter 0.3s ease",
      }}
    >
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 16,
          backgroundColor: "rgba(0, 0, 0, 0.6)",
          borderRadius: 20,
          padding: "32px 48px",
          border: "2px solid rgba(255,215,0,0.5)",
          boxShadow: "0 0 40px rgba(255,215,0,0.2)",
        }}
      >
        <div
          style={{
            fontSize: 16,
            color: "#a0a890",
            textTransform: "uppercase",
            letterSpacing: 2,
          }}
        >
          Auction Complete
        </div>

        <div
          style={{
            fontSize: 28,
            fontWeight: "bold",
            color: "#ffd700",
            textAlign: "center",
            textShadow: "0 2px 8px rgba(255,215,0,0.3)",
          }}
        >
          {roundWinner.winnerName} won {roundWinner.cardLabel}!
        </div>

        {roundWinner.coins.length > 0 && (
          <>
            <div
              style={{
                fontSize: 14,
                color: "#a0a890",
                textTransform: "uppercase",
                letterSpacing: 1,
                marginTop: 4,
              }}
            >
              Coins spent
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
              {roundWinner.coins
                .slice()
                .sort((a, b) => a - b)
                .map((v, i) => (
                  <Coin key={i} value={v} size="small" />
                ))}
            </div>
            <div
              style={{
                fontSize: 22,
                fontWeight: "bold",
                color: "#e0d8b0",
              }}
            >
              Total: ${roundWinner.coins.reduce((s, v) => s + v, 0).toLocaleString()}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
