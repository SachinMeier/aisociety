import type { PublicTable } from "../types";
import type { RoundWinnerInfo } from "../routes/GameTablePage";
import PlayerSeat from "./PlayerSeat";
import StatusCard from "./StatusCard";
import Coin from "./MoneyCard";
import { SEAT_LAYOUTS } from "./seatLayout";

interface TableLayoutProps {
  publicTable: PublicTable;
  activePlayerId: number | null;
  roundWinner?: RoundWinnerInfo | null;
}

// Minimum container dimensions
const MIN_CONTAINER_WIDTH = 900;
const MIN_CONTAINER_HEIGHT = 700;

export default function TableLayout({
  publicTable,
  activePlayerId,
  roundWinner,
}: TableLayoutProps) {
  const { players, status_card, round, revealed_status_cards } = publicTable;
  const playerCount = players.length;

  // Get the layout for this player count, fallback to 5-player layout
  const layout = SEAT_LAYOUTS[playerCount] ?? SEAT_LAYOUTS[5];

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        maxWidth: 1600,
        maxHeight: 1200,
        minWidth: MIN_CONTAINER_WIDTH,
        minHeight: MIN_CONTAINER_HEIGHT,
        margin: "0 auto",
      }}
    >
      {/* Center table content - shifted up slightly to balance with lowered bottom seats */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: "calc(50% - 20px)",
          transform: "translate(-50%, -50%)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 10,
          zIndex: 1,
        }}
      >
        {/* Label */}
        <span
          style={{
            fontSize: 11,
            color: "rgba(176, 200, 160, 0.7)",
            textTransform: "uppercase",
            letterSpacing: 2,
            fontWeight: 600,
            fontFamily: "'Georgia', serif",
          }}
        >
          Up for Auction
        </span>

        {/* Current status card */}
        {status_card != null ? (
          <StatusCard card={status_card} size="large" />
        ) : (
          <div
            style={{
              width: 150,
              height: 210,
              borderRadius: 14,
              border: "2px dashed rgba(255,255,255,0.15)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              color: "rgba(128, 128, 112, 0.6)",
              fontStyle: "italic",
              fontFamily: "'Georgia', serif",
            }}
          >
            No card
          </div>
        )}

        {/* Bid info */}
        {round != null && round.highest_bid > 0 && (
          <div
            style={{
              fontSize: 16,
              color: "#e0d8b0",
              textAlign: "center",
              lineHeight: 1.4,
              fontFamily: "'Georgia', serif",
              marginTop: 2,
            }}
          >
            <span style={{ fontSize: 11, color: "rgba(176, 200, 160, 0.6)", textTransform: "uppercase", letterSpacing: 1 }}>
              Leading bid
            </span>
            <div>
              <strong style={{ color: "#ffd700", fontSize: 20 }}>
                ${round.highest_bid.toLocaleString()}
              </strong>
              <span style={{ fontSize: 14, opacity: 0.7, marginLeft: 6 }}>
                by{" "}
                {players.find((p) => p.id === round.highest_bidder)?.name ??
                  `Player ${round.highest_bidder}`}
              </span>
            </div>
          </div>
        )}

        {/* Revealed (discarded) status cards */}
        {revealed_status_cards.length > 0 && (
          <div
            style={{
              display: "flex",
              gap: 4,
              marginTop: 4,
              opacity: 0.7,
            }}
          >
            {revealed_status_cards.map((card, i) => (
              <StatusCard key={i} card={card} size="small" />
            ))}
          </div>
        )}
      </div>

      {/* Round winner overlay */}
      {roundWinner && (
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 10,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 12,
            backgroundColor: "rgba(0, 0, 0, 0.88)",
            borderRadius: 20,
            padding: "24px 36px",
            border: "2px solid rgba(255,215,0,0.5)",
            boxShadow: "0 0 30px rgba(255,215,0,0.3)",
            animation: "fadeIn 0.3s ease",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: "bold",
              color: "#ffd700",
              fontFamily: "'Georgia', serif",
              textAlign: "center",
            }}
          >
            {roundWinner.winnerName} won {roundWinner.cardLabel}!
          </div>
          {roundWinner.coins.length > 0 && (
            <>
              <div
                style={{
                  fontSize: 13,
                  color: "#a0a890",
                  textTransform: "uppercase",
                  letterSpacing: 1,
                  fontFamily: "'Georgia', serif",
                }}
              >
                Coins spent
              </div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center" }}>
                {roundWinner.coins
                  .slice()
                  .sort((a, b) => a - b)
                  .map((v, i) => (
                    <Coin key={i} value={v} size="small" />
                  ))}
              </div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: "bold",
                  color: "#e0d8b0",
                  fontFamily: "'Georgia', serif",
                }}
              >
                Total: ${roundWinner.coins.reduce((s, v) => s + v, 0).toLocaleString()}
              </div>
            </>
          )}
        </div>
      )}

      {/* Player seats */}
      {players.map((player, index) => (
        <PlayerSeat
          key={player.id}
          player={player}
          isActive={player.id === activePlayerId}
          isHighestBidder={round != null && player.id === round.highest_bidder}
          position={layout[index]}
        />
      ))}
    </div>
  );
}
