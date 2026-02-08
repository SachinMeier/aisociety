import type { PlayerData } from "../types";
import type { SeatPosition } from "./seatLayout";
import { ANCHOR_TRANSFORMS } from "./seatLayout";
import StatusCard from "./StatusCard";
import Coin from "./MoneyCard";

interface PlayerSeatProps {
  player: PlayerData;
  isActive: boolean;
  isHighestBidder: boolean;
  position: SeatPosition;
}

// Fixed heights to prevent layout jitter when content changes
// Sized for up to 5 coins (36px each + gaps) and 5 status cards (60px each + gaps)
const BID_AREA_HEIGHT = 90;
const POSSESSIONS_AREA_HEIGHT = 100;
const SEAT_WIDTH = 340;

export default function PlayerSeat({
  player,
  isActive,
  isHighestBidder,
  position,
}: PlayerSeatProps) {

  const totalBid = player.open_bid.reduce((sum, v) => sum + v, 0);
  const hasPassed = player.money_count === -1;

  return (
    <div
      style={{
        position: "absolute",
        left: position.left,
        top: position.top,
        transform: ANCHOR_TRANSFORMS[position.anchor],
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
        width: SEAT_WIDTH,
        minWidth: SEAT_WIDTH,
      }}
    >
      {/* Name plate - stable position at top of fixed-size container */}
      <div
        style={{
          position: "relative",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          backgroundColor: isActive
            ? "rgba(212, 175, 55, 0.95)"
            : hasPassed
            ? "rgba(80, 80, 70, 0.85)"
            : "rgba(255, 255, 255, 0.92)",
          borderRadius: 20,
          padding: "7px 18px",
          boxShadow: isActive
            ? "0 0 20px rgba(255,215,0,0.4), 0 2px 8px rgba(0,0,0,0.3)"
            : "0 2px 8px rgba(0,0,0,0.2)",
          border: isActive
            ? "2px solid #e8c840"
            : hasPassed
            ? "2px solid rgba(100,100,90,0.5)"
            : "2px solid rgba(200,200,190,0.4)",
          transition: "all 0.2s ease",
          flexShrink: 0,
        }}
      >
        {/* Active indicator dot */}
        {isActive && (
          <div style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            backgroundColor: "#3a2e00",
            flexShrink: 0,
            animation: "pulse 1.5s ease-in-out infinite",
          }} />
        )}

        <span
          style={{
            fontWeight: "bold",
            fontSize: 17,
            color: isActive ? "#3a2e00" : hasPassed ? "#999" : "#1a1a1a",
            whiteSpace: "nowrap",
            fontFamily: "'Georgia', serif",
            letterSpacing: 0.3,
            textDecoration: hasPassed ? "line-through" : "none",
          }}
        >
          {player.name}
        </span>

        {hasPassed && (
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: "#888",
              textTransform: "uppercase",
              letterSpacing: 1,
              fontFamily: "'Georgia', serif",
            }}
          >
            OUT
          </span>
        )}
      </div>

      {/* Possessions area - fixed height container to prevent layout shift */}
      <div
        style={{
          minHeight: POSSESSIONS_AREA_HEIGHT,
          width: "100%",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {player.owned_status_cards.length > 0 && (
          <div style={{
            display: "flex",
            gap: 4,
            flexWrap: "wrap",
            justifyContent: "center",
          }}>
            {player.owned_status_cards.map((card, i) => (
              <StatusCard key={i} card={card} size="small" />
            ))}
          </div>
        )}
      </div>

      {/* Bid area - fixed height container to prevent layout shift */}
      <div
        style={{
          minHeight: BID_AREA_HEIGHT,
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
          flexShrink: 0,
        }}
      >
        {totalBid > 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 4,
              padding: "6px 10px",
              borderRadius: 10,
              backgroundColor: isHighestBidder
                ? "rgba(255,215,0,0.12)"
                : "transparent",
              border: isHighestBidder
                ? "1.5px solid rgba(255,215,0,0.4)"
                : "1.5px solid transparent",
            }}
          >
            <div style={{
              display: "flex",
              gap: 4,
              flexWrap: "wrap",
              justifyContent: "center",
            }}>
              {player.open_bid
                .slice()
                .sort((a, b) => a - b)
                .map((v, i) => (
                  <Coin key={i} value={v} size="tiny" />
                ))}
            </div>
            <div
              style={{
                color: isHighestBidder ? "#ffd700" : "#c0b890",
                fontWeight: "bold",
                fontSize: 14,
                fontFamily: "'Georgia', serif",
                textShadow: isHighestBidder ? "0 0 6px rgba(255,215,0,0.3)" : "none",
              }}
            >
              ${totalBid.toLocaleString()}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
