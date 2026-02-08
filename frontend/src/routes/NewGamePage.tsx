import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createGame, type SeatSpec } from "../api/client";
import Divider from "../components/Divider";
import styles from "./NewGamePage.module.css";

const SEAT_TYPES = ["human", "easy", "medium", "hard", "expert"] as const;
const SEAT_LABELS: Record<string, string> = {
  human: "Human",
  easy: "Easy Bot",
  medium: "Medium Bot",
  hard: "Hard Bot",
  expert: "Expert Bot",
};

const BOT_NAMES = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"];

function getNextBotName(existingSeats: SeatSpec[]): string {
  const usedNames = new Set(existingSeats.map((s) => s.name));
  for (const name of BOT_NAMES) {
    const fullName = `Bot ${name}`;
    if (!usedNames.has(fullName)) {
      return fullName;
    }
  }
  // Fallback to numbered bots if all names are taken
  let i = 1;
  while (usedNames.has(`Bot ${i}`)) i++;
  return `Bot ${i}`;
}

const DEFAULT_SEATS: SeatSpec[] = [
  { type: "human", name: "Player 1" },
  { type: "medium", name: "Bot Alpha" },
  { type: "medium", name: "Bot Beta" },
];

export default function NewGamePage() {
  const navigate = useNavigate();
  const [seats, setSeats] = useState<SeatSpec[]>(DEFAULT_SEATS);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateSeat(index: number, patch: Partial<SeatSpec>) {
    setSeats((prev) =>
      prev.map((s, i) => (i === index ? { ...s, ...patch } : s))
    );
  }

  async function handleStart() {
    setError(null);
    setLoading(true);
    try {
      const cleanSeats = seats.map((s) => ({
        type: s.type,
        ...(s.name ? { name: s.name } : {}),
      }));
      const { game_id } = await createGame(cleanSeats);
      navigate(`/game/${game_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create game");
      setLoading(false);
    }
  }

  function addSeat() {
    if (seats.length < 5) {
      setSeats((prev) => [...prev, { type: "medium", name: getNextBotName(prev) }]);
    }
  }

  function removeSeat(index: number) {
    if (seats.length > 3) {
      setSeats((prev) => prev.filter((_, i) => i !== index));
    }
  }

  return (
    <div className={styles.container}>
      <h1 className={styles.heading}>Set Up Game</h1>
      <Divider />

      <div className={styles.seatList}>
        {seats.map((seat, i) => (
          <div key={i} className={styles.seatRow}>
            <select
              className={styles.typeSelect}
              value={seat.type}
              onChange={(e) =>
                updateSeat(i, {
                  type: e.target.value as SeatSpec["type"],
                })
              }
            >
              {SEAT_TYPES.map((t) => (
                <option key={t} value={t}>
                  {SEAT_LABELS[t]}
                </option>
              ))}
            </select>
            <input
              className={styles.nameInput}
              type="text"
              placeholder={seat.type === "human" ? `Player ${i + 1}` : "Name (optional)"}
              value={seat.name ?? ""}
              onChange={(e) => updateSeat(i, { name: e.target.value })}
            />
            {seats.length > 3 && (
              <button
                className={styles.removeSeatBtn}
                onClick={() => removeSeat(i)}
                aria-label="Remove seat"
              >
                &times;
              </button>
            )}
          </div>
        ))}

        {seats.length < 5 && (
          <button className={styles.addSeatBtn} onClick={addSeat}>
            + Add Seat
          </button>
        )}
      </div>

      {error && <p className={styles.error}>{error}</p>}

      <div className={styles.actions}>
        <button
          className={styles.backBtn}
          onClick={() => navigate("/")}
          disabled={loading}
        >
          Back
        </button>
        <button
          className={styles.startBtn}
          onClick={handleStart}
          disabled={loading}
        >
          {loading ? "Starting..." : "Start Game"}
        </button>
      </div>
    </div>
  );
}
