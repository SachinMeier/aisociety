import { useState, useCallback, useEffect } from "react";
import type { LegalAction } from "../types";
import Coin from "./MoneyCard";
import Divider from "./Divider";

interface ActionPanelProps {
  legalActions: LegalAction[];
  privateHand: number[] | null;
  currentBid?: number[];
  highestBid?: number;
  onSubmit: (action: LegalAction) => void;
}

function formatMoney(value: number): string {
  if (value >= 1000) {
    return `$${(value / 1000).toFixed(0)}k`;
  }
  return `$${value}`;
}

/**
 * Check whether a selected set of cards matches any legal bid action.
 * Cards are compared as sorted arrays.
 */
function findMatchingBid(
  selectedCards: number[],
  legalActions: LegalAction[]
): LegalAction | null {
  const sorted = selectedCards.slice().sort((a, b) => a - b);
  for (const action of legalActions) {
    if (action.kind !== "bid" || !action.cards) continue;
    const actionSorted = action.cards.slice().sort((a, b) => a - b);
    if (actionSorted.length !== sorted.length) continue;
    if (actionSorted.every((v, i) => v === sorted[i])) {
      return action;
    }
  }
  return null;
}

const ALL_DENOMINATIONS = [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 25000];

export default function ActionPanel({
  legalActions,
  privateHand,
  currentBid = [],
  highestBid = 0,
  onSubmit,
}: ActionPanelProps) {
  const [selectedValues, setSelectedValues] = useState<Set<number>>(
    new Set()
  );
  const [error, setError] = useState<string | null>(null);

  const canPass = legalActions.some((a) => a.kind === "pass");
  const discardActions = legalActions.filter(
    (a) => a.kind === "discard_possession" && a.possession_value != null
  );
  const hasBidActions = legalActions.some((a) => a.kind === "bid");

  const handSet = new Set(privateHand ?? []);

  const toggleCard = useCallback(
    (value: number) => {
      setError(null);
      setSelectedValues((prev) => {
        const next = new Set(prev);
        if (next.has(value)) {
          next.delete(value);
        } else {
          next.add(value);
        }
        return next;
      });
    },
    []
  );

  const selectedCards = ALL_DENOMINATIONS.filter((v) => selectedValues.has(v) && handSet.has(v));
  const selectedTotal = selectedCards.reduce((s, v) => s + v, 0);
  const existingBidTotal = currentBid.reduce((s, v) => s + v, 0);
  const newBidTotal = existingBidTotal + selectedTotal;
  const beatsHighest = selectedCards.length > 0 && newBidTotal > highestBid;

  function handleBid() {
    if (selectedCards.length === 0) {
      setError("Select at least one coin to bid.");
      return;
    }
    const match = findMatchingBid(selectedCards, legalActions);
    if (match) {
      onSubmit(match);
      setSelectedValues(new Set());
      setError(null);
    } else {
      setError(
        `Bid of ${selectedCards.map(formatMoney).join(" + ")} (${formatMoney(selectedTotal)}) is not a valid action.`
      );
    }
  }

  function handlePass() {
    setError(null);
    setSelectedValues(new Set());
    onSubmit({ kind: "pass" });
  }

  function handleDiscard(action: LegalAction) {
    setError(null);
    setSelectedValues(new Set());
    onSubmit(action);
  }

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.code === "Space" && canPass) {
        e.preventDefault();
        handlePass();
      } else if (e.code === "Enter" && hasBidActions && selectedCards.length > 0) {
        e.preventDefault();
        handleBid();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  return (
    <div
      style={{
        backgroundColor: "rgba(20, 30, 20, 0.9)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 12,
        padding: "14px 10px",
        flex: 1,
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
      }}
    >
      {/* Error toast */}
      {error && (
        <div
          style={{
            backgroundColor: "rgba(180, 40, 40, 0.9)",
            color: "#fff",
            padding: "10px 16px",
            borderRadius: 8,
            marginBottom: 14,
            fontSize: 14,
            fontFamily: "'Georgia', serif",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span>{error}</span>
          <button
            onClick={() => setError(null)}
            style={{
              background: "none",
              border: "none",
              color: "#fff",
              fontSize: 18,
              cursor: "pointer",
              padding: "0 0 0 12px",
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Coins — always show all denominations, grey out spent ones */}
      {privateHand != null && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
            {ALL_DENOMINATIONS.map((value) => {
              const available = handSet.has(value);
              const isSelected = selectedValues.has(value) && available;
              return (
                <Coin
                  key={value}
                  value={value}
                  spent={!available}
                  selected={isSelected}
                  selectedWeak={isSelected && !beatsHighest}
                  onClick={available && hasBidActions ? () => toggleCard(value) : undefined}
                />
              );
            })}
          </div>
          {selectedCards.length > 0 && (
            <div
              style={{
                marginTop: 8,
                fontSize: 13,
                color: beatsHighest ? "#4caf50" : "#999",
                fontFamily: "'Georgia', serif",
              }}
            >
              {existingBidTotal > 0 ? (
                <>
                  Adding: {selectedCards.map(formatMoney).join(" + ")} ({formatMoney(selectedTotal)})
                  <br />
                  <span style={{ color: "#a0a890" }}>
                    Already bid: {formatMoney(existingBidTotal)}
                  </span>
                  {" → "}
                  <strong>New total: {formatMoney(newBidTotal)}</strong>
                </>
              ) : (
                <>
                  Selected: {selectedCards.map(formatMoney).join(" + ")} ={" "}
                  <strong>{formatMoney(selectedTotal)}</strong>
                </>
              )}
              {/* Warning when bidding all remaining coins */}
              {privateHand && privateHand.length > 0 && selectedCards.length === privateHand.length && (
                <div
                  style={{
                    color: "#e53935",
                    fontSize: 13,
                    fontFamily: "'Georgia', serif",
                    marginTop: 6,
                  }}
                >
                  Having zero money guarantees a loss.
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <div
        style={{
          marginTop: "auto",
        }}
      >
        <Divider />
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Pass button */}
          {canPass && (
            <button
              onClick={handlePass}
              style={{
                padding: "10px 24px",
                borderRadius: 8,
                border: "2px solid rgba(255,255,255,0.2)",
                backgroundColor: "rgba(255,255,255,0.05)",
                color: "#d0d0c0",
                fontSize: 15,
                fontWeight: "bold",
                cursor: "pointer",
                fontFamily: "'Georgia', serif",
                transition: "all 0.1s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#ffd700";
                e.currentTarget.style.color = "#ffd700";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
                e.currentTarget.style.color = "#d0d0c0";
              }}
            >
              Pass [Space]
            </button>
          )}

          {/* Bid button — enabled only when selection beats the highest bid */}
          {hasBidActions && (
            <button
              onClick={handleBid}
              disabled={!beatsHighest}
              style={{
                padding: "10px 24px",
                borderRadius: 8,
                border: "none",
                backgroundColor: beatsHighest ? "#2d6b2d" : "#333",
                color: beatsHighest ? "#fff" : "#888",
                fontSize: 15,
                fontWeight: "bold",
                cursor: beatsHighest ? "pointer" : "not-allowed",
                fontFamily: "'Georgia', serif",
                boxShadow: beatsHighest
                  ? "0 4px 12px rgba(45,107,45,0.4)"
                  : "none",
                transition: "all 0.15s ease",
              }}
              onMouseEnter={(e) => {
                if (beatsHighest) {
                  e.currentTarget.style.backgroundColor = "#3a8a3a";
                }
              }}
              onMouseLeave={(e) => {
                if (beatsHighest) {
                  e.currentTarget.style.backgroundColor = "#2d6b2d";
                }
              }}
            >
              Bid{selectedCards.length > 0 ? ` ${formatMoney(newBidTotal)}` : ""} [Enter]
            </button>
          )}

          {/* Discard possession options */}
          {discardActions.map((action, i) => (
            <button
              key={`discard-${i}`}
              onClick={() => handleDiscard(action)}
              style={{
                padding: "10px 24px",
                borderRadius: 8,
                border: "2px solid rgba(255,255,255,0.2)",
                backgroundColor: "rgba(255,255,255,0.05)",
                color: "#d0d0c0",
                fontSize: 14,
                cursor: "pointer",
                fontFamily: "'Georgia', serif",
                transition: "all 0.1s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "#ffd700";
                e.currentTarget.style.color = "#ffd700";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.2)";
                e.currentTarget.style.color = "#d0d0c0";
              }}
            >
              Discard possession <strong>{action.possession_value}</strong>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
