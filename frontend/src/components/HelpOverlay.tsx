import { useState, useEffect } from "react";

interface HelpTip {
  label: string;
  description: string;
  position: { top?: string; bottom?: string; left?: string; right?: string };
  anchor: "top" | "bottom" | "left" | "right" | "none";
}

const TIPS: HelpTip[] = [
  {
    label: "Auction Card",
    description: "This is the card currently up for auction. Bid your coins to win it, or pass to save your money.",
    position: { top: "42%", left: "50%" },
    anchor: "top",
  },
  {
    label: "Player Seats",
    description: "Each player sits around the table. The glowing name plate shows whose turn it is. Coins they've bid are shown below their name.",
    position: { top: "22%", left: "50%" },
    anchor: "bottom",
  },
  {
    label: "Your Coins",
    description: "Your money cards are shown in the sidebar. Tap coins to select them for a bid. Spent coins appear darkened.",
    position: { top: "30%", right: "22%" },
    anchor: "right",
  },
  {
    label: "Bid & Pass",
    description: "Select coins then press Bid (or Enter) to place your bid. Press Pass (or Space) to drop out of the current auction.",
    position: { bottom: "8%", right: "22%" },
    anchor: "right",
  },
  {
    label: "Leading Bid",
    description: "Shows the current highest bid and who placed it. Your bid must exceed this amount to take the lead.",
    position: { top: "62%", left: "50%" },
    anchor: "none",
  },
];

// Inline styles for HelpButton - matches the quitBtn style from GameTablePage.module.css
const helpButtonStyle: React.CSSProperties = {
  width: "50%",
  padding: "0.4rem 0.75rem",
  fontSize: "0.8rem",
  fontFamily: "'Georgia', serif",
  color: "#8a8a7a",
  background: "rgba(255, 255, 255, 0.03)",
  border: "1px solid rgba(255, 255, 255, 0.08)",
  borderRadius: "6px",
  cursor: "pointer",
  transition: "all 0.15s ease",
};

const helpButtonHoverStyle: React.CSSProperties = {
  ...helpButtonStyle,
  color: "#b8a890",
  background: "rgba(255, 255, 255, 0.06)",
  borderColor: "rgba(255, 255, 255, 0.15)",
};

export function HelpButton({ onClick }: { onClick: () => void }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <button
      onClick={onClick}
      aria-label="Help"
      style={isHovered ? helpButtonHoverStyle : helpButtonStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      Help
    </button>
  );
}

export function HelpOverlay({ onClose }: { onClose: () => void }) {
  const [activeTip, setActiveTip] = useState(0);
  const tip = TIPS[activeTip];

  // ESC key handler to dismiss the overlay
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  function next() {
    if (activeTip < TIPS.length - 1) {
      setActiveTip(activeTip + 1);
    } else {
      onClose();
    }
  }

  function prev() {
    if (activeTip > 0) {
      setActiveTip(activeTip - 1);
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        zIndex: 999,
        cursor: "pointer",
      }}
    >
      {/* Tooltip card */}
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          position: "absolute",
          ...tip.position,
          transform: getTransform(tip.anchor),
          backgroundColor: "rgba(20, 30, 20, 0.95)",
          border: "1.5px solid rgba(212, 175, 55, 0.5)",
          borderRadius: 14,
          padding: "16px 20px",
          maxWidth: 300,
          zIndex: 1001,
          animation: "fadeInCenter 0.2s ease",
          cursor: "default",
          boxShadow: "0 4px 20px rgba(0,0,0,0.4), 0 0 20px rgba(212,175,55,0.1)",
        }}
      >
        {/* Arrow indicator */}
        {tip.anchor !== "none" && (
          <div style={{
            position: "absolute",
            ...getArrowPosition(tip.anchor),
            width: 12,
            height: 12,
            backgroundColor: "rgba(20, 30, 20, 0.95)",
            border: "1.5px solid rgba(212, 175, 55, 0.5)",
            transform: "rotate(45deg)",
            ...getArrowBorderClip(tip.anchor),
          }} />
        )}

        <div style={{
          fontSize: 11,
          color: "#d4af37",
          textTransform: "uppercase",
          letterSpacing: 1.5,
          fontWeight: 600,
          marginBottom: 6,
          fontFamily: "'Georgia', serif",
        }}>
          {tip.label}
        </div>

        <div style={{
          fontSize: 14,
          color: "#d0c8b0",
          lineHeight: 1.5,
          fontFamily: "'Georgia', serif",
        }}>
          {tip.description}
        </div>

        {/* Navigation */}
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginTop: 14,
          paddingTop: 10,
          borderTop: "1px solid rgba(255,255,255,0.08)",
        }}>
          <div style={{ display: "flex", gap: 6 }}>
            {TIPS.map((_, i) => (
              <div
                key={i}
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  backgroundColor: i === activeTip ? "#d4af37" : "rgba(255,255,255,0.2)",
                  transition: "background-color 0.15s",
                }}
              />
            ))}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {activeTip > 0 && (
              <button
                onClick={prev}
                style={{
                  background: "none",
                  border: "none",
                  color: "#8a7a60",
                  fontSize: 13,
                  cursor: "pointer",
                  fontFamily: "'Georgia', serif",
                  padding: "2px 8px",
                }}
              >
                Back
              </button>
            )}
            <button
              onClick={next}
              style={{
                background: "rgba(212, 175, 55, 0.15)",
                border: "1px solid rgba(212, 175, 55, 0.3)",
                color: "#d4af37",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "'Georgia', serif",
                padding: "4px 14px",
                borderRadius: 6,
                transition: "all 0.1s",
              }}
            >
              {activeTip === TIPS.length - 1 ? "Got it" : "Next"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function getTransform(anchor: string): string {
  switch (anchor) {
    case "top": return "translate(-50%, 20px)";
    case "bottom": return "translate(-50%, calc(-100% - 20px))";
    case "left": return "translate(calc(-100% - 20px), -50%)";
    case "right": return "translate(20px, -50%)";
    case "none": return "translate(-50%, -50%)";
    default: return "translate(-50%, -50%)";
  }
}

function getArrowPosition(anchor: string): Record<string, string | number> {
  switch (anchor) {
    case "top": return { top: -7, left: "50%", marginLeft: -6 };
    case "bottom": return { bottom: -7, left: "50%", marginLeft: -6 };
    case "left": return { left: -7, top: "50%", marginTop: -6 };
    case "right": return { right: -7, top: "50%", marginTop: -6 };
    default: return {};
  }
}

function getArrowBorderClip(anchor: string): Record<string, string> {
  switch (anchor) {
    case "top": return { borderRight: "none", borderBottom: "none" };
    case "bottom": return { borderLeft: "none", borderTop: "none" };
    case "left": return { borderTop: "none", borderRight: "none" };
    case "right": return { borderBottom: "none", borderLeft: "none" };
    default: return {};
  }
}
