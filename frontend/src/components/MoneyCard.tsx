interface CoinProps {
  value: number;
  size?: "tiny" | "compact" | "small" | "normal";
  selected?: boolean;
  /** When selected, use grey instead of green (bid doesn't beat highest) */
  selectedWeak?: boolean;
  /** Coin has been spent — render as a dark shadow */
  spent?: boolean;
  onClick?: () => void;
}

function formatMoney(value: number): string {
  if (value >= 1000) {
    return `$${(value / 1000).toFixed(0)}k`;
  }
  return `$${value}`;
}

export { formatMoney };

export default function Coin({
  value,
  size = "normal",
  selected = false,
  selectedWeak = false,
  spent = false,
  onClick,
}: CoinProps) {
  const d =
    size === "tiny"
      ? { s: 36, font: 10, border: 2 }
      : size === "compact"
      ? { s: 48, font: 13, border: 2 }
      : size === "small"
      ? { s: 64, font: 16, border: 3 }
      : { s: 80, font: 20, border: 3 };

  const isGrey = selected && selectedWeak;
  const isGreen = selected && !selectedWeak;

  const bg = spent
    ? "radial-gradient(circle at 35% 35%, #3a3a3a 0%, #2a2a2a 60%, #1a1a1a 100%)"
    : isGreen
    ? "radial-gradient(circle at 35% 35%, #a8e6a0 0%, #4caf50 60%, #2e7d32 100%)"
    : isGrey
    ? "radial-gradient(circle at 35% 35%, #d0d0d0 0%, #9e9e9e 60%, #707070 100%)"
    : "radial-gradient(circle at 35% 35%, #fff4c2 0%, #f0d060 40%, #c8a830 80%, #a08020 100%)";

  const borderColor = spent
    ? "#2a2a2a"
    : isGreen
    ? "#2e7d32"
    : isGrey
    ? "#606060"
    : "#8a6d1b";

  const glowColor = isGreen
    ? "#28a745"
    : isGrey
    ? "#888"
    : "transparent";

  const textColor = spent ? "#555" : selected ? "#fff" : "#5a4a10";

  return (
    <div
      onClick={onClick}
      style={{
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: d.s,
        height: d.s,
        borderRadius: "50%",
        background: bg,
        border: `${d.border}px solid ${borderColor}`,
        boxShadow: selected
          ? `0 0 0 2px ${glowColor}, 0 2px 8px rgba(0,0,0,0.25), inset 0 1px 2px rgba(255,255,255,0.4)`
          : "0 2px 8px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.5)",
        cursor: onClick ? "pointer" : "default",
        fontFamily: "'Georgia', serif",
        userSelect: "none",
        transition: "all 0.15s ease",
        transform: selected ? "translateY(-4px)" : "none",
      }}
    >
      <span
        style={{
          fontSize: d.font,
          fontWeight: "bold",
          color: textColor,
          textShadow: selected
            ? "0 1px 2px rgba(0,0,0,0.3)"
            : "0 1px 0 rgba(255,255,255,0.4)",
        }}
      >
        {formatMoney(value)}
      </span>
    </div>
  );
}
