import type { StatusCardData } from "../types";

interface StatusCardProps {
  card: StatusCardData;
  size?: "small" | "normal" | "large";
}

interface CardTheme {
  bg: string;
  border: string;
  text: string;
  accent: string;
  glow: string;
  icon: string;
}

const themes: Record<string, CardTheme> = {
  possession: {
    bg: "linear-gradient(145deg, #fdf6e3 0%, #f5e6c8 40%, #e8d5a8 100%)",
    border: "#c9a84c",
    text: "#5a4a1e",
    accent: "#d4af37",
    glow: "rgba(212, 175, 55, 0.15)",
    icon: "\u2666", // diamond
  },
  title: {
    bg: "linear-gradient(145deg, #e8f4e8 0%, #c8e0c8 40%, #a8c8a8 100%)",
    border: "#4a8c4a",
    text: "#1a4a1a",
    accent: "#d4af37",
    glow: "rgba(74, 140, 74, 0.15)",
    icon: "\u265b", // crown/queen
  },
  misfortune: {
    bg: "linear-gradient(145deg, #f5e0e0 0%, #e0c0c0 40%, #c8a0a0 100%)",
    border: "#9c4040",
    text: "#5a1e1e",
    accent: "#c04040",
    glow: "rgba(156, 64, 64, 0.15)",
    icon: "\u2620", // skull
  },
};

function getImpactLabel(card: StatusCardData): string {
  if (card.kind === "possession" && card.value != null) return String(card.value);
  if (card.kind === "title") return "\u00d72";
  if (card.kind === "misfortune" && card.misfortune) {
    switch (card.misfortune) {
      case "scandal": return "\u00d7\u00bd";
      case "debt": return "\u22125";
      case "theft": return "Cancel";
      default: return "\u00d7\u00bd";
    }
  }
  return "?";
}

function getSubtitle(card: StatusCardData): string {
  if (card.kind === "possession") return "";
  if (card.kind === "title") return "Title";
  if (card.kind === "misfortune" && card.misfortune) {
    switch (card.misfortune) {
      case "scandal": return "Scandal";
      case "debt": return "Debt";
      case "theft": return "Theft";
      default: return "Misfortune";
    }
  }
  return "";
}

function getFlavorText(card: StatusCardData): string | null {
  if (card.kind === "title") return "Doubles all possessions";
  if (card.kind === "misfortune" && card.misfortune) {
    switch (card.misfortune) {
      case "scandal": return "Halves your score";
      case "debt": return "Lose 5 points";
      case "theft": return "Lose a possession";
    }
  }
  return null;
}

const dims = {
  small: { w: 60, h: 82, valueFont: 22, subFont: 8, iconFont: 10, pad: 4, radius: 6, borderW: 1.5, flavorFont: 0, cornerFont: 7 },
  normal: { w: 100, h: 140, valueFont: 38, subFont: 11, iconFont: 14, pad: 8, radius: 10, borderW: 2, flavorFont: 9, cornerFont: 10 },
  large: { w: 150, h: 210, valueFont: 56, subFont: 15, iconFont: 20, pad: 14, radius: 14, borderW: 3, flavorFont: 11, cornerFont: 14 },
};

export default function StatusCard({ card, size = "normal" }: StatusCardProps) {
  const theme = themes[card.kind] ?? themes.possession;
  const d = dims[size];
  const label = getImpactLabel(card);
  const subtitle = getSubtitle(card);
  const flavor = getFlavorText(card);

  const isLongLabel = label.length > 3;
  const mainFontSize = isLongLabel ? d.valueFont * 0.55 : d.valueFont;

  return (
    <div
      style={{
        position: "relative",
        display: "inline-flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: d.w,
        height: d.h,
        background: theme.bg,
        border: `${d.borderW}px solid ${theme.border}`,
        borderRadius: d.radius,
        boxShadow: `0 2px 8px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.5), 0 0 0 1px ${theme.glow}`,
        padding: d.pad,
        fontFamily: "'Georgia', serif",
        userSelect: "none",
        overflow: "hidden",
      }}
    >
      {/* Decorative corner icons */}
      {size !== "small" && (
        <>
          <span style={{
            position: "absolute",
            top: d.pad,
            left: d.pad + 2,
            fontSize: d.cornerFont,
            color: theme.accent,
            opacity: 0.6,
            lineHeight: 1,
          }}>
            {theme.icon}
          </span>
          <span style={{
            position: "absolute",
            bottom: d.pad,
            right: d.pad + 2,
            fontSize: d.cornerFont,
            color: theme.accent,
            opacity: 0.6,
            lineHeight: 1,
            transform: "rotate(180deg)",
          }}>
            {theme.icon}
          </span>
        </>
      )}

      {/* Subtitle at top */}
      <span
        style={{
          fontSize: d.subFont,
          color: theme.text,
          opacity: 0.65,
          textTransform: "uppercase",
          letterSpacing: size === "small" ? 0.3 : 1,
          textAlign: "center",
          lineHeight: 1,
          marginBottom: size === "small" ? 1 : 2,
        }}
      >
        {subtitle}
      </span>

      {/* Main value */}
      <span
        style={{
          fontSize: mainFontSize,
          fontWeight: "bold",
          color: theme.text,
          textAlign: "center",
          lineHeight: 1.1,
          textShadow: `0 1px 0 rgba(255,255,255,0.4)`,
        }}
      >
        {label}
      </span>

      {/* Flavor text for larger sizes */}
      {flavor && d.flavorFont > 0 && (
        <span
          style={{
            fontSize: d.flavorFont,
            color: theme.text,
            opacity: 0.5,
            textAlign: "center",
            fontStyle: "italic",
            lineHeight: 1.2,
            marginTop: 4,
            maxWidth: d.w - d.pad * 2 - 8,
          }}
        >
          {flavor}
        </span>
      )}

      {/* Subtle inner border */}
      <div
        style={{
          position: "absolute",
          inset: d.borderW + 2,
          borderRadius: d.radius - 2,
          border: `1px solid rgba(255,255,255,0.3)`,
          pointerEvents: "none",
        }}
      />
    </div>
  );
}

/** Compact label for discard pile thumbnails. */
export function StatusCardLabel({ card }: { card: StatusCardData }) {
  return <>{getImpactLabel(card)}</>;
}
