import type { CSSProperties } from "react";

interface DividerProps {
  className?: string;
}

const containerStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 12,
  margin: "0.5rem 0",
};

const lineStyle: CSSProperties = {
  display: "block",
  width: 60,
  height: 1,
  background: "linear-gradient(90deg, transparent, #d4af37, transparent)",
};

const iconStyle: CSSProperties = {
  color: "#d4af37",
  fontSize: "0.8rem",
  opacity: 0.6,
};

export default function Divider({ className }: DividerProps) {
  return (
    <div style={containerStyle} className={className}>
      <span style={lineStyle} />
      <span style={iconStyle}>&#9830;</span>
      <span style={lineStyle} />
    </div>
  );
}
