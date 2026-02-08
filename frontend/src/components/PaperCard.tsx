import type { CSSProperties, ReactNode } from "react";

interface PaperCardProps {
  children: ReactNode;
  style?: CSSProperties;
  className?: string;
}

const baseStyle: CSSProperties = {
  background: "linear-gradient(170deg, #e8dcc8, #d4c4a8, #c9b896)",
  border: "1px solid #b8a480",
  borderRadius: 12,
  padding: "1.25rem 1.5rem",
};

export default function PaperCard({ children, style, className }: PaperCardProps) {
  return (
    <div style={{ ...baseStyle, ...style }} className={className}>
      {children}
    </div>
  );
}
