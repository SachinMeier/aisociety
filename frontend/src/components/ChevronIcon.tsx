interface ChevronIconProps {
  direction: "up" | "down" | "left" | "right";
  size?: number;
  color?: string;
}

export default function ChevronIcon({
  direction,
  size = 20,
  color = "currentColor",
}: ChevronIconProps) {
  const rotations: Record<string, number> = {
    up: 180,
    down: 0,
    left: 90,
    right: -90,
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ transform: `rotate(${rotations[direction]}deg)` }}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
