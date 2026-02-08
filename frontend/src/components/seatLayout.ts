// Fixed position layouts for each player count
// Positions form regular polygons around the center auction card
// Calculated using: x = 50% + radius * sin(angle), y = 50% - radius * cos(angle)

export type VerticalAnchor = "top" | "center" | "bottom";

export interface SeatPosition {
  left: string;    // CSS percentage from left edge
  top: string;     // CSS percentage from top edge
  anchor: VerticalAnchor;  // How seat expands from position point
}

// Anchor behavior:
// - top: Seat anchored at top edge, expands downward (toward center)
// - center: Seat centered on position point
// - bottom: Seat anchored at bottom edge, expands upward (toward center)

// 3 players: Equilateral triangle (120° apart)
// Angles from top: 0°, 120°, 240° with radius ~40%
const LAYOUT_3: SeatPosition[] = [
  { left: "50%", top: "8%", anchor: "top" },        // top (0°)
  { left: "15%", top: "78%", anchor: "bottom" },    // bottom-left (240°)
  { left: "85%", top: "78%", anchor: "bottom" },    // bottom-right (120°)
];

// 4 players: Diamond/square (90° apart)
// Angles from top: 0°, 90°, 180°, 270° with radius ~42%
const LAYOUT_4: SeatPosition[] = [
  { left: "50%", top: "6%", anchor: "top" },        // top (0°)
  { left: "92%", top: "50%", anchor: "center" },    // right (90°)
  { left: "50%", top: "94%", anchor: "bottom" },    // bottom (180°)
  { left: "8%", top: "50%", anchor: "center" },     // left (270°)
];

// 5 players: Regular pentagon (72° apart)
// Angles from top: 0°, 72°, 144°, 216°, 288° with radius ~42%
const LAYOUT_5: SeatPosition[] = [
  { left: "50%", top: "6%", anchor: "top" },        // top (0°)
  { left: "90%", top: "35%", anchor: "center" },    // top-right (72°)
  { left: "75%", top: "85%", anchor: "bottom" },    // bottom-right (144°)
  { left: "25%", top: "85%", anchor: "bottom" },    // bottom-left (216°)
  { left: "10%", top: "35%", anchor: "center" },    // top-left (288°)
];

export const SEAT_LAYOUTS: Record<number, SeatPosition[]> = {
  3: LAYOUT_3,
  4: LAYOUT_4,
  5: LAYOUT_5,
};

// CSS transforms for each anchor type
export const ANCHOR_TRANSFORMS: Record<VerticalAnchor, string> = {
  top: "translateX(-50%)",           // centered horizontally, anchored at top
  center: "translate(-50%, -50%)",   // centered both ways
  bottom: "translate(-50%, -100%)",  // centered horizontally, anchored at bottom
};
