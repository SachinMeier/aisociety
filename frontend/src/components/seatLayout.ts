// Fixed position layouts for each player count
// Seats pushed to container edges to maximize clearance from center card area

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

// 3 players: Triangle — top + two bottom corners
const LAYOUT_3: SeatPosition[] = [
  { left: "50%", top: "0%", anchor: "top" },        // top
  { left: "17%", top: "97%", anchor: "bottom" },   // bottom-left
  { left: "83%", top: "97%", anchor: "bottom" },   // bottom-right
];

// 4 players: Diamond — top, right, bottom, left
const LAYOUT_4: SeatPosition[] = [
  { left: "50%", top: "0%", anchor: "top" },        // top
  { left: "84%", top: "50%", anchor: "center" },    // right
  { left: "50%", top: "97%", anchor: "bottom" },   // bottom
  { left: "16%", top: "50%", anchor: "center" },    // left
];

// 5 players: Pentagon — top, upper-right, lower-right, lower-left, upper-left
const LAYOUT_5: SeatPosition[] = [
  { left: "50%", top: "0%", anchor: "top" },        // top
  { left: "84%", top: "38%", anchor: "center" },    // top-right
  { left: "75%", top: "97%", anchor: "bottom" },   // bottom-right
  { left: "25%", top: "97%", anchor: "bottom" },   // bottom-left
  { left: "16%", top: "38%", anchor: "center" },    // top-left
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
