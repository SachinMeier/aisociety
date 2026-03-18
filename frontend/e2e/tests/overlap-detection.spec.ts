import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-api";
import { checkPairwiseOverlaps, checkViewportClipping, collectLayoutTestIds } from "../helpers/overlap";
import { saveScreenshot } from "../helpers/screenshots";
import { turn3pLargeBids } from "../fixtures/turn-3p-large-bids";
import { turn5pBasic } from "../fixtures/turn-5p-basic";

test.describe("Overlap detection - 3 players, large bids", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page, turn3pLargeBids);
    await page.goto("/game/test-game-001");
    // Wait for the table to render with player seats
    await page.getByTestId("player-seat-0").waitFor({ state: "visible", timeout: 10_000 });
  });

  test("player seats do not overlap each other", async ({ page }) => {
    await saveScreenshot(page, "3p-large-bids");

    const testIds = await collectLayoutTestIds(page);
    const overlaps = await checkPairwiseOverlaps(page, testIds);

    if (overlaps.length > 0) {
      const details = overlaps
        .map((o) => `  ${o.a} overlaps ${o.b} by ${o.overlapPx}px (${Math.round(o.overlapArea.width)}x${Math.round(o.overlapArea.height)})`)
        .join("\n");
      expect(overlaps, `Found overlapping elements:\n${details}`).toHaveLength(0);
    }
  });

  test("no player seats clipped by viewport", async ({ page }) => {
    const testIds = await collectLayoutTestIds(page);
    const clipped = await checkViewportClipping(page, testIds);

    if (clipped.length > 0) {
      const details = clipped
        .map((c) => `  ${c.testId} clipped: ${c.clippedBy}`)
        .join("\n");
      expect(clipped, `Elements clipped by viewport:\n${details}`).toHaveLength(0);
    }
  });

  test("center card does not overlap any player seat", async ({ page }) => {
    const seatIds = (await collectLayoutTestIds(page)).filter((id) => id.startsWith("player-seat-"));
    const centerOverlaps = await checkPairwiseOverlaps(page, [...seatIds, "center-card"]);
    const centerConflicts = centerOverlaps.filter(
      (o) => o.a === "center-card" || o.b === "center-card",
    );

    if (centerConflicts.length > 0) {
      const details = centerConflicts
        .map((o) => `  ${o.a} overlaps ${o.b} by ${o.overlapPx}px`)
        .join("\n");
      expect(centerConflicts, `Center card overlaps seats:\n${details}`).toHaveLength(0);
    }
  });
});

test.describe("Overlap detection - 5 players", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page, turn5pBasic);
    await page.goto("/game/test-game-001");
    await page.getByTestId("player-seat-0").waitFor({ state: "visible", timeout: 10_000 });
  });

  test("player seats do not overlap each other", async ({ page }) => {
    await saveScreenshot(page, "5p-basic");

    const testIds = await collectLayoutTestIds(page);
    const overlaps = await checkPairwiseOverlaps(page, testIds);

    if (overlaps.length > 0) {
      const details = overlaps
        .map((o) => `  ${o.a} overlaps ${o.b} by ${o.overlapPx}px (${Math.round(o.overlapArea.width)}x${Math.round(o.overlapArea.height)})`)
        .join("\n");
      expect(overlaps, `Found overlapping elements:\n${details}`).toHaveLength(0);
    }
  });

  test("no player seats clipped by viewport", async ({ page }) => {
    const testIds = await collectLayoutTestIds(page);
    const clipped = await checkViewportClipping(page, testIds);

    if (clipped.length > 0) {
      const details = clipped
        .map((c) => `  ${c.testId} clipped: ${c.clippedBy}`)
        .join("\n");
      expect(clipped, `Elements clipped by viewport:\n${details}`).toHaveLength(0);
    }
  });

  test("center card does not overlap any player seat", async ({ page }) => {
    const seatIds = (await collectLayoutTestIds(page)).filter((id) => id.startsWith("player-seat-"));
    const centerOverlaps = await checkPairwiseOverlaps(page, [...seatIds, "center-card"]);
    const centerConflicts = centerOverlaps.filter(
      (o) => o.a === "center-card" || o.b === "center-card",
    );

    if (centerConflicts.length > 0) {
      const details = centerConflicts
        .map((o) => `  ${o.a} overlaps ${o.b} by ${o.overlapPx}px`)
        .join("\n");
      expect(centerConflicts, `Center card overlaps seats:\n${details}`).toHaveLength(0);
    }
  });
});
