import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-api";
import { checkPairwiseOverlaps, getRect } from "../helpers/overlap";
import { saveScreenshot } from "../helpers/screenshots";
import { turn3pLargeBids } from "../fixtures/turn-3p-large-bids";

test.describe("Sidebar collapse/expand", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page, turn3pLargeBids);
    await page.goto("/game/test-game-001");
    await page.getByTestId("player-seat-0").waitFor({ state: "visible", timeout: 10_000 });
  });

  test("sidebar is visible by default", async ({ page }) => {
    const sidePanel = page.getByTestId("side-panel");
    await expect(sidePanel).toBeVisible();

    const rect = await getRect(page, "side-panel");
    expect(rect).not.toBeNull();
    // Sidebar should have meaningful width (not collapsed)
    expect(rect!.width).toBeGreaterThan(200);

    await saveScreenshot(page, "sidebar-expanded");
  });

  test("sidebar does not overlap game area", async ({ page }) => {
    const overlaps = await checkPairwiseOverlaps(page, ["game-area", "side-panel"]);

    if (overlaps.length > 0) {
      const details = overlaps
        .map((o) => `  ${o.a} overlaps ${o.b} by ${o.overlapPx}px`)
        .join("\n");
      expect(overlaps, `Sidebar overlaps game area:\n${details}`).toHaveLength(0);
    }
  });

  test("collapsed sidebar is narrower", async ({ page }) => {
    // Click the collapse toggle
    const toggle = page.locator("[aria-label='Collapse sidebar']");
    await toggle.click();

    // Wait for transition
    await page.waitForTimeout(300);

    const rect = await getRect(page, "side-panel");
    expect(rect).not.toBeNull();
    // Collapsed sidebar should be narrow
    expect(rect!.width).toBeLessThan(120);

    await saveScreenshot(page, "sidebar-collapsed");
  });

  test("expand after collapse restores width", async ({ page }) => {
    // Collapse
    await page.locator("[aria-label='Collapse sidebar']").click();
    await page.waitForTimeout(300);

    // Expand
    await page.locator("[aria-label='Expand sidebar']").click();
    await page.waitForTimeout(300);

    const rect = await getRect(page, "side-panel");
    expect(rect).not.toBeNull();
    expect(rect!.width).toBeGreaterThan(200);

    await saveScreenshot(page, "sidebar-re-expanded");
  });
});
