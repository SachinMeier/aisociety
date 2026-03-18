import type { Page, Locator } from "@playwright/test";

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
  right: number;
  bottom: number;
}

export interface OverlapResult {
  a: string;
  b: string;
  overlapPx: number;
  overlapArea: { width: number; height: number };
}

/** Intersection area of two rectangles, or null if they don't overlap. */
function intersect(a: Rect, b: Rect): { width: number; height: number } | null {
  const xOverlap = Math.min(a.right, b.right) - Math.max(a.x, b.x);
  const yOverlap = Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y);
  if (xOverlap <= 0 || yOverlap <= 0) return null;
  return { width: xOverlap, height: yOverlap };
}

/** Get bounding rect for an element by test ID. */
async function getRect(page: Page, testId: string): Promise<Rect | null> {
  const el = page.getByTestId(testId);
  if ((await el.count()) === 0) return null;
  const box = await el.boundingBox();
  if (!box) return null;
  return {
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    right: box.x + box.width,
    bottom: box.y + box.height,
  };
}

/** Get bounding rect from a locator. */
async function getRectFromLocator(locator: Locator): Promise<Rect | null> {
  const box = await locator.boundingBox();
  if (!box) return null;
  return {
    x: box.x,
    y: box.y,
    width: box.width,
    height: box.height,
    right: box.x + box.width,
    bottom: box.y + box.height,
  };
}

/**
 * Check all pairs of elements with given testIds for bounding-box overlaps.
 * Returns list of overlapping pairs with pixel details.
 */
export async function checkPairwiseOverlaps(
  page: Page,
  testIds: string[],
): Promise<OverlapResult[]> {
  const rects = new Map<string, Rect>();
  for (const id of testIds) {
    const r = await getRect(page, id);
    if (r) rects.set(id, r);
  }

  const overlaps: OverlapResult[] = [];
  const ids = [...rects.keys()];
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = rects.get(ids[i])!;
      const b = rects.get(ids[j])!;
      const overlap = intersect(a, b);
      if (overlap) {
        overlaps.push({
          a: ids[i],
          b: ids[j],
          overlapPx: Math.round(overlap.width * overlap.height),
          overlapArea: overlap,
        });
      }
    }
  }
  return overlaps;
}

/**
 * Check if any elements are clipped by the viewport (extend beyond visible area).
 */
export async function checkViewportClipping(
  page: Page,
  testIds: string[],
): Promise<{ testId: string; clippedBy: string }[]> {
  const viewport = page.viewportSize();
  if (!viewport) return [];

  const clipped: { testId: string; clippedBy: string }[] = [];
  for (const id of testIds) {
    const r = await getRect(page, id);
    if (!r) continue;

    const issues: string[] = [];
    if (r.x < 0) issues.push(`left by ${Math.round(-r.x)}px`);
    if (r.y < 0) issues.push(`top by ${Math.round(-r.y)}px`);
    if (r.right > viewport.width) issues.push(`right by ${Math.round(r.right - viewport.width)}px`);
    if (r.bottom > viewport.height) issues.push(`bottom by ${Math.round(r.bottom - viewport.height)}px`);

    if (issues.length > 0) {
      clipped.push({ testId: id, clippedBy: issues.join(", ") });
    }
  }
  return clipped;
}

/**
 * Find all elements matching a testId prefix pattern on the page.
 * E.g. prefix "player-seat-" finds player-seat-0, player-seat-1, etc.
 */
export async function findTestIdsByPrefix(page: Page, prefix: string): Promise<string[]> {
  return page.evaluate((pfx) => {
    const els = document.querySelectorAll(`[data-testid^="${pfx}"]`);
    return Array.from(els).map((el) => el.getAttribute("data-testid")!);
  }, prefix);
}

/**
 * Collect all seat + center-card test IDs from the page.
 */
export async function collectLayoutTestIds(page: Page): Promise<string[]> {
  const seatIds = await findTestIdsByPrefix(page, "player-seat-");
  const centerCard = (await page.getByTestId("center-card").count()) > 0 ? ["center-card"] : [];
  return [...seatIds, ...centerCard];
}

export { getRect, getRectFromLocator };
