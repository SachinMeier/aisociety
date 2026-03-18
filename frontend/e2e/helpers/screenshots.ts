import type { Page } from "@playwright/test";
import * as path from "path";

// Playwright resolves relative to CWD (frontend/)
const SCREENSHOT_DIR = path.resolve("test-results/screenshots");

/**
 * Save a full-page screenshot to the test-results directory.
 * File path: test-results/screenshots/{name}-{WxH}.png
 */
export async function saveScreenshot(page: Page, name: string): Promise<string> {
  const viewport = page.viewportSize();
  const suffix = viewport ? `${viewport.width}x${viewport.height}` : "unknown";
  const filename = `${name}-${suffix}.png`;
  const filepath = path.join(SCREENSHOT_DIR, filename);

  await page.screenshot({ path: filepath, fullPage: false });
  return filepath;
}

/**
 * Save a screenshot of a specific element by test ID.
 */
export async function saveElementScreenshot(
  page: Page,
  testId: string,
  name: string,
): Promise<string> {
  const viewport = page.viewportSize();
  const suffix = viewport ? `${viewport.width}x${viewport.height}` : "unknown";
  const filename = `${name}-${suffix}.png`;
  const filepath = path.join(SCREENSHOT_DIR, filename);

  const el = page.getByTestId(testId);
  await el.screenshot({ path: filepath });
  return filepath;
}
