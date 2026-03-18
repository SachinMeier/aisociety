import type { Page } from "@playwright/test";
import type { TurnResponse } from "../../src/types";

/**
 * Intercept all /api/* routes and return fixture data.
 * The turn endpoint returns the same response every time, making the UI stable.
 */
export async function mockApiRoutes(page: Page, turnData: TurnResponse): Promise<void> {
  // Intercept GET /api/local-games/:id/turn
  await page.route("**/api/local-games/*/turn", (route) => {
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(turnData),
    });
  });

  // Intercept GET /api/local-games/:id (game state)
  await page.route("**/api/local-games/*", (route, request) => {
    if (request.method() === "GET" && !request.url().includes("/turn") && !request.url().includes("/actions")) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(turnData),
      });
    }
    return route.continue();
  });

  // Intercept POST /api/local-games/:id/actions - return same turn data
  await page.route("**/api/local-games/*/actions", (route) => {
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(turnData),
    });
  });

  // Intercept POST /api/local-games (create game)
  await page.route("**/api/local-games", (route, request) => {
    if (request.method() === "POST") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ game_id: "test-game-001" }),
      });
    }
    return route.continue();
  });
}
