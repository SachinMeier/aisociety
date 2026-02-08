# High Society — Local Pass-and-Play React Frontend Plan

## Work Request
Build a local pass-and-play experience with a friendly React UI, backed by the existing Python game engine, with support for multiple humans on one computer plus configurable bot opponents.

## Scope
- In scope:
  - Local single-machine play (browser + local Python server).
  - Multiple human seats in the same game (pass-and-play).
  - Mixed human/bot tables (3-5 players total).
  - New friendly frontend flow: Home -> New Game -> Table -> Results.
  - Fixed difficulty presets (no advanced config editing in UI).
- Out of scope:
  - Online matchmaking/lobbies.
  - Persistence across server restarts.
  - Training workflows and ops dashboard integration.

## Key Decisions
1. Keep game rules and turn legality in existing domain/server code (`highsociety/domain`, `highsociety/server`).
2. Do not replace the existing `GameRunner` for this feature; add a new interactive application service for pause/resume turn control.
3. Build frontend in React for a richer and less technical UX.
4. Keep backend state in memory for local pass-and-play.
5. Expose only simple, opinionated setup options in UI.
6. Use short Base32 game IDs (not UUIDs) for local sessions.

## Why Interactive Service Is Needed (Not a New Rules Loop)
- Current runner executes a full game in one blocking call (`highsociety/app/runner.py`), where `player.act(...)` must return immediately.
- Browser humans are asynchronous: actions arrive later via HTTP requests.
- Therefore we keep the same rules/engine but add an orchestration service that:
  - advances bot turns automatically,
  - stops when a human decision is required,
  - resumes after the submitted human action.

## UX and Product Requirements

### Player Setup Simplicity
- UI must not expose raw run specs, checkpoints, temperatures, or any advanced knobs.
- New Game form only includes:
  - player count (3-5),
  - seat type per player,
  - optional display name.

### Difficulty Presets (Fixed Mapping)
- `Easy` -> `static`
- `Medium` -> `heuristic` with style `balanced`
- `Hard` -> `mlp` with checkpoint `checkpoints/mlp/mlp_v3`
- `Expert` -> `heuristic` with style `cautious`

### Pass-and-Play Privacy Model
- Public for all players:
  - every player's owned status cards (possessions and title/misfortune markers),
  - open bids, turn marker, revealed status cards, discard summaries.
- Private per player:
  - money hand cards.
- Before each human turn:
  - show a handoff gate ("Pass to Player X"),
  - keep private hand hidden until player confirms reveal.

### Visual Layout Requirements
- Table should look like an online poker table:
  - bird's-eye table center,
  - players arranged in a circle around the table.
- All hands should render right-side up (no upside-down/rotated card text).
- Tone should be friendly and game-like, not technical/ops-like.

## Target Architecture

### Backend (Python)
- New module: `highsociety/app/local_game_service.py`
- Responsibility:
  - create and track local interactive sessions,
  - map seat presets to player implementations,
  - auto-play bot turns until next human turn or game end,
  - provide player-scoped turn payloads for UI.
- Game ID format:
  - uppercase Base32 token, 8 chars (example: `K7Q2M8P4`),
  - generated from secure random bytes,
  - collision-safe via regenerate-until-unique against in-memory session map.

### API Layer (Python)
- New module: `highsociety/server/local_api.py` (FastAPI)
- Endpoints:
  - `POST /api/local-games`
  - `GET /api/local-games/{game_id}`
  - `GET /api/local-games/{game_id}/turn`
  - `POST /api/local-games/{game_id}/actions`
  - `GET /api/health`

### Frontend (React)
- New folder: `frontend/`
- Pages:
  - Home (`/`)
  - New Game (`/new`)
  - Table (`/game/:id`)
  - Result (`/game/:id/result`)

## Data Contract (API Sketch)

### Create Game Request
```json
{
  "seats": [
    {"type": "human", "name": "Player 1"},
    {"type": "easy"},
    {"type": "hard"}
  ],
  "seed": null
}
```

### Turn Response
```json
{
  "game_id": "K7Q2M8P4",
  "status": "awaiting_human_action",
  "active_player_id": 1,
  "active_player_name": "Player 2",
  "requires_handoff": true,
  "public_table": {
    "status_card": {"kind": "possession", "value": 8},
    "round": {"highest_bid": 6000, "highest_bidder": 0, "turn_player": 1},
    "players": [
      {
        "id": 0,
        "name": "Player 1",
        "open_bid": [6000],
        "owned_status_cards": [{"kind": "title"}],
        "money_count": 10
      }
    ],
    "revealed_status_cards": []
  },
  "private_hand": [25000, 20000, 15000, 12000, 10000, 8000, 6000, 4000, 3000, 2000, 1000],
  "legal_actions": [
    {"kind": "pass"},
    {"kind": "bid", "cards": [1000]},
    {"kind": "bid", "cards": [2000]}
  ]
}
```

### Submit Action Request
```json
{
  "player_id": 1,
  "action": {"kind": "bid", "cards": [2000, 1000]}
}
```

### Submit Action Response
- Returns updated turn payload after auto-advancing bots.
- If terminal: response status `finished` plus scored results.

## Implementation Plan (Phased)

### Phase 1: Backend Interactive Session Core
- Add `LocalSeatSpec` and `LocalGameSession` dataclasses.
- Implement `LocalGameService` with:
  - `create_game(...)`
  - `get_turn_view(...)`
  - `submit_human_action(...)`
  - `_advance_until_human_or_terminal(...)`
- Add `_new_game_id()` helper:
  - secure random short Base32 (8 chars),
  - regenerate on collision.
- Add preset resolver:
  - `easy -> static`
  - `medium -> heuristic balanced`
  - `hard -> mlp checkpoints/mlp/mlp_v3`
  - `expert -> heuristic cautious`
- Keep per-game state in memory.

Deliverable:
- Service can create mixed tables and drive bots automatically until a human decision is required.

### Phase 2: Public/Private View Model
- Introduce UI-focused serializers:
  - `build_public_table_view(state, manifest)`
  - `build_private_hand_view(state, player_id)`
  - `build_legal_action_view(actions)`
- Ensure public payload includes all owned status cards for every player.
- Ensure private payload includes only active player's money hand.

Deliverable:
- Deterministic JSON payload for table rendering with correct information boundaries.

### Phase 3: FastAPI Endpoints
- Add FastAPI app with typed request/response models.
- Wire endpoints to `LocalGameService`.
- Add lightweight validation and clear error responses.
- Add script entrypoint to run local web server.

Deliverable:
- Browser client can create game, query turn state, and submit actions.

### Phase 4: React App Scaffold
- Initialize React app (Vite + TypeScript).
- Add routing and API client layer.
- Build pages:
  - Home: title + "New Game" button.
  - New Game: seat configuration only (friendly labels).
  - Table: poker-style circular layout + center board + action controls.
  - Result: winners and summary stats.

Deliverable:
- End-to-end playable UI against local API.

### Phase 5: Pass-and-Play UX Hardening
- Add handoff gate modal between human turns.
- Add hidden-hand state until "Reveal my hand" confirmation.
- Add turn ownership checks to prevent wrong-seat action submission.
- Add clear invalid-action feedback and retry affordance.

Deliverable:
- Multiple humans can safely share one machine without accidental hand leaks.

### Phase 6: Polish and Accessibility
- Improve typography/colors/spacing for friendlier feel.
- Ensure keyboard-accessible controls for action selection.
- Add responsive table behavior for common laptop widths.
- Keep cards upright in all seat positions.

Deliverable:
- Friendly, non-technical play experience that feels game-first.

## File-Level Change Plan

### New Backend Files
- `highsociety/app/local_game_service.py`
- `highsociety/app/local_views.py` (optional serializer/view models)
- `highsociety/server/local_api.py`
- `scripts/play_local.py` (or `scripts/web_play.py`)

### Modified Backend Files
- `highsociety/players/registry.py` (preset mapping helpers or factories)
- `README.md` (new local play instructions)

### New Frontend Files (Representative)
- `frontend/package.json`
- `frontend/src/main.tsx`
- `frontend/src/App.tsx`
- `frontend/src/routes/HomePage.tsx`
- `frontend/src/routes/NewGamePage.tsx`
- `frontend/src/routes/GameTablePage.tsx`
- `frontend/src/routes/ResultPage.tsx`
- `frontend/src/components/TableLayout.tsx`
- `frontend/src/components/HandoffOverlay.tsx`
- `frontend/src/components/ActionPanel.tsx`
- `frontend/src/api/client.ts`

## Testing Plan

### Backend Tests
- Service flow:
  - game creation with mixed human/bot seats,
  - bot auto-advance to next human,
  - terminal scoring path.
- Difficulty mapping:
  - each preset resolves to expected bot and params.
- Visibility rules:
  - public view contains all players' owned status cards,
  - private hand only for active player.
- Turn security:
  - reject actions submitted for non-active player.

Suggested files:
- `tests/test_local_game_service.py`
- `tests/test_local_api.py`
- `tests/test_local_views.py`

### Frontend Tests
- New Game form:
  - enforces 3-5 seats,
  - maps friendly seat types correctly.
- Table view:
  - circular seat layout renders,
  - center table state visible.
- Pass-and-play behavior:
  - handoff overlay appears between human turns,
  - hand hidden until reveal.

Suggested files:
- `frontend/src/routes/__tests__/NewGamePage.test.tsx`
- `frontend/src/routes/__tests__/GameTablePage.test.tsx`

## Acceptance Criteria
1. Users can start a game from Home -> New Game with 3-5 seats.
2. Seat types are limited to Human/Easy/Medium/Hard/Expert with fixed preset mapping.
3. During gameplay, all players can see each player's owned status cards.
4. Money hand cards are visible only for current human after handoff confirmation.
5. Bot seats auto-play without manual steps.
6. Table layout is poker-like: circular seats, center table view, upright cards.
7. No advanced technical config is exposed in the UI.
8. Game IDs are short Base32 tokens (not UUIDs) and are unique within active local sessions.
9. Game can be completed end-to-end locally from browser.

## Risks and Mitigations
- Risk: MLP checkpoint load failure for `Hard`.
  - Mitigation: validate checkpoint at server startup and fail fast with actionable error.
- Risk: Large legal action list causing UI clutter.
  - Mitigation: group bid actions by total and allow quick select patterns.
- Risk: Privacy leak in pass-and-play.
  - Mitigation: strict handoff gate + private-hand payload only for active player.

## Rollout Strategy
1. Ship backend service + API first with test coverage.
2. Land minimal React flow (Home/New Game/Table/Result).
3. Add pass-and-play hardening and polish before defaulting to this UX.


## Agent Breakout

This plan is very parallelizable if you lock the API contract first.

Use this file as the source of truth and split like this:

Agent A: Interactive backend core
Files: local_game_service.py, test_local_game_service.py
Scope: seat presets, bot auto-advance, short Base32 IDs, turn orchestration.

Agent B: API surface
Files: local_api.py, play_local.py, test_local_api.py
Scope: FastAPI endpoints + request/response models matching the plan contract.

Agent C: Frontend app shell
Files: main.tsx, App.tsx, route pages
Scope: Home/New Game/Game/Result routing and API client wiring.

Agent D: Table UX and pass-and-play privacy
Files: TableLayout.tsx, HandoffOverlay.tsx, ActionPanel.tsx
Scope: poker-style circular layout, upright cards, handoff/reveal flow.

Agent E: Integration and verification
Files: test_local_views.py, frontend tests, README.md
Scope: visibility rules, active-player enforcement, end-to-end sanity checks, docs.

Best merge order:

Contract freeze (turn payload and action schema).
Agent A + B merge backend.
Agent C + D merge frontend.
Agent E runs integration and fixes seams.
Main conflict risk:

API payload shape drift between backend and frontend.
Seat preset naming mismatch (easy/medium/hard/expert).
Privacy leakage if frontend renders private hand outside active seat/handoff gate.