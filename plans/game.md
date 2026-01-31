# High Society — DDD Implementation Plan (Python, Core Game Only)

## Implementation Decisions
- **Language**: Python only.
- **No frontend** for now; engine and APIs are pure library/CLI use.
- Use dataclasses + enums + type hints for clarity and a stable public API.
- Inject a deterministic RNG (seedable) into shuffles and any tie‑breaking.
- **Scope**: Core game concepts + tests + game loop + multi‑game server. No AI/API logic (mark TODO hooks only).

## Tools (Python)
Core (standard library):
- `dataclasses`, `enum`, `typing`, `random`, `itertools`, `collections`
- `logging` (engine diagnostics)
- `json` or `pickle` (state snapshots)

Dev & testing:
- `pytest` (unit tests for rules and edge cases)
- `ruff` or `black` (formatting/linting; choose one)

## Core Concepts
- Players each have a fixed hand of 11 money cards (no change, no drawing).
- Each round reveals 1 status card; players bid or pass in clockwise order.
- Possession/Title cards go to the *last remaining bidder*; Misfortune cards go to the *first passer*.
- Bids are cumulative: you only add money cards; you can never retract while still in the round.
- **Bids are explicit sets of money cards**, not just a numeric amount. In this game, each player has one of each denomination, so the **denomination values can serve as card IDs** in the API.
- Game ends immediately when the 4th red‑edged card is revealed; that card and the rest of the deck are ignored.
- Final winner must not be the poorest player (least money remaining).
- **Bids are explicit sets of money cards**, not just a numeric amount. Denomination values can act as IDs.

## Constants / Card Lists
- Money values (per player, 11 cards): `[25000, 20000, 15000, 12000, 10000, 8000, 6000, 4000, 3000, 2000, 1000]`.
- Status deck (16 cards total):
  - Possessions: values 1–10.
  - Titles: 3 cards (no numeric value; each doubles possession total).
  - Misfortunes: Scandal (halve total), Gambling Debt (-5), Theft (discard one possession).
- Red‑edged cards: all 3 Titles + Scandal.

## Data Model (Types / Structs)
Suggested minimal structures; adapt to language:

- `MoneyCard { value: int }` (values are unique per player and can act as IDs)
- `StatusCard { kind: possession|title|misfortune, value?: int, misfortune?: scandal|debt|theft, red: bool }`
- `PlayerState {
    id: int,
    hand: MoneyCard[],
    open_bid: MoneyCard[],
    passed: bool,
    possessions: StatusCard[], // possession cards only
    titles: int,
    scandal: bool,
    debt: bool,
    theft_pending: bool,
    money_discarded: MoneyCard[] // optional bookkeeping
  }`
- `RoundState {
    card: StatusCard,
    starting_player: int,
    turn_player: int,
    highest_bidder?: int,
    highest_bid: int,
    any_bid: bool,
    first_passer?: int
  }`
- `GameState {
    players: PlayerState[],
    status_deck: StatusCard[],
    status_discard: StatusCard[],
    money_discard: MoneyCard[],
    red_revealed: int,
    round: RoundState | null,
    game_over: bool
  }`

## DDD Implementation Plan

### 1) Domain Layer (pure rules, no I/O)
**Goal:** model rules and state transitions with strict validation.

Entities / Value Objects:
- `MoneyCard(value)` (immutable)
- `StatusCard(kind, value, misfortune, red)` (immutable)
- `PlayerState` (entity with hand, open_bid, possessions, flags)
- `RoundState`
- `GameState`

Domain services:
- `RulesEngine` (pure functions): `start_round`, `apply_action`, `end_round`, `score_game`, `is_terminal`

Domain errors:
- `RuleViolation`, `InvalidAction`, `InvalidState` (use for robust error handling)

**Constructor validation tests (first):**
- Validate money card values are from the allowed set.
- Validate status cards have consistent kind/value/misfortune.
- Validate game state invariants (unique money cards per player, no duplicates, red_revealed range, etc.).

### 2) Application Layer (use‑case orchestration)
**Goal:** expose safe, deterministic operations to drive a game.

Use‑cases:
- `new_game(player_manifest, seed) -> GameId` (creates a new GameState)
- `get_game(game_id) -> GameState`
- `legal_actions(game_id, player_id) -> list[Action]`
- `apply_action(game_id, player_id, action) -> GameState`
- `advance_round_if_needed(game_id) -> GameState`

**TODO hooks** (no AI/API logic):
- `# TODO(api): validate player manifest with external registry`
- `# TODO(api): expose game events over websocket/REST`

### 3) Infrastructure Layer (in‑memory server)
**Goal:** run multiple games concurrently and isolate failures.

Components:
- `GameRepository` (in‑memory store: `dict[GameId, GameState]`)
- `GameServer` (sync, manages multiple active games)
  - `new_game(manifest, seed) -> game_id`
  - `step(game_id, player_id, action) -> result`
  - `get_state(game_id) -> GameState`
  - `list_games() -> list[game_id]`

Error isolation:
- Catch domain exceptions per game and return structured error without crashing server.
- If a single game fails, mark it `errored` and keep server running.

### 4) Game Loop (core only, no AI/API)
**Goal:** deterministic turn loop driven by external actions.

Loop responsibilities:
- Track whose turn it is.
- Validate and apply actions.
- Detect round end and apply awards.
- Detect game end and compute results.

**Notes:**
- The loop should be pure with respect to domain logic; no networking.
- All player inputs are provided externally (TODO hook for API).

### 5) Tests (strict, non‑circular)
**Testing order:**
1. Constructor validation (invalid values, invalid status cards, invalid states).
2. Round logic (bidding increases, pass rules, misfortune first‑pass behavior).
3. Theft logic (discard now vs pending).
4. Red‑edge termination (4th red card ends immediately, no award).
5. Scoring and poorest elimination.

**Non‑circular guidance:**
- Tests should operate on explicit, known states (no tests that re‑use production logic to compute expectations).
- Use small, deterministic fixtures with explicit hands/cards.

### 6) Game Server Tests
- `new_game` creates unique game ids and independent state.
- Parallel games do not affect each other.
- Invalid action in one game doesn’t crash server and doesn’t affect other games.

## API Surface (Game Engine)
Suggested function set for a simple app + bots:
- `new_game(player_manifest, seed) -> game_id`
- `get_state(game_id) -> GameState`
- `legal_actions(game_id, player_id) -> list[Action]` (explicit card sets)
- `apply_action(game_id, player_id, action) -> GameState`
- `is_terminal(game_id) -> bool`
- `score_game(game_id) -> Results`

Simple `Action` shape (Python):
- `Action(kind: PASS|BID, cards: tuple[int, ...])` where `cards` are denominations from the player’s hand.

Notes:
- `theft_pending` means the player must discard their *next* possession card (and the Theft card itself).
- `titles` can be an int count; no need to store separate title cards unless UI needs it.

## Round Flow (State Machine)
1. **Start Round**
   - Reveal top status card.
   - If it is red‑edged, increment `red_revealed`.
   - If `red_revealed == 4`, end game immediately (do **not** award this card).
   - Initialize `RoundState` with `starting_player`, `turn_player`, and `highest_bid = 0`.

2. **Player Turn** (clockwise)
   - **Action: Bid**
     - Player selects one or more **specific** money cards from hand (explicit set).
     - Add to `open_bid`; update `highest_bid` and `highest_bidder`.
     - Constraint: new total must be strictly greater than current highest.
   - **Action: Pass**
     - Set `passed = true`.
     - If possession/title round: return `open_bid` to hand.
     - If misfortune round: mark `first_passer` if unset; return `open_bid` to hand.
   - Continue to next active player.

3. **End Round**
   - **Possession/Title card**
     - End when only one non‑passed player remains.
     - Winner pays: move their `open_bid` to `money_discard`.
     - If `any_bid == false` (everyone passed immediately), the *last player to act* gets the card for free.
   - **Misfortune card**
     - End immediately when the first player passes.
     - First passer receives the misfortune card; they keep their money (their bid returns to hand).
     - All other players discard their `open_bid`.

4. **Apply Card Effects**
   - **Possession**: add to `possessions` unless `theft_pending` is true.
     - If `theft_pending`, discard this possession immediately and clear `theft_pending`.
   - **Title**: `titles += 1`.
   - **Scandal**: `scandal = true`.
   - **Gambling Debt**: `debt = true`.
   - **Theft**: if player has a possession, discard one immediately; else set `theft_pending = true`.

5. **Next Round**
   - New `starting_player` = player who received the status card this round.
   - Reset each player’s `open_bid` and `passed`.

## End Game & Scoring
1. **Poorest Elimination**
   - Sum remaining money in each player’s hand.
   - Player(s) with the least money are **out** and cannot win.

2. **Status Total** (for remaining players)
   - Base = sum of possession values.
   - If `debt`, subtract 5.
   - Multiply by `2 ^ titles`.
   - If `scandal`, multiply by 0.5 (integer handling: keep as float or use rational).

3. **Winner**
   - Highest status total wins.
   - Tiebreaker: more money in hand.
   - Further tie: all tied players win.

## API Surface (Game Engine)
Suggested function set for a simple app + bots:
- `newGame(playerCount, seed) -> GameState`
- `startRound(game) -> GameState` (reveals card, checks game end)
- `legalActions(game, playerId) -> list[Action]` (includes explicit card sets)
- `applyAction(game, playerId, action) -> GameState`
- `getObservation(game, playerId) -> Observation` (partial info)
- `isTerminal(game) -> bool`
- `scoreGame(game) -> Results`

Simple `Action` shape (Python):
- `Action(kind: PASS|BID, cards: tuple[int, ...])` where `cards` are denominations from the player’s hand.

## Edge Cases & Tests
- All players pass without any bid on a possession/title card: last to act gets it for free.
- Misfortune: round ends on first pass; all other bidders discard their open money.
- Theft when no possessions: mark pending, discard the next possession received.
- 4th red‑edged card revealed: immediate end, card not awarded.
- Ties in score and money are allowed; multiple winners possible.
