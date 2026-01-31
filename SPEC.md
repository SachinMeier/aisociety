# High Society — Source of Truth Spec

This spec is the authoritative reference for rules, actions, and observation schema.

## Core Rules
- **Players**: 3–5.
- **Money cards**: each player has exactly one of each denomination:
  `[25000, 20000, 15000, 12000, 10000, 8000, 6000, 4000, 3000, 2000, 1000]`.
- **Status deck** (16 cards):
  - 10 Possessions (values 1–10)
  - 3 Titles (no value)
  - 3 Misfortunes: Scandal, Gambling Debt, Theft
- **Red‑edged cards**: all 3 Titles + Scandal (4 total).

## Round Flow
1. Reveal top status card.
2. Starting player acts, then clockwise.
3. On your turn you either:
   - **Bid**: play a set of specific money cards (face‑up), added to your open bid.
   - **Pass**: take back your open bid and exit the round.
4. Bids must strictly increase the highest bid and are **cumulative** (no removing cards).

### End of Round
- **Possession/Title**:
  - Round ends when only one player has not passed.
  - That player takes the card and pays their open bid (discarded).
  - If all players pass without bidding, the **last player to act** gets the card for free.
- **Misfortune**:
  - Round ends immediately when the **first** player passes.
  - That player takes the misfortune card and **keeps** their money.
  - All other players discard their open bids.

### Theft
- If the player has a possession, they must discard one immediately.
- If not, they must discard the **first possession they later receive** (theft pending).

## Game End
- The game ends immediately when the **4th red‑edged card** is revealed.
- That card is **not awarded** and the remaining deck is ignored.

## Scoring
1. **Poorest elimination**: player(s) with least remaining money cannot win.
2. **Status total** for remaining players:
   - Sum possession values.
  - Subtract 5 per Gambling Debt card.
  - Multiply by `2^titles`.
  - Halve once per Scandal card.
3. Highest total wins.
4. Ties: higher remaining money wins; if still tied, all tied players win.

## Action Schema
- **PASS**: no payload.
- **BID**: explicit **set of money cards** (denominations). Denomination values act as card IDs.
- **DISCARD_POSSESSION**: possession value to discard (used only for Theft resolution).

## Observation Schema (Info‑set Safe)
A player sees all public info plus their own hand:
- Current status card (type + value/misfortune).
- Public round info: highest bid, highest bidder, open bids per player, passed flags, turn seat.
- Private: your hand, possessions, titles, misfortune counts, theft pending count.
- Public history: revealed status cards, remaining deck counts, red‑edge count, money discarded per player.

## Determinism
- Shuffling and any randomness must be seeded.
- Tests use fixed decks and seeds for reproducibility.
