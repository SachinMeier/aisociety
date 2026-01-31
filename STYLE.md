# Code Style & Consistency Standards

This document defines strict standards for clean, DRY, modular code in this repo. All contributors and subagents should follow these rules.

## 1) Design Principles
- Prefer **small, composable modules** with single responsibility.
- Keep **domain logic pure** (no I/O, no ML dependencies in domain).
- Prefer **explicit data structures** over implicit state.
- Favor **determinism**: seed RNG, avoid hidden global state.
- Use **clear boundaries** between Domain, Application, Players, ML, Ops, Infra.

## 2) File & Module Organization
- One primary class or concept per file.
- Folder names reflect domain boundaries (e.g., `domain`, `players`, `ml`, `ops`).
- Avoid circular imports; use local imports only when needed.

## 3) Naming Conventions
- Classes: `CamelCase`.
- Functions/methods/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Tests: `test_<behavior>_<context>()`.

## 4) Type Safety & Validation
- **Type hints are mandatory for all functions, methods, and class attributes**.
- Validate inputs at module boundaries.
- Raise domain‑specific errors (`InvalidAction`, `InvalidState`) instead of generic exceptions.

## 5) Error Handling
- Fail fast with clear error messages.
- Never crash the server on a single game error.
- Prefer structured error returns for recoverable errors.

## 6) DRY Rules
- Do not duplicate logic across modules.
- Shared logic goes into domain services or helpers.
- Test fixtures should be minimal and explicit.

## 7) Testing Standards
- Tests must validate behavior, not just setup.
- Avoid circular tests (don’t compute expected values with the same production logic being tested).
- Include edge cases explicitly.
- Use deterministic seeds in tests.

## 8) Documentation
- **Docstrings are mandatory for every module, class, and function** (public and private).
- Add comments only when logic is non‑obvious.
- Avoid redundant comments.

## 9) Formatting & Linting
- Follow `ruff` rules and keep formatting consistent.
- Line length: 100.
- No unused imports or variables.

## 10) Checkpoints & Artifacts
- Artifacts must be stored in structured folders (`checkpoints/`, `runs/`).
- Each checkpoint includes weights + config + metadata.

## 11) API Consistency
- Action interfaces must remain explicit about card sets (no target sums).
- All player implementations use the same `Player` protocol.
- Observations must be info‑set safe.
