# High Society — Bot + RL Plan (Python‑only)

## Implementation Decisions
- **Language**: Python only for engine, API, and ML.
- **No frontend** for now; optional CLI/terminal input for human players.
- **Domain purity**: core game rules have **no** dependency on ML libraries.
- **Determinism**: seedable RNG and reproducible matches.

## Tools (Python)
Core (standard library):
- `dataclasses`, `enum`, `typing`, `random`, `itertools`, `collections`
- `logging`, `json`, `pickle` (telemetry and checkpoints)

ML & numeric:
- `numpy` (feature encoding, baselines)
- `torch` (PyTorch) for MLP policy/value models

Training utilities:
- `tqdm` (progress bars)
- `matplotlib` (optional, quick eval plots)

Dev & testing:
- `pytest` (unit + integration tests)
- `ruff` (formatting/linting; choose this)
- `mypy` (optional static typing)

CLI / config:
- `typer` (training & eval scripts)
- `pydantic` (run spec validation)

Data/reporting (optional but recommended for eval):
- `pandas` (tables + CSV summaries)

## Architecture Goals
- Plug‑and‑play players: any mix of human, heuristic, and ML bots in the same game.
- Clean boundaries: domain logic ≠ game orchestration ≠ player logic ≠ ML training.
- Simple, DRY APIs with minimal coupling so components can be swapped.
- Operator‑friendly runs: configure, launch, and monitor batches of games/training without hardcoding.

## Layered Design (DDD Boundaries)
- **Domain** (`highsociety/domain`): cards, actions, rules, scoring, state transitions.
- **Application** (`highsociety/app`): game runner, turn loop, observations, legal actions.
- **Players** (`highsociety/players`): player interface + concrete implementations.
- **ML** (`highsociety/ml`): encoders, policies, models, training loops.
- **Infrastructure** (`highsociety/infra`): CLI adapters, logging, serialization, registries.
- **Ops** (`highsociety/ops`): run manager, experiment configs, metrics, dashboards.

## Core Interfaces

### Action / Observation
- `Action` dataclass: `kind: PASS | BID`, payload is **explicit money cards** (denominations). Each player has one of each denomination, so **values act as card IDs**.
- `Observation` dataclass: info‑set‑safe (no hidden opponent hands).
- `LegalActionSet`: list + optional mask for ML models.

### Player API (runtime)
Single interface used by the game runner for all players:

```python
class Player(Protocol):
    name: str
    def reset(self, game_config: GameConfig, player_id: int, seat: int) -> None: ...
    def act(self, obs: Observation, legal_actions: list[Action]) -> Action: ...
    def on_game_end(self, result: GameResult) -> None: ...
```

- `HumanCLIPlayer` implements the same API via terminal prompts.
- `HeuristicBot`, `LinearRLBot`, `MLPPolicyBot` also implement this API.

### Bot / Policy / Model (internal composition)
- `Policy`: `select_action(obs, legal_actions) -> Action`
- `Encoder`: `encode(obs) -> np.ndarray`, `action_mask(legal_actions) -> np.ndarray`
- `Model`: `forward(encoded_obs) -> logits, value`
- `BotPlayer(Player)`: wires `Encoder + Model + ActionSelector` into `act()`

This keeps training separate from runtime usage.

## Game Runner + Environment Adapter
- **GameRunner** orchestrates turn order and calls `Player.act()`.
- **EnvAdapter** wraps the runner with an RL‑style API:
  - `reset(seed) -> state`
  - `observe(player_id) -> observation`
  - `legal_actions(player_id) -> Action[]` (explicit card sets)
  - `step(player_id, action) -> (next_state, reward, done, info)` (action must include the exact cards played)
  - `clone()` and `serialize()` for debugging/search

## Plug‑and‑Play Composition
Use a registry + config‑driven factory so player mixes are trivial.

Example config:
```python
players = [
  {"type": "human_cli"},
  {"type": "heuristic", "style": "cautious"},
  {"type": "heuristic", "style": "aggressive"},
  {"type": "linear_rl", "checkpoint": "checkpoints/lin_v3.pkl"},
  {"type": "mlp", "checkpoint": "checkpoints/mlp_v2.pt"},
]
```

The game runner only depends on the Player API, so any combination works.

## Detailed File Structure (proposed)
```
highsociety/
  app/
    runner.py            # GameRunner (single game loop)
    env_adapter.py       # RL-style wrapper over GameRunner
    observations.py      # Observation dataclass + builders
    legal_actions.py     # Action generation helpers
  players/
    base.py              # Player Protocol + base utilities
    registry.py          # PlayerFactory + registry
    human_cli.py         # HumanCLIPlayer (terminal input)
    random_bot.py        # RandomBot
    heuristic_bot.py     # HeuristicBot (parametrized)
    linear_rl_bot.py     # LinearRLBot (loads linear model)
    mlp_bot.py           # MLPPolicyBot (loads torch model)
  ml/
    encoders/
      base.py            # Encoder interface
      basic.py           # Feature vector encoder
    models/
      linear.py          # Linear policy/Q model
      mlp.py             # Torch MLP (policy + value)
    policies/
      base.py            # Policy interface
      epsilon_greedy.py  # Exploration wrapper
      softmax.py         # Temperature sampler
    training/
      linear_train.py    # Linear RL training loop
      mlp_train.py       # MLP training loop
    checkpoints.py       # Save/load + metadata
  ops/
    spec.py              # RunSpec + PlayerSpec (pydantic)
    runner.py            # RunManager (batch runs)
    metrics.py           # Metrics aggregation
    artifacts.py         # CSV/JSON writers
    elo.py               # ELO calculator (optional)
    dashboard/           # Streamlit or FastAPI+HTMX app (phase 2)
  infra/
    logging.py           # Structured logging helpers
    serialization.py     # JSON helpers for artifacts
scripts/
  run.py                 # RunManager CLI
  eval.py                # Evaluation CLI
  train.py               # Training CLI
  dashboard.py           # Optional dashboard entrypoint
```

## Key Classes & Types (concrete)
- `PlayerSpec` (pydantic): `type`, `params`, `checkpoint`, `name`.
- `RunSpec` (pydantic): `mode`, `seed`, `num_games`, `players`, `rules`.
- `RunResult`: summary metrics + artifact paths.
- `Observation`: public + private fields (explicit schema).
- `Action`: explicit card set for bids.
- `PlayerRegistry`: `register(type, factory)` and `create(spec)`.
- `GameRunner`: single-game turn loop, uses `GameServer` or domain rules.
- `EnvAdapter`: RL step/observe API with action index mapping.
- `Encoder`, `Policy`, `Model`: ML interfaces.
- `Checkpoint`: `save(path, model, encoder_cfg, metrics)` / `load(path)`.

## Operator API (runs, training, evaluation)
Goal: avoid hardcoding agents in code. Use config files + a simple runner API.

### Run Spec (YAML/JSON)
Defines a **batch** of games or a training run:
- `mode`: `play` | `eval` | `train`
- `seed`: base RNG seed
- `num_games`: number of games to run
- `players`: list of player specs (type + params)
- `rules`: optional flags (if variants added later)

Example:
```yaml
mode: eval
seed: 42
num_games: 100
players:
  - type: heuristic
    style: cautious
  - type: linear_rl
    checkpoint: checkpoints/lin_v3.pkl
  - type: mlp
    checkpoint: checkpoints/mlp_v2.pt
  - type: random
```

### Run Manager
`RunManager.run(spec)`:
- Instantiates players via registry
- Executes games via `GameRunner`
- Streams metrics (wins, average score, poorest rate)
- Saves a summary report + per‑game logs

## Evaluation Artifacts (tables, charts, tracking)
**Goal:** make results comparable, repeatable, and easy to inspect after large batches.

Artifacts per run:
- `summary.json`: overall win rates, average scores, poorest rate, finish rate.
- `results.csv`: per‑game outcomes (winner, scores, remaining money, round count).
- `head_to_head.csv`: win matrix between agent types.
- `elo.json` (optional): ELO ratings over time.

Charts (optional):
- Win‑rate over time (rolling window).
- Average score over time.
- ELO progression per bot.

## Evaluation CLI / Reports
- `scripts/eval.py --spec runs/eval_1000.yaml`
- Outputs artifacts to `runs/<timestamp>/`.
- Optional plotting via `matplotlib` into `runs/<timestamp>/plots/`.

## Dashboard (Run Control + Results)
**Phase 1 (lightweight):** Streamlit or FastAPI+HTMX.
**Phase 2 (richer):** simple web dashboard with run queue + charts.

Core dashboard features:
- Launch runs from a saved spec (or fill a simple form).
- Monitor progress (games completed, live win rates).
- View run artifacts (tables + charts).
- Compare runs (diff win rates or ELO changes).

Implementation notes:
- Dashboard talks to `RunManager` (same API as CLI) to avoid duplicate logic.
- Store all artifacts in `runs/` so CLI and dashboard share results.

### CLI / Dashboard
Phase 1 (CLI):
- `python -m scripts.run --spec runs/eval_100.yaml`
- `python -m scripts.train --spec runs/train_linear.yaml`
- `python -m scripts.play_cli --players runs/humans_vs_bots.yaml`

Phase 2 (simple dashboard):
- Minimal web UI (FastAPI + HTMX or Streamlit) to submit specs and watch progress.
- Not required now; CLI first.

## Observation Design (info‑set safe)
- Current status card (type + value or misfortune kind).
- Public round info: current highest bid, highest bidder, open bids per player, passed flags, turn seat.
- Private: your hand (sorted), possessions, titles, misfortune flags, theft pending.
- Public history: revealed status cards, remaining deck counts, red‑edge count, money discarded per player.

### Observation Schema (concrete, human‑readable)
The observation is a **player‑centric view** of the game state. It contains **everything public** plus the player’s private hand:

```python
Observation(
  player_id: int,
  status_card: {kind, value|misfortune},
  round: {
    turn_player: int,
    highest_bid: int,
    highest_bidder: int | None,
    open_bids: dict[player_id, tuple[int, ...]],  # explicit card values
    passed: dict[player_id, bool],
  },
  self: {
    hand: tuple[int, ...],                # your remaining money cards
    possessions: tuple[int, ...],         # values only
    titles: int,
    scandal: bool,
    debt: bool,
    theft_pending: bool,
  },
  public: {
    revealed_status: tuple[StatusCard, ...],
    remaining_counts: {possession: int, title: int, misfortune: int},
    red_revealed: int,
    money_discarded: dict[player_id, tuple[int, ...]],
  },
)
```

Notes:
- **No hidden opponent hands** are exposed.
- Bids are **explicit card sets** because cards are face‑up in the rules.
- `remaining_counts` can be derived from revealed cards + known deck composition.

### Feature Encoders
- **Counts**: money card counts by value.
- **Totals**: possession sum, titles count, remaining money sum.
- **Round**: highest bid, gap to lead, number of active players.
- **Deck**: remaining counts of possessions/titles/misfortunes.

## Action Encoding Strategy
**Bids must specify the exact money cards spent** because denominations matter and are not interchangeable.

Recommended v1:
- `BID(cards)` where `cards` are denominations chosen from the player’s current hand.
- Legal action generation enumerates **all subsets** of the hand that strictly increase the current bid.
- For ML, use an **action index** that maps to a specific subset (the `legal_actions` list for that observation).

## Bot Types
- **RandomBot**: baseline for sanity checks.
- **HeuristicBot**: rule‑based with tunable risk parameters.
- **LinearRLBot**: linear policy or Q‑function on discretized features.
- **MLPPolicyBot**: small PyTorch MLP with policy + value heads.

## What is a “Bot” Artifact?
For ML bots, a **checkpoint** is a directory or file containing:
- **Weights** (e.g., `.pt` for PyTorch, `.pkl` for linear models)
- **Encoder config** (feature layout, action encoding version)
- **Model config** (input size, hidden sizes, activation)
- **Training metadata** (git hash, seed, win‑rate, date)

Recommended structure:
```
checkpoints/
  mlp_v2/
    model.pt
    encoder.json
    config.json
    metrics.json
```

Heuristic bots don’t need weights; just config params.

## Reward Design
- Terminal only: `+1` win, `0` loss.
- Light shaping optional, but keep off by default for stable learning.

## Training Loop (self‑play)
1. Sample mixed agents (baseline + current policy).
2. Play episodes and collect trajectories.
3. Update model (batch updates every K episodes).
4. Evaluate vs baselines and older snapshots (ELO or win‑rate).
5. Save checkpoints with metadata (model config + encoder version).

## Train → Use Flow
1. **Train** with `scripts/train.py` using a run spec.
2. **Checkpoint** saved to `checkpoints/<name>/`.
3. **Register** the bot in a run spec (by path).
4. **Run games** (eval or play) with that bot in the `players` list.
5. **Review** metrics/logs in `runs/<timestamp>/`.

## Skill Levels (difficulty settings)
- Epsilon‑greedy exploration.
- Softmax temperature for stochastic action selection.
- Budget caps on bid size or lookahead depth.
- Policy mixing (e.g., 70% model, 30% heuristic).

## Suggested Package Layout
- `highsociety/domain/` — rules, cards, actions, scoring
- `highsociety/app/` — game runner, env adapter, observations
- `highsociety/players/` — Player API + human/heuristic/bot implementations
- `highsociety/ml/` — encoders, models, training code
- `highsociety/infra/` — CLI adapters, logging, config, registry
- `scripts/` — `train.py`, `eval.py`, `play_cli.py`

## PR‑Sized Work Breakdown (parallelizable)
Each item is a small, distinct slice with clear boundaries.

### Cutoff: “Full Run” with Random/Heuristic Bots
You can run **arbitrary batch runs** and track performance once PRs **1–6, 9, 10** are complete.
Everything below that is **post‑cutoff** (RL/ML work).

1) **Player API + Registry** *(pre‑cutoff)*
   - Add `players/base.py` Protocol, `players/registry.py` factory.
   - Minimal tests for registry creation.
   - Depends on: none (can be built standalone).

2) **Observation + Legal Actions** *(pre‑cutoff)*
   - `app/observations.py`, `app/legal_actions.py`.
   - Tests for info‑set correctness + explicit bid sets.
   - Depends on: domain types (`Action`, `GameState`) from core game.

3) **GameRunner + EnvAdapter** *(pre‑cutoff)*
   - `app/runner.py` wraps `GameServer` for a single game loop.
   - `app/env_adapter.py` for RL step/observe.
   - Tests for turn order + legal action exposure.
   - Depends on: domain + `GameServer`, plus Observation/legal actions (PR #2).

4) **Baseline Players** *(pre‑cutoff)*
   - `players/random_bot.py`, `players/heuristic_bot.py`.
   - Tests for action validity and determinism under seed.
   - Depends on: Player API (PR #1), Observation/legal actions (PR #2).

5) **RunSpec + RunManager** *(pre‑cutoff)*
   - `ops/spec.py`, `ops/runner.py`.
   - Tests for spec validation + multi‑game batching.
   - Depends on: Player Registry (PR #1) + GameRunner (PR #3).

6) **Metrics + Artifacts** *(pre‑cutoff)*
   - `ops/metrics.py`, `ops/artifacts.py` (CSV/JSON).
   - Tests for summary correctness and output schema.
   - Depends on: RunManager outputs (PR #5).

7) **Evaluation CLI** *(pre‑cutoff)*
   - `scripts/eval.py`, `scripts/run.py`.
   - Tests for dry‑run and artifact path creation.
   - Depends on: RunManager (PR #5) + Metrics/Artifacts (PR #6).

8) **Dashboard (optional)** *(pre‑cutoff)*
   - `ops/dashboard/` minimal Streamlit or FastAPI+HTMX.
   - Uses same RunManager and artifacts.
   - Depends on: RunManager (PR #5) + Metrics/Artifacts (PR #6).

--- 
**Post‑cutoff (RL/ML work begins here)**  

9) **Linear RL Bot** *(post‑cutoff)*
   - `ml/models/linear.py`, `ml/training/linear_train.py`.
   - `players/linear_rl_bot.py` loader.
   - Depends on: Encoder base + Observation schema (PR #2), Player API (PR #1).

10) **MLP Bot (Torch)** *(post‑cutoff)*
   - `ml/models/mlp.py`, `ml/training/mlp_train.py`.
   - `players/mlp_bot.py` loader.
   - Depends on: Encoder base + Observation schema (PR #2), Player API (PR #1).

## Minimal Deliverables
- Pure‑Python game engine with deterministic seed support.
- Player API + built‑in `HumanCLIPlayer`, `RandomBot`, `HeuristicBot`.
- RL harness with linear baseline and MLP option.
- Config‑driven match runner to mix any players in a single game.
