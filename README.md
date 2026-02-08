# High Society AI

## Introduction
High Society AI is a Python project for simulating the card game High Society (Reiner Knizia).
It includes a deterministic game engine plus a couple of baseline bots, so you can run batches
of games and see who wins, without needing any machine learning background.

## Setup (from scratch)
1) Install Python 3.10 or newer.
2) Create and activate a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3) Install development tools (tests + linting):
   - `pip install -r requirements-dev.txt`
4) Optional: install PyYAML if you want YAML run specs:
   - `pip install pyyaml`

There is no build step. Run commands from the repo root.

## Local development
To match the GitHub Actions CI checks locally, install and run pre-commit:
- Install dev dependencies (includes pre-commit):
  - `pip install -r requirements-dev.txt`
- Install hooks:
  - `pre-commit install`
- Run the CI-parity checks on demand:
  - `pre-commit run --all-files`

CI-parity commands (run manually if you prefer):
- Lint: `ruff check .`
- Build: `python -m compileall highsociety`
- Test: `pytest`

## How to use it (run specs)
Games are configured with a JSON or YAML **run spec**. A spec describes:
- `mode`: label for the run (`play`, `eval`, or `train`)
- `seed`: base random seed for reproducibility
- `num_games`: how many games to run
- `players`: list of player types and options
- `rules`: optional settings (currently empty)

Example `spec.json`:
```json
{
  "mode": "eval",
  "seed": 42,
  "num_games": 50,
  "players": [
    {"type": "random", "name": "Rand A"},
    {"type": "random", "name": "Rand B"},
    {"type": "heuristic", "name": "Heuristic", "params": {"style": "balanced"}}
  ],
  "rules": {}
}
```

## Run games
- Run a batch:
  - `python -m scripts.run --spec spec.json`
- Run an evaluation batch (requires `mode: eval`):
  - `python -m scripts.eval --spec spec.json`
- Choose an output directory:
  - `python -m scripts.run --spec spec.json --output runs/my_run`
- Validate without running:
  - `python -m scripts.run --spec spec.json --dry-run`

Outputs (default: `runs/<timestamp>/`):
- `summary.json` with aggregate win rates and averages
- `results.csv` with per-game outcomes
- `spec.json` with the exact run spec used (for reproducibility)

## Streamlit dashboard (run + inspect)
The dashboard lets you run specs and inspect past runs from a web UI.

1) Ensure Streamlit is installed:
   - `pip install -r requirements-dev.txt`
   - (or) `pip install streamlit`
2) Start the app:
   - `streamlit run scripts/dashboard.py`
3) Open the URL Streamlit prints (usually `http://localhost:8501`).
4) To run a new spec from the UI, place a JSON run spec under `runs/`
   (the dropdown lists `runs/*.json`).

Dashboard pages include:
- Run from spec
- Build run
- Train bots
- Inspect runs
- Inspect trainings (per-training learner progress charts)

## Bots (how to use them)
Available bot types today:
- `random`: picks a legal move at random
- `heuristic`: simple rule-based bidding with a risk style
- `static`: bids with fixed budgets by card type
- `mlp`: PyTorch MLP policy/value model loaded from a checkpoint
- `linear_rl`: linear feature model loaded from a checkpoint

Heuristic styles:
- `cautious`
- `balanced`
- `aggressive`

Example player config:
```json
{
  "type": "heuristic",
  "name": "SafeBidder",
  "params": {"style": "cautious", "seed": 7}
}
```

MLP options:
- `checkpoint` (required): path to a `.pt` file or a checkpoint directory
- `params.temperature`: sampling temperature (0 = greedy, higher = more stochastic)
- `params.seed`: RNG seed for action sampling
- `params.device`: torch device string (e.g. `cpu`, `cuda`)

Example MLP player config:
```json
{
  "type": "mlp",
  "name": "MLP v1",
  "checkpoint": "checkpoints/mlp_v1",
  "params": {"temperature": 0.2, "seed": 7, "device": "cpu"}
}
```

## Training MLP bots
Training uses a simple policy-gradient loop with configurable opponents.

Requirements:
- PyTorch installed (CPU or GPU build).
- Optional: PyYAML if you want YAML training specs.

1) Create a training spec (JSON or YAML). Example `trainings/mlp/example.json`:
```json
{
  "episodes": 2000,
  "seed": 3,
  "player_count": 3,
  "learning_rate": 0.001,
  "learning_rate_final": 0.0001,
  "hidden_sizes": [128, 128],
  "activation": "relu",
  "dropout": 0.0,
  "temperature": 1.0,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "max_grad_norm": 1.0,
  "checkpoint_path": "checkpoints/mlp_v1",
  "checkpoint_every": 250,
  "artifacts_path": "trainings/history/mlp_v1",
  "resume": "checkpoints/mlp_v0",
  "opponents": [
    {"type": "heuristic", "params": {"style": "balanced"}},
    {"type": "heuristic", "params": {"style": "aggressive"}},
    {"type": "heuristic", "params": {"style": "cautious"}},
    {"type": "static"}
  ],
  "opponent_weights": [0.35, 0.25, 0.25, 0.15],
  "device": "cpu"
}
```

2) Train the model:
- `python -m scripts.train --spec trainings/mlp/example.json`

3) Use `resume` to continue from an existing MLP checkpoint, or leave it empty for a new run.

4) Use the checkpoint in a run spec (see below).

Optional learning-rate scheduling:
- Set `learning_rate_final` to linearly decay the optimizer learning rate from `learning_rate` to
  `learning_rate_final` over the full training run.

Training artifacts (when `artifacts_path` is set):
- `training_summary.json`
- `training_games.csv` (winners, poorest, score/money/status snapshots per game)
- `training_players.csv` (per-player rows with cumulative wins/poorest)

Checkpoint paths can be either:
- A directory (e.g. `checkpoints/mlp_v1/`) containing `model.pt`, `config.json`, and
  `encoder.json`.
- A single `.pt` file containing the model state + configs.

## Training linear bots
Training uses a linear model updated via self-play against a configurable opponent mix.

Requirements:
- Optional: PyYAML if you want YAML training specs.

1) Create a training spec (JSON or YAML). Example `trainings/linear/example.json`:
```json
{
  "num_games": 1000,
  "player_count": 4,
  "seed": 1,
  "epsilon": 0.15,
  "learning_rate": 0.05,
  "log_every": 1000,
  "output": "checkpoints/linear_v1.pkl",
  "artifacts_path": "trainings/history/linear_v1",
  "opponents": [
    {"type": "heuristic", "params": {"style": "balanced"}},
    {"type": "heuristic", "params": {"style": "aggressive"}},
    {"type": "heuristic", "params": {"style": "cautious"}},
    {"type": "static"}
  ],
  "opponent_weights": [0.25, 0.25, 0.25, 0.25],
  "learner_seats": [0]
}
```

2) Train the model:
- `python -m scripts.train_linear --spec trainings/linear/example.json`

3) Use the checkpoint in a run spec (see below).

You can also pass `--artifacts-path trainings/history/<run_name>` when using CLI flags.

Linear checkpoints can be either:
- A `.pkl` file (e.g. `checkpoints/linear_v1.pkl`).
- A directory (e.g. `checkpoints/linear_v1/`) containing `linear.pkl`.

## Run specs with ML bots
To run the MLP bot in a batch, include it in the `players` list with its checkpoint:

```json
{
  "mode": "eval",
  "seed": 42,
  "num_games": 50,
  "players": [
    {"type": "mlp", "name": "MLP v1", "checkpoint": "checkpoints/mlp_v1"},
    {"type": "heuristic", "name": "Heuristic", "params": {"style": "balanced"}},
    {"type": "random", "name": "Rand"}
  ],
  "rules": {}
}
```

To run the linear bot, use type `linear_rl` with a `.pkl` checkpoint:

```json
{
  "mode": "eval",
  "seed": 42,
  "num_games": 50,
  "players": [
    {
      "type": "linear_rl",
      "name": "Linear v1",
      "checkpoint": "checkpoints/linear_v1.pkl",
      "params": {"epsilon": 0.05, "seed": 7}
    },
    {"type": "heuristic", "name": "Heuristic", "params": {"style": "balanced"}},
    {"type": "random", "name": "Rand"}
  ],
  "rules": {}
}
```

## Local Play (pass-and-play in browser)

Play High Society locally against bots (or other humans on the same machine) through a browser UI.

### Prerequisites
- Python backend dependencies installed (see Setup above).
- FastAPI and uvicorn: `pip install fastapi uvicorn`
- Node.js 18+ and npm (for the frontend).

### Start the backend
```bash
python scripts/play_local.py
```
This starts the API server on `http://localhost:8000`. Options:
- `--port 8080` to change the port.
- `--reload` for auto-reload during development.

### Start the frontend
```bash
cd frontend
npm install
npm run dev
```
This starts the Vite dev server (usually `http://localhost:5173`) with API requests proxied to the backend.

For a production build: `npm run build` and serve the `frontend/dist/` directory.

### How to play
1. Open the frontend URL in your browser.
2. Click **New Game** and configure 3-5 seats (Human, Easy/Medium/Hard/Expert bots).
3. Click **Start Game** to begin.
4. For pass-and-play with multiple humans, the game shows a handoff screen between turns so players can pass the device privately.
5. Select an action (Pass or Bid) and click **Confirm**.
6. The game ends when all status cards are dealt; results are shown on the final screen.

### Difficulty presets
| Label  | Bot type   | Notes                          |
|--------|-----------|--------------------------------|
| Easy   | Static    | Fixed budget bidding           |
| Medium | Heuristic | Balanced risk style            |
| Hard   | MLP       | Neural net (requires checkpoint) |
| Expert | Heuristic | Cautious risk style            |

### API endpoints
- `POST /api/local-games` -- create a new game
- `GET /api/local-games/{game_id}` -- get current game state
- `GET /api/local-games/{game_id}/turn` -- get current turn view
- `POST /api/local-games/{game_id}/actions` -- submit a player action
- `GET /api/health` -- health check

## Where to look next
- Original game rules text: `RULES.md`
- Code entry points: `scripts/run.py`, `scripts/eval.py`, `scripts/train.py`
- Local play: `highsociety/app/local_game_service.py`, `highsociety/server/local_api.py`, `frontend/`
