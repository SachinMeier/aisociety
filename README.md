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

## Bots (how to use them)
Available bot types today:
- `random`: picks a legal move at random
- `heuristic`: simple rule-based bidding with a risk style
- `static`: bids with fixed budgets by card type
- `mlp`: PyTorch MLP policy/value model loaded from a checkpoint

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
Training uses a simple policy-gradient loop against random opponents.

Requirements:
- PyTorch installed (CPU or GPU build).
- Optional: PyYAML if you want YAML training specs.

1) Create a training spec (JSON or YAML). Example `runs/train_mlp.json`:
```json
{
  "episodes": 2000,
  "seed": 3,
  "player_count": 3,
  "learning_rate": 0.001,
  "hidden_sizes": [128, 128],
  "activation": "relu",
  "dropout": 0.0,
  "temperature": 1.0,
  "entropy_coef": 0.01,
  "value_coef": 0.5,
  "max_grad_norm": 1.0,
  "checkpoint_path": "checkpoints/mlp_v1",
  "checkpoint_every": 250,
  "device": "cpu"
}
```

2) Train the model:
- `python -m scripts.train --spec runs/train_mlp.json`

3) Use the checkpoint in a run spec (see below).

Checkpoint paths can be either:
- A directory (e.g. `checkpoints/mlp_v1/`) containing `model.pt`, `config.json`, and
  `encoder.json`.
- A single `.pt` file containing the model state + configs.

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

## Where to look next
- Game rules source of truth: `SPEC.md`
- Original game rules text: `RULES.md`
- Code entry points: `scripts/run.py`, `scripts/eval.py`, `scripts/train.py`
