"""Streamlit dashboard for running and inspecting games."""

from __future__ import annotations

import inspect
import json
import secrets
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from highsociety.ops.cli import execute_spec, load_spec, resolve_output_dir
from highsociety.ops.metrics import compute_summary
from highsociety.players.colors import DEFAULT_BOT_COLOR, resolve_bot_color

_TRAIN_BOT_TYPES = ("mlp", "linear")
_TRAIN_MLP_ACTIVATIONS = ("relu", "tanh", "gelu")
_TRAIN_DEVICES = ("cpu", "cuda", "mps")
_STOP_GRACE_SECONDS = 10
_TRAINING_HISTORY_ROOT = Path("trainings") / "history"


def _load_json(path: Path) -> dict[str, object]:
    """Load a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def _info_box(streamlit, message: str) -> None:
    """Render a green-accented info callout."""
    streamlit.markdown(
        f"""
        <div class="hs-info">
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _stretch_width_kwargs(streamlit_callable: object) -> dict[str, object]:
    """Prefer width='stretch' while supporting older Streamlit signatures."""
    try:
        params = inspect.signature(streamlit_callable).parameters
    except (TypeError, ValueError):
        return {"use_container_width": True}
    if "width" in params:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _maybe_spinner(streamlit, message: str):
    """Return a spinner context manager when available."""
    spinner = getattr(streamlit, "spinner", None)
    if callable(spinner):
        return spinner(message)
    return nullcontext()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _list_spec_files(spec_root: Path) -> list[Path]:
    """Return sorted JSON spec files from a root folder."""
    if not spec_root.exists():
        return []
    return sorted(
        [path for path in spec_root.iterdir() if path.is_file() and path.suffix == ".json"],
        key=lambda path: path.name,
    )


def _next_gui_spec_path(spec_root: Path, prefix: str = "gui") -> Path:
    """Create a unique spec file path under the runs folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = spec_root / f"{prefix}_{timestamp}.json"
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = spec_root / f"{prefix}_{timestamp}_{index}.json"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to allocate a unique spec filename.")


def _checkpoint_options(bot_type: str) -> list[str]:
    """List checkpoint paths from checkpoints/<bot_type>."""
    root = Path("checkpoints") / bot_type
    if not root.exists() or not root.is_dir():
        return []
    options: list[str] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.name.startswith("."):
            continue
        if path.is_file() or path.is_dir():
            options.append(str(path))
    return options


def _default_checkpoint_name(bot_type: str) -> str:
    """Return a default checkpoint name for a training run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{bot_type}_{timestamp}"


def _resolve_training_checkpoint_path(
    bot_type: str,
    checkpoint_name: str,
    output_path: str,
) -> Path:
    """Resolve the checkpoint output path from form inputs."""
    explicit_output = output_path.strip()
    if explicit_output:
        return Path(explicit_output)
    name = checkpoint_name.strip()
    if not name:
        raise ValueError("Checkpoint name is required when output path is blank.")
    if bot_type not in _TRAIN_BOT_TYPES:
        raise ValueError(f"Unsupported bot type: {bot_type}")
    return Path("checkpoints") / bot_type / name


def _training_spec_root(bot_type: str) -> Path:
    """Return the output folder used for generated training specs."""
    if bot_type not in _TRAIN_BOT_TYPES:
        raise ValueError(f"Unsupported bot type: {bot_type}")
    return Path("trainings") / bot_type


def _training_command(bot_type: str, spec_path: Path) -> list[str]:
    """Return the CLI command used for a training job."""
    if bot_type == "mlp":
        module = "scripts.train"
    elif bot_type == "linear":
        module = "scripts.train_linear"
    else:
        raise ValueError(f"Unsupported bot type: {bot_type}")
    return [sys.executable, "-m", module, "--spec", str(spec_path)]


def _parse_optional_json_list(
    raw_text: str,
    field_label: str,
) -> tuple[list[object] | None, str | None]:
    """Parse an optional JSON list from a text input."""
    text = raw_text.strip()
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, f"{field_label} must be valid JSON."
    if not isinstance(parsed, list):
        return None, f"{field_label} must be a JSON list."
    return parsed, None


def _parse_int_csv(
    raw_text: str,
    field_label: str,
    *,
    allow_empty: bool,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Parse comma-separated integer values from free text."""
    text = raw_text.strip()
    if not text:
        if allow_empty:
            return (), None
        return None, f"{field_label} cannot be empty."
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        if allow_empty:
            return (), None
        return None, f"{field_label} cannot be empty."
    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            return None, f"{field_label} must contain integers separated by commas."
        if value < 0:
            return None, f"{field_label} values must be zero or positive."
        values.append(value)
    return tuple(values), None


def _training_jobs(session_state: dict[str, object]) -> list[dict[str, object]]:
    """Return the normalized list of training job records."""
    raw_jobs = session_state.get("training_jobs")
    if not isinstance(raw_jobs, list):
        raw_jobs = []
        session_state["training_jobs"] = raw_jobs
    jobs = [job for job in raw_jobs if isinstance(job, dict)]
    if len(jobs) != len(raw_jobs):
        session_state["training_jobs"] = jobs
    return jobs


def _new_training_job_id(bot_type: str) -> str:
    """Create a short unique id for a training job."""
    return f"{bot_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


def _staging_checkpoint_path(job_id: str, target_path: Path) -> Path:
    """Return an isolated staging checkpoint path for a job."""
    stage_root = Path("trainings") / "staging" / job_id
    if target_path.suffix:
        return stage_root / f"checkpoint{target_path.suffix}"
    return stage_root / "checkpoint"


def _training_history_path(job_id: str) -> Path:
    """Return the training history artifact directory for a job."""
    return _TRAINING_HISTORY_ROOT / job_id


def _remove_path(path: Path) -> None:
    """Remove a file or directory path when it exists."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _promote_staged_checkpoint(job: dict[str, object]) -> str | None:
    """Move staged checkpoint artifacts to their final path."""
    staging_value = job.get("staging_checkpoint_path")
    target_value = job.get("checkpoint_path")
    if not isinstance(staging_value, str) or not staging_value:
        return "Missing staged checkpoint path."
    if not isinstance(target_value, str) or not target_value:
        return "Missing target checkpoint path."
    source = Path(staging_value)
    target = Path(target_value)
    if not source.exists():
        return f"Staged checkpoint not found: {source}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _remove_path(target)
    try:
        source.replace(target)
    except OSError:
        shutil.move(str(source), str(target))
    stage_root = source.parent
    if stage_root.exists() and not any(stage_root.iterdir()):
        stage_root.rmdir()
    return None


def _cleanup_staged_checkpoint(job: dict[str, object]) -> None:
    """Delete staged checkpoint artifacts for a job."""
    staging_value = job.get("staging_checkpoint_path")
    if not isinstance(staging_value, str) or not staging_value:
        return
    source = Path(staging_value)
    _remove_path(source)
    stage_root = source.parent
    if stage_root.exists() and not any(stage_root.iterdir()):
        stage_root.rmdir()


def _request_stop_training_job(job: dict[str, object], *, force: bool = False) -> bool:
    """Request termination of a running training job."""
    process = job.get("process")
    if not isinstance(process, subprocess.Popen):
        return False
    if process.poll() is not None:
        return False
    if force:
        process.kill()
    else:
        process.terminate()
    job["stop_requested"] = True
    job["stop_requested_at"] = time.time()
    job["status"] = "stopping"
    return True


def _maybe_force_stop(job: dict[str, object]) -> None:
    """Force-kill a job after grace period once stop has been requested."""
    if job.get("status") != "stopping":
        return
    process = job.get("process")
    if not isinstance(process, subprocess.Popen):
        return
    if process.poll() is not None:
        return
    requested_at = job.get("stop_requested_at")
    if not isinstance(requested_at, (int, float)):
        return
    if (time.time() - float(requested_at)) < _STOP_GRACE_SECONDS:
        return
    process.kill()


def _refresh_training_jobs(session_state: dict[str, object]) -> list[dict[str, object]]:
    """Update job statuses by polling subprocesses."""
    jobs = _training_jobs(session_state)
    for job in jobs:
        if job.get("status") not in {"running", "stopping"}:
            continue
        process = job.get("process")
        if not isinstance(process, subprocess.Popen):
            continue
        _maybe_force_stop(job)
        return_code = process.poll()
        if return_code is None:
            continue
        job["exit_code"] = int(return_code)
        job["completed_at"] = datetime.now().isoformat(timespec="seconds")
        stop_requested = bool(job.get("stop_requested"))
        if stop_requested:
            job["status"] = "stopped"
            _cleanup_staged_checkpoint(job)
        elif return_code == 0:
            promotion_error = _promote_staged_checkpoint(job)
            if promotion_error is None:
                job["status"] = "completed"
            else:
                job["status"] = "failed"
                job["promotion_error"] = promotion_error
        else:
            job["status"] = "failed"
            _cleanup_staged_checkpoint(job)
    return jobs


def _start_training_job(
    bot_type: str,
    job_id: str,
    spec_path: Path,
    checkpoint_path: Path,
    staging_checkpoint_path: Path,
    training_artifacts_path: Path,
) -> dict[str, object]:
    """Launch a background training process and return its job record."""
    logs_root = Path("trainings") / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / f"{job_id}.log"
    command = _training_command(bot_type, spec_path)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=str(Path.cwd()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    return {
        "id": job_id,
        "bot_type": bot_type,
        "status": "running",
        "pid": process.pid,
        "command": " ".join(command),
        "spec_path": str(spec_path),
        "checkpoint_path": str(checkpoint_path),
        "staging_checkpoint_path": str(staging_checkpoint_path),
        "training_artifacts_path": str(training_artifacts_path),
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
        "exit_code": None,
        "stop_requested": False,
        "stop_requested_at": None,
        "process": process,
    }


def _tail_text(path: Path, max_lines: int = 60) -> str:
    """Return the tail of a log file for dashboard display."""
    if not path.exists():
        return "Log file not available yet."
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    if not lines:
        return "(no log output yet)"
    return "".join(lines[-max_lines:]).strip()


def _running_training_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return jobs that still have an active process."""
    return [job for job in jobs if job.get("status") in {"running", "stopping"}]


def _random_seed() -> int:
    """Return a random seed suitable for the run spec."""
    return secrets.randbelow(2_147_483_647)


def _format_params(params: object) -> str:
    """Format params dict into a short label."""
    if not isinstance(params, dict) or not params:
        return ""
    preferred = ["style", "seed"]
    parts = [f"{key}={params[key]}" for key in preferred if key in params]
    for key in sorted(params):
        if key in preferred:
            continue
        parts.append(f"{key}={params[key]}")
    return ", ".join(str(part) for part in parts if part)


def _normalize_player_params(player: dict[str, object]) -> dict[str, object]:
    """Normalize a player spec into a params dict."""
    params = player.get("params")
    if not isinstance(params, dict):
        params = {}
    else:
        params = dict(params)
    if "style" in player and "style" not in params:
        params["style"] = player["style"]
    if "seed" in player and "seed" not in params:
        params["seed"] = player["seed"]
    return params


def _append_players(
    players: list[dict[str, object]],
    count: int,
    player_type: str,
    name_prefix: str,
    params: dict[str, object] | None = None,
    checkpoint: str | None = None,
) -> None:
    """Append repeated player specs to a list."""
    if count <= 0:
        return
    for index in range(count):
        entry: dict[str, object] = {
            "type": player_type,
            "name": f"{name_prefix} {index + 1}",
        }
        if checkpoint:
            entry["checkpoint"] = checkpoint
        if params:
            entry["params"] = dict(params)
        players.append(entry)


def _player_label(index: int, spec_data: dict[str, object] | None) -> str:
    """Build a label for a player id using the run spec."""
    if spec_data:
        players = spec_data.get("players")
        if isinstance(players, list) and 0 <= index < len(players):
            player = players[index] if isinstance(players[index], dict) else {}
            name = player.get("name")
            player_type = player.get("type")
            params = _normalize_player_params(player)
            display = str(name) if name else (str(player_type) if player_type else f"Player {index}")
            details: list[str] = []
            if name and player_type:
                details.append(str(player_type))
            params_label = _format_params(params)
            if params_label:
                details.append(params_label)
            if details:
                display = f"{display} ({', '.join(details)})"
            return f"{index}: {display}"
    return f"{index}: Player {index}"


def _player_color(index: int, spec_data: dict[str, object] | None) -> str | None:
    """Return the configured bot color for a player id."""
    if spec_data:
        players = spec_data.get("players")
        if isinstance(players, list) and 0 <= index < len(players):
            player = players[index] if isinstance(players[index], dict) else {}
            player_type = player.get("type")
            if isinstance(player_type, str) and player_type:
                params = _normalize_player_params(player)
                return resolve_bot_color(player_type, params, default=DEFAULT_BOT_COLOR)
    return None


def _parse_player_index(key: object) -> int:
    """Convert a player id key into an int for sorting."""
    try:
        return int(key)
    except (TypeError, ValueError):
        return -1


def _chart_frame(
    data: dict[str, object],
    labels: dict[int, str],
    colors: dict[int, str] | None = None,
):
    """Build a DataFrame for charting."""
    import pandas as pd

    rows = []
    for key, value in data.items():
        index = _parse_player_index(key)
        label = labels.get(index, str(key))
        row = {"player": label, "value": value, "order": index}
        if colors is not None:
            row["color"] = colors.get(index, DEFAULT_BOT_COLOR)
        rows.append(row)
    rows.sort(key=lambda row: row["order"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame


def _chart_dataframe(data: dict[str, object], labels: dict[int, str]):
    """Build a DataFrame for a Streamlit bar chart."""
    frame = _chart_frame(data, labels)
    if frame.empty:
        return frame
    frame = frame.set_index("player")
    return frame.drop(columns=["order"])


def _count_rate_dataframe(
    counts: dict[str, object],
    rates: dict[str, object] | None,
    labels: dict[int, str],
    denominator: int,
    colors: dict[int, str] | None = None,
):
    """Build a DataFrame containing count/rate pairs."""
    import pandas as pd

    rows = []
    rate_lookup = rates if isinstance(rates, dict) else {}
    for key, value in counts.items():
        index = _parse_player_index(key)
        label = labels.get(index, str(key))
        count_value = float(value) if isinstance(value, (int, float)) else 0.0
        rate_value: float | None = None
        if isinstance(rate_lookup, dict):
            raw_rate = rate_lookup.get(key)
            if raw_rate is None and index != -1:
                raw_rate = rate_lookup.get(str(index), rate_lookup.get(index))
            if isinstance(raw_rate, (int, float)):
                rate_value = float(raw_rate)
        if rate_value is None:
            rate_value = count_value / denominator if denominator else 0.0
        row = {"player": label, "count": count_value, "rate": rate_value, "order": index}
        if colors is not None:
            row["color"] = colors.get(index, DEFAULT_BOT_COLOR)
        rows.append(row)
    rows.sort(key=lambda row: row["order"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop(columns=["order"])


def _count_rate_chart(
    counts: dict[str, object],
    rates: dict[str, object] | None,
    labels: dict[int, str],
    colors: dict[int, str] | None,
    games_finished: object,
    count_title: str,
    rate_title: str,
):
    """Build a dual-axis bar chart for count/rate pairs."""
    try:
        import altair as alt
    except ImportError:
        return None
    denominator = 1
    if isinstance(games_finished, (int, float)) and games_finished > 0:
        denominator = int(games_finished)
    frame = _count_rate_dataframe(counts, rates, labels, denominator, colors)
    if frame.empty:
        return None
    player_order = frame["player"].tolist()
    encodings = {
        "x": alt.X("player:N", sort=player_order, axis=alt.Axis(title="Player")),
        "y": alt.Y(
            "count:Q",
            axis=alt.Axis(title=count_title, orient="left", format="d"),
        ),
        "tooltip": [
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("count:Q", title=count_title, format="d"),
            alt.Tooltip("rate:Q", title=rate_title, format=".1%"),
        ],
    }
    if "color" in frame.columns:
        encodings["color"] = alt.Color("color:N", scale=None, legend=None)
    base = alt.Chart(frame).mark_bar().encode(**encodings)
    rate_axis = alt.Chart(frame).mark_rule(opacity=0).encode(
        y=alt.Y(
            "count:Q",
            axis=alt.Axis(
                title=rate_title,
                orient="right",
                labelExpr=f"format(datum.value / {denominator}, '.0%')",
            ),
        )
    )
    return alt.layer(base, rate_axis).resolve_scale(y="shared").resolve_axis(y="independent")


def _bar_chart(
    data: dict[str, object],
    labels: dict[int, str],
    colors: dict[int, str] | None,
    value_title: str,
):
    """Build a single-metric bar chart with optional colors."""
    try:
        import altair as alt
    except ImportError:
        return None
    frame = _chart_frame(data, labels, colors)
    if frame.empty:
        return None
    player_order = frame["player"].tolist()
    encodings = {
        "x": alt.X("player:N", sort=player_order, axis=alt.Axis(title="Player")),
        "y": alt.Y("value:Q", axis=alt.Axis(title=value_title)),
        "tooltip": [
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("value:Q", title=value_title),
        ],
    }
    if "color" in frame.columns:
        encodings["color"] = alt.Color("color:N", scale=None, legend=None)
    return alt.Chart(frame).mark_bar().encode(**encodings)


def _special_card_frames(
    card_counts: dict[str, dict[str, object]],
    labels: dict[int, str],
    card_order: list[str],
    colors: dict[int, str] | None = None,
):
    """Build long/wide DataFrames for special card counts."""
    import pandas as pd

    player_ids = sorted(labels)
    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    for pid in player_ids:
        label = labels[pid]
        wide_row: dict[str, object] = {"player": label}
        for card in card_order:
            data = card_counts.get(card)
            if not isinstance(data, dict):
                data = {}
            raw = data.get(str(pid), data.get(pid, 0))
            count_value = float(raw) if isinstance(raw, (int, float)) else 0.0
            row = {"player": label, "card": card, "count": count_value, "order": pid}
            if colors is not None:
                row["color"] = colors.get(pid, DEFAULT_BOT_COLOR)
            long_rows.append(row)
            wide_row[card] = count_value
        wide_rows.append(wide_row)
    long_frame = pd.DataFrame(long_rows)
    if long_frame.empty:
        return long_frame, pd.DataFrame()
    long_frame = long_frame.sort_values("order").drop(columns=["order"])
    wide_frame = pd.DataFrame(wide_rows).set_index("player")
    return long_frame, wide_frame


def _special_card_chart(
    card_counts: dict[str, dict[str, object]],
    labels: dict[int, str],
    card_order: list[str],
    colors: dict[int, str] | None = None,
):
    """Build a grouped bar chart for special card counts."""
    try:
        import altair as alt
    except ImportError:
        return None
    frame, _ = _special_card_frames(card_counts, labels, card_order, colors)
    if frame.empty:
        return None
    player_order = [labels[pid] for pid in sorted(labels)]
    encodings = {
        "x": alt.X("player:N", sort=player_order, axis=alt.Axis(title="Player")),
        "xOffset": alt.XOffset("card:N", sort=card_order),
        "y": alt.Y("count:Q", axis=alt.Axis(title="Card count", format="d")),
        "tooltip": [
            alt.Tooltip("player:N", title="Player"),
            alt.Tooltip("card:N", title="Card"),
            alt.Tooltip("count:Q", title="Count", format="d"),
        ],
    }
    if "color" in frame.columns:
        encodings["color"] = alt.Color("color:N", scale=None, legend=None)
    return alt.Chart(frame).mark_bar().encode(**encodings)


def _collect_player_ids(summary_data: dict[str, object]) -> set[int]:
    """Collect player ids from summary sections."""
    ids: set[int] = set()
    for key in (
        "win_counts",
        "win_rates",
        "poorest_counts",
        "poorest_rates",
        "title_counts",
        "scandal_counts",
        "debt_counts",
        "theft_counts",
        "cancel_counts",
    ):
        section = summary_data.get(key)
        if isinstance(section, dict):
            for player_id in section.keys():
                try:
                    ids.add(int(player_id))
                except (TypeError, ValueError):
                    continue
    return ids


def _has_completed_games(summary_data: dict[str, object]) -> bool:
    """Return True when summary metrics report at least one finished game."""
    games_finished = summary_data.get("games_finished")
    if isinstance(games_finished, bool):
        return False
    if isinstance(games_finished, (int, float)):
        return games_finished > 0
    return True


def _has_numeric_chart_data(data: object) -> bool:
    """Return True when a chart section has at least one numeric value."""
    if not isinstance(data, dict) or not data:
        return False
    for value in data.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return True
    return False


def _session_state(streamlit):
    """Return Streamlit session state, creating one for simple test doubles."""
    state = getattr(streamlit, "session_state", None)
    if state is None:
        state = {}
        streamlit.session_state = state
    return state


def _pop_session_message(streamlit, key: str) -> str | None:
    """Pop a one-time string message from session state."""
    message = _session_state(streamlit).pop(key, None)
    if isinstance(message, str) and message:
        return message
    return None


def _reset_run_from_spec_state(streamlit) -> None:
    """Clear pending state for the run-from-spec action."""
    session_state = _session_state(streamlit)
    session_state["run_spec_running"] = False
    session_state.pop("run_spec_pending_path", None)
    session_state.pop("run_spec_pending_output_root", None)


def _reset_build_run_state(streamlit) -> None:
    """Clear pending state for the build-run action."""
    session_state = _session_state(streamlit)
    session_state["build_run_running"] = False
    session_state.pop("build_run_pending_spec_path", None)
    session_state.pop("build_run_pending_output_root", None)
    session_state.pop("build_run_pending_seed_label", None)
    session_state.pop("build_run_pending_seed_value", None)
    session_state.pop("build_run_pending_spec_data", None)


def _page_run_from_spec(streamlit) -> None:
    """Run a spec selected from the runs folder."""
    streamlit.subheader("Run from spec")
    session_state = _session_state(streamlit)
    run_error = _pop_session_message(streamlit, "run_spec_error")
    if run_error:
        streamlit.error(run_error)
    is_running = bool(session_state.get("run_spec_running", False))
    spec_root = Path("runs")
    spec_files = _list_spec_files(spec_root)
    if not spec_files:
        if is_running:
            _reset_run_from_spec_state(streamlit)
            session_state["run_spec_error"] = "Run failed to execute."
            streamlit.rerun()
        _info_box(streamlit, f"No spec JSON files found in {spec_root}.")
        return
    spec_path = streamlit.selectbox(
        "Spec file",
        options=spec_files,
        format_func=lambda path: str(path.relative_to(spec_root)),
        key="run_spec_path",
    )
    output_root = streamlit.text_input(
        "Output directory (optional)",
        value="",
        key="run_spec_output_root",
    )
    dry_run = streamlit.checkbox("Dry run", value=False, key="run_spec_dry_run")
    run_clicked = streamlit.button("Run spec", key="run_spec_run", disabled=is_running)
    if run_clicked and not is_running:
        if not spec_path.exists():
            streamlit.error(f"Spec file not found: {spec_path}")
            return
        try:
            spec = load_spec(spec_path)
        except (OSError, ValueError) as exc:
            streamlit.error(f"Failed to load spec: {exc}")
            return
        if dry_run:
            streamlit.success(f"Spec validated: {spec.mode} ({spec.num_games} games)")
            return
        session_state["run_spec_running"] = True
        session_state["run_spec_pending_path"] = str(spec_path)
        session_state["run_spec_pending_output_root"] = output_root.strip()
        streamlit.rerun()
    if not is_running:
        _info_box(streamlit, "Provide a spec path and click Run to start.")
        return
    pending_spec_path = session_state.get("run_spec_pending_path")
    if not isinstance(pending_spec_path, str) or not pending_spec_path:
        _reset_run_from_spec_state(streamlit)
        session_state["run_spec_error"] = "Run failed to execute."
        streamlit.rerun()
    pending_output_root = session_state.get("run_spec_pending_output_root", "")
    output_root_value = pending_output_root if isinstance(pending_output_root, str) else ""
    output_dir = resolve_output_dir(Path(output_root_value) if output_root_value else None)
    spec_path = Path(pending_spec_path)
    try:
        result = execute_spec(spec_path, output_dir=output_dir, dry_run=False, write_outputs=True)
    except Exception as exc:  # noqa: BLE001
        _reset_run_from_spec_state(streamlit)
        session_state["run_spec_error"] = f"Run failed to execute: {exc}"
        streamlit.rerun()
    if result is None:
        _reset_run_from_spec_state(streamlit)
        session_state["run_spec_error"] = "Run failed to execute."
        streamlit.rerun()
    summary = compute_summary(result)
    _reset_run_from_spec_state(streamlit)
    streamlit.success(f"Run complete: {output_dir}")
    streamlit.json(summary.to_dict())
    _redirect_to_inspect(streamlit, output_dir)


def _page_build_run(streamlit) -> None:
    """Build a spec via GUI and run it."""
    streamlit.subheader("Build a run spec")
    session_state = _session_state(streamlit)
    build_error = _pop_session_message(streamlit, "build_run_error")
    if build_error:
        streamlit.error(build_error)
    spec_root = Path("runs")
    spec_root.mkdir(parents=True, exist_ok=True)
    is_running = bool(session_state.get("build_run_running", False))
    with streamlit.form("build_spec_form"):
        mode = streamlit.selectbox("Mode", ("eval", "play"), index=0, key="build_mode")
        num_games = streamlit.number_input(
            "Number of games",
            min_value=1,
            value=50,
            step=1,
            key="build_num_games",
        )
        seed_text = streamlit.text_input("Seed (optional)", value="", key="build_seed")
        streamlit.markdown("**Bots**")
        col1, col2, col3 = streamlit.columns(3)
        random_count = col1.number_input(
            "Random bots",
            min_value=0,
            value=1,
            step=1,
            key="build_random",
        )
        static_count = col3.number_input(
            "Static bots",
            min_value=0,
            value=1,
            step=1,
            key="build_static",
        )
        streamlit.markdown("**Heuristic styles**")
        hcol1, hcol2, hcol3 = streamlit.columns(3)
        heuristic_balanced = hcol1.number_input(
            "Balanced bots",
            min_value=0,
            value=1,
            step=1,
            key="build_heuristic_balanced",
        )
        heuristic_cautious = hcol2.number_input(
            "Cautious bots",
            min_value=0,
            value=0,
            step=1,
            key="build_heuristic_cautious",
        )
        heuristic_aggressive = hcol3.number_input(
            "Aggressive bots",
            min_value=0,
            value=0,
            step=1,
            key="build_heuristic_aggressive",
        )
        streamlit.markdown("---")
        streamlit.markdown("**Advanced bots**")
        mlp_count = streamlit.number_input(
            "MLP bots",
            min_value=0,
            value=0,
            step=1,
            key="build_mlp",
        )
        mlp_checkpoint_options = ["", *_checkpoint_options("mlp")]
        mlp_checkpoint = streamlit.selectbox(
            "MLP checkpoint path",
            options=mlp_checkpoint_options,
            format_func=lambda path: path if path else "Select an MLP checkpoint",
            key="build_mlp_checkpoint_select",
        )
        streamlit.caption("Options loaded from `checkpoints/mlp`.")
        with streamlit.expander("MLP advanced settings"):
            mlp_temperature = streamlit.number_input(
                "MLP temperature",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="build_mlp_temperature",
            )
            streamlit.caption(
                "Controls sampling randomness; 0 is greedy, higher values are more stochastic."
            )
            mlp_device = streamlit.selectbox(
                "MLP device",
                ("cpu", "cuda", "mps"),
                index=0,
                key="build_mlp_device",
            )
            streamlit.caption(
                "Torch device for model tensors; use cpu unless you have a CUDA or Apple GPU."
            )
        linear_count = streamlit.number_input(
            "Linear RL bots",
            min_value=0,
            value=0,
            step=1,
            key="build_linear",
        )
        linear_checkpoint_options = ["", *_checkpoint_options("linear")]
        linear_checkpoint = streamlit.selectbox(
            "Linear RL checkpoint path",
            options=linear_checkpoint_options,
            format_func=lambda path: path if path else "Select a Linear RL checkpoint",
            key="build_linear_checkpoint_select",
        )
        streamlit.caption("Options loaded from `checkpoints/linear`.")
        with streamlit.expander("Linear RL advanced settings"):
            linear_epsilon = streamlit.number_input(
                "Linear RL epsilon",
                min_value=0.0,
                value=0.0,
                step=0.05,
                key="build_linear_epsilon",
            )
            streamlit.caption(
                "Exploration rate; higher values choose random actions more often."
            )
        output_root = streamlit.text_input(
            "Output directory (optional)",
            value="",
            key="build_output_root",
        )
        submitted = streamlit.form_submit_button("Run", disabled=is_running)
    if submitted and not is_running:
        errors: list[str] = []
        seed_value: int | None
        seed_label: str | None = None
        seed_text = seed_text.strip()
        if seed_text:
            try:
                seed_value = int(seed_text)
            except ValueError:
                seed_value = None
                errors.append("Seed must be an integer.")
        else:
            seed_value = _random_seed()
            seed_label = "Seed was not provided; a random seed was generated."
        total_players = int(
            random_count
            + heuristic_balanced
            + heuristic_cautious
            + heuristic_aggressive
            + static_count
            + mlp_count
            + linear_count
        )
        if total_players < 3 or total_players > 5:
            errors.append("Player count must be between 3 and 5.")
        if mlp_count > 0 and not mlp_checkpoint.strip():
            errors.append("MLP bots require a checkpoint path.")
        if linear_count > 0 and not linear_checkpoint.strip():
            errors.append("Linear RL bots require a checkpoint path.")
        if errors:
            for error in errors:
                streamlit.error(error)
            return
        players: list[dict[str, object]] = []
        _append_players(players, int(random_count), "random", "Random")
        _append_players(
            players,
            int(heuristic_balanced),
            "heuristic",
            "Heuristic Balanced",
            params={"style": "balanced"},
        )
        _append_players(
            players,
            int(heuristic_cautious),
            "heuristic",
            "Heuristic Cautious",
            params={"style": "cautious"},
        )
        _append_players(
            players,
            int(heuristic_aggressive),
            "heuristic",
            "Heuristic Aggressive",
            params={"style": "aggressive"},
        )
        _append_players(players, int(static_count), "static", "Static")
        if mlp_count > 0:
            mlp_params: dict[str, object] = {}
            if mlp_temperature:
                mlp_params["temperature"] = float(mlp_temperature)
            if mlp_device.strip():
                mlp_params["device"] = mlp_device.strip()
            _append_players(
                players,
                int(mlp_count),
                "mlp",
                "MLP",
                params=mlp_params,
                checkpoint=mlp_checkpoint.strip(),
            )
        if linear_count > 0:
            linear_params: dict[str, object] = {}
            if linear_epsilon:
                linear_params["epsilon"] = float(linear_epsilon)
            _append_players(
                players,
                int(linear_count),
                "linear_rl",
                "Linear RL",
                params=linear_params,
                checkpoint=linear_checkpoint.strip(),
            )
        spec_data: dict[str, object] = {
            "mode": mode,
            "seed": seed_value if seed_value is not None else 0,
            "num_games": int(num_games),
            "players": players,
            "rules": {},
        }
        spec_path = _next_gui_spec_path(spec_root)
        try:
            _write_json(spec_path, spec_data)
        except OSError as exc:
            streamlit.error(f"Failed to write spec file: {exc}")
            return
        try:
            spec = load_spec(spec_path)
        except (OSError, ValueError) as exc:
            streamlit.error(f"Failed to load spec: {exc}")
            return
        session_state["build_run_running"] = True
        session_state["build_run_pending_spec_path"] = str(spec_path)
        session_state["build_run_pending_output_root"] = output_root.strip()
        session_state["build_run_pending_seed_label"] = seed_label or ""
        session_state["build_run_pending_seed_value"] = seed_value
        session_state["build_run_pending_spec_data"] = spec.to_mapping()
        streamlit.rerun()
    if not is_running:
        return
    pending_spec_path = session_state.get("build_run_pending_spec_path")
    if not isinstance(pending_spec_path, str) or not pending_spec_path:
        _reset_build_run_state(streamlit)
        session_state["build_run_error"] = "Run failed to execute."
        streamlit.rerun()
    pending_output_root = session_state.get("build_run_pending_output_root", "")
    output_root_value = pending_output_root if isinstance(pending_output_root, str) else ""
    output_dir = resolve_output_dir(Path(output_root_value) if output_root_value else None)
    spec_path = Path(pending_spec_path)
    try:
        result = execute_spec(spec_path, output_dir=output_dir, dry_run=False, write_outputs=True)
    except Exception as exc:  # noqa: BLE001
        _reset_build_run_state(streamlit)
        session_state["build_run_error"] = f"Run failed to execute: {exc}"
        streamlit.rerun()
    if result is None:
        _reset_build_run_state(streamlit)
        session_state["build_run_error"] = "Run failed to execute."
        streamlit.rerun()
    summary = compute_summary(result)
    raw_seed_label = session_state.get("build_run_pending_seed_label", "")
    seed_label = raw_seed_label if isinstance(raw_seed_label, str) and raw_seed_label else None
    seed_value = session_state.get("build_run_pending_seed_value")
    spec_data = session_state.get("build_run_pending_spec_data")
    _reset_build_run_state(streamlit)
    streamlit.success(f"Run complete: {output_dir}")
    streamlit.caption(f"Spec saved to: {spec_path}")
    if seed_label and isinstance(seed_value, int):
        _info_box(streamlit, f"{seed_label} Seed: {seed_value}")
    streamlit.json(summary.to_dict())
    if isinstance(spec_data, dict):
        with streamlit.expander("Spec JSON"):
            streamlit.json(spec_data)
    _redirect_to_inspect(streamlit, output_dir)


def _page_train_bots(streamlit) -> None:
    """Build training specs and launch learning bots in the background."""
    streamlit.subheader("Train learning bots")
    session_state = _session_state(streamlit)
    train_error = _pop_session_message(streamlit, "train_bot_error")
    train_message = _pop_session_message(streamlit, "train_bot_message")
    if train_error:
        streamlit.error(train_error)
    if train_message:
        streamlit.success(train_message)

    jobs = _refresh_training_jobs(session_state)
    running_jobs = sum(1 for job in jobs if job.get("status") == "running")
    if running_jobs > 0:
        _info_box(
            streamlit,
            f"{running_jobs} training job(s) running in background; the UI stays responsive.",
        )
    else:
        _info_box(streamlit, "Configure a training spec and start a background job.")

    bot_type = streamlit.selectbox(
        "Learning bot type",
        options=_TRAIN_BOT_TYPES,
        format_func=lambda value: "MLP" if value == "mlp" else "Linear RL",
        key="train_bot_type",
    )
    checkpoint_name_key = f"train_checkpoint_name_{bot_type}"
    checkpoint_output_key = f"train_checkpoint_output_{bot_type}"
    if checkpoint_name_key not in session_state:
        session_state[checkpoint_name_key] = _default_checkpoint_name(bot_type)
    if checkpoint_output_key not in session_state:
        session_state[checkpoint_output_key] = ""

    with streamlit.form("train_bot_form"):
        streamlit.markdown("**Checkpoint**")
        checkpoint_name = streamlit.text_input("Checkpoint name", key=checkpoint_name_key)
        streamlit.caption(
            "Names the checkpoint folder/file when output path is left blank."
        )
        checkpoint_output = streamlit.text_input(
            "Checkpoint output path (optional)",
            key=checkpoint_output_key,
        )
        try:
            preview_checkpoint = _resolve_training_checkpoint_path(
                bot_type,
                checkpoint_name,
                checkpoint_output,
            )
            streamlit.caption(f"Resolved checkpoint path: `{preview_checkpoint}`")
        except ValueError as exc:
            streamlit.caption(str(exc))

        streamlit.markdown("**Common settings**")
        train_player_count = streamlit.number_input(
            "Players per game",
            min_value=3,
            max_value=5,
            value=3,
            step=1,
            key=f"train_{bot_type}_player_count",
        )
        streamlit.caption("Controls game size during self-play and opponent sampling.")
        train_seed = streamlit.number_input(
            "Seed",
            min_value=0,
            value=0,
            step=1,
            key=f"train_{bot_type}_seed",
        )
        streamlit.caption("Sets the base RNG seed so training runs are reproducible.")
        opponents_raw = streamlit.text_area(
            "Opponents JSON (optional list)",
            value="",
            key=f"train_{bot_type}_opponents_json",
            height=100,
        )
        streamlit.caption("Provide a JSON list of player specs to train against custom opponents.")
        opponent_weights_raw = streamlit.text_input(
            "Opponent weights JSON (optional list)",
            value="",
            key=f"train_{bot_type}_opponent_weights_json",
        )
        streamlit.caption("Optional per-opponent sampling weights; order matches the opponents list.")

        if bot_type == "mlp":
            streamlit.markdown("**MLP settings**")
            episodes = streamlit.number_input(
                "Episodes",
                min_value=1,
                value=1000,
                step=50,
                key="train_mlp_episodes",
            )
            streamlit.caption("Total self-play episodes used to optimize the policy/value network.")
            learning_rate = streamlit.number_input(
                "Learning rate",
                min_value=0.000001,
                value=0.001,
                step=0.0001,
                format="%.6f",
                key="train_mlp_learning_rate",
            )
            streamlit.caption("Scales each optimizer update, where larger values change weights faster.")
            use_lr_decay = streamlit.checkbox(
                "Use learning-rate decay",
                value=False,
                key="train_mlp_use_lr_decay",
            )
            learning_rate_final: float | None = None
            if use_lr_decay:
                default_final = max(0.000001, float(learning_rate) * 0.1)
                learning_rate_final = streamlit.number_input(
                    "Final learning rate",
                    min_value=0.000001,
                    max_value=float(learning_rate),
                    value=min(float(learning_rate), default_final),
                    step=0.0001,
                    format="%.6f",
                    key="train_mlp_learning_rate_final",
                )
                streamlit.caption(
                    "Linearly decays the optimizer learning rate from the start to this value."
                )
            hidden_sizes_raw = streamlit.text_input(
                "Hidden sizes (comma-separated)",
                value="128,128",
                key="train_mlp_hidden_sizes",
            )
            streamlit.caption("Defines MLP layer widths from input to output.")
            activation = streamlit.selectbox(
                "Activation",
                options=_TRAIN_MLP_ACTIVATIONS,
                index=0,
                key="train_mlp_activation",
            )
            streamlit.caption("Chooses the non-linearity applied between hidden layers.")
            dropout = streamlit.number_input(
                "Dropout",
                min_value=0.0,
                value=0.0,
                step=0.05,
                key="train_mlp_dropout",
            )
            streamlit.caption("Randomly drops hidden activations each step to reduce overfitting.")
            temperature = streamlit.number_input(
                "Temperature",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="train_mlp_temperature",
            )
            streamlit.caption(
                "Controls policy sampling randomness; 0 is greedy and higher values explore more."
            )
            entropy_coef = streamlit.number_input(
                "Entropy coefficient",
                min_value=0.0,
                value=0.01,
                step=0.005,
                format="%.4f",
                key="train_mlp_entropy_coef",
            )
            streamlit.caption("Rewards broader action distributions so the policy explores longer.")
            value_coef = streamlit.number_input(
                "Value coefficient",
                min_value=0.0,
                value=0.5,
                step=0.1,
                key="train_mlp_value_coef",
            )
            streamlit.caption("Balances value-loss magnitude against policy-loss magnitude.")
            max_grad_norm = streamlit.number_input(
                "Max gradient norm",
                min_value=0.1,
                value=1.0,
                step=0.1,
                key="train_mlp_max_grad_norm",
            )
            streamlit.caption("Clips gradient norm each update to stabilize optimization.")
            checkpoint_every = streamlit.number_input(
                "Checkpoint every N episodes",
                min_value=1,
                value=250,
                step=25,
                key="train_mlp_checkpoint_every",
            )
            streamlit.caption("Writes intermediate checkpoints at this episode interval.")
            device = streamlit.selectbox(
                "Device",
                options=_TRAIN_DEVICES,
                index=0,
                key="train_mlp_device",
            )
            streamlit.caption("Selects the torch device used for forward/backward passes.")
            resume = streamlit.text_input(
                "Resume checkpoint (optional)",
                value="",
                key="train_mlp_resume",
            )
            submitted = streamlit.form_submit_button("Start background training")
        else:
            streamlit.markdown("**Linear RL settings**")
            num_games = streamlit.number_input(
                "Self-play games",
                min_value=1,
                value=500,
                step=50,
                key="train_linear_num_games",
            )
            streamlit.caption("Total games sampled for linear value updates.")
            learning_rate = streamlit.number_input(
                "Learning rate",
                min_value=0.000001,
                value=0.05,
                step=0.005,
                format="%.6f",
                key="train_linear_learning_rate",
            )
            streamlit.caption("Scales each linear-model weight update from observed rewards.")
            epsilon = streamlit.number_input(
                "Epsilon",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="train_linear_epsilon",
            )
            streamlit.caption("Sets epsilon-greedy exploration so random moves happen with this chance.")
            log_every = streamlit.number_input(
                "Log every N games",
                min_value=0,
                value=0,
                step=100,
                key="train_linear_log_every",
            )
            streamlit.caption("Prints periodic progress logs every N games; 0 disables interval logs.")
            learner_seats_raw = streamlit.text_input(
                "Learner seats (optional comma list)",
                value="",
                key="train_linear_learner_seats",
            )
            streamlit.caption(
                "Pins learner-controlled seats by id (e.g. 0 or 0,2) instead of using every seat."
            )
            resume = streamlit.text_input(
                "Resume checkpoint (optional)",
                value="",
                key="train_linear_resume",
            )
            submitted = streamlit.form_submit_button("Start background training")

    if submitted:
        errors: list[str] = []
        job_id = ""
        staging_checkpoint_path = Path("trainings/staging")
        training_artifacts_path = _TRAINING_HISTORY_ROOT
        try:
            checkpoint_path = _resolve_training_checkpoint_path(
                bot_type,
                checkpoint_name,
                checkpoint_output,
            )
            job_id = _new_training_job_id(bot_type)
            staging_checkpoint_path = _staging_checkpoint_path(job_id, checkpoint_path)
            training_artifacts_path = _training_history_path(job_id)
        except ValueError as exc:
            errors.append(str(exc))
            checkpoint_path = Path("checkpoints")

        opponents, opponents_error = _parse_optional_json_list(opponents_raw, "Opponents JSON")
        if opponents_error:
            errors.append(opponents_error)
        opponent_weights, weights_error = _parse_optional_json_list(
            opponent_weights_raw,
            "Opponent weights JSON",
        )
        if weights_error:
            errors.append(weights_error)
        parsed_weights: list[float] | None = None
        if opponent_weights is not None:
            parsed_weights = []
            for value in opponent_weights:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append("Opponent weights must contain numeric values.")
                    break
                if value <= 0:
                    errors.append("Opponent weights must be positive.")
                    break
                parsed_weights.append(float(value))
            if opponents is None:
                errors.append("Opponent weights require opponents.")
            elif parsed_weights and len(parsed_weights) != len(opponents):
                errors.append("Opponent weights must match the opponents list length.")

        spec_data: dict[str, object] = {}
        if bot_type == "mlp":
            hidden_sizes, hidden_error = _parse_int_csv(
                hidden_sizes_raw,
                "Hidden sizes",
                allow_empty=False,
            )
            if hidden_error:
                errors.append(hidden_error)
            elif hidden_sizes is not None and any(size <= 0 for size in hidden_sizes):
                errors.append("Hidden sizes must be positive integers.")
            spec_data = {
                "episodes": int(episodes),
                "seed": int(train_seed),
                "player_count": int(train_player_count),
                "learning_rate": float(learning_rate),
                "hidden_sizes": list(hidden_sizes or ()),
                "activation": activation,
                "dropout": float(dropout),
                "temperature": float(temperature),
                "entropy_coef": float(entropy_coef),
                "value_coef": float(value_coef),
                "max_grad_norm": float(max_grad_norm),
                "checkpoint_path": str(staging_checkpoint_path),
                "checkpoint_every": int(checkpoint_every),
                "artifacts_path": str(training_artifacts_path),
                "device": str(device),
            }
            if learning_rate_final is not None:
                spec_data["learning_rate_final"] = float(learning_rate_final)
        else:
            learner_seats, seats_error = _parse_int_csv(
                learner_seats_raw,
                "Learner seats",
                allow_empty=True,
            )
            if seats_error:
                errors.append(seats_error)
            elif learner_seats:
                for seat in learner_seats:
                    if seat >= int(train_player_count):
                        errors.append("Learner seats must be valid player ids for this player count.")
                        break
            spec_data = {
                "num_games": int(num_games),
                "player_count": int(train_player_count),
                "seed": int(train_seed),
                "epsilon": float(epsilon),
                "learning_rate": float(learning_rate),
                "log_every": int(log_every),
                "output": str(staging_checkpoint_path),
                "artifacts_path": str(training_artifacts_path),
            }
            if learner_seats:
                spec_data["learner_seats"] = list(learner_seats)

        resume_value = resume.strip()
        if resume_value:
            spec_data["resume"] = resume_value
        if opponents is not None:
            spec_data["opponents"] = opponents
        if parsed_weights is not None:
            spec_data["opponent_weights"] = parsed_weights

        if errors:
            for error in errors:
                streamlit.error(error)
        else:
            spec_root = _training_spec_root(bot_type)
            spec_root.mkdir(parents=True, exist_ok=True)
            spec_path = _next_gui_spec_path(spec_root, prefix="gui")
            try:
                _write_json(spec_path, spec_data)
                job = _start_training_job(
                    bot_type,
                    job_id,
                    spec_path,
                    checkpoint_path,
                    staging_checkpoint_path,
                    training_artifacts_path,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                session_state["train_bot_error"] = f"Failed to start training: {exc}"
                streamlit.rerun()
            jobs = _training_jobs(session_state)
            jobs.insert(0, job)
            session_state["train_bot_message"] = (
                f"Started {bot_type.upper()} training job {job['id']}."
            )
            streamlit.rerun()

    streamlit.markdown("---")
    streamlit.markdown("**Training jobs**")
    auto_refresh = streamlit.checkbox(
        "Auto-refresh logs while jobs are running",
        value=False,
        key="train_jobs_auto_refresh",
    )
    refresh_seconds = streamlit.number_input(
        "Refresh interval (seconds)",
        min_value=1,
        max_value=30,
        value=2,
        step=1,
        key="train_jobs_refresh_seconds",
    )
    if streamlit.button("Refresh training status", key="train_jobs_refresh"):
        streamlit.rerun()
    jobs = _refresh_training_jobs(session_state)
    if not jobs:
        streamlit.caption("No training jobs started in this browser session yet.")
        return
    running = _running_training_jobs(jobs)
    if running:
        streamlit.markdown("**Running trainings**")
        for job in running:
            job_id = str(job.get("id", "unknown"))
            pid = job.get("pid")
            pid_label = str(pid) if isinstance(pid, int) else "n/a"
            checkpoint = str(job.get("checkpoint_path", ""))
            artifacts_dir = str(job.get("training_artifacts_path", ""))
            status = str(job.get("status", "unknown"))
            streamlit.caption(f"- `{job_id}` [{status}] (pid {pid_label}) -> `{checkpoint}`")
            if artifacts_dir:
                streamlit.caption(f"  artifacts -> `{artifacts_dir}`")
    else:
        streamlit.caption("No trainings are currently running.")
    streamlit.markdown("**Job details + logs**")
    for job in jobs:
        status = str(job.get("status", "unknown"))
        marker_lookup = {
            "running": "[RUNNING]",
            "stopping": "[STOPPING]",
            "stopped": "[STOPPED]",
            "completed": "[COMPLETED]",
            "failed": "[FAILED]",
        }
        marker = marker_lookup.get(status, "[UNKNOWN]")
        job_id = str(job.get("id", "unknown"))
        bot_label = "MLP" if str(job.get("bot_type")) == "mlp" else "Linear RL"
        with streamlit.expander(f"{marker} {job_id} | {bot_label} | {status}", expanded=False):
            streamlit.caption(f"Started: {job.get('started_at', 'n/a')}")
            streamlit.caption(f"PID: {job.get('pid', 'n/a')}")
            if job.get("completed_at"):
                streamlit.caption(f"Completed: {job.get('completed_at')}")
            if job.get("exit_code") is not None:
                streamlit.caption(f"Exit code: {job.get('exit_code')}")
            if job.get("stop_requested"):
                streamlit.caption("Stop requested: final checkpoint publish disabled for this job.")
            streamlit.caption(f"Command: {job.get('command', '')}")
            streamlit.caption(f"Spec: {job.get('spec_path', '')}")
            streamlit.caption(f"Checkpoint target: {job.get('checkpoint_path', '')}")
            streamlit.caption(f"Checkpoint staging: {job.get('staging_checkpoint_path', '')}")
            streamlit.caption(f"Training artifacts: {job.get('training_artifacts_path', '')}")
            if job.get("promotion_error"):
                streamlit.error(str(job.get("promotion_error")))
            if status in {"running", "stopping"}:
                force_stop = status == "stopping"
                button_label = "Force kill now" if force_stop else "Stop job safely"
                if streamlit.button(button_label, key=f"train_stop_{job_id}"):
                    stopped = _request_stop_training_job(job, force=force_stop)
                    if stopped:
                        session_state["train_bot_message"] = (
                            f"Stop requested for {job_id}. Final checkpoint will not be published."
                        )
                    else:
                        session_state["train_bot_error"] = f"Unable to stop {job_id}; it may have exited."
                    streamlit.rerun()
            log_path_value = job.get("log_path", "")
            streamlit.caption(f"Log: {log_path_value}")
            if isinstance(log_path_value, str) and log_path_value:
                streamlit.code(_tail_text(Path(log_path_value)), language="text")
    if auto_refresh and running:
        streamlit.caption(f"Live log mode: refreshing every {int(refresh_seconds)} second(s).")
        time.sleep(int(refresh_seconds))
        streamlit.rerun()


def _page_inspect_runs(streamlit) -> None:
    """Inspect existing run outputs."""
    streamlit.subheader("Inspect existing runs")
    runs_root = streamlit.text_input("Runs root", value="runs", key="inspect_runs_root")
    if not runs_root:
        _info_box(streamlit, "Enter a runs root to browse existing runs.")
        return
    root = Path(runs_root)
    if not root.exists():
        _info_box(streamlit, f"Runs root not found: {root}")
        return
    run_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()],
        reverse=True,
    )
    if not run_dirs:
        _info_box(streamlit, "No runs found with summary.json.")
        return
    selected_dir = streamlit.selectbox(
        "Run directory",
        options=run_dirs,
        format_func=lambda path: str(path.relative_to(root)),
        key="inspect_run_dir",
    )
    summary_path = selected_dir / "summary.json"
    try:
        summary_data = _load_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        streamlit.error(f"Failed to read summary.json: {exc}")
        return
    spec_path = selected_dir / "spec.json"
    spec_data: dict[str, object] | None = None
    if spec_path.exists():
        try:
            spec_data = _load_json(spec_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            streamlit.warning(f"Failed to read spec.json: {exc}")
    streamlit.caption("Player IDs map to the player order in the spec file (0-based).")
    streamlit.json(summary_data)
    if spec_data:
        with streamlit.expander("Spec used for this run"):
            streamlit.json(spec_data)
    if not _has_completed_games(summary_data):
        _info_box(streamlit, "No completed games in this run yet, so charts are hidden.")
        return
    win_counts_data = summary_data.get("win_counts")
    win_rates_data = summary_data.get("win_rates")
    poorest_counts_data = summary_data.get("poorest_counts")
    poorest_rates_data = summary_data.get("poorest_rates")
    cancel_counts_data = summary_data.get("cancel_counts")
    if not isinstance(cancel_counts_data, dict):
        cancel_counts_data = summary_data.get("theft_counts")
    debt_counts_data = summary_data.get("debt_counts")
    title_counts_data = summary_data.get("title_counts")
    scandal_counts_data = summary_data.get("scandal_counts")
    sections = (
        win_counts_data,
        win_rates_data,
        poorest_counts_data,
        poorest_rates_data,
        cancel_counts_data,
        debt_counts_data,
        title_counts_data,
        scandal_counts_data,
    )
    if not any(_has_numeric_chart_data(section) for section in sections):
        _info_box(streamlit, "No chart data is available for this run.")
        return
    player_ids = sorted(_collect_player_ids(summary_data))
    players = {index: _player_label(index, spec_data) for index in player_ids}
    player_colors: dict[int, str] | None = None
    if spec_data:
        player_colors = {
            index: (_player_color(index, spec_data) or DEFAULT_BOT_COLOR) for index in player_ids
        }
    games_finished = summary_data.get("games_finished")
    chart_width_kwargs = _stretch_width_kwargs(streamlit.altair_chart)
    if _has_numeric_chart_data(win_counts_data) and _has_numeric_chart_data(win_rates_data):
        streamlit.subheader("Wins (count + rate)")
        win_chart = _count_rate_chart(
            win_counts_data,
            win_rates_data,
            players,
            player_colors,
            games_finished,
            "Win count",
            "Win rate",
        )
        if win_chart is not None:
            streamlit.altair_chart(win_chart, **chart_width_kwargs)
        else:
            win_counts_chart = _bar_chart(win_counts_data, players, player_colors, "Win count")
            if win_counts_chart is not None:
                streamlit.altair_chart(win_counts_chart, **chart_width_kwargs)
            else:
                win_counts = _chart_dataframe(win_counts_data, players)
                streamlit.bar_chart(win_counts)
            win_rates_chart = _bar_chart(win_rates_data, players, player_colors, "Win rate")
            if win_rates_chart is not None:
                streamlit.altair_chart(win_rates_chart, **chart_width_kwargs)
            else:
                win_rates = _chart_dataframe(win_rates_data, players)
                streamlit.bar_chart(win_rates)
    else:
        if _has_numeric_chart_data(win_counts_data):
            streamlit.subheader("Win counts")
            win_counts_chart = _bar_chart(win_counts_data, players, player_colors, "Win count")
            if win_counts_chart is not None:
                streamlit.altair_chart(win_counts_chart, **chart_width_kwargs)
            else:
                win_counts = _chart_dataframe(win_counts_data, players)
                streamlit.bar_chart(win_counts)
        if _has_numeric_chart_data(win_rates_data):
            streamlit.subheader("Win rates")
            win_rates_chart = _bar_chart(win_rates_data, players, player_colors, "Win rate")
            if win_rates_chart is not None:
                streamlit.altair_chart(win_rates_chart, **chart_width_kwargs)
            else:
                win_rates = _chart_dataframe(win_rates_data, players)
                streamlit.bar_chart(win_rates)
    if _has_numeric_chart_data(poorest_counts_data) and _has_numeric_chart_data(poorest_rates_data):
        streamlit.subheader("Poorest (count + rate)")
        poorest_chart = _count_rate_chart(
            poorest_counts_data,
            poorest_rates_data,
            players,
            player_colors,
            games_finished,
            "Poorest count",
            "Poorest rate",
        )
        if poorest_chart is not None:
            streamlit.altair_chart(poorest_chart, **chart_width_kwargs)
        else:
            poorest_counts_chart = _bar_chart(
                poorest_counts_data, players, player_colors, "Poorest count"
            )
            if poorest_counts_chart is not None:
                streamlit.altair_chart(poorest_counts_chart, **chart_width_kwargs)
            else:
                poorest_counts = _chart_dataframe(poorest_counts_data, players)
                streamlit.bar_chart(poorest_counts)
            poorest_rates_chart = _bar_chart(
                poorest_rates_data, players, player_colors, "Poorest rate"
            )
            if poorest_rates_chart is not None:
                streamlit.altair_chart(poorest_rates_chart, **chart_width_kwargs)
            else:
                poorest_rates = _chart_dataframe(poorest_rates_data, players)
                streamlit.bar_chart(poorest_rates)
    else:
        if _has_numeric_chart_data(poorest_counts_data):
            streamlit.subheader("Poorest counts")
            poorest_counts_chart = _bar_chart(
                poorest_counts_data, players, player_colors, "Poorest count"
            )
            if poorest_counts_chart is not None:
                streamlit.altair_chart(poorest_counts_chart, **chart_width_kwargs)
            else:
                poorest_counts = _chart_dataframe(poorest_counts_data, players)
                streamlit.bar_chart(poorest_counts)
        if _has_numeric_chart_data(poorest_rates_data):
            streamlit.subheader("Poorest rates")
            poorest_rates_chart = _bar_chart(
                poorest_rates_data, players, player_colors, "Poorest rate"
            )
            if poorest_rates_chart is not None:
                streamlit.altair_chart(poorest_rates_chart, **chart_width_kwargs)
            else:
                poorest_rates = _chart_dataframe(poorest_rates_data, players)
                streamlit.bar_chart(poorest_rates)
    card_order = ["Cancel", "Debt", "Title", "Scandal"]
    card_counts = {
        "Cancel": cancel_counts_data,
        "Debt": debt_counts_data,
        "Title": title_counts_data,
        "Scandal": scandal_counts_data,
    }
    if all(_has_numeric_chart_data(value) for value in card_counts.values()):
        streamlit.subheader("Special cards (count)")
        special_chart = _special_card_chart(card_counts, players, card_order, player_colors)
        if special_chart is not None:
            streamlit.altair_chart(special_chart, **chart_width_kwargs)
        else:
            _, wide_frame = _special_card_frames(card_counts, players, card_order)
            if not wide_frame.empty:
                streamlit.bar_chart(wide_frame)
    results_path = selected_dir / "results.csv"
    if results_path.exists():
        streamlit.caption(f"results.csv: {results_path}")


def _list_training_artifact_dirs(root: Path) -> list[Path]:
    """Return directories that contain training artifacts."""
    if not root.exists():
        return []
    dirs = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if (path / "training_players.csv").exists() or (path / "training_summary.json").exists():
            dirs.append(path)
    return sorted(dirs, reverse=True)


@lru_cache(maxsize=2)
def _cached_training_players_frame(path_value: str, mtime_ns: int, size_bytes: int):
    """Load and normalize per-player training outcomes from CSV (cached by file signature)."""
    del mtime_ns, size_bytes
    import pandas as pd

    frame = pd.read_csv(path_value)
    required = {
        "game_index",
        "seed",
        "player_id",
        "is_learner",
        "won",
        "poorest",
        "score",
        "money_remaining",
        "titles",
        "scandal",
        "debt",
        "theft",
        "cumulative_wins",
        "cumulative_poorest",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training_players.csv is missing columns: {', '.join(missing)}")
    int_columns = [
        "game_index",
        "seed",
        "player_id",
        "is_learner",
        "won",
        "poorest",
        "money_remaining",
        "titles",
        "scandal",
        "debt",
        "theft",
        "cumulative_wins",
        "cumulative_poorest",
    ]
    for column in int_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0).astype(float)
    return frame.sort_values(["game_index", "player_id"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _cached_training_learners_with_rolling(
    path_value: str,
    mtime_ns: int,
    size_bytes: int,
    rolling_window: int,
):
    """Return learner rows augmented with rolling metrics (cached by file signature + window)."""
    players_frame = _cached_training_players_frame(path_value, mtime_ns, size_bytes)
    learner_frame = players_frame.loc[players_frame["is_learner"] == 1].copy()
    if learner_frame.empty:
        return learner_frame
    learner_frame["learner_label"] = learner_frame["player_id"].map(
        lambda value: f"Learner {int(value)}"
    )
    return _training_with_rolling_rates(learner_frame, rolling_window)


def _load_training_players_frame(path: Path):
    """Load and normalize per-player training outcomes from CSV."""
    stat = path.stat()
    return _cached_training_players_frame(str(path), stat.st_mtime_ns, stat.st_size)


def _training_with_rolling_rates(frame, window_size: int):
    """Return learner rows with rolling win/poorest rates by learner."""
    window = max(1, int(window_size))
    if frame.empty:
        result = frame.copy()
        result["rolling_win_rate"] = 0.0
        result["rolling_poorest_rate"] = 0.0
        return result
    ordered = frame.sort_values(["player_id", "game_index"]).reset_index(drop=True)
    grouped = ordered.groupby("player_id", sort=False)
    ordered["rolling_win_rate"] = (
        grouped["won"].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    ordered["rolling_poorest_rate"] = (
        grouped["poorest"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return ordered.sort_values(["game_index", "player_id"]).reset_index(drop=True)


def _downsample_training_games(frame, max_games: int):
    """Downsample a training frame to a maximum number of game indices."""
    if frame.empty:
        return frame
    limit = max(1, int(max_games))
    games = sorted(frame["game_index"].unique().tolist())
    if len(games) <= limit:
        return frame
    if limit == 1:
        selected_games = {games[-1]}
    else:
        last_index = len(games) - 1
        selected_games = {games[int(i * last_index / (limit - 1))] for i in range(limit)}
    return frame.loc[frame["game_index"].isin(selected_games)].copy()


def _training_cumulative_and_rolling_chart(
    frame,
    cumulative_column: str,
    cumulative_title: str,
    rolling_column: str,
    rolling_title: str,
    *,
    show_points: bool = True,
):
    """Build a dual-axis learner chart with cumulative and rolling metrics."""
    try:
        import altair as alt
    except ImportError:
        return None
    if frame.empty:
        return None
    cumulative_line = alt.Chart(frame).mark_line(point=bool(show_points)).encode(
        x=alt.X("game_index:Q", axis=alt.Axis(title="Game")),
        y=alt.Y(f"{cumulative_column}:Q", axis=alt.Axis(title=cumulative_title, orient="left")),
        color=alt.Color("learner_label:N", legend=alt.Legend(title="Learner")),
        tooltip=[
            alt.Tooltip("game_index:Q", title="Game"),
            alt.Tooltip("learner_label:N", title="Learner"),
            alt.Tooltip(f"{cumulative_column}:Q", title=cumulative_title, format=".0f"),
            alt.Tooltip(f"{rolling_column}:Q", title=rolling_title, format=".1%"),
        ],
    )
    rolling_line = alt.Chart(frame).mark_line(point=False, opacity=0.9).encode(
        x=alt.X("game_index:Q", axis=alt.Axis(title="Game")),
        y=alt.Y(
            f"{rolling_column}:Q",
            axis=alt.Axis(title=rolling_title, orient="right", format=".0%"),
        ),
        detail=alt.Detail("learner_label:N"),
        color=alt.value("#2e7d32"),
        tooltip=[
            alt.Tooltip("game_index:Q", title="Game"),
            alt.Tooltip("learner_label:N", title="Learner"),
            alt.Tooltip(f"{cumulative_column}:Q", title=cumulative_title, format=".0f"),
            alt.Tooltip(f"{rolling_column}:Q", title=rolling_title, format=".1%"),
        ],
    )
    return alt.layer(cumulative_line, rolling_line).resolve_scale(y="independent")


def _training_grouped_bar_chart(frame, value_column: str, value_title: str):
    """Build a grouped per-game bar chart for learner metrics."""
    try:
        import altair as alt
    except ImportError:
        return None
    if frame.empty:
        return None
    return alt.Chart(frame).mark_bar().encode(
        x=alt.X("game_index:O", axis=alt.Axis(title="Game")),
        xOffset=alt.XOffset("learner_label:N"),
        y=alt.Y(f"{value_column}:Q", axis=alt.Axis(title=value_title)),
        color=alt.Color("learner_label:N", legend=alt.Legend(title="Learner")),
        tooltip=[
            alt.Tooltip("game_index:Q", title="Game"),
            alt.Tooltip("learner_label:N", title="Learner"),
            alt.Tooltip(f"{value_column}:Q", title=value_title, format=".2f"),
        ],
    )


def _training_value_line_chart(frame, value_column: str, value_title: str, *, show_points: bool = True):
    """Build a per-game line chart for learner metrics."""
    try:
        import altair as alt
    except ImportError:
        return None
    if frame.empty:
        return None
    return alt.Chart(frame).mark_line(point=bool(show_points)).encode(
        x=alt.X("game_index:Q", axis=alt.Axis(title="Game")),
        y=alt.Y(f"{value_column}:Q", axis=alt.Axis(title=value_title)),
        color=alt.Color("learner_label:N", legend=alt.Legend(title="Learner")),
        tooltip=[
            alt.Tooltip("game_index:Q", title="Game"),
            alt.Tooltip("learner_label:N", title="Learner"),
            alt.Tooltip(f"{value_column}:Q", title=value_title, format=".2f"),
        ],
    )


def _training_special_cards_cumulative_frame(frame, *, max_games: int | None = None):
    """Build a long frame of cumulative special-card counts by learner/game."""
    if frame.empty:
        return frame
    card_columns = {
        "Title": "titles",
        "Debt": "debt",
        "Theft": "theft",
        "Scandal": "scandal",
    }
    special_frame = frame[
        ["game_index", "player_id", "learner_label", *card_columns.values()]
    ].copy()
    special_frame = special_frame.sort_values(["player_id", "game_index"]).reset_index(drop=True)
    grouped = special_frame.groupby("player_id", sort=False)
    cumulative_cols = []
    for column in card_columns.values():
        cumulative_column = f"cumulative_{column}"
        cumulative_cols.append(cumulative_column)
        special_frame[cumulative_column] = grouped[column].cumsum().astype(int)
    if max_games is not None:
        special_frame = _downsample_training_games(special_frame, max_games)
    counts_long = special_frame.melt(
        id_vars=["game_index", "player_id", "learner_label"],
        value_vars=list(card_columns.values()),
        var_name="card_key",
        value_name="count",
    )
    cumulative_long = special_frame.melt(
        id_vars=["game_index", "player_id", "learner_label"],
        value_vars=cumulative_cols,
        var_name="cumulative_key",
        value_name="cumulative_count",
    )
    card_lookup = {value: key for key, value in card_columns.items()}
    cumulative_lookup = {f"cumulative_{value}": key for key, value in card_columns.items()}
    counts_long["card"] = counts_long["card_key"].map(card_lookup)
    cumulative_long["card"] = cumulative_long["cumulative_key"].map(cumulative_lookup)
    cumulative_long = cumulative_long.drop(columns=["cumulative_key"])
    long_frame = counts_long.merge(
        cumulative_long,
        on=["game_index", "player_id", "learner_label", "card"],
        how="left",
    )
    long_frame["series_label"] = long_frame["learner_label"] + " - " + long_frame["card"].astype(
        str
    )
    long_frame = long_frame.sort_values(["player_id", "card", "game_index"]).reset_index(drop=True)
    return long_frame.drop(columns=["card_key"])


def _training_special_cards_line_chart(frame, *, show_points: bool = True):
    """Build a learner line chart for cumulative special-card counts."""
    try:
        import altair as alt
    except ImportError:
        return None
    if frame.empty:
        return None
    card_order = ["Title", "Debt", "Theft", "Scandal"]
    return alt.Chart(frame).mark_line(point=bool(show_points)).encode(
        x=alt.X("game_index:Q", axis=alt.Axis(title="Game")),
        y=alt.Y("cumulative_count:Q", axis=alt.Axis(title="Cumulative special cards")),
        color=alt.Color("card:N", sort=card_order, legend=alt.Legend(title="Card")),
        strokeDash=alt.StrokeDash("learner_label:N", legend=alt.Legend(title="Learner")),
        tooltip=[
            alt.Tooltip("game_index:Q", title="Game"),
            alt.Tooltip("learner_label:N", title="Learner"),
            alt.Tooltip("card:N", title="Card"),
            alt.Tooltip("cumulative_count:Q", title="Cumulative count", format="d"),
        ],
    )


def _page_inspect_trainings(streamlit) -> None:
    """Inspect individual training runs and chart learner progress over games."""
    streamlit.subheader("Inspect training runs")
    root_value = streamlit.text_input(
        "Training history root",
        value=str(_TRAINING_HISTORY_ROOT),
        key="inspect_trainings_root",
    )
    if not root_value:
        _info_box(streamlit, "Enter a training history root to browse artifacts.")
        return
    root = Path(root_value)
    if not root.exists():
        _info_box(streamlit, f"Training history root not found: {root}")
        return
    run_dirs = _list_training_artifact_dirs(root)
    if not run_dirs:
        _info_box(streamlit, "No training artifacts found.")
        return
    selected_dir = streamlit.selectbox(
        "Training directory",
        options=run_dirs,
        format_func=lambda path: str(path.relative_to(root)),
        key="inspect_training_dir",
    )

    summary_path = selected_dir / "training_summary.json"
    summary_data: dict[str, object] | None = None
    if summary_path.exists():
        try:
            summary_data = _load_json(summary_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            streamlit.warning(f"Failed to read training_summary.json: {exc}")
    if summary_data:
        streamlit.json(summary_data)

    spec_path = selected_dir / "training_spec.json"
    if spec_path.exists():
        try:
            spec_data = _load_json(spec_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            streamlit.warning(f"Failed to read training_spec.json: {exc}")
        else:
            with streamlit.expander("Training spec"):
                streamlit.json(spec_data)

    players_path = selected_dir / "training_players.csv"
    if not players_path.exists():
        _info_box(streamlit, "This training run has no training_players.csv yet.")
        return
    try:
        players_stat = players_path.stat()
        with _maybe_spinner(streamlit, "Loading training players..."):
            players_frame = _cached_training_players_frame(
                str(players_path),
                players_stat.st_mtime_ns,
                players_stat.st_size,
            )
    except (ImportError, OSError, ValueError) as exc:
        streamlit.error(f"Failed to read training_players.csv: {exc}")
        return
    learner_rows = players_frame.loc[players_frame["is_learner"] == 1, ["game_index", "player_id"]]
    if learner_rows.empty:
        _info_box(streamlit, "No learner rows found in this training artifact.")
        return
    total_games = max(1, int(learner_rows["game_index"].nunique()))
    max_window = max(1, int(learner_rows["game_index"].max()))
    num_learners = max(1, int(learner_rows["player_id"].nunique()))
    default_window = min(500, max_window)
    rolling_window = int(
        streamlit.number_input(
            "Rolling average window (games)",
            min_value=1,
            max_value=max_window,
            value=default_window,
            step=1,
            key="inspect_trainings_rolling_window",
        )
    )
    safe_max_plot_games = min(total_games, max(1, int(4500 / num_learners)))
    default_plot_games = min(2000, safe_max_plot_games)
    max_plot_games = int(
        streamlit.number_input(
            "Max games to plot (downsampled)",
            min_value=1,
            max_value=safe_max_plot_games,
            value=default_plot_games,
            step=50,
            key="inspect_trainings_plot_games",
        )
    )
    with _maybe_spinner(streamlit, "Computing rolling learner metrics..."):
        learner_frame = _cached_training_learners_with_rolling(
            str(players_path),
            players_stat.st_mtime_ns,
            players_stat.st_size,
            rolling_window,
        )
        if learner_frame.empty:
            _info_box(streamlit, "No learner rows found in this training artifact.")
            return
    chart_frame = _downsample_training_games(learner_frame, max_plot_games)
    plotted_games = max(1, int(chart_frame["game_index"].nunique()))
    if plotted_games < total_games:
        streamlit.caption(
            f"Charts are downsampled to {plotted_games} games (of {total_games}) for performance."
        )
    streamlit.caption(
        "Learner-colored lines are cumulative counts; green lines are rolling averages."
    )
    show_points = plotted_games <= 250
    chart_width_kwargs = _stretch_width_kwargs(streamlit.altair_chart)

    streamlit.subheader("Cumulative wins and rolling win rate over games")
    with _maybe_spinner(streamlit, "Rendering win-rate chart..."):
        wins_line = _training_cumulative_and_rolling_chart(
            chart_frame,
            "cumulative_wins",
            "Cumulative wins",
            "rolling_win_rate",
            f"Rolling win rate ({rolling_window} games)",
            show_points=show_points,
        )
        if wins_line is not None:
            try:
                streamlit.altair_chart(wins_line, **chart_width_kwargs)
            except Exception:
                wins_line = None
        if wins_line is None:
            wins_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="cumulative_wins",
                aggfunc="max",
            ).sort_index()
            streamlit.line_chart(wins_wide)
            rolling_wins_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="rolling_win_rate",
                aggfunc="mean",
            ).sort_index()
            streamlit.line_chart(rolling_wins_wide)

    streamlit.subheader("Cumulative poorest and rolling poorest rate over games")
    with _maybe_spinner(streamlit, "Rendering poorest-rate chart..."):
        poorest_line = _training_cumulative_and_rolling_chart(
            chart_frame,
            "cumulative_poorest",
            "Cumulative poorest",
            "rolling_poorest_rate",
            f"Rolling poorest rate ({rolling_window} games)",
            show_points=show_points,
        )
        if poorest_line is not None:
            try:
                streamlit.altair_chart(poorest_line, **chart_width_kwargs)
            except Exception:
                poorest_line = None
        if poorest_line is None:
            poorest_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="cumulative_poorest",
                aggfunc="max",
            ).sort_index()
            streamlit.line_chart(poorest_wide)
            rolling_poorest_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="rolling_poorest_rate",
                aggfunc="mean",
            ).sort_index()
            streamlit.line_chart(rolling_poorest_wide)

    streamlit.subheader("Money remaining by game (learners)")
    use_bars = plotted_games <= 200
    with _maybe_spinner(streamlit, "Rendering money chart..."):
        if use_bars:
            money_chart = _training_grouped_bar_chart(
                chart_frame, "money_remaining", "Money remaining"
            )
        else:
            money_chart = _training_value_line_chart(
                chart_frame,
                "money_remaining",
                "Money remaining",
                show_points=show_points,
            )
        if money_chart is not None:
            try:
                streamlit.altair_chart(money_chart, **chart_width_kwargs)
            except Exception:
                money_chart = None
        if money_chart is None:
            money_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="money_remaining",
                aggfunc="mean",
            ).sort_index()
            if use_bars:
                streamlit.bar_chart(money_wide)
            else:
                streamlit.line_chart(money_wide)

    streamlit.subheader("Score by game (learners)")
    with _maybe_spinner(streamlit, "Rendering score chart..."):
        if use_bars:
            score_chart = _training_grouped_bar_chart(chart_frame, "score", "Score")
        else:
            score_chart = _training_value_line_chart(
                chart_frame,
                "score",
                "Score",
                show_points=show_points,
            )
        if score_chart is not None:
            try:
                streamlit.altair_chart(score_chart, **chart_width_kwargs)
            except Exception:
                score_chart = None
        if score_chart is None:
            score_wide = chart_frame.pivot_table(
                index="game_index",
                columns="learner_label",
                values="score",
                aggfunc="mean",
            ).sort_index()
            if use_bars:
                streamlit.bar_chart(score_wide)
            else:
                streamlit.line_chart(score_wide)

    streamlit.subheader("Cumulative special cards by game (learners)")
    special_max_games = min(max_plot_games, max(1, int(1000 / num_learners)))
    with _maybe_spinner(streamlit, "Rendering special-card chart..."):
        special_cards_frame = _training_special_cards_cumulative_frame(
            learner_frame,
            max_games=special_max_games,
        )
        special_chart = _training_special_cards_line_chart(
            special_cards_frame,
            show_points=show_points,
        )
        if special_chart is not None:
            try:
                streamlit.altair_chart(special_chart, **chart_width_kwargs)
            except Exception:
                special_chart = None
        if special_chart is None:
            special_wide = special_cards_frame.pivot_table(
                index="game_index",
                columns="series_label",
                values="cumulative_count",
                aggfunc="max",
            ).sort_index()
            streamlit.line_chart(special_wide)

    games_path = selected_dir / "training_games.csv"
    if games_path.exists():
        streamlit.caption(f"training_games.csv: {games_path}")
    streamlit.caption(f"training_players.csv: {players_path}")


def _redirect_to_inspect(streamlit, output_dir: Path) -> None:
    """Jump to the inspect page and preselect the latest run."""
    streamlit.session_state["dashboard_page"] = "Inspect runs"
    streamlit.session_state["inspect_runs_root"] = str(output_dir.parent)
    streamlit.session_state["inspect_run_dir"] = output_dir
    streamlit.rerun()


def _apply_theme(streamlit) -> None:
    """Apply a custom dark/light theme to the Streamlit app."""
    streamlit.markdown(
        """
        <style>
        @import url("https://fonts.googleapis.com/css2?family=Cinzel:wght@600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap");

        :root {
            --bg: #0b1611;
            --surface: #0f1f17;
            --surface-2: #13261d;
            --chart-bg: #07110c;
            --text: #f4f6f1;
            --muted: #c1c9bc;
            --accent: #c9a227;
            --accent-soft: #a67f1b;
            --border: #234234;
            --nav-active: rgba(201, 162, 39, 0.18);
            --nav-hover: rgba(255, 255, 255, 0.06);
            --info-bar: #2f7d4a;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg: #ffffff;
                --surface: #f4f7f3;
                --surface-2: #eef3ee;
                --chart-bg: #e6efe6;
                --text: #1b1f1c;
                --muted: #4a5a50;
                --accent: #1f6f43;
                --accent-soft: #b9922a;
                --border: #d8e1d7;
                --nav-active: rgba(31, 111, 67, 0.12);
                --nav-hover: rgba(31, 111, 67, 0.08);
                --info-bar: #1f6f43;
            }
        }

        html, body, .stApp {
            background-color: var(--bg);
            color: var(--text);
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp h1, .stApp h2, .stApp h3 {
            font-family: "Cinzel", serif;
            color: var(--text);
            letter-spacing: 0.4px;
        }

        .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background-color: var(--surface);
        }

        .block-container {
            background: transparent;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div {
            background-color: var(--surface);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
        }

        [data-baseweb="select"] div {
            color: var(--text);
        }

        .stCheckbox label, .stRadio label {
            color: var(--text);
        }

        .stButton > button,
        .stFormSubmitButton > button,
        [data-testid="stFormSubmitButton"] > button {
            background-color: var(--accent);
            color: #ffffff;
            border: 1px solid var(--accent-soft);
            border-radius: 0.6rem;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            background-color: var(--accent-soft);
            border-color: var(--accent-soft);
            color: #ffffff;
        }

        .stButton > button:disabled,
        .stFormSubmitButton > button:disabled,
        [data-testid="stFormSubmitButton"] > button:disabled {
            opacity: 0.58;
            filter: brightness(0.72);
            cursor: not-allowed;
        }

        .stCaption, .stMarkdown small {
            color: var(--muted);
        }

        .stAlert {
            border-radius: 0.6rem;
            background: var(--surface-2);
            border: 1px solid var(--accent-soft);
            color: var(--text);
        }

        .stAlert svg {
            color: var(--accent);
            fill: var(--accent);
        }

        .vega-embed,
        .vega-embed canvas,
        .vega-embed svg {
            background: var(--chart-bg) !important;
        }

        .hs-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            border-radius: 0.6rem;
            border: 1px solid var(--border);
            border-left: 0.35rem solid var(--info-bar);
            background: var(--surface-2);
            color: var(--text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sidebar_nav(streamlit) -> str:
    """Render the sidebar navigation and return the active page."""
    streamlit.markdown(
        """
        <style>
        [data-testid="stSidebar"] button[kind] {
            width: 100%;
            justify-content: flex-start;
            border-radius: 0.6rem;
            padding: 0.55rem 0.75rem;
            margin: 0.15rem 0;
            border: 1px solid transparent;
            background: transparent;
            color: inherit;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: var(--nav-hover);
        }
        [data-testid="stSidebar"] button[kind="primary"] {
            background: var(--nav-active);
            border-color: var(--nav-active);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "dashboard_page" not in streamlit.session_state:
        streamlit.session_state["dashboard_page"] = "Run from spec"
    button_width_kwargs = _stretch_width_kwargs(streamlit.sidebar.button)

    def nav_button(label: str, key: str) -> None:
        is_active = streamlit.session_state["dashboard_page"] == label
        if streamlit.sidebar.button(
            label,
            **button_width_kwargs,
            type="primary" if is_active else "secondary",
            key=key,
        ):
            if not is_active:
                streamlit.session_state["dashboard_page"] = label
                streamlit.rerun()

    nav_button("Run from spec", "nav_run_spec")
    nav_button("Build run", "nav_build_run")
    nav_button("Train bots", "nav_train_bots")
    nav_button("Inspect runs", "nav_inspect_runs")
    nav_button("Inspect trainings", "nav_inspect_trainings")
    return str(streamlit.session_state["dashboard_page"])


def run_dashboard() -> None:
    """Run the Streamlit dashboard."""
    try:
        import streamlit
    except ImportError as exc:
        raise ImportError("streamlit is required to run the dashboard") from exc
    streamlit.title("High Society Run + Train Dashboard")
    _apply_theme(streamlit)
    page = _sidebar_nav(streamlit)
    if page == "Run from spec":
        _page_run_from_spec(streamlit)
    elif page == "Build run":
        _page_build_run(streamlit)
    elif page == "Train bots":
        _page_train_bots(streamlit)
    elif page == "Inspect trainings":
        _page_inspect_trainings(streamlit)
    else:
        _page_inspect_runs(streamlit)
