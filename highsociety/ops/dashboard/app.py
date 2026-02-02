"""Streamlit dashboard for running and inspecting games."""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

from highsociety.ops.cli import execute_spec, load_spec, resolve_output_dir
from highsociety.ops.metrics import compute_summary
from highsociety.players.colors import DEFAULT_BOT_COLOR, resolve_bot_color


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


def _page_run_from_spec(streamlit) -> None:
    """Run a spec selected from the runs folder."""
    streamlit.subheader("Run from spec")
    spec_root = Path("runs")
    spec_files = _list_spec_files(spec_root)
    if not spec_files:
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
    run_clicked = streamlit.button("Run spec", key="run_spec_run")
    if not run_clicked:
        _info_box(streamlit, "Provide a spec path and click Run to start.")
        return
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
    output_dir = resolve_output_dir(Path(output_root) if output_root else None)
    try:
        result = execute_spec(spec_path, output_dir=output_dir, dry_run=False, write_outputs=True)
    except Exception as exc:  # noqa: BLE001
        streamlit.error(f"Run failed to execute: {exc}")
        return
    if result is None:
        streamlit.error("Run failed to execute.")
        return
    summary = compute_summary(result)
    streamlit.success(f"Run complete: {output_dir}")
    streamlit.json(summary.to_dict())
    _redirect_to_inspect(streamlit, output_dir)


def _page_build_run(streamlit) -> None:
    """Build a spec via GUI and run it."""
    streamlit.subheader("Build a run spec")
    spec_root = Path("runs")
    spec_root.mkdir(parents=True, exist_ok=True)
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
        mlp_checkpoint = streamlit.text_input(
            "MLP checkpoint path",
            value="",
            key="build_mlp_checkpoint",
        )
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
        linear_checkpoint = streamlit.text_input(
            "Linear RL checkpoint path",
            value="",
            key="build_linear_checkpoint",
        )
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
        submitted = streamlit.form_submit_button("Run")
    if not submitted:
        return
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
    output_dir = resolve_output_dir(Path(output_root) if output_root else None)
    try:
        result = execute_spec(spec_path, output_dir=output_dir, dry_run=False, write_outputs=True)
    except Exception as exc:  # noqa: BLE001
        streamlit.error(f"Run failed to execute: {exc}")
        return
    if result is None:
        streamlit.error("Run failed to execute.")
        return
    summary = compute_summary(result)
    streamlit.success(f"Run complete: {output_dir}")
    streamlit.caption(f"Spec saved to: {spec_path}")
    if seed_label:
        _info_box(streamlit, f"{seed_label} Seed: {seed_value}")
    streamlit.json(summary.to_dict())
    with streamlit.expander("Spec JSON"):
        streamlit.json(spec.to_mapping())
    _redirect_to_inspect(streamlit, output_dir)


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
    player_ids = sorted(_collect_player_ids(summary_data))
    players = {index: _player_label(index, spec_data) for index in player_ids}
    player_colors: dict[int, str] | None = None
    if spec_data:
        player_colors = {
            index: (_player_color(index, spec_data) or DEFAULT_BOT_COLOR) for index in player_ids
        }
    games_finished = summary_data.get("games_finished")
    win_counts_data = summary_data.get("win_counts")
    win_rates_data = summary_data.get("win_rates")
    if isinstance(win_counts_data, dict) and isinstance(win_rates_data, dict):
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
            streamlit.altair_chart(win_chart, use_container_width=True)
        else:
            win_counts_chart = _bar_chart(win_counts_data, players, player_colors, "Win count")
            if win_counts_chart is not None:
                streamlit.altair_chart(win_counts_chart, use_container_width=True)
            else:
                win_counts = _chart_dataframe(win_counts_data, players)
                streamlit.bar_chart(win_counts)
            win_rates_chart = _bar_chart(win_rates_data, players, player_colors, "Win rate")
            if win_rates_chart is not None:
                streamlit.altair_chart(win_rates_chart, use_container_width=True)
            else:
                win_rates = _chart_dataframe(win_rates_data, players)
                streamlit.bar_chart(win_rates)
    else:
        if isinstance(win_counts_data, dict):
            streamlit.subheader("Win counts")
            win_counts_chart = _bar_chart(win_counts_data, players, player_colors, "Win count")
            if win_counts_chart is not None:
                streamlit.altair_chart(win_counts_chart, use_container_width=True)
            else:
                win_counts = _chart_dataframe(win_counts_data, players)
                streamlit.bar_chart(win_counts)
        if isinstance(win_rates_data, dict):
            streamlit.subheader("Win rates")
            win_rates_chart = _bar_chart(win_rates_data, players, player_colors, "Win rate")
            if win_rates_chart is not None:
                streamlit.altair_chart(win_rates_chart, use_container_width=True)
            else:
                win_rates = _chart_dataframe(win_rates_data, players)
                streamlit.bar_chart(win_rates)
    poorest_counts_data = summary_data.get("poorest_counts")
    poorest_rates_data = summary_data.get("poorest_rates")
    if isinstance(poorest_counts_data, dict) and isinstance(poorest_rates_data, dict):
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
            streamlit.altair_chart(poorest_chart, use_container_width=True)
        else:
            poorest_counts_chart = _bar_chart(
                poorest_counts_data, players, player_colors, "Poorest count"
            )
            if poorest_counts_chart is not None:
                streamlit.altair_chart(poorest_counts_chart, use_container_width=True)
            else:
                poorest_counts = _chart_dataframe(poorest_counts_data, players)
                streamlit.bar_chart(poorest_counts)
            poorest_rates_chart = _bar_chart(
                poorest_rates_data, players, player_colors, "Poorest rate"
            )
            if poorest_rates_chart is not None:
                streamlit.altair_chart(poorest_rates_chart, use_container_width=True)
            else:
                poorest_rates = _chart_dataframe(poorest_rates_data, players)
                streamlit.bar_chart(poorest_rates)
    else:
        if isinstance(poorest_counts_data, dict):
            streamlit.subheader("Poorest counts")
            poorest_counts_chart = _bar_chart(
                poorest_counts_data, players, player_colors, "Poorest count"
            )
            if poorest_counts_chart is not None:
                streamlit.altair_chart(poorest_counts_chart, use_container_width=True)
            else:
                poorest_counts = _chart_dataframe(poorest_counts_data, players)
                streamlit.bar_chart(poorest_counts)
        if isinstance(poorest_rates_data, dict):
            streamlit.subheader("Poorest rates")
            poorest_rates_chart = _bar_chart(
                poorest_rates_data, players, player_colors, "Poorest rate"
            )
            if poorest_rates_chart is not None:
                streamlit.altair_chart(poorest_rates_chart, use_container_width=True)
            else:
                poorest_rates = _chart_dataframe(poorest_rates_data, players)
                streamlit.bar_chart(poorest_rates)
    cancel_counts_data = summary_data.get("cancel_counts")
    if not isinstance(cancel_counts_data, dict):
        cancel_counts_data = summary_data.get("theft_counts")
    debt_counts_data = summary_data.get("debt_counts")
    title_counts_data = summary_data.get("title_counts")
    scandal_counts_data = summary_data.get("scandal_counts")
    card_order = ["Cancel", "Debt", "Title", "Scandal"]
    card_counts = {
        "Cancel": cancel_counts_data,
        "Debt": debt_counts_data,
        "Title": title_counts_data,
        "Scandal": scandal_counts_data,
    }
    if all(isinstance(value, dict) for value in card_counts.values()):
        streamlit.subheader("Special cards (count)")
        special_chart = _special_card_chart(card_counts, players, card_order, player_colors)
        if special_chart is not None:
            streamlit.altair_chart(special_chart, use_container_width=True)
        else:
            _, wide_frame = _special_card_frames(card_counts, players, card_order)
            if not wide_frame.empty:
                streamlit.bar_chart(wide_frame)
    results_path = selected_dir / "results.csv"
    if results_path.exists():
        streamlit.caption(f"results.csv: {results_path}")


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

    def nav_button(label: str, key: str) -> None:
        is_active = streamlit.session_state["dashboard_page"] == label
        if streamlit.sidebar.button(
            label,
            use_container_width=True,
            type="primary" if is_active else "secondary",
            key=key,
        ):
            if not is_active:
                streamlit.session_state["dashboard_page"] = label
                streamlit.rerun()

    nav_button("Run from spec", "nav_run_spec")
    nav_button("Build run", "nav_build_run")
    nav_button("Inspect runs", "nav_inspect_runs")
    return str(streamlit.session_state["dashboard_page"])


def run_dashboard() -> None:
    """Run the Streamlit dashboard."""
    try:
        import streamlit
    except ImportError as exc:
        raise ImportError("streamlit is required to run the dashboard") from exc
    streamlit.title("High Society Run Dashboard")
    _apply_theme(streamlit)
    page = _sidebar_nav(streamlit)
    if page == "Run from spec":
        _page_run_from_spec(streamlit)
    elif page == "Build run":
        _page_build_run(streamlit)
    else:
        _page_inspect_runs(streamlit)
