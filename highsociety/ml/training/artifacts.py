"""Per-game artifact writers for ML training runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from highsociety.domain.rules import GameResult
from highsociety.domain.state import PlayerState


@dataclass(frozen=True)
class TrainingArtifactPaths:
    """Filesystem paths for generated training artifacts."""

    root: Path
    summary_json: Path
    games_csv: Path
    players_csv: Path
    spec_json: Path


class TrainingArtifactLogger:
    """Append per-game outcomes and write a summary at the end of training."""

    def __init__(
        self,
        output_dir: Path | str,
        *,
        bot_type: str,
        learner_seats: Sequence[int],
        player_count: int,
        spec: Mapping[str, object] | None = None,
    ) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.paths = TrainingArtifactPaths(
            root=root,
            summary_json=root / "training_summary.json",
            games_csv=root / "training_games.csv",
            players_csv=root / "training_players.csv",
            spec_json=root / "training_spec.json",
        )
        self._bot_type = bot_type
        self._learner_seats = tuple(sorted({int(seat) for seat in learner_seats}))
        self._win_counts = {pid: 0 for pid in range(max(0, int(player_count)))}
        self._poorest_counts = {pid: 0 for pid in range(max(0, int(player_count)))}
        self._games_logged = 0
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._closed = False

        self._games_handle = self.paths.games_csv.open("w", encoding="utf-8", newline="")
        self._players_handle = self.paths.players_csv.open("w", encoding="utf-8", newline="")
        self._games_writer = csv.DictWriter(
            self._games_handle,
            fieldnames=[
                "game_index",
                "seed",
                "winners",
                "poorest",
                "learner_winners",
                "learner_poorest",
                "scores",
                "money_remaining",
                "titles",
                "scandal",
                "debt",
                "theft",
            ],
        )
        self._players_writer = csv.DictWriter(
            self._players_handle,
            fieldnames=[
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
            ],
        )
        self._games_writer.writeheader()
        self._players_writer.writeheader()

        if spec is not None:
            with self.paths.spec_json.open("w", encoding="utf-8") as handle:
                json.dump(dict(spec), handle, indent=2, sort_keys=True)

    def record_game(
        self,
        *,
        game_index: int,
        seed: int,
        result: GameResult,
        players: Sequence[PlayerState],
    ) -> None:
        """Append a single game's aggregate and per-player outcome rows."""
        winners = tuple(sorted(int(pid) for pid in result.winners))
        poorest = tuple(sorted(int(pid) for pid in result.poorest))
        winner_set = set(winners)
        poorest_set = set(poorest)
        learner_set = set(self._learner_seats)
        player_ids = sorted({int(player.id) for player in players})
        for pid in player_ids:
            self._win_counts.setdefault(pid, 0)
            self._poorest_counts.setdefault(pid, 0)
        for pid in winners:
            self._win_counts[pid] = self._win_counts.get(pid, 0) + 1
        for pid in poorest:
            self._poorest_counts[pid] = self._poorest_counts.get(pid, 0) + 1
        self._games_logged += 1

        scores = {pid: float(result.scores.get(pid, 0.0)) for pid in player_ids}
        money_remaining = {pid: int(result.money_remaining.get(pid, 0)) for pid in player_ids}
        titles = {int(player.id): int(player.titles) for player in players}
        scandal = {int(player.id): int(player.scandal) for player in players}
        debt = {int(player.id): int(player.debt) for player in players}
        theft = {int(player.id): int(player.theft) for player in players}
        learner_winners = tuple(sorted(pid for pid in winners if pid in learner_set))
        learner_poorest = tuple(sorted(pid for pid in poorest if pid in learner_set))

        self._games_writer.writerow(
            {
                "game_index": int(game_index),
                "seed": int(seed),
                "winners": ",".join(str(pid) for pid in winners),
                "poorest": ",".join(str(pid) for pid in poorest),
                "learner_winners": ",".join(str(pid) for pid in learner_winners),
                "learner_poorest": ",".join(str(pid) for pid in learner_poorest),
                "scores": json.dumps(_stringify_keys(scores), sort_keys=True),
                "money_remaining": json.dumps(_stringify_keys(money_remaining), sort_keys=True),
                "titles": json.dumps(_stringify_keys(titles), sort_keys=True),
                "scandal": json.dumps(_stringify_keys(scandal), sort_keys=True),
                "debt": json.dumps(_stringify_keys(debt), sort_keys=True),
                "theft": json.dumps(_stringify_keys(theft), sort_keys=True),
            }
        )

        for player in sorted(players, key=lambda item: int(item.id)):
            player_id = int(player.id)
            self._players_writer.writerow(
                {
                    "game_index": int(game_index),
                    "seed": int(seed),
                    "player_id": player_id,
                    "is_learner": 1 if player_id in learner_set else 0,
                    "won": 1 if player_id in winner_set else 0,
                    "poorest": 1 if player_id in poorest_set else 0,
                    "score": float(result.scores.get(player_id, 0.0)),
                    "money_remaining": int(result.money_remaining.get(player_id, 0)),
                    "titles": int(player.titles),
                    "scandal": int(player.scandal),
                    "debt": int(player.debt),
                    "theft": int(player.theft),
                    "cumulative_wins": int(self._win_counts.get(player_id, 0)),
                    "cumulative_poorest": int(self._poorest_counts.get(player_id, 0)),
                }
            )
        self._games_handle.flush()
        self._players_handle.flush()

    def finalize(
        self,
        *,
        total_games: int,
        training_metrics: Mapping[str, object] | None = None,
        status: str = "completed",
    ) -> dict[str, object]:
        """Write summary JSON and close open handles."""
        games_logged = int(self._games_logged)
        win_rates = {
            pid: (count / games_logged) if games_logged else 0.0
            for pid, count in self._win_counts.items()
        }
        poorest_rates = {
            pid: (count / games_logged) if games_logged else 0.0
            for pid, count in self._poorest_counts.items()
        }
        learner_win_counts = {pid: self._win_counts.get(pid, 0) for pid in self._learner_seats}
        learner_poorest_counts = {
            pid: self._poorest_counts.get(pid, 0) for pid in self._learner_seats
        }
        learner_win_rates = {
            pid: (count / games_logged) if games_logged else 0.0
            for pid, count in learner_win_counts.items()
        }
        learner_poorest_rates = {
            pid: (count / games_logged) if games_logged else 0.0
            for pid, count in learner_poorest_counts.items()
        }
        summary: dict[str, object] = {
            "bot_type": self._bot_type,
            "status": status,
            "started_at": self._started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "games_total": int(total_games),
            "games_logged": games_logged,
            "player_ids": sorted(int(pid) for pid in self._win_counts.keys()),
            "learner_seats": list(self._learner_seats),
            "win_counts": _stringify_keys(self._win_counts),
            "win_rates": _stringify_keys(win_rates),
            "poorest_counts": _stringify_keys(self._poorest_counts),
            "poorest_rates": _stringify_keys(poorest_rates),
            "learner_win_counts": _stringify_keys(learner_win_counts),
            "learner_win_rates": _stringify_keys(learner_win_rates),
            "learner_poorest_counts": _stringify_keys(learner_poorest_counts),
            "learner_poorest_rates": _stringify_keys(learner_poorest_rates),
        }
        if training_metrics is not None:
            summary["training_metrics"] = dict(training_metrics)
        with self.paths.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        self.close()
        return summary

    def close(self) -> None:
        """Close open file handles."""
        if self._closed:
            return
        self._games_handle.close()
        self._players_handle.close()
        self._closed = True


def _stringify_keys(data: Mapping[int, object]) -> dict[str, object]:
    """Return a mapping with string keys for JSON serialization."""
    return {str(key): value for key, value in data.items()}
