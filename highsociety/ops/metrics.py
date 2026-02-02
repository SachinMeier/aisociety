"""Metrics computation for run results."""

from __future__ import annotations

from dataclasses import dataclass

from highsociety.ops.runner import RunResult


@dataclass(frozen=True)
class SummaryMetrics:
    """Summary metrics aggregated over a batch run."""

    games_total: int
    games_finished: int
    games_errored: int
    finish_rate: float
    win_counts: dict[int, int]
    win_rates: dict[int, float]
    poorest_counts: dict[int, int]
    poorest_rates: dict[int, float]
    title_counts: dict[int, int]
    scandal_counts: dict[int, int]
    debt_counts: dict[int, int]
    theft_counts: dict[int, int]
    average_scores: dict[int, float]
    average_money: dict[int, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dict of the metrics."""
        return {
            "games_total": self.games_total,
            "games_finished": self.games_finished,
            "games_errored": self.games_errored,
            "finish_rate": self.finish_rate,
            "win_counts": _stringify_keys(self.win_counts),
            "win_rates": _stringify_keys(self.win_rates),
            "poorest_counts": _stringify_keys(self.poorest_counts),
            "poorest_rates": _stringify_keys(self.poorest_rates),
            "title_counts": _stringify_keys(self.title_counts),
            "scandal_counts": _stringify_keys(self.scandal_counts),
            "debt_counts": _stringify_keys(self.debt_counts),
            "theft_counts": _stringify_keys(self.theft_counts),
            "average_scores": _stringify_keys(self.average_scores),
            "average_money": _stringify_keys(self.average_money),
        }


def compute_summary(run_result: RunResult) -> SummaryMetrics:
    """Compute summary metrics for a run result."""
    games_total = run_result.spec.num_games
    games_finished = len(run_result.games)
    games_errored = len(run_result.errors)
    finish_rate = (games_finished / games_total) if games_total else 0.0
    player_ids = sorted(_collect_player_ids(run_result))
    win_counts = {pid: 0 for pid in player_ids}
    poorest_counts = {pid: 0 for pid in player_ids}
    title_counts = {pid: 0 for pid in player_ids}
    scandal_counts = {pid: 0 for pid in player_ids}
    debt_counts = {pid: 0 for pid in player_ids}
    theft_counts = {pid: 0 for pid in player_ids}
    score_sums = {pid: 0.0 for pid in player_ids}
    score_counts = {pid: 0 for pid in player_ids}
    money_sums = {pid: 0.0 for pid in player_ids}
    for game in run_result.games:
        for pid in game.result.winners:
            win_counts[pid] += 1
        for pid in game.result.poorest:
            poorest_counts[pid] += 1
        for pid, score in game.result.scores.items():
            score_sums[pid] += float(score)
            score_counts[pid] += 1
        for pid, money in game.result.money_remaining.items():
            money_sums[pid] += float(money)
        for player in game.state.players:
            pid = player.id
            if pid not in title_counts:
                title_counts[pid] = 0
                scandal_counts[pid] = 0
                debt_counts[pid] = 0
                theft_counts[pid] = 0
            title_counts[pid] += player.titles
            scandal_counts[pid] += player.scandal
            debt_counts[pid] += player.debt
            theft_counts[pid] += player.theft
    win_rates = {
        pid: (win_counts[pid] / games_finished) if games_finished else 0.0
        for pid in player_ids
    }
    poorest_rates = {
        pid: (poorest_counts[pid] / games_finished) if games_finished else 0.0
        for pid in player_ids
    }
    average_scores = {
        pid: (score_sums[pid] / score_counts[pid]) if score_counts[pid] else 0.0
        for pid in player_ids
    }
    average_money = {
        pid: (money_sums[pid] / games_finished) if games_finished else 0.0
        for pid in player_ids
    }
    return SummaryMetrics(
        games_total=games_total,
        games_finished=games_finished,
        games_errored=games_errored,
        finish_rate=finish_rate,
        win_counts=win_counts,
        win_rates=win_rates,
        poorest_counts=poorest_counts,
        poorest_rates=poorest_rates,
        title_counts=title_counts,
        scandal_counts=scandal_counts,
        debt_counts=debt_counts,
        theft_counts=theft_counts,
        average_scores=average_scores,
        average_money=average_money,
    )


def _collect_player_ids(run_result: RunResult) -> set[int]:
    """Collect all player ids from a run result."""
    player_ids: set[int] = set()
    for game in run_result.games:
        player_ids.update(game.result.money_remaining.keys())
    return player_ids


def _stringify_keys(data: dict[int, object]) -> dict[str, object]:
    """Convert dict keys to strings for JSON serialization."""
    return {str(key): value for key, value in data.items()}
