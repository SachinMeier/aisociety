"""Tests for training-page helpers in the dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from highsociety.ops.dashboard.app import (
    _checkpoint_options,
    _parse_int_csv,
    _parse_optional_json_list,
    _refresh_training_jobs,
    _request_stop_training_job,
    _resolve_training_checkpoint_path,
    _staging_checkpoint_path,
    _training_command,
)


def test_training_checkpoint_path_defaults_to_bot_folder() -> None:
    """Blank output path should resolve to checkpoints/<bot>/<name>."""
    path = _resolve_training_checkpoint_path("mlp", "alpha", "")

    assert path == Path("checkpoints/mlp/alpha")


def test_checkpoint_options_lists_entries_in_bot_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checkpoint options should come from checkpoints/<bot_type>."""
    root = tmp_path / "checkpoints" / "mlp"
    root.mkdir(parents=True)
    (root / "b_model").mkdir()
    (root / "a_model").mkdir()
    (root / ".hidden").mkdir()

    monkeypatch.chdir(tmp_path)

    options = _checkpoint_options("mlp")

    assert options == ["checkpoints/mlp/a_model", "checkpoints/mlp/b_model"]


def test_checkpoint_options_returns_empty_when_folder_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing checkpoints folder should produce no options."""
    monkeypatch.chdir(tmp_path)

    assert _checkpoint_options("linear") == []


def test_training_checkpoint_path_uses_explicit_output() -> None:
    """Explicit output path should be returned unchanged."""
    path = _resolve_training_checkpoint_path("linear", "ignored", "custom/linear_v2.pkl")

    assert path == Path("custom/linear_v2.pkl")


def test_training_checkpoint_path_requires_name_when_output_blank() -> None:
    """Checkpoint name is required when no explicit output path is provided."""
    with pytest.raises(ValueError):
        _resolve_training_checkpoint_path("mlp", "", "")


def test_training_command_targets_expected_module() -> None:
    """Command builder should point each bot type at its training module."""
    mlp_command = _training_command("mlp", Path("trainings/mlp/spec.json"))
    linear_command = _training_command("linear", Path("trainings/linear/spec.json"))

    assert mlp_command[:3] == [sys.executable, "-m", "scripts.train"]
    assert linear_command[:3] == [sys.executable, "-m", "scripts.train_linear"]


def test_parse_int_csv_parses_comma_separated_values() -> None:
    """Comma-separated integers should parse into a tuple."""
    values, error = _parse_int_csv("1, 2,5", "Hidden sizes", allow_empty=False)

    assert error is None
    assert values == (1, 2, 5)


def test_parse_optional_json_list_rejects_non_list_values() -> None:
    """Only JSON lists are accepted for optional list fields."""
    values, error = _parse_optional_json_list('{"foo": 1}', "Opponents JSON")

    assert values is None
    assert error == "Opponents JSON must be a JSON list."


def test_staging_checkpoint_path_keeps_target_suffix() -> None:
    """Staging checkpoint path should preserve file suffix for file outputs."""
    staging = _staging_checkpoint_path("job_1", Path("checkpoints/model.pkl"))

    assert staging.name == "checkpoint.pkl"


def test_refresh_training_jobs_promotes_staged_checkpoint_on_success(tmp_path: Path) -> None:
    """Successful jobs should publish staged checkpoints to target path."""
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])  # noqa: S603
    process.wait(timeout=5)
    target = tmp_path / "checkpoints" / "linear.pkl"
    staging = tmp_path / "trainings" / "staging" / "job_1" / "checkpoint.pkl"
    staging.parent.mkdir(parents=True)
    staging.write_text("new-checkpoint", encoding="utf-8")
    session_state: dict[str, object] = {
        "training_jobs": [
            {
                "id": "job_1",
                "status": "running",
                "process": process,
                "checkpoint_path": str(target),
                "staging_checkpoint_path": str(staging),
                "stop_requested": False,
                "completed_at": None,
                "exit_code": None,
            }
        ]
    }

    jobs = _refresh_training_jobs(session_state)

    assert len(jobs) == 1
    assert jobs[0]["status"] == "completed"
    assert jobs[0]["exit_code"] == 0
    assert isinstance(jobs[0]["completed_at"], str)
    assert target.read_text(encoding="utf-8") == "new-checkpoint"
    assert not staging.exists()


def test_refresh_training_jobs_stopped_job_does_not_publish(tmp_path: Path) -> None:
    """Stopped jobs should not overwrite the final checkpoint path."""
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(0)"])  # noqa: S603
    process.wait(timeout=5)
    target = tmp_path / "checkpoints" / "linear.pkl"
    target.parent.mkdir(parents=True)
    target.write_text("old-checkpoint", encoding="utf-8")
    staging = tmp_path / "trainings" / "staging" / "job_1" / "checkpoint.pkl"
    staging.parent.mkdir(parents=True)
    staging.write_text("new-checkpoint", encoding="utf-8")
    session_state: dict[str, object] = {
        "training_jobs": [
            {
                "id": "job_1",
                "status": "stopping",
                "process": process,
                "checkpoint_path": str(target),
                "staging_checkpoint_path": str(staging),
                "stop_requested": True,
                "completed_at": None,
                "exit_code": None,
            }
        ]
    }

    jobs = _refresh_training_jobs(session_state)

    assert jobs[0]["status"] == "stopped"
    assert target.read_text(encoding="utf-8") == "old-checkpoint"
    assert not staging.exists()


def test_request_stop_training_job_marks_stopping() -> None:
    """Stop requests should mark job state and signal the process."""
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    job: dict[str, object] = {"status": "running", "process": process}

    stopped = _request_stop_training_job(job)

    assert stopped is True
    assert job["status"] == "stopping"
    assert job["stop_requested"] is True
    process.wait(timeout=5)
