from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest

from app.training.subprocess_runner import TrainingSubprocessRunner


def _blocking_target(stop_event, marker_file: str) -> None:
    path = Path(marker_file)
    path.write_text("started", encoding="utf-8")
    while not stop_event.is_set():
        time.sleep(0.01)


def test_subprocess_runner_start_and_stop(tmp_path: Path) -> None:
    marker_file = tmp_path / "marker.txt"
    runner = TrainingSubprocessRunner(start_method=_test_start_method())

    handle = runner.start(_blocking_target, str(marker_file))
    _wait_for_file(marker_file)
    assert handle.is_alive()

    runner.request_stop()
    handle.wait(timeout=2.0)
    if handle.is_alive():
        runner.stop(timeout=1.0)

    assert not handle.is_alive()


def test_subprocess_runner_rejects_concurrent_start(tmp_path: Path) -> None:
    marker_file = tmp_path / "marker.txt"
    runner = TrainingSubprocessRunner(start_method=_test_start_method())

    _ = runner.start(_blocking_target, str(marker_file))
    _wait_for_file(marker_file)

    with pytest.raises(RuntimeError, match="already running"):
        runner.start(_blocking_target, str(marker_file))

    runner.stop(timeout=2.0)


def _wait_for_file(path: Path, *, timeout_s: float = 3.0) -> None:
    start = time.time()
    while not path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for file: {path}")
        time.sleep(0.01)


def _test_start_method() -> str:
    available = mp.get_all_start_methods()
    if "fork" in available:
        return "fork"
    return "spawn"
