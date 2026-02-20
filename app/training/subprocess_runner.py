from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass
from typing import Any


def _process_entrypoint(
    target,
    stop_event,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    target(stop_event, *args, **kwargs)


@dataclass(slots=True)
class TrainingProcessHandle:
    """Handle for a running training subprocess."""

    process: mp.Process
    stop_event: Any

    def is_alive(self) -> bool:
        """Return whether the subprocess is still running."""
        return self.process.is_alive()

    def request_stop(self) -> None:
        """Signal the subprocess to stop gracefully."""
        self.stop_event.set()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for subprocess termination."""
        self.process.join(timeout)

    def terminate(self) -> None:
        """Forcefully terminate the subprocess."""
        self.process.terminate()
        self.process.join()


class TrainingSubprocessRunner:
    """Manage lifecycle for a single active training subprocess."""

    def __init__(self, *, start_method: str | None = None) -> None:
        method = start_method or _default_start_method()
        self._context = mp.get_context(method)
        self._lock = threading.Lock()
        self._active: TrainingProcessHandle | None = None

    def start(self, target, *args: Any, **kwargs: Any) -> TrainingProcessHandle:
        """Start a new training process.

        The `target` callable must accept `stop_event` as its first argument.
        """
        with self._lock:
            if self._active is not None and self._active.is_alive():
                raise RuntimeError("A training process is already running.")

            stop_event = self._context.Event()
            process = self._context.Process(
                target=_process_entrypoint,
                args=(target, stop_event, args, kwargs),
                daemon=False,
            )
            process.start()
            self._active = TrainingProcessHandle(
                process=process,
                stop_event=stop_event,
            )
            return self._active

    def get_active(self) -> TrainingProcessHandle | None:
        """Return the active process handle when still alive."""
        with self._lock:
            if self._active is None:
                return None
            if self._active.is_alive():
                return self._active
            self._active = None
            return None

    def request_stop(self) -> None:
        """Request graceful stop for the active process."""
        with self._lock:
            if self._active is None or not self._active.is_alive():
                return
            self._active.request_stop()

    def stop(self, *, timeout: float = 10.0) -> None:
        """Stop the active process gracefully, then terminate if needed."""
        with self._lock:
            handle = self._active
        if handle is None:
            return
        if not handle.is_alive():
            with self._lock:
                self._active = None
            return

        handle.request_stop()
        handle.wait(timeout)
        if handle.is_alive():
            handle.terminate()

        with self._lock:
            self._active = None


def _default_start_method() -> str:
    available = mp.get_all_start_methods()
    if "spawn" in available:
        return "spawn"
    if "forkserver" in available:
        return "forkserver"
    if "fork" in available:
        return "fork"
    raise RuntimeError("No multiprocessing start method is available.")


__all__ = ["TrainingProcessHandle", "TrainingSubprocessRunner"]
