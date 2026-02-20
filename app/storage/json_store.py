from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO, TypeVar

_T = TypeVar("_T")


class JsonStore:
    """JSON persistence helper with atomic writes and lock protection."""

    _thread_locks: dict[Path, threading.Lock] = {}
    _thread_locks_guard = threading.Lock()

    def read(self, path: Path, *, default: _T | None = None) -> Any | _T:
        """Read JSON data from disk."""
        if not path.exists():
            if default is None:
                raise FileNotFoundError(path)
            return copy.deepcopy(default)

        with self._file_lock(path):
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    def write(self, path: Path, data: Any) -> None:
        """Write JSON to disk atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(data)
        with self._file_lock(path):
            temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(path)

    def update(
        self,
        path: Path,
        updater: Callable[[Any], _T],
        *,
        default_factory: Callable[[], Any] | None = None,
    ) -> _T:
        """Atomically read, transform, and rewrite a JSON document."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock(path):
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    current = json.load(handle)
            elif default_factory is not None:
                current = default_factory()
            else:
                raise FileNotFoundError(path)

            updated = updater(copy.deepcopy(current))
            payload = self._serialize(updated)
            temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(path)

        return updated

    def _serialize(self, data: Any) -> str:
        """Serialize JSON using stable formatting."""
        return f"{json.dumps(data, indent=2, ensure_ascii=True, sort_keys=False)}\n"

    @classmethod
    def _get_thread_lock(cls, lock_path: Path) -> threading.Lock:
        with cls._thread_locks_guard:
            lock = cls._thread_locks.get(lock_path)
            if lock is None:
                lock = threading.Lock()
                cls._thread_locks[lock_path] = lock
            return lock

    @contextmanager
    def _file_lock(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(f"{path.suffix}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        thread_lock = self._get_thread_lock(lock_path)
        with thread_lock:
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                self._acquire_os_lock(lock_handle)
                try:
                    yield
                finally:
                    self._release_os_lock(lock_handle)

    def _acquire_os_lock(self, handle: TextIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _release_os_lock(self, handle: TextIO) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
