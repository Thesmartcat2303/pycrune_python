"""Manage a reusable process running the matching CPython interpreter."""

from __future__ import annotations

import marshal
import os
import subprocess
from contextlib import suppress
from pathlib import Path
from threading import RLock

from ._support import DecompilationError, FailFastException
from ._wire import MARSHAL_FORMAT, unpack_model
from .bytecode import optimization_level
from .interpreter import interpreter_for

_BOOTSTRAP = (
    "import sys; sys.path.insert(0, sys.argv[1]); from pycrune._worker import serve; serve()"
)


class PyBridge:
    """Read normalized code models and compile source with one interpreter."""

    def __init__(self, executable):
        self.executable = os.fspath(executable)
        self._process = None
        self._lock = RLock()

    @classmethod
    def for_path(cls, path, python=None):
        return cls(interpreter_for(path, python))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        with self._lock:
            process = self._process
            if process is None:
                return
            self._process = None
            with suppress(OSError):
                process.stdin.close()
            process.stdout.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _start(self):
        package_parent = Path(__file__).resolve().parent.parent
        return subprocess.Popen(
            [self.executable, "-I", "-X", "utf8", "-c", _BOOTSTRAP, str(package_parent)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def _request(self, operation, target, optimize, views):
        with self._lock:
            if self._process is None:
                self._process = self._start()
            process = self._process
            try:
                marshal.dump((operation, target, optimize, views), process.stdin, MARSHAL_FORMAT)
                process.stdin.flush()
                ok, payload = marshal.load(process.stdout)
            except (EOFError, OSError, TypeError, ValueError) as exc:
                self.close()
                raise DecompilationError(
                    f"The CPython worker {self.executable!r} stopped "
                    f"(exit code {process.returncode}): {exc}"
                ) from exc
        if not ok:
            kind, message = payload
            error = FailFastException if kind == "FailFastException" else DecompilationError
            raise error(f"{kind}: {message}")
        return tuple(unpack_model(data) for data in payload)

    def load_views(self, path, *, optimize=None, views=(True,)):
        """Load normalized (True) or original (False) views from one file read."""
        file = Path(path).absolute()
        optimize = optimization_level(file, optimize)
        return self._request("load", str(file), optimize, views)

    def load(self, path, *, optimize=None, normalize=True):
        return self.load_views(path, optimize=optimize, views=(normalize,))[0]

    def compile(self, source, *, optimize=0, normalize=True):
        return self._request("compile", source, optimize, (normalize,))[0]

    def check_syntax(self, source, *, optimize=0):
        self._request("compile", source, optimize, ())
