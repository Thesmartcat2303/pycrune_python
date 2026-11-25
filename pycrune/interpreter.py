"""Locate the CPython interpreter that owns a bytecode dialect."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache
from importlib.util import MAGIC_NUMBER
from pathlib import Path

from ._support import DecompilationError

MAGIC_VERSIONS = {
    3413: "3.8",
    3425: "3.9",
    3439: "3.10",
    3495: "3.11",
    3531: "3.12",
    3571: "3.13",
    3627: "3.14",
    3666: "3.15",
}
HOST_MAGIC = int.from_bytes(MAGIC_NUMBER[:2], "little")
HOST_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
HOST_RELEASE = sys.version.split()[0]

_PROBE = (
    "import importlib.util, sys; "
    "print(sys.implementation.name); "
    "print(int.from_bytes(importlib.util.MAGIC_NUMBER[:2], 'little')); "
    "print(sys.executable)"
)


def file_magic(file, *, header=None):
    if file.suffix.lower() != ".pyc":
        return None
    if header is None:
        with file.open("rb") as stream:
            header = stream.read(4)
    if len(header) == 4 and header[2:] == b"\r\n":
        return int.from_bytes(header[:2], "little")
    return None


@lru_cache(maxsize=32)
def _probe(command):
    try:
        run = subprocess.run(
            [*command, "-I", "-X", "utf8", "-c", _PROBE],
            capture_output=True,
            encoding="utf-8",
            timeout=10,
            check=True,
        )
        implementation, magic, executable = run.stdout.strip().splitlines()
        if implementation == "cpython" and executable:
            return int(magic), executable
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def same_interpreter(executable):
    return os.path.normcase(os.path.abspath(executable)) == os.path.normcase(
        os.path.abspath(sys.executable)
    )


def interpreter_for(path, python=None):
    file = Path(path)
    magic = file_magic(file)
    version = MAGIC_VERSIONS.get(magic)
    if python is not None:
        result = _probe((os.fspath(python),))
        if result is None:
            raise DecompilationError(f"Cannot start CPython using {os.fspath(python)!r}")
        if version and result[0] != magic:
            raise DecompilationError(f"{file} needs CPython {version}. Check --python.")
        return result[1]
    if sys.implementation.name != "cpython":
        raise DecompilationError("The running interpreter must be CPython.")
    if magic == HOST_MAGIC or version is None:
        return sys.executable
    for command in ((f"python{version}",), ("py", f"-{version}")):
        executable = shutil.which(command[0])
        if executable:
            result = _probe((executable, *command[1:]))
            if result is not None and result[0] == magic:
                return result[1]
    raise DecompilationError(
        f"CPython {version} is required to read {file}. Install it on PATH or use "
        f"--python /path/to/python{version}."
    )
