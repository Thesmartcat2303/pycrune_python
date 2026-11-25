"""Read or compile user files with the interpreter selected by the bridge."""

from __future__ import annotations

import marshal
import sys
from pathlib import Path

from ._wire import MARSHAL_FORMAT, pack_model
from .bytecode import model_code, read_code


def _dispatch(operation, target, optimize, views):
    if operation == "load":
        code = read_code(Path(target), optimize)
    elif operation == "compile":
        code = compile(target, "<decompiled>", "exec", dont_inherit=True, optimize=optimize)
        if not views:
            return []
    else:
        raise ValueError(f"Unknown bridge operation {operation!r}")
    return [pack_model(model_code(code, normalize=normalize)) for normalize in views]


def serve():
    incoming = sys.stdin.buffer
    outgoing = sys.stdout.buffer
    while True:
        try:
            request = marshal.load(incoming)
        except EOFError:
            return
        try:
            models = _dispatch(*request)
            response = marshal.dumps((True, models), MARSHAL_FORMAT)
        except Exception as exc:
            response = marshal.dumps((False, (type(exc).__name__, str(exc))), MARSHAL_FORMAT)
        outgoing.write(response)
        outgoing.flush()
