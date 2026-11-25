"""Public decompilation entry points."""

from __future__ import annotations

from pathlib import Path
from types import CodeType

from .bridge import PyBridge
from .bytecode import model_code


def load_bytecode(path, *, optimize=None, python=None):
    """Return the normalized instruction model from the matching CPython."""
    with PyBridge.for_path(path, python) as bridge:
        return bridge.load(path, optimize=optimize)


def decompile_code(code, *, progress=None):
    """Decompile a native code object or a normalized model to source text."""
    from .emitter import PythonEmitter
    from .structural import StructuralDecompiler

    model = model_code(code, normalize=True) if isinstance(code, CodeType) else code
    report_position = None
    if progress is not None:
        progress(25)
        positions = {}
        pending = [model]
        total = 0
        while pending:
            current = pending.pop()
            key = id(current)
            if key not in positions:
                positions[key] = 0
                total += len(current.instructions)
                pending.extend(current.nested_codes)

        total = max(total, 1)
        processed = 0

        def report_instruction(current, position):
            nonlocal processed
            key = id(current)
            previous = positions[key]
            if position > previous:
                positions[key] = position
                processed += position - previous
                progress(25 + 55 * processed // total)

        report_position = report_instruction

    return PythonEmitter.emit(
        StructuralDecompiler(model, report_position=report_position).decompile()
    )


def decompile_file(path, *, optimize=None, python=None):
    """Reconstruct source from a .pyc or a .py file."""
    return decompile_code(load_bytecode(Path(path), optimize=optimize, python=python))
