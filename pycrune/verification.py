"""Compare bytecode recursively and optionally run programs."""

from __future__ import annotations

import os
import subprocess
import tempfile
import tokenize
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from io import BytesIO
from pathlib import Path

from .bridge import PyBridge
from .bytecode import model_code, optimization_level
from .interpreter import interpreter_for
from .model import _same_value

_TRANSPARENT_OPS = {"NOP", "EXTENDED_ARG", "NOT_TAKEN", "CACHE"}


@dataclass
class DiffResult:
    total_original: int = 0
    total_decompiled: int = 0
    discrepancies: list[str] = field(default_factory=list)

    @property
    def is_identical(self):
        return not self.discrepancies


def _first_line_load(instructions, index):
    return (
        index + 1 < len(instructions)
        and instructions[index].op_name == "LOAD_CONST"
        and (instructions[index + 1].op_name == "STORE_NAME")
        and (instructions[index + 1].arg_val == "__firstlineno__")
    )


def _comparison_view(code):
    insns = [ins for ins in code.instructions if ins.op_name not in _TRANSPARENT_OPS]
    first_line = {i for i in range(len(insns)) if _first_line_load(insns, i)}
    constant_ops = {"LOAD_CONST", "RETURN_CONST", "KW_NAMES"}
    used_elsewhere = {
        ins.op_arg
        for i, ins in enumerate(insns)
        if i not in first_line and ins.op_name in constant_ops
    }
    skip = sorted(
        {insns[i].op_arg for i in first_line if type(insns[i].arg_val) is int} - used_elsewhere
    )
    constants = [value for i, value in enumerate(code.constants) if i not in skip]
    if skip:
        insns = [
            (
                replace(ins, op_arg=ins.op_arg - bisect_left(skip, ins.op_arg))
                if ins.op_name in constant_ops and ins.op_arg is not None
                else ins
            )
            for ins in insns
        ]
    return (insns, constants, first_line)


def _compare_objects(original, candidate, result, where):
    diffs = result.discrepancies

    def difference(label, left, right):
        if not _same_value(left, right):
            diffs.append(f"{where} {label}: {left!r} != {right!r}")

    for key in (
        "name",
        "qual_name",
        "arg_count",
        "pos_only_arg_count",
        "kw_only_arg_count",
        "flags",
        "names",
        "var_names",
        "free_vars",
        "cell_vars",
    ):
        difference(key, getattr(original, key), getattr(candidate, key))
    left, left_constants, left_lines = _comparison_view(original)
    right, right_constants, right_lines = _comparison_view(candidate)
    difference("constant count", len(left_constants), len(right_constants))
    for i, (a, b) in enumerate(zip(left_constants, right_constants)):
        difference(f"constant #{i}", a, b)
    difference("stack size", original.stack_size, candidate.stack_size)
    result.total_original += len(left)
    result.total_decompiled += len(right)
    difference("instruction count", len(left), len(right))
    left_offsets = [i.offset for i in left]
    right_offsets = [i.offset for i in right]
    for i, (a, b) in enumerate(zip(left, right)):
        label = f"instruction #{i}"
        if a.op_name != b.op_name:
            difference(label + " opcode", a.op_name, b.op_name)
        elif i in left_lines and i in right_lines:
            continue
        elif a.target_offset is not None or b.target_offset is not None:
            a_target = (
                bisect_left(left_offsets, a.target_offset) if a.target_offset is not None else None
            )
            b_target = (
                bisect_left(right_offsets, b.target_offset) if b.target_offset is not None else None
            )
            difference(label + " jump target", a_target, b_target)
        else:
            if a.op_name != "LOAD_CONST" or not _same_value(a.arg_val, b.arg_val):
                difference(label + " argument", a.op_arg, b.op_arg)
            difference(label + " value", a.arg_val, b.arg_val)
    difference(
        "exception table length", len(original.exception_table), len(candidate.exception_table)
    )
    for i, (a, b) in enumerate(zip(original.exception_table, candidate.exception_table)):
        boundaries_a = tuple((bisect_left(left_offsets, x) for x in (a.start, a.end, a.target)))
        boundaries_b = tuple((bisect_left(right_offsets, x) for x in (b.start, b.end, b.target)))
        difference(
            f"exception table #{i}",
            (*boundaries_a, a.depth, a.last_i),
            (*boundaries_b, b.depth, b.last_i),
        )
    difference("nested code count", len(original.nested_codes), len(candidate.nested_codes))
    for i, (a, b) in enumerate(zip(original.nested_codes, candidate.nested_codes)):
        _compare_objects(a, b, result, f"{where}/{a.name}#{i}")


def compare_models(original, candidate):
    result = DiffResult()
    if original.python_version != candidate.python_version:
        result.discrepancies.append(
            f"CPython version: {original.python_version} != {candidate.python_version}"
        )
        return result
    _compare_objects(original, candidate, result, "<module>")
    return result


def compare_code(original, candidate):
    """Compare two native code objects recursively, ignoring source locations."""
    return compare_models(
        model_code(original, normalize=False), model_code(candidate, normalize=False)
    )


def compare_file(path, source, *, optimize=None, python=None):
    """Compile the source with the matching interpreter and compare bytecode."""
    file = Path(path)
    optimize = optimization_level(file, optimize)
    with PyBridge.for_path(file, python) as bridge:
        original = bridge.load(file, optimize=optimize, normalize=False)
        candidate = bridge.compile(source, optimize=optimize, normalize=False)
    return compare_models(original, candidate)


def verify_execution(path, source, *, optimize=None, timeout=60, python=None):
    """Execute a file and its reconstructed source on trusted inputs."""
    file = Path(path)
    optimize = optimization_level(file, optimize)
    interpreter = interpreter_for(file, python)
    command = [interpreter]
    if optimize:
        command.append("-" + "O" * optimize)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONOPTIMIZE", None)
    with tempfile.TemporaryDirectory(prefix="decompiler-verify-") as directory:
        destination = Path(directory, file.stem + ".py")
        utf8 = source.encode("utf-8")
        encoding, _ = tokenize.detect_encoding(BytesIO(utf8).readline)
        destination.write_bytes(utf8 if encoding == "utf-8" else source.encode(encoding))
        paths = [str(file.resolve().parent)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        try:
            runs = [
                subprocess.run(
                    [*command, str(target)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=timeout,
                )
                for target in (file.resolve(), destination)
            ]
        except subprocess.TimeoutExpired:
            return (False, f"Execution exceeded {timeout:g} seconds.")
    original, candidate = runs
    if original.returncode != candidate.returncode:
        details = [run.stderr[:600].decode("utf-8", "replace") for run in runs]
        return (
            False,
            f"Exit codes differ ({original.returncode} != {candidate.returncode}).\nOriginal stderr: {details[0]}\nReconstructed stderr: {details[1]}",
        )
    if original.stdout != candidate.stdout:
        return (
            False,
            f"Stdout differs.\nOriginal: {original.stdout[:500]!r}\nReconstructed: {candidate.stdout[:500]!r}",
        )
    return (True, f"Exit code and stdout matched (exit code {original.returncode}).")
