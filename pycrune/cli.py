"""Command-line entry points for decompilation, inspection and verification."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import tokenize
from contextlib import ExitStack
from io import StringIO
from pathlib import Path
from time import monotonic, sleep

from ._support import DecompilationError
from .api import decompile_code
from .bridge import PyBridge
from .bytecode import optimization_level
from .cfg import ControlFlowGraph, DominatorTree, ExceptionScope, LoopAnalysis
from .interpreter import interpreter_for
from .verification import compare_models, verify_execution


def _show_diff(result, *, stream=None):
    if stream is None:
        stream = sys.stderr
    if result.is_identical:
        print(f"[PASS] Bytecode identical ({result.total_original} instructions).", file=stream)
        return
    print(
        f"[FAIL] {len(result.discrepancies)} bytecode difference(s), {result.total_original} original vs {result.total_decompiled} reconstructed instructions.",
        file=stream,
    )
    for difference in result.discrepancies[:12]:
        print(f"  {difference}", file=stream)
    if len(result.discrepancies) > 12:
        print(f"  … {len(result.discrepancies) - 12} more", file=stream)


def _verify_results(file, source, result, args, interpreter, *, stream=None):
    if stream is None:
        stream = sys.stderr
    _show_diff(result, stream=stream)
    passed = result.is_identical
    if args.verify:
        same, message = verify_execution(
            file, source, optimize=args.optimize, timeout=args.timeout, python=interpreter
        )
        print(f"[{('PASS' if same else 'FAIL')}] {message}", file=stream)
        passed = passed and same
    return passed


_FUNCTIONS = ("decompile", "compare", "verify", "inspect", "batch")


_FUNCTION_FLAGS = {f"--{name}": name for name in _FUNCTIONS}


class _DecompileProgress:
    _BAR_WIDTH = 40
    _FRAME_INTERVAL = 0.016
    _COMPLETE_HOLD = 0.18

    def __init__(self, stream):
        self.stream = stream
        self.enabled = stream is not None and stream.isatty() and os.getenv("PYCRUNE_TRACE") != "1"
        self.percent = -1
        self.line = ""
        self.next_frame_time = monotonic()
        self.label = "Decompile "
        self.bar_width = self._BAR_WIDTH
        if self.enabled:
            try:
                columns = os.get_terminal_size(stream.fileno()).columns
            except (AttributeError, OSError, ValueError):
                columns = 0
            if columns:
                available = columns - len("[] 100%") - 1
                if available <= len(self.label):
                    self.label = ""
                self.bar_width = max(1, min(self.bar_width, available - len(self.label)))

    def update(self, percent):
        if not self.enabled or percent <= self.percent:
            return
        for value in range(self.percent + 1, percent + 1):
            delay = self.next_frame_time - monotonic()
            if delay > 0:
                sleep(delay)
            self.percent = value
            filled = value * self.bar_width // 100
            bar = "=" * filled + "-" * (self.bar_width - filled)
            self.line = f"{self.label}[{bar}] {value:3d}%"
            self.stream.write("\r" + self.line)
            self.stream.flush()
            self.next_frame_time = monotonic() + self._FRAME_INTERVAL
        if self.percent == 100:
            sleep(self._COMPLETE_HOLD)

    def clear(self):
        if self.line:
            self.stream.write("\r" + " " * len(self.line) + "\r")
            self.stream.flush()
            self.line = ""


def _write_source(path, source):
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix="." + destination.name + ".",
            delete=False,
        ) as out:
            temporary = Path(out.name)
            out.write(source)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _source_text(path):
    with tokenize.open(path) as stream:
        return stream.read()


def _decompile(path, output, args, bridge, *, autosave=False, display=True):
    file = Path(path)
    optimize = optimization_level(file, args.optimize)
    progress = _DecompileProgress(sys.stderr if display else None)
    report = StringIO() if progress.enabled else None
    write_report = sys.stderr if report is None else report
    try:
        progress.update(0)
        check = args.check or args.verify
        views = (True, False) if check else (True,)
        model, *raw = bridge.load_views(file, optimize=optimize, views=views)
        if autosave and output is None:
            version = ".".join(model.python_version.split(".")[:2])
            output = file.with_name(f"dcp{version}-{file.stem}.py")
        if output is not None and output.resolve() == file.resolve():
            raise DecompilationError("Choose an output path different from the input.")
        progress.update(12)
        source = decompile_code(model, progress=progress.update if progress.enabled else None)
        progress.update(82)
        if check:
            candidate = bridge.compile(source, optimize=optimize, normalize=False)
            diff = compare_models(raw[0], candidate)
            progress.update(94)
            if display or not diff.is_identical or args.verify:
                if not _verify_results(
                    file, source, diff, args, bridge.executable, stream=write_report
                ):
                    return 1
        else:
            bridge.check_syntax(source, optimize=optimize)
        progress.update(97)
        if output is not None:
            _write_source(output, source)
        progress.update(100)
    finally:
        progress.clear()
        if report is not None:
            sys.stderr.write(report.getvalue())
    if not display:
        return 0
    if output is None:
        sys.stdout.write(source)
    else:
        print(f"Saved to {output}", file=sys.stderr)
    return 0


def _handle_decompile(args, bridge):
    return _decompile(args.input, args.output, args, bridge, autosave=args.autosave)


def _handle_compare(args, bridge):
    source = _source_text(args.candidate)
    optimize = optimization_level(args.input, args.optimize)
    original = bridge.load(args.input, optimize=optimize, normalize=False)
    candidate = bridge.compile(source, optimize=optimize, normalize=False)
    result = compare_models(original, candidate)
    return 0 if _verify_results(args.input, source, result, args, bridge.executable) else 1


def _handle_inspect(args, bridge):
    code = bridge.load(args.input, optimize=args.optimize)
    graph = ControlFlowGraph.build(code)
    dom_tree = DominatorTree(graph)
    scopes = ExceptionScope.analyze(graph, dom_tree)
    loops = LoopAnalysis(graph, dom_tree).loops
    print(f"CPython {code.python_version} | {code.qual_name}")
    print(
        f"CFG blocks: {len(graph.blocks)} | exception scopes: {len(scopes)} | natural loops: {len(loops)}"
    )
    for block in graph.blocks:
        successors = ", ".join((str(item.id_) for item in block.successors)) or "-"
        kind = " [handler]" if block.is_exception_target else ""
        print(f"Block {block.id_} [{block.start_offset}, {block.end_offset}] -> {successors}{kind}")
        if args.instructions:
            for ins in block.instructions:
                print(f"  {ins}")
    for scope in scopes:
        exit_id = scope.exit_block.id_ if scope.exit_block is not None else "-"
        print(
            f"Scope {scope.id_} [{scope.start_offset}, {scope.end_offset}) | handlers: {len(scope.handlers)} | finally: {scope.has_finally} | exit: {exit_id}"
        )
    for loop in loops:
        body = ", ".join((str(block.id_) for block in sorted(loop.body, key=lambda item: item.id_)))
        print(f"Loop {loop.header.id_} | blocks: {body}")
    return 0


def _handle_batch(args):
    directory = Path(args.directory)
    if not directory.is_dir():
        raise DecompilationError(f"{directory}: expected a directory")
    files = sorted(
        item for item in directory.rglob("*") if item.is_file() and item.suffix.lower() == ".pyc"
    )
    passing = 0
    with ExitStack() as resources:
        bridges = {}
        for file in files:
            try:
                interpreter = interpreter_for(file, args.python)
                if interpreter not in bridges:
                    bridges[interpreter] = resources.enter_context(PyBridge(interpreter))
                output = None
                if args.output_dir is not None:
                    output = (args.output_dir / file.relative_to(directory)).with_suffix(".py")
                    output.parent.mkdir(parents=True, exist_ok=True)
                status = _decompile(file, output, args, bridges[interpreter], display=False)
                if status:
                    print(f"[FAIL] {file}", file=sys.stderr)
                    continue
            except (OSError, ValueError, SyntaxError, DecompilationError, RecursionError) as exc:
                print(f"[FAIL] {file}: {exc}", file=sys.stderr)
                continue
            passing += 1
            print(f"[OK] {file}")
    print(f"{passing}/{len(files)} files decompiled.", file=sys.stderr)
    return int(passing != len(files))


def _configure_common(parser):
    parser.add_argument("--python", metavar="EXECUTABLE", help="CPython executable")
    parser.add_argument("--optimize", type=int, choices=(0, 1, 2), help="Optimization level")


def _configure_verification(parser, *, allow_check=False):
    if allow_check:
        checks = parser.add_mutually_exclusive_group()
        checks.add_argument(
            "--check",
            action="store_true",
            default=True,
            help="Check bytecode (default)",
        )
        checks.add_argument(
            "--unchecked",
            action="store_false",
            dest="check",
            help="Skip bytecode check",
        )
        parser.add_argument(
            "--verify", action="store_true", help="Run both programs and compare bytecode"
        )
    parser.add_argument("--timeout", type=float, default=60, help="Timeout in seconds (default 60)")


def _parser():
    parser = argparse.ArgumentParser(
        prog="decompiler", description="Reconstruct Python source from CPython bytecode."
    )
    commands = parser.add_subparsers(dest="command", required=True, title="commands")
    decompile = commands.add_parser("decompile", help="Decompile a file")
    decompile.add_argument("input", type=Path)
    decompile.add_argument("-o", "--output", type=Path, help="Output file")
    _configure_common(decompile)
    _configure_verification(decompile, allow_check=True)
    decompile.set_defaults(handler=_handle_decompile)
    for name, help_ in (
        ("compare", "Compare bytecode"),
        ("verify", "Check bytecode and execution"),
    ):
        compare = commands.add_parser(name, help=help_)
        compare.add_argument("input", type=Path)
        compare.add_argument("candidate", type=Path)
        _configure_common(compare)
        if name == "verify":
            _configure_verification(compare)
        compare.set_defaults(handler=_handle_compare, verify=name == "verify", timeout=60)
    inspect = commands.add_parser("inspect", help="Show control flow")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--instructions", action="store_true", help="Show instructions")
    _configure_common(inspect)
    inspect.set_defaults(handler=_handle_inspect)
    batch = commands.add_parser("batch", help="Decompile a directory")
    batch.add_argument("directory", type=Path)
    batch.add_argument(
        "-o", "--output-dir", type=Path, help="Write decompiled files into this directory"
    )
    _configure_common(batch)
    _configure_verification(batch, allow_check=True)
    batch.set_defaults(handler=_handle_batch)
    return parser


def _command_arguments(argv, parser):
    if not argv:
        return ["--help"]
    if argv[0] in _FUNCTIONS or all(arg in ("-h", "--help") for arg in argv):
        return argv

    end = argv.index("--") if "--" in argv else len(argv)
    functions = [
        (i, _FUNCTION_FLAGS[arg]) for i, arg in enumerate(argv[:end]) if arg in _FUNCTION_FLAGS
    ]
    if any(name in ("decompile", "batch") for _, name in functions):
        # These functions also accept --verify as an execution check.
        functions = [(i, name) for i, name in functions if name != "verify"]
    if len(functions) > 1:
        parser.error("choose one function flag")
    if functions:
        index, command = functions[0]
        return [command, *argv[:index], *argv[index + 1 :]]
    return ["decompile", *argv]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(_command_arguments(argv, parser))
    if not 0 < getattr(args, "timeout", 60) < math.inf:
        parser.error("--timeout must be positive")
    args.autosave = args.command == "decompile" and argv[0] != "decompile"
    try:
        if args.command == "batch":
            return args.handler(args)
        with PyBridge.for_path(args.input, args.python) as bridge:
            return args.handler(args, bridge)
    except (OSError, SyntaxError, ValueError, DecompilationError, RecursionError) as exc:
        if os.getenv("PYCRUNE_TRACE") == "1":
            raise
        print(f"decompiler: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
