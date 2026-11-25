"""Round-trip regressions for the Python port."""

import marshal
import py_compile
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from pycrune import (
    PyBridge,
    compare_code,
    compare_file,
    decompile_code,
    decompile_file,
    verify_execution,
)


def compile_source(source):
    return compile(textwrap.dedent(source), "<sample>", "exec", dont_inherit=True)


class RoundTripTests(unittest.TestCase):
    def assert_roundtrip(self, source):
        original = compile_source(source)
        recovered = decompile_code(original)
        result = compare_code(original, compile_source(recovered))
        self.assertTrue(result.is_identical, "\n".join(result.discrepancies[:8]))
        return recovered

    def test_import_forms(self):
        self.assert_roundtrip(
            """
            import a.b
            import a.b as a
            import a.b.c as other
            from .site import item as alias, child
            from package import *
            """
        )

    @unittest.skipUnless(sys.version_info >= (3, 10), "pattern matching needs CPython 3.10")
    def test_mixed_patterns(self):
        self.assert_roundtrip(
            """
            def describe(value):
                match value:
                    case [a, b] if a > 0:
                        return a + b
                    case {"x": x}:
                        return x
                    case 1 | 2:
                        return "small"
                    case _:
                        return None
            """
        )

    @unittest.skipUnless(sys.version_info >= (3, 11), "legacy with cleanup uses other opcodes")
    def test_return_inside_multi_item_with(self):
        self.assert_roundtrip(
            """
            def read_values():
                with A("first") as a, B("second") as b:
                    return a.read() + b.read()
            """
        )

    def test_bridge_load_and_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "name with spaces.py"
            source.write_text('result = ("text", -0.0, 2j, b"\\x00", ...)\n', encoding="utf-8")
            bytecode = root / "name with spaces.pyc"
            py_compile.compile(
                str(source),
                str(bytecode),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
            )
            with PyBridge.for_path(bytecode) as bridge:
                model, raw = bridge.load_views(bytecode, views=(True, False))
                self.assertEqual(model.name, "<module>")
                self.assertEqual(raw.python_version, model.python_version)
                self.assertEqual(bridge.load(source).constants, model.constants)
                self.assertEqual(bridge.compile(decompile_code(model)).constants, model.constants)
            self.assertTrue(compare_file(bytecode, decompile_file(bytecode)).is_identical)

    def test_bridge_preserves_fstring_conversions(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "format.py"
            file.write_text('name = "demo"\nprint(f"{name!r}")\n', encoding="utf-8")
            result = compare_file(file, decompile_file(file))
            self.assertTrue(result.is_identical, result.discrepancies)

    def test_stripped_pyc_with_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            bytecode = Path(directory) / "stripped.pyc"
            bytecode.write_bytes(b"prefix\x00\x00\x00" + marshal.dumps(compile_source("print(23)")))
            result = compare_file(bytecode, decompile_file(bytecode))
            self.assertTrue(result.is_identical, result.discrepancies)

    def test_verify_source_encoding(self):
        source = '# coding: latin-1\nprint("é")\n'
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "test.py"
            original.write_bytes(source.encode("latin-1"))
            passed, report = verify_execution(original, source)
            self.assertTrue(passed, report)

    def test_comparison_checks_nested_code(self):
        left = compile_source("def f():\n    return 1\n")
        right = compile_source("def f():\n    return 2\n")
        result = compare_code(left, right)
        self.assertFalse(result.is_identical)
        self.assertTrue(any("/f#0" in line for line in result.discrepancies))
