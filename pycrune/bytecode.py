"""Read native bytecode and map each CPython dialect to the decompiler's opcodes."""

from __future__ import annotations

import dis
import marshal
import re
from dataclasses import replace
from io import BytesIO
from types import CodeType

from ._support import DecompilationError, FailFastException
from .interpreter import (
    HOST_MAGIC,
    HOST_RELEASE,
    HOST_VERSION,
    MAGIC_VERSIONS,
    file_magic,
)
from .model import (
    ExceptionTableEntry,
    FrozenSetValue,
    OpCodes,
    PythonCodeObject,
    PythonInstruction,
    SliceValue,
)

_OP_CODES = {key: value for key, value in vars(OpCodes).items() if key.isupper()}


_LEGACY_BINARY_OPERATORS = (
    ("ADD", "+"),
    ("AND", "&"),
    ("FLOOR_DIVIDE", "//"),
    ("LSHIFT", "<<"),
    ("MATRIX_MULTIPLY", "@"),
    ("MULTIPLY", "*"),
    ("MODULO", "%"),
    ("OR", "|"),
    ("POWER", "**"),
    ("RSHIFT", ">>"),
    ("SUBTRACT", "-"),
    ("TRUE_DIVIDE", "/"),
    ("XOR", "^"),
)


_LEGACY_BINARY = {
    prefix + opcode: (i + base, symbol + suffix)
    for i, (opcode, symbol) in enumerate(_LEGACY_BINARY_OPERATORS)
    for prefix, base, suffix in (("BINARY_", 0, ""), ("INPLACE_", 13, "="))
}


_LEGACY_OPCODES = {
    "SETUP_FINALLY": "LEGACY_SETUP_FINALLY",
    "SETUP_EXCEPT": "LEGACY_SETUP_FINALLY",
    "SETUP_WITH": "LEGACY_SETUP_WITH",
    "SETUP_ASYNC_WITH": "LEGACY_SETUP_WITH",
    "CALL_KW": "LEGACY_CALL_FUNCTION_KW",
    **{name[7:]: name for name in _OP_CODES if name.startswith("LEGACY_")},
}


_DISPLAY_OPCODES = {
    "SETUP_EXCEPT": "SETUP_FINALLY",
    "SETUP_ASYNC_WITH": "SETUP_WITH",
    "CALL_KW": "CALL_FUNCTION_KW",
}


_NOOP_FOR_PARSER = {
    "PRECALL",
    "POP_ITER",
    "GEN_START",
    "BEGIN_FINALLY",
    "POP_FINALLY",
    "WITH_CLEANUP_START",
    "WITH_CLEANUP_FINISH",
    "TO_BOOL",
}


_JUMP_OPCODES = (
    set(getattr(dis, "hasjump", ()))
    | set(getattr(dis, "hasjrel", ()))
    | set(getattr(dis, "hasjabs", ()))
)


def optimization_level(file, optimize=None):
    if optimize is not None:
        if type(optimize) is not int or optimize not in (0, 1, 2):
            raise ValueError("optimize must be 0, 1 or 2")
        return optimize
    match = re.search("\\.opt-([12])\\.pyc$", file.name, re.IGNORECASE)
    return int(match[1]) if match else 0


def _require_matching_magic(file, magic):
    if magic in MAGIC_VERSIONS and magic != HOST_MAGIC:
        raise DecompilationError(
            f"{file} uses CPython {MAGIC_VERSIONS[magic]}. The running interpreter is {HOST_VERSION}."
        )


def read_code(file, optimize):
    data = file.read_bytes()
    if file.suffix.lower() != ".pyc":
        return compile(data, str(file), "exec", dont_inherit=True, optimize=optimize)
    magic = file_magic(file, header=data[:4])
    _require_matching_magic(file, magic)
    offsets = dict.fromkeys((16, 0, 8, 12) if magic else (0, 8, 12, 16))
    for offset, tag in enumerate(data[:4096]):
        if len(offsets) >= 256:
            break
        if tag & 127 == ord("c"):
            offsets[offset] = None
    payload = BytesIO(data)
    for offset in offsets:
        if offset >= len(data) or data[offset] & 127 != ord("c"):
            continue
        payload.seek(offset)
        try:
            candidate = marshal.load(payload)
        except (EOFError, ValueError, TypeError, OverflowError):
            continue
        if isinstance(candidate, CodeType):
            return candidate
    raise DecompilationError(f"{file}: no readable marshal code object for CPython {HOST_VERSION}")


def _convert_constant(value, markers):
    if isinstance(value, CodeType):
        return markers.get(id(value), f"<code:{value.co_name}>")
    if isinstance(value, tuple):
        return [_convert_constant(item, markers) for item in value]
    if isinstance(value, frozenset):
        items = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        return FrozenSetValue(tuple(_convert_constant(item, markers) for item in items))
    if isinstance(value, slice):
        return SliceValue(
            _convert_constant(value.start, markers),
            _convert_constant(value.stop, markers),
            _convert_constant(value.step, markers),
        )
    return value


def model_code(code, *, normalize):
    constants = []
    nested = []
    markers = {}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            marker = f"<code:{value.co_name}:{len(nested)}>"
            markers.setdefault(id(value), marker)
            nested.append(model_code(value, normalize=normalize))
    for value in code.co_consts:
        constants.append(_convert_constant(value, markers))
    instructions = []
    starts = dict(dis.findlinestarts(code))
    for instruction in dis.get_instructions(code):
        value = instruction.argval
        if instruction.opname == "KW_NAMES" and instruction.arg is not None:
            value = code.co_consts[instruction.arg]
        target = None
        if instruction.opcode in _JUMP_OPCODES:
            target = getattr(instruction, "jump_target", None)
            if target is None and type(value) is int:
                target = value
        line = instruction.starts_line
        if type(line) is bool:
            line = starts.get(instruction.offset) if line else None
        instructions.append(
            PythonInstruction(
                instruction.offset,
                instruction.opcode,
                instruction.opname,
                instruction.arg,
                _convert_constant(value, markers),
                instruction.argrepr,
                target,
                getattr(instruction, "is_jump_target", False),
                line,
            )
        )
    entries = [
        ExceptionTableEntry(entry.start, entry.end, entry.target, entry.depth, entry.lasti)
        for entry in getattr(dis.Bytecode(code), "exception_entries", ())
    ]
    return PythonCodeObject(
        name=code.co_name,
        qual_name=getattr(code, "co_qualname", code.co_name),
        python_version=HOST_RELEASE,
        file_name=code.co_filename,
        first_line_no=code.co_firstlineno,
        arg_count=code.co_argcount,
        pos_only_arg_count=code.co_posonlyargcount,
        kw_only_arg_count=code.co_kwonlyargcount,
        stack_size=code.co_stacksize,
        flags=code.co_flags,
        constants=constants,
        names=list(code.co_names),
        var_names=list(code.co_varnames),
        free_vars=list(code.co_freevars),
        cell_vars=list(code.co_cellvars),
        instructions=_normalize_instructions(instructions, code) if normalize else instructions,
        exception_table=entries,
        nested_codes=nested,
    )


def _mapped_instruction(instruction, name, display=None, **updates):
    return replace(instruction, op_code=_OP_CODES[name], op_name=display or name, **updates)


def _local_instruction(instruction, opcode, index, code):
    name = (
        code._varname_from_oparg(index)
        if hasattr(code, "_varname_from_oparg")
        else code.co_varnames[index]
    )
    return _mapped_instruction(instruction, opcode, op_arg=index, arg_val=name, arg_repr=name)


def _fused_locals(instruction, code):
    values = instruction.arg_val
    if isinstance(values, (list, tuple)) and len(values) == 2:
        try:
            return tuple((code.co_varnames.index(name) for name in values))
        except ValueError:
            pass
    arg = instruction.op_arg
    return (arg >> 4, arg & 15)


def _normalize_instruction(ins, code):
    name = ins.op_name
    if name in _LEGACY_BINARY:
        argument, symbol = _LEGACY_BINARY[name]
        yield _mapped_instruction(
            ins, "BINARY_OP", op_arg=argument, arg_val=argument, arg_repr=symbol
        )
    elif name in _NOOP_FOR_PARSER:
        yield _mapped_instruction(ins, "NOP")
    elif name.startswith("POP_JUMP_FORWARD_IF_") or name.startswith("POP_JUMP_BACKWARD_IF_"):
        yield _mapped_instruction(ins, "POP_JUMP_IF_" + name.split("_IF_", 1)[1])
    elif name in {"JUMP_ABSOLUTE", "JUMP"} and ins.target_offset is not None:
        yield _mapped_instruction(
            ins, "JUMP_BACKWARD" if ins.target_offset < ins.offset else "JUMP_FORWARD"
        )
    elif name in {"JUMP_IF_TRUE_OR_POP", "JUMP_IF_FALSE_OR_POP"}:
        yield _mapped_instruction(
            ins, "COPY", op_arg=1, arg_val=1, arg_repr="1", target_offset=None
        )
        yield _mapped_instruction(
            ins, "POP_JUMP_IF_TRUE" if "TRUE" in name else "POP_JUMP_IF_FALSE"
        )
        yield _mapped_instruction(ins, "POP_TOP", target_offset=None)
    elif name in _LEGACY_OPCODES:
        yield _mapped_instruction(ins, _LEGACY_OPCODES[name], _DISPLAY_OPCODES.get(name, name))
    elif name in {"CALL_FUNCTION", "CALL_METHOD"}:
        yield _mapped_instruction(ins, "CALL")
    elif name == "LOAD_METHOD":
        yield _mapped_instruction(ins, "LOAD_ATTR")
    elif name in {"LOAD_FAST_CHECK", "LOAD_FAST_BORROW"}:
        yield _local_instruction(ins, "LOAD_FAST", ins.op_arg, code)
    elif name in {
        "LOAD_FAST_LOAD_FAST",
        "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
        "STORE_FAST_LOAD_FAST",
        "STORE_FAST_STORE_FAST",
    }:
        first, second = _fused_locals(ins, code)
        yield _local_instruction(
            ins, "STORE_FAST" if name.startswith("STORE") else "LOAD_FAST", first, code
        )
        yield _local_instruction(
            ins, "STORE_FAST" if name.endswith("STORE_FAST") else "LOAD_FAST", second, code
        )
    elif name == "LOAD_SMALL_INT":
        yield _mapped_instruction(
            ins, "LOAD_CONST", name, arg_val=ins.op_arg, arg_repr=str(ins.op_arg)
        )
    elif name == "LOAD_COMMON_CONSTANT":
        if ins.arg_val is AssertionError or ins.arg_repr == "AssertionError":
            yield _mapped_instruction(ins, "LOAD_ASSERTION_ERROR")
        elif isinstance(ins.arg_val, type) and ins.arg_val.__module__ == "builtins":
            symbol = ins.arg_val.__qualname__
            yield _mapped_instruction(ins, "LOAD_NAME", name, arg_val=symbol, arg_repr=symbol)
        else:
            yield _mapped_instruction(ins, "LOAD_CONST", name)
    elif name == "COMPARE_OP":
        arg_repr = ins.arg_val if isinstance(ins.arg_val, str) else ins.arg_repr
        yield _mapped_instruction(ins, name, arg_repr=arg_repr or "==")
    elif name == "BINARY_OP" and ins.arg_repr == "[]":
        yield _mapped_instruction(ins, "BINARY_SUBSCR")
    elif name in _OP_CODES:
        yield _mapped_instruction(ins, name)
    else:
        raise FailFastException(
            f"Unsupported opcode {name!r} (arg {ins.op_arg}) at offset {ins.offset}"
        )


def _normalize_instructions(instructions, code):
    result = []
    previous = None
    for ins in instructions:
        if previous == "END_FOR" and ins.op_name in {"POP_TOP", "POP_ITER"}:
            previous = ins.op_name
            continue
        result.extend(_normalize_instruction(ins, code))
        previous = ins.op_name
    return result
