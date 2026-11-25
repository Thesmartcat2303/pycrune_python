"""Encode code models using only cross-version marshal primitives."""

from __future__ import annotations

import builtins
from dataclasses import fields
from types import BuiltinFunctionType

from .model import (
    ExceptionTableEntry,
    FrozenSetValue,
    PythonCodeObject,
    PythonInstruction,
    SliceValue,
)

MARSHAL_FORMAT = 4
_INSTRUCTION_FIELDS = tuple(field.name for field in fields(PythonInstruction))


def _pack_value(value):
    if isinstance(value, FrozenSetValue):
        return {"frozenset": tuple(_pack_value(item) for item in value.items)}
    if isinstance(value, SliceValue):
        return {"slice": tuple(_pack_value(item) for item in (value.start, value.stop, value.step))}
    if isinstance(value, (type, BuiltinFunctionType)) and value.__module__ == "builtins":
        return {"builtin": value.__qualname__}
    if isinstance(value, dict):
        return {"mapping": [(_pack_value(key), _pack_value(item)) for key, item in value.items()]}
    if isinstance(value, list):
        return [_pack_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pack_value(item) for item in value)
    return value


def _unpack_value(value):
    if isinstance(value, dict):
        if "frozenset" in value:
            return FrozenSetValue(tuple(_unpack_value(item) for item in value["frozenset"]))
        if "slice" in value:
            return SliceValue(*(_unpack_value(item) for item in value["slice"]))
        if "builtin" in value:
            name = value["builtin"]
            return getattr(builtins, name, value)
        if "mapping" in value:
            return {_unpack_value(key): _unpack_value(item) for key, item in value["mapping"]}
    if isinstance(value, list):
        return [_unpack_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_unpack_value(item) for item in value)
    return value


def pack_model(code):
    instructions = []
    for instruction in code.instructions:
        data = vars(instruction).copy()
        data["arg_val"] = _pack_value(instruction.arg_val)
        instructions.append(tuple(data[name] for name in _INSTRUCTION_FIELDS))
    return {
        **vars(code),
        "constants": [_pack_value(value) for value in code.constants],
        "instructions": instructions,
        "exception_table": [vars(entry) for entry in code.exception_table],
        "nested_codes": [pack_model(nested) for nested in code.nested_codes],
    }


def unpack_model(data):
    code = dict(data)
    code["constants"] = [_unpack_value(value) for value in data["constants"]]
    code["instructions"] = []
    for values in data["instructions"]:
        instruction = dict(zip(_INSTRUCTION_FIELDS, values))
        instruction["arg_val"] = _unpack_value(instruction["arg_val"])
        code["instructions"].append(PythonInstruction(**instruction))
    code["exception_table"] = [ExceptionTableEntry(**entry) for entry in data["exception_table"]]
    code["nested_codes"] = [unpack_model(nested) for nested in data["nested_codes"]]
    return PythonCodeObject(**code)
