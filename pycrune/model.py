"""Code objects, instructions, exception entries and normalized opcodes."""

from __future__ import annotations

import struct
from dataclasses import dataclass


def _same_value(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is float:
        return struct.pack(">d", a) == struct.pack(">d", b)
    if type(a) is complex:
        return _same_value(a.real, b.real) and _same_value(a.imag, b.imag)
    if isinstance(a, FrozenSetValue):
        return _same_items(a.items, b.items)
    if isinstance(a, (set, frozenset)):
        return _same_items(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, SliceValue):
        return (
            _same_value(a.start, b.start)
            and _same_value(a.stop, b.stop)
            and _same_value(a.step, b.step)
        )
    return a == b


def _same_items(left, right):
    return len(left) == len(right) and all(
        any(_same_value(item, other) for other in right) for item in left
    )


@dataclass
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    last_i: bool

    def encloses(self, offset):
        return offset >= self.start and offset < self.end


@dataclass
class FrozenSetValue:
    items: tuple[object, ...]


@dataclass
class SliceValue:
    start: object | None
    stop: object | None
    step: object | None


@dataclass
class PythonInstruction:
    offset: int
    op_code: int
    op_name: str
    op_arg: int | None
    arg_val: object | None
    arg_repr: str | None
    target_offset: int | None
    is_jump_target: bool
    starts_line: int | None

    def __str__(self):
        target = f" -> {self.target_offset}" if self.target_offset is not None else ""
        operand = self.arg_repr if self.arg_repr is not None else self.arg_val
        if operand is None:
            operand = ""
        argument = f" {self.op_arg} ({operand})" if self.op_arg is not None else ""
        return f"{self.offset:6}: {self.op_name:26}{argument}{target}"


@dataclass(eq=False)
class PythonCodeObject:
    name: str = None
    qual_name: str = None
    python_version: str = ""
    file_name: str = None
    first_line_no: int = 0
    arg_count: int = 0
    pos_only_arg_count: int = 0
    kw_only_arg_count: int = 0
    stack_size: int = 0
    flags: int = 0
    constants: list[object | None] = None
    names: list[str] = None
    var_names: list[str] = None
    free_vars: list[str] = None
    cell_vars: list[str] = None
    instructions: list[PythonInstruction] = None
    exception_table: list[ExceptionTableEntry] = None
    nested_codes: list[PythonCodeObject] = None


class OpCodes:
    CACHE = 0
    POP_TOP = 1
    PUSH_NULL = 2
    INTERPRETER_EXIT = 3
    END_FOR = 4
    END_SEND = 5
    NOP = 9
    UNARY_POSITIVE = 10
    UNARY_NEGATIVE = 11
    UNARY_NOT = 12
    UNARY_INVERT = 15
    BINARY_SUBSCR = 25
    BINARY_SLICE = 26
    STORE_SLICE = 27
    GET_LEN = 30
    MATCH_MAPPING = 31
    MATCH_SEQUENCE = 32
    MATCH_KEYS = 33
    PUSH_EXC_INFO = 35
    CHECK_EXC_MATCH = 36
    CHECK_EG_MATCH = 37
    WITH_EXCEPT_START = 49
    GET_AITER = 50
    GET_ANEXT = 51
    BEFORE_ASYNC_WITH = 52
    BEFORE_WITH = 53
    END_ASYNC_FOR = 54
    CLEANUP_THROW = 55
    STORE_SUBSCR = 60
    DELETE_SUBSCR = 61
    GET_ITER = 68
    GET_YIELD_FROM_ITER = 69
    LOAD_BUILD_CLASS = 71
    LOAD_ASSERTION_ERROR = 74
    RETURN_GENERATOR = 75
    RETURN_VALUE = 83
    IMPORT_STAR = 84
    SETUP_ANNOTATIONS = 85
    LEGACY_SETUP_FINALLY = 1000
    LEGACY_POP_BLOCK = 1001
    LEGACY_END_FINALLY = 1002
    LEGACY_JUMP_IF_NOT_EXC_MATCH = 1003
    LEGACY_DUP_TOP = 1004
    LEGACY_CALL_FUNCTION_KW = 1005
    LEGACY_BUILD_MAP_UNPACK = 1006
    LEGACY_SETUP_WITH = 1007
    LEGACY_YIELD_FROM = 1008
    SET_FUNCTION_ATTRIBUTE = 1009
    CONVERT_VALUE = 1010
    FORMAT_SIMPLE = 1011
    FORMAT_WITH_SPEC = 1012
    LOAD_SPECIAL = 1013
    NOT_TAKEN = 1014
    LEGACY_ROT_TWO = 1015
    LEGACY_ROT_THREE = 1016
    LEGACY_ROT_FOUR = 1017
    LEGACY_DUP_TOP_TWO = 1018
    LEGACY_LIST_TO_TUPLE = 1019
    LEGACY_BUILD_TUPLE_UNPACK = 1020
    LEGACY_BUILD_LIST_UNPACK = 1021
    LEGACY_BUILD_TUPLE_UNPACK_WITH_CALL = 1022
    LEGACY_CALL_FINALLY = 1023
    LEGACY_COPY_DICT_WITHOUT_KEYS = 1024
    LEGACY_ROT_N = 1025
    LEGACY_PREP_RERAISE_STAR = 1026
    LEGACY_ASYNC_GEN_WRAP = 1027
    LEGACY_BUILD_MAP_UNPACK_WITH_CALL = 1028
    LEGACY_BUILD_SET_UNPACK = 1029
    LOAD_LOCALS = 87
    POP_EXCEPT = 89
    STORE_NAME = 90
    DELETE_NAME = 91
    UNPACK_SEQUENCE = 92
    FOR_ITER = 93
    UNPACK_EX = 94
    STORE_ATTR = 95
    DELETE_ATTR = 96
    STORE_GLOBAL = 97
    DELETE_GLOBAL = 98
    SWAP = 99
    LOAD_CONST = 100
    LOAD_NAME = 101
    BUILD_TUPLE = 102
    BUILD_LIST = 103
    BUILD_SET = 104
    BUILD_MAP = 105
    LOAD_ATTR = 106
    COMPARE_OP = 107
    IMPORT_NAME = 108
    IMPORT_FROM = 109
    JUMP_FORWARD = 110
    POP_JUMP_IF_FALSE = 114
    POP_JUMP_IF_TRUE = 115
    LOAD_GLOBAL = 116
    IS_OP = 117
    CONTAINS_OP = 118
    RERAISE = 119
    COPY = 120
    RETURN_CONST = 121
    BINARY_OP = 122
    SEND = 123
    LOAD_FAST = 124
    STORE_FAST = 125
    DELETE_FAST = 126
    LOAD_FAST_CHECK = 127
    POP_JUMP_IF_NOT_NONE = 128
    POP_JUMP_IF_NONE = 129
    RAISE_VARARGS = 130
    GET_AWAITABLE = 131
    MAKE_FUNCTION = 132
    BUILD_SLICE = 133
    JUMP_BACKWARD_NO_INTERRUPT = 134
    MAKE_CELL = 135
    LOAD_CLOSURE = 136
    LOAD_DEREF = 137
    STORE_DEREF = 138
    DELETE_DEREF = 139
    JUMP_BACKWARD = 140
    LOAD_SUPER_ATTR = 141
    CALL_FUNCTION_EX = 142
    LOAD_FAST_AND_CLEAR = 143
    EXTENDED_ARG = 144
    LIST_APPEND = 145
    SET_ADD = 146
    MAP_ADD = 147
    COPY_FREE_VARS = 149
    YIELD_VALUE = 150
    RESUME = 151
    MATCH_CLASS = 152
    FORMAT_VALUE = 155
    BUILD_CONST_KEY_MAP = 156
    BUILD_STRING = 157
    LIST_EXTEND = 162
    SET_UPDATE = 163
    DICT_MERGE = 164
    DICT_UPDATE = 165
    CALL = 171
    KW_NAMES = 172
    CALL_INTRINSIC_1 = 173
    CALL_INTRINSIC_2 = 174
    LOAD_FROM_DICT_OR_GLOBALS = 175
    LOAD_FROM_DICT_OR_DEREF = 176

    @staticmethod
    def get_cache_entries(opcode):
        if opcode == OpCodes.BINARY_SUBSCR:
            return 1
        if opcode == OpCodes.STORE_SUBSCR:
            return 1
        if opcode == OpCodes.UNPACK_SEQUENCE:
            return 1
        if opcode == OpCodes.FOR_ITER:
            return 1
        if opcode == OpCodes.STORE_ATTR:
            return 4
        if opcode == OpCodes.LOAD_ATTR:
            return 9
        if opcode == OpCodes.COMPARE_OP:
            return 1
        if opcode == OpCodes.LOAD_GLOBAL:
            return 4
        if opcode == OpCodes.BINARY_OP:
            return 1
        if opcode == OpCodes.SEND:
            return 1
        if opcode == OpCodes.LOAD_SUPER_ATTR:
            return 1
        if opcode == OpCodes.CALL:
            return 3
        return 0

    @staticmethod
    def is_relative_forward_jump(opcode):
        return opcode in (
            OpCodes.JUMP_FORWARD,
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
            OpCodes.POP_JUMP_IF_NONE,
        )

    @staticmethod
    def is_conditional_jump(opcode):
        return opcode in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
            OpCodes.POP_JUMP_IF_NONE,
        )

    @staticmethod
    def is_relative_backward_jump(opcode):
        return opcode in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)

    @staticmethod
    def is_jump(opcode):
        return (
            OpCodes.is_relative_forward_jump(opcode)
            or OpCodes.is_relative_backward_jump(opcode)
            or opcode == OpCodes.FOR_ITER
        )

    @staticmethod
    def is_terminator(opcode):
        return opcode in (
            OpCodes.RETURN_VALUE,
            OpCodes.RETURN_CONST,
            OpCodes.RAISE_VARARGS,
            OpCodes.RERAISE,
            OpCodes.JUMP_FORWARD,
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
        )
