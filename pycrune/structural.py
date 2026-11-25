"""Reconstruct Python syntax from normalized bytecode."""

from __future__ import annotations

import math
import os
import sys
from bisect import bisect_left
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field, replace
from functools import cmp_to_key

from ._support import (
    _MISSING,
    DecompilationError,
    FailFastException,
    PatParseException,
    _add,
    _all,
    _any,
    _cast_int,
    _compare_to,
    _concat,
    _count,
    _default_if_empty,
    _distinct,
    _empty,
    _empty_string,
    _find_last_index,
    _first,
    _first_required,
    _get_range,
    _group_by,
    _item_field,
    _iter_values,
    _join,
    _of_type,
    _order_by,
    _order_by_descending,
    _parse_int,
    _raise,
    _range,
    _remove,
    _remove_all,
    _remove_range,
    _select,
    _select_indexed,
    _skip,
    _sort_ascending_descending,
    _split,
    _take,
    _trace,
    _trim,
    _try_add,
    _where,
    _zip,
)
from .cleanup import SyntheticCleanupPass
from .emitter import PythonEmitter
from .model import (
    FrozenSetValue,
    OpCodes,
    PythonCodeObject,
    SliceValue,
    _same_value,
)
from .nodes import (
    AnnAssignStmt,
    AssertionErrorExpr,
    AssertStmt,
    AssignStmt,
    AttributeExpr,
    AugAssignStmt,
    AwaitExpr,
    BinaryExpr,
    BoolOpExpr,
    BreakStmt,
    CallExpr,
    ClassDefStmt,
    ClassExpr,
    CompareExpr,
    ComprehensionClause,
    ConstantExpr,
    ContinueStmt,
    DeleteStmt,
    DictCompExpr,
    DictExpr,
    DoubleStarredExpr,
    ExceptClause,
    ExprNode,
    ExprStmt,
    FormattedValueExpr,
    ForStmt,
    FunctionDefStmt,
    FunctionExpr,
    GeneratorExpr,
    GenericBaseExpr,
    GlobalStmt,
    IfStmt,
    ImportAlias,
    ImportFromStmt,
    ImportStmt,
    JoinedStrExpr,
    LambdaExpr,
    ListCompExpr,
    ListExpr,
    MatchAsPattern,
    MatchCase,
    MatchClassPattern,
    MatchKeywordPattern,
    MatchMappingPattern,
    MatchOrPattern,
    MatchSequencePattern,
    MatchStarPattern,
    MatchStmt,
    MatchValuePattern,
    MatchWildcardPattern,
    NamedExpr,
    NameExpr,
    NonlocalStmt,
    PassStmt,
    RaiseStmt,
    ReturnStmt,
    SetCompExpr,
    SetExpr,
    SliceExpr,
    StarredExpr,
    SubscriptExpr,
    TernaryExpr,
    TryStmt,
    TupleExpr,
    TypeAliasExpr,
    TypeAliasStmt,
    UnaryExpr,
    WhileStmt,
    WithItem,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)
from .stack import SymbolicStack


def _annotation_text(value, flags):
    if flags & 16777216 and isinstance(value, ConstantExpr) and (type(value.value) is str):
        return value.value
    return PythonEmitter.emit_expr(value)


@dataclass(eq=False)
class TryScope:
    start_offset: int = 0
    body_end_offset: int = 0
    except_handlers: list[int] = field(default_factory=list)
    else_start_offset: int | None = None
    else_end_offset: int | None = None
    post_else_resume_offset: int | None = None
    finally_handler: int | None = None
    end_offset: int = 0
    depth: int = 0


@dataclass
class HandlerCompletion:
    next_idx: int
    next_offset: int
    jump_target: int | None


@dataclass
class Pat2:
    pass


@dataclass
class Pat2Val(Pat2):
    value: ExprNode


@dataclass
class Pat2Wild(Pat2):
    pass


@dataclass
class Pat2Cap(Pat2):
    slot: PatSlot


@dataclass
class Pat2Star(Pat2):
    slot: PatSlot


@dataclass
class Pat2Seq(Pat2):
    items: list[Pat2]
    is_list: bool


@dataclass
class Pat2Map(Pat2):
    entries: list[tuple[ExprNode, Pat2]]
    rest: Pat2 | None


@dataclass
class Pat2Class(Pat2):
    cls: ExprNode
    args: list[Pat2]


@dataclass
class Pat2Kw(Pat2):
    attr: str
    pat: Pat2


@dataclass
class Pat2Or(Pat2):
    alts: list[Pat2Alt]


@dataclass
class Pat2Alt:
    pat: Pat2
    caps: list[PatSlot]


@dataclass(eq=False)
class PatSlot:
    pat: Pat2 | None = None
    name: str | None = None
    is_capture: bool = False
    is_star: bool = False
    is_pseudo: bool = False
    is_keys: bool = False


@dataclass(eq=False)
class PatCtx:
    stack: list[PatSlot] = field(default_factory=list)
    i: int = 0
    fail_targets: list[int] = field(default_factory=list)
    pending_popped: list[PatSlot] = field(default_factory=list)
    case_start: int = -1
    depth: int = 0


@dataclass(eq=False)
class ConditionJump:
    instr_index: int = 0
    condition: ExprNode = None
    true_offset: int = 0
    false_offset: int = 0


class StructuralDecompiler:
    python_keywords = {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }

    def __init__(self, code, *, report_position=None):
        self._code = None
        self._instrs = None
        self._offset_to_index = {}
        self._try_scopes = []
        self._active_scopes = set()
        self._handler_stop_indices = set()
        self._active_loop_headers = []
        self._active_loop_exits = []
        self._cleared_fast_vars = set()
        self._type_params = []
        self._active_kw_names = None
        self._last_inlined_exit_start = -1
        self._last_inlined_exit_length = -1
        self._pending_break_exit_start = -1
        self._pending_break_exit_length = -1
        self._used_const_indices = None
        self._first_const_load_positions = None
        self._late_fold_constants_used = set()
        self._folded_operand_used = False
        self._code = code
        version = code.python_version
        self._version = tuple(map(int, version.split(".")[:2])) if version else sys.version_info[:2]
        self._instrs = code.instructions
        self._report_position = report_position
        i = 0
        while i < len(self._instrs):
            _try_add(self._offset_to_index, self._instrs[i].offset, i)
            if self._instrs[i].op_code == OpCodes.LOAD_FAST_AND_CLEAR:
                clear_name = (
                    _value_3
                    if (
                        _value_3 := (
                            str(_subject_2)
                            if (
                                _subject_2 := (
                                    _value_1
                                    if (_value_1 := self._instrs[i].arg_repr) is not None
                                    else self._instrs[i].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                if clear_name.startswith("NULL + "):
                    clear_name = clear_name[7:]
                if not _empty_string(clear_name):
                    _add(self._cleared_fast_vars, clear_name)
            i += 1
        self.build_try_scopes(code.exception_table)
        self.extend_finally_return_regions()

    def extend_finally_return_regions(self):
        by_handler = {}
        for entry in self._code.exception_table:
            if not entry.last_i:
                by_handler.setdefault((entry.target, entry.depth), []).append(entry)
        offsets = [ins.offset for ins in self._instrs]
        for scope in self._try_scopes:
            if scope.except_handlers or scope.finally_handler is None:
                continue
            entries = sorted(
                (
                    entry
                    for entry in by_handler.get((scope.finally_handler, scope.depth), ())
                    if scope.start_offset <= entry.start < entry.end <= scope.finally_handler
                ),
                key=lambda entry: entry.start,
            )
            if len(entries) < 2 or entries[-1].end != scope.finally_handler:
                continue
            if (
                scope.body_end_offset <= entries[0].start
                or scope.body_end_offset >= entries[-1].end
            ):
                continue
            has_return_gap = False
            for earlier, later in zip(entries, entries[1:]):
                boundary = bisect_left(offsets, later.start)
                if boundary and self._instrs[boundary - 1].offset >= earlier.end:
                    has_return_gap |= self._instrs[boundary - 1].op_code in (
                        OpCodes.RETURN_VALUE,
                        OpCodes.RETURN_CONST,
                    )
            if has_return_gap:
                scope.body_end_offset = entries[-1].end
                scope.end_offset = max(scope.end_offset, scope.body_end_offset)

    def is_active_loop_header(self, offset):
        if offset in self._active_loop_headers:
            return True
        if not (idx := self._offset_to_index.get(offset, _MISSING)) is not _MISSING:
            return False
        while idx < len(self._instrs) and self._instrs[idx].op_code == OpCodes.EXTENDED_ARG:
            idx += 1
        return idx < len(self._instrs) and self._instrs[idx].offset in self._active_loop_headers

    def skip_synthetic_var_cleanup(self, p):
        if (
            p + 2 < len(self._instrs)
            and self._instrs[p].op_code == OpCodes.LOAD_CONST
            and (self._instrs[p].arg_val is None)
            and (
                (_subject_1 := self._instrs[p + 1].op_code),
                _subject_1 == OpCodes.STORE_FAST or _subject_1 == OpCodes.STORE_NAME,
            )[-1]
            and (self._instrs[p + 2].op_code in (OpCodes.DELETE_FAST, OpCodes.DELETE_NAME))
        ):
            s = (
                str(_subject_4)
                if (
                    _subject_4 := (
                        _value_3
                        if (_value_3 := self._instrs[p + 1].arg_repr) is not None
                        else self._instrs[p + 1].arg_val
                    )
                )
                is not None
                else None
            )
            d = (
                str(_subject_6)
                if (
                    _subject_6 := (
                        _value_5
                        if (_value_5 := self._instrs[p + 2].arg_repr) is not None
                        else self._instrs[p + 2].arg_val
                    )
                )
                is not None
                else None
            )
            if s == d:
                p += 3
        return (None, p)

    def is_pep709_cleanup_reraise(self, idx):
        if idx < 1 or idx >= len(self._instrs):
            return False
        if self._instrs[idx].op_code != OpCodes.RERAISE or self._instrs[idx].op_arg != 0:
            return False
        prev = self._instrs[idx - 1]
        if not prev.op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME, OpCodes.STORE_DEREF):
            return False
        name = (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (
                        _subject_1 := (prev.arg_repr if prev.arg_repr is not None else prev.arg_val)
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if _empty_string(name) or not name in self._cleared_fast_vars:
            return False
        return (
            idx + 2 < len(self._instrs)
            and self._instrs[idx + 1].op_code == OpCodes.SWAP
            and (self._instrs[idx + 2].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME))
            and (
                (
                    _value_6
                    if (
                        _value_6 := (
                            str(_subject_5)
                            if (
                                _subject_5 := (
                                    _value_4
                                    if (_value_4 := self._instrs[idx + 2].arg_repr) is not None
                                    else self._instrs[idx + 2].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                == name
            )
        )

    def find_except_handler_region_end(self, handler_offset, upper_bound, fallback_end):
        max_end = fallback_end
        targets = []
        for e in _iter_values(self._code.exception_table):
            if (
                _item_field(e, "start", 0) < handler_offset
                or _item_field(e, "start", 0) >= upper_bound
            ):
                continue
            if _item_field(e, "end", 1) > max_end:
                max_end = _item_field(e, "end", 1)
            _add(targets, e.target)
        for t in _iter_values(_distinct(targets)):
            if not (t_idx := self._offset_to_index.get(t, _MISSING)) is not _MISSING:
                continue
            if self._instrs[t_idx].op_code == OpCodes.PUSH_EXC_INFO:
                continue
            q = t_idx
            while q < len(self._instrs):
                if self._instrs[q].op_code == OpCodes.RERAISE:
                    after = self._instrs[q].offset + 2
                    if after > max_end:
                        max_end = after
                    break
                q += 1
        return max_end

    def build_try_scopes(self, entries):
        self._try_scopes.clear()
        user_entries = list(
            _sort_ascending_descending(
                _where(
                    entries,
                    lambda e: (
                        not e.last_i
                        and (t_idx := self._offset_to_index.get(e.target, _MISSING)) is not _MISSING
                        and (self._instrs[t_idx].op_code == OpCodes.PUSH_EXC_INFO)
                        and (
                            t_idx + 1 >= len(self._instrs)
                            or self._instrs[t_idx + 1].op_code != OpCodes.WITH_EXCEPT_START
                        )
                    ),
                ),
                lambda e: _item_field(e, "start", 0),
                lambda e: _item_field(e, "end", 1),
            )
        )
        if len(user_entries) == 0 and (
            not _any(self._instrs, lambda i: i.op_code == OpCodes.PUSH_EXC_INFO)
        ):
            return
        else_starts = set()
        for e1 in _iter_values(user_entries):
            if not self.is_finally_handler(e1.target):
                for e2 in _iter_values(user_entries):
                    if (
                        self.is_finally_handler(e2.target)
                        and _item_field(e2, "start", 0) >= _item_field(e1, "end", 1)
                        and (_item_field(e2, "start", 0) < e1.target)
                    ):
                        if (
                            (
                                es_idx := self._offset_to_index.get(
                                    _item_field(e2, "start", 0), _MISSING
                                )
                            )
                            is not _MISSING
                            and (
                                ee_idx := self._offset_to_index.get(
                                    _item_field(e2, "end", 1), _MISSING
                                )
                            )
                            is not _MISSING
                            and StructuralDecompiler.has_real_statements_between(
                                self._instrs, es_idx, ee_idx
                            )
                        ):
                            _add(else_starts, _item_field(e2, "start", 0))
        entry_starts = set(
            _select(
                _where(user_entries, lambda e: not _item_field(e, "start", 0) in else_starts),
                lambda e: _item_field(e, "start", 0),
            )
        )
        jump_targets = set()
        for ins in _iter_values(self._instrs):
            if ins.target_offset is not None and ins.op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_FALSE,
            ):
                _add(jump_targets, ins.target_offset)
        clusters = []
        i = 0
        while i < len(self._instrs):
            if self._instrs[i].op_code == OpCodes.NOP:
                k = i
                nops = []
                while k < len(self._instrs) and self._instrs[k].op_code == OpCodes.NOP:
                    if k > i and (
                        self._instrs[k].offset in jump_targets
                        or self._instrs[k].offset in entry_starts
                    ):
                        break
                    _add(nops, self._instrs[k].offset)
                    k += 1
                if (
                    k < len(self._instrs)
                    and self._instrs[k].offset in entry_starts
                    and (len(nops) > 0)
                ):
                    next_off = self._instrs[k].offset
                    if _any(
                        user_entries,
                        lambda e: self.is_finally_handler(e.target) and e.encloses(nops[0]),
                    ):
                        i = k
                        continue
                    if i > 0 and self._instrs[i - 1].op_code in (
                        OpCodes.RERAISE,
                        OpCodes.PUSH_EXC_INFO,
                    ):
                        if _any(
                            user_entries,
                            lambda e: (
                                self.is_finally_handler(e.target)
                                and _item_field(e, "start", 0) < next_off
                                and (e.target > next_off)
                            ),
                        ):
                            i += 1
                            continue
                    _add(clusters, (nops, next_off))
                    i = k
                    continue
            i += 1
        all_except_targets = set(
            _select(
                _where(user_entries, lambda e: not self.is_finally_handler(e.target)),
                lambda e: e.target,
            )
        )
        for ins in _iter_values(self._instrs):
            if ins.op_code == OpCodes.PUSH_EXC_INFO and (not self.is_finally_handler(ins.offset)):
                _add(all_except_targets, ins.offset)
        except_targets = list(_order_by(all_except_targets, lambda t: t))
        except_handler_ends = {}
        for h in _iter_values(except_targets):
            if (h_idx := self._offset_to_index.get(h, _MISSING)) is not _MISSING:
                scan = h_idx
                nesting = 0
                end_off = -1
                while scan < len(self._instrs):
                    if self._instrs[scan].op_code == OpCodes.PUSH_EXC_INFO and scan != h_idx:
                        nesting += 1
                    elif self._instrs[scan].op_code == OpCodes.POP_EXCEPT:
                        is_cleanup_tail = (
                            scan + 1 < len(self._instrs)
                            and self._instrs[scan + 1].op_code == OpCodes.RERAISE
                        )
                        if is_cleanup_tail:
                            if nesting > 0:
                                nesting -= 1
                        elif nesting == 0:
                            end_off = self._instrs[scan].offset
                        elif nesting > 0:
                            nesting -= 1
                    elif self._instrs[scan].op_code == OpCodes.RERAISE and nesting == 0:
                        if end_off >= 0:
                            break
                    scan += 1
                if end_off >= 0:
                    except_handler_ends[h] = end_off
        scopes = []
        claimed_targets = set()
        cluster_pairs = []
        waiting_outers = []
        c_idx = 0
        while c_idx < len(clusters):
            cluster = clusters[c_idx]
            nops = _item_field(cluster, "nops", 0)
            bs = _item_field(cluster, "body_start", 1)
            nops_start = nops[0]
            next_bs = (
                _item_field(clusters[c_idx + 1], "nops", 0)[0]
                if c_idx + 1 < len(clusters)
                else self._instrs[-1].offset + 2
            )
            enclosing_h = list(
                _where(
                    except_targets,
                    lambda h: (
                        h < nops_start
                        and (eh_end := except_handler_ends.get(h, _MISSING)) is not _MISSING
                        and (nops_start < eh_end)
                    ),
                )
            )
            enc_end = (
                _cast_int(e_end)
                if len(enclosing_h) > 0
                and (e_end := except_handler_ends.get(enclosing_h[-1], _MISSING)) is not _MISSING
                else None
            )
            limit = min(next_bs, enc_end) if enc_end is not None else next_bs
            c_entries = list(
                _where(
                    user_entries,
                    lambda e: (
                        bs <= _item_field(e, "start", 0)
                        and _item_field(e, "start", 0) < limit
                        and (not e.target in claimed_targets)
                    ),
                )
            )
            targets = []
            for e in _iter_values(c_entries):
                if not e.target in targets:
                    _add(targets, e.target)
            bs_depth_entries = list(
                _select(
                    _where(
                        user_entries,
                        lambda e: (
                            _item_field(e, "start", 0) <= bs and bs < _item_field(e, "end", 1)
                        ),
                    ),
                    lambda e: e.depth,
                )
            )
            bs_depth = bs_depth_entries[0] if len(bs_depth_entries) > 0 else 0
            if len(nops) == 1:
                s = TryScope(
                    start_offset=nops[0], body_end_offset=bs, end_offset=bs, depth=bs_depth
                )
                for t in _iter_values(targets):
                    if not self.is_finally_handler(t) and len(s.except_handlers) == 0:
                        _add(s.except_handlers, t)
                        _add(claimed_targets, t)
                    elif self.is_finally_handler(t) and (not s.finally_handler is not None):
                        s.finally_handler = t
                        _add(claimed_targets, t)
                if len(s.except_handlers) == 0 and s.finally_handler is not None:
                    intermediate_except = _first(
                        except_targets,
                        lambda h: (
                            h > bs
                            and h < s.finally_handler
                            and (not h in claimed_targets)
                            and _any(
                                user_entries,
                                lambda e: e.target == h and _item_field(e, "start", 0) == bs,
                            )
                        ),
                        default=0,
                    )
                    if intermediate_except > 0:
                        _add(s.except_handlers, intermediate_except)
                        _add(claimed_targets, intermediate_except)
                s_t = s.except_handlers[0] if len(s.except_handlers) > 0 else s.finally_handler
                if s_t is not None:
                    nested_handler_regions = []
                    for t in _iter_values(
                        _distinct(
                            _select(_where(entries, lambda e: e.target < s_t), lambda e: e.target)
                        )
                    ):
                        if (
                            not (nested_idx := self._offset_to_index.get(t, _MISSING))
                            is not _MISSING
                        ):
                            continue
                        if self._instrs[nested_idx].op_code != OpCodes.PUSH_EXC_INFO:
                            continue
                        nested_exit = self.find_handler_normal_exit(nested_idx, len(self._instrs))
                        if nested_exit < 0:
                            continue
                        _add(nested_handler_regions, (t, self._instrs[nested_exit].offset))
                    target_entries = list(_where(user_entries, lambda e: e.target == s_t))
                    s_ends = list(
                        _select(
                            _where(
                                _where(target_entries, lambda e: not self.is_cleanup_only_entry(e)),
                                lambda e: (
                                    not _any(
                                        nested_handler_regions,
                                        lambda r: (
                                            _item_field(e, "start", 0) >= _item_field(r, "start", 0)
                                            and _item_field(e, "start", 0)
                                            <= _item_field(r, "end", 1)
                                            and (
                                                _item_field(e, "end", 1) <= _item_field(r, "end", 1)
                                            )
                                        ),
                                    )
                                ),
                            ),
                            lambda e: _item_field(e, "end", 1),
                        )
                    )
                    if len(s_ends) == 0:
                        s_ends = list(
                            _select(
                                _where(target_entries, lambda e: not self.is_cleanup_only_entry(e)),
                                lambda e: _item_field(e, "end", 1),
                            )
                        )
                    max_end = max(_iter_values(s_ends)) if len(s_ends) > 0 else bs
                    if not max_end in self._offset_to_index:
                        max_end = s_t
                    if (
                        s.finally_handler is not None
                        and len(s.except_handlers) == 0
                        and (
                            (fh_idx := self._offset_to_index.get(s.finally_handler, _MISSING))
                            is not _MISSING
                        )
                    ):
                        fh_start = fh_idx
                        if (
                            fh_start < len(self._instrs)
                            and self._instrs[fh_start].op_code == OpCodes.PUSH_EXC_INFO
                        ):
                            fh_start += 1
                        fh_end = fh_start
                        nesting = 0
                        while fh_end < len(self._instrs):
                            if self._instrs[fh_end].op_code == OpCodes.PUSH_EXC_INFO:
                                nesting += 1
                            elif self._instrs[fh_end].op_code == OpCodes.POP_EXCEPT:
                                if nesting > 0:
                                    nesting -= 1
                            elif self._instrs[fh_end].op_code in (
                                OpCodes.RERAISE,
                                OpCodes.RETURN_VALUE,
                                OpCodes.RETURN_CONST,
                            ):
                                if nesting == 0:
                                    break
                            fh_end += 1
                        fin_len = fh_end - fh_start
                        if fin_len > 0:
                            min_scan = (
                                b_idx
                                if (b_idx := self._offset_to_index.get(bs, _MISSING))
                                is not _MISSING
                                else 0
                            )
                            max_scan = min(fh_start - 1, len(self._instrs) - fin_len)
                            scan = max_scan
                            while scan >= min_scan:
                                match_ = True
                                if scan + fin_len > fh_start:
                                    scan -= 1
                                    continue
                                m = 0
                                while m < fin_len:
                                    if self._instrs[scan + m].op_code != self._instrs[
                                        fh_start + m
                                    ].op_code and (
                                        not (
                                            self._instrs[scan + m].op_code == OpCodes.JUMP_FORWARD
                                            and self._instrs[fh_start + m].op_code
                                            == OpCodes.RERAISE
                                        )
                                    ):
                                        match_ = False
                                        break
                                    m += 1
                                if match_:
                                    normal_fin_offset = self._instrs[scan].offset
                                    if normal_fin_offset > bs:
                                        max_end = normal_fin_offset
                                    break
                                scan -= 1
                    s.body_end_offset = max_end
                    if len(s.except_handlers) > 0 and (not s.finally_handler is not None):
                        fin_candidate = _first(
                            _select(
                                _order_by(
                                    _group_by(
                                        _where(
                                            user_entries,
                                            lambda e: (
                                                self.is_finally_handler(e.target)
                                                and (not e.target in claimed_targets)
                                                and (e.target > s.except_handlers[0])
                                                and (
                                                    _item_field(e, "start", 0) >= s.body_end_offset
                                                )
                                                and (
                                                    _item_field(e, "start", 0)
                                                    < s.except_handlers[0]
                                                )
                                            ),
                                        ),
                                        lambda e: e.target,
                                    ),
                                    lambda g: min(_select(g, lambda e: _item_field(e, "start", 0))),
                                ),
                                lambda g: _item_field(g, "key", 0),
                            )
                        )
                        if fin_candidate is not None and fin_candidate > 0:
                            s.finally_handler = fin_candidate
                            _add(claimed_targets, fin_candidate)
                    _add(scopes, s)
            elif len(nops) == 2:
                outer_s = TryScope(
                    start_offset=nops[0], body_end_offset=bs, end_offset=bs, depth=bs_depth
                )
                inner_s = TryScope(
                    start_offset=nops[1], body_end_offset=bs, end_offset=bs, depth=bs_depth
                )
                _add(cluster_pairs, (outer_s, inner_s))
                c_excs = list(_where(targets, lambda t: not self.is_finally_handler(t)))
                c_fins = list(_where(targets, self.is_finally_handler))
                if len(c_excs) >= 2 and len(c_fins) >= 2:
                    _add(inner_s.except_handlers, c_excs[0])
                    _add(claimed_targets, c_excs[0])
                    inner_s.finally_handler = c_fins[0]
                    _add(claimed_targets, c_fins[0])
                    _add(outer_s.except_handlers, c_excs[1])
                    _add(claimed_targets, c_excs[1])
                    outer_s.finally_handler = c_fins[1]
                    _add(claimed_targets, c_fins[1])
                    inner_bs_ends = list(
                        _select(
                            _where(
                                user_entries,
                                lambda e: (
                                    e.target == c_excs[0]
                                    and _item_field(e, "start", 0) <= bs
                                    and (bs < _item_field(e, "end", 1))
                                ),
                            ),
                            lambda e: _item_field(e, "end", 1),
                        )
                    )
                    inner_s.body_end_offset = (
                        max(_iter_values(inner_bs_ends)) if len(inner_bs_ends) > 0 else bs
                    )
                elif len(targets) == 2:
                    first_t = targets[0]
                    second_t = targets[1]
                    if self.is_finally_handler(first_t):
                        inner_s.finally_handler = first_t
                    else:
                        _add(inner_s.except_handlers, first_t)
                    _add(claimed_targets, first_t)
                    if self.is_finally_handler(second_t):
                        outer_s.finally_handler = second_t
                    else:
                        _add(outer_s.except_handlers, second_t)
                    _add(claimed_targets, second_t)
                elif len(targets) == 1:
                    first_t = targets[0]
                    if self.is_finally_handler(first_t):
                        inner_s.finally_handler = first_t
                    else:
                        _add(inner_s.except_handlers, first_t)
                    _add(claimed_targets, first_t)
                    _add(waiting_outers, outer_s)
                elif len(targets) >= 3:
                    first_t = targets[0]
                    if self.is_finally_handler(first_t):
                        inner_s.finally_handler = first_t
                    else:
                        _add(inner_s.except_handlers, first_t)
                    _add(claimed_targets, first_t)
                    rem = list(_skip(targets, 1))
                    rem_excs = list(_where(rem, lambda t: not self.is_finally_handler(t)))
                    rem_fins = list(_where(rem, self.is_finally_handler))
                    if len(rem_excs) > 0:
                        first_exc = rem_excs[0]
                        _add(outer_s.except_handlers, first_exc)
                        _add(claimed_targets, first_exc)
                        for f in _iter_values(rem_fins):
                            if f < first_exc and (not inner_s.finally_handler is not None):
                                inner_s.finally_handler = f
                                _add(claimed_targets, f)
                            elif f > first_exc and (not outer_s.finally_handler is not None):
                                outer_s.finally_handler = f
                                _add(claimed_targets, f)
                    else:
                        for f in _iter_values(rem_fins):
                            if not outer_s.finally_handler is not None:
                                outer_s.finally_handler = f
                                _add(claimed_targets, f)
                inner_t = (
                    inner_s.except_handlers[0]
                    if len(inner_s.except_handlers) > 0
                    else inner_s.finally_handler
                )
                if inner_t is not None:
                    inner_bs_ends = list(
                        _select(
                            _where(user_entries, lambda e: e.target == inner_t),
                            lambda e: _item_field(e, "end", 1),
                        )
                    )
                    inner_s.body_end_offset = (
                        max(_iter_values(inner_bs_ends)) if len(inner_bs_ends) > 0 else bs
                    )
                _add(scopes, inner_s)
                _add(scopes, outer_s)
            c_idx += 1
        w_idx = len(waiting_outers) - 1
        while w_idx >= 0:
            outer_s = waiting_outers[w_idx]
            inner_s = _item_field(
                _first_required(cluster_pairs, lambda p: _item_field(p, "outer", 0) == outer_s),
                "inner",
                1,
            )
            post_targets = list(
                _order_by(
                    _distinct(
                        _select(
                            _where(
                                user_entries,
                                lambda e: (
                                    not e.target in claimed_targets
                                    and e.target > outer_s.start_offset
                                    and (e.target < outer_s.start_offset + 600)
                                ),
                            ),
                            lambda e: e.target,
                        )
                    ),
                    lambda t: t,
                )
            )
            post_excs = list(_where(post_targets, lambda t: not self.is_finally_handler(t)))
            post_fins = list(_where(post_targets, self.is_finally_handler))
            if len(post_excs) > 0:
                first_exc = post_excs[0]
                _add(outer_s.except_handlers, first_exc)
                _add(claimed_targets, first_exc)
                for f in _iter_values(post_fins):
                    if f < first_exc and (not inner_s.finally_handler is not None):
                        inner_s.finally_handler = f
                        _add(claimed_targets, f)
                    elif f > first_exc and (not outer_s.finally_handler is not None):
                        outer_s.finally_handler = f
                        _add(claimed_targets, f)
            else:
                for f in _iter_values(post_fins):
                    if not outer_s.finally_handler is not None:
                        outer_s.finally_handler = f
                        _add(claimed_targets, f)
            w_idx -= 1
        _remove_all(
            scopes, lambda sc: len(sc.except_handlers) == 0 and (not sc.finally_handler is not None)
        )
        for s in _iter_values(scopes):
            if len(s.except_handlers) > 0 and (
                not _any(cluster_pairs, lambda p: _item_field(p, "outer", 0) == s)
            ):
                first_h = s.except_handlers[0]
                if (h_idx := self._offset_to_index.get(first_h, _MISSING)) is not _MISSING and (
                    body_end_idx := self._offset_to_index.get(s.body_end_offset, _MISSING)
                ) is not _MISSING:
                    k = h_idx - 1
                    while k > body_end_idx:
                        if (
                            self._instrs[k].op_code == OpCodes.JUMP_FORWARD
                            and (
                                _value_3
                                if (_value_3 := self._instrs[k].target_offset) is not None
                                else 0
                            )
                            > first_h
                            and StructuralDecompiler.has_real_statements_between(
                                self._instrs, body_end_idx, k
                            )
                        ):
                            s.else_start_offset = s.body_end_offset
                            s.else_end_offset = self._instrs[k].offset
                            break
                        k -= 1
                    if not s.else_start_offset is not None and s.finally_handler is not None:
                        else_entry = _first(
                            user_entries,
                            lambda e: (
                                e.target == s.finally_handler
                                and _item_field(e, "start", 0) >= s.body_end_offset
                                and (_item_field(e, "start", 0) < first_h)
                                and (
                                    (
                                        es_idx := self._offset_to_index.get(
                                            _item_field(e, "start", 0), _MISSING
                                        )
                                    )
                                    is not _MISSING
                                )
                                and (
                                    (
                                        ee_idx := self._offset_to_index.get(
                                            _item_field(e, "end", 1), _MISSING
                                        )
                                    )
                                    is not _MISSING
                                )
                                and StructuralDecompiler.has_real_statements_between(
                                    self._instrs, es_idx, ee_idx
                                )
                            ),
                        )
                        if else_entry is not None:
                            s.else_start_offset = _item_field(else_entry, "start", 0)
                            s.else_end_offset = _item_field(else_entry, "end", 1)
                    if not s.else_start_offset is not None:
                        else_entry = _first(
                            user_entries,
                            lambda e: (
                                _item_field(e, "start", 0) == s.body_end_offset
                                and e.target != first_h
                                and self.is_finally_handler(e.target)
                                and (_item_field(e, "start", 0) < first_h)
                                and (
                                    (
                                        es_idx := self._offset_to_index.get(
                                            _item_field(e, "start", 0), _MISSING
                                        )
                                    )
                                    is not _MISSING
                                )
                                and (
                                    (
                                        ee_idx := self._offset_to_index.get(
                                            _item_field(e, "end", 1), _MISSING
                                        )
                                    )
                                    is not _MISSING
                                )
                                and StructuralDecompiler.has_real_statements_between(
                                    self._instrs, es_idx, ee_idx
                                )
                            ),
                        )
                        if else_entry is not None:
                            s.else_start_offset = _item_field(else_entry, "start", 0)
                            s.else_end_offset = _item_field(else_entry, "end", 1)
        for s in _iter_values(scopes):
            if (
                len(s.except_handlers) == 0
                or s.else_start_offset is not None
                or s.finally_handler is not None
            ):
                continue
            if _any(cluster_pairs, lambda p: _item_field(p, "outer", 0) == s):
                continue
            first_h = s.except_handlers[0]
            if not (h_idx := self._offset_to_index.get(first_h, _MISSING)) is not _MISSING:
                continue
            if (
                not (body_end_idx := self._offset_to_index.get(s.body_end_offset, _MISSING))
                is not _MISSING
            ):
                continue
            if h_idx <= body_end_idx:
                continue
            if not StructuralDecompiler.has_real_statements_between(
                self._instrs, body_end_idx, h_idx
            ):
                continue
            if not (pop_offset := except_handler_ends.get(first_h, _MISSING)) is not _MISSING:
                continue
            if not (pop_idx := self._offset_to_index.get(pop_offset, _MISSING)) is not _MISSING:
                continue
            cont_idx = pop_idx + 1
            (
                (_parsed_5 := self.skip_synthetic_var_cleanup(cont_idx)),
                (cont_idx := _parsed_5[1]),
                _parsed_5[0],
            )[-1]
            if cont_idx >= len(self._instrs):
                continue
            cont = self._instrs[cont_idx]
            region_end = h_idx
            if (
                cont.op_code in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and type(cont.target_offset) is int
                and ((back_target := cont.target_offset) is not None)
                and ((back_idx := self._offset_to_index.get(back_target, _MISSING)) is not _MISSING)
            ):
                if back_idx >= body_end_idx and back_idx < h_idx:
                    continue
                if back_idx < body_end_idx:
                    k = body_end_idx
                    while k < h_idx:
                        op = self._instrs[k].op_code
                        if (
                            op
                            in (
                                OpCodes.JUMP_BACKWARD,
                                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                OpCodes.JUMP_FORWARD,
                            )
                            and (
                                (_subject_6 := self._instrs[k].target_offset),
                                type(_subject_6) is int and (t := _subject_6) is not None,
                            )[-1]
                            and ((t_idx := self._offset_to_index.get(t, _MISSING)) is not _MISSING)
                            and (t_idx < body_end_idx or t_idx >= h_idx)
                        ):
                            region_end = k
                            break
                        k += 1
            if region_end <= body_end_idx:
                continue
            run_end = cont_idx
            nesting = 0
            while run_end < len(self._instrs):
                if self._instrs[run_end].op_code == OpCodes.PUSH_EXC_INFO:
                    nesting += 1
                elif self._instrs[run_end].op_code == OpCodes.POP_EXCEPT:
                    if nesting > 0:
                        nesting -= 1
                elif self._instrs[run_end].op_code == OpCodes.RERAISE and nesting == 0:
                    break
                run_end += 1
            region_len = region_end - body_end_idx
            max_dup = min(region_len, run_end - cont_idx)
            dup_len = 0
            len_ = 1
            while len_ <= max_dup:
                match_ = True
                m = 0
                while m < len_:
                    a = self._instrs[region_end - len_ + m]
                    b = self._instrs[cont_idx + m]
                    if a.op_code != b.op_code:
                        match_ = False
                        break
                    if not a.op_code in (
                        OpCodes.JUMP_FORWARD,
                        OpCodes.JUMP_BACKWARD,
                        OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                    ) and (not _same_value(a.arg_val, b.arg_val)):
                        match_ = False
                        break
                    m += 1
                if match_:
                    dup_len = len_
                len_ += 1
            if dup_len >= region_len:
                continue
            else_end_idx = region_end - dup_len
            s.else_start_offset = self._instrs[body_end_idx].offset
            s.else_end_offset = self._instrs[else_end_idx].offset
            if dup_len > 0:
                s.post_else_resume_offset = s.else_end_offset
        for s in _iter_values(scopes):
            max_end = max(
                s.body_end_offset, s.else_end_offset if s.else_end_offset is not None else 0
            )
            for h in _iter_values(s.except_handlers):
                if (h_idx := self._offset_to_index.get(h, _MISSING)) is not _MISSING:
                    if h_idx > 0 and self._instrs[h_idx - 1].op_code == OpCodes.JUMP_FORWARD:
                        h_jump_target = (
                            _value_7
                            if (_value_7 := self._instrs[h_idx - 1].target_offset) is not None
                            else 0
                        )
                        if h_jump_target > max_end:
                            max_end = h_jump_target
                    cur_idx = h_idx
                    nesting = 0
                    while cur_idx < len(self._instrs):
                        cur = self._instrs[cur_idx]
                        nxt = self._instrs[cur_idx + 1] if cur_idx + 1 < len(self._instrs) else None
                        if cur.op_code == OpCodes.PUSH_EXC_INFO and cur_idx != h_idx:
                            nesting += 1
                        elif cur.op_code == OpCodes.JUMP_FORWARD and nesting == 0:
                            target = cur.target_offset if cur.target_offset is not None else 0
                            if target > max_end:
                                max_end = target
                        elif cur.op_code == OpCodes.POP_EXCEPT:
                            if nxt is not None and nxt.op_code == OpCodes.RERAISE:
                                if nesting > 0:
                                    nesting -= 1
                                    cur_idx += 2
                                    continue
                                else:
                                    after = nxt.offset + 2
                                    if after > max_end:
                                        max_end = after
                                    break
                            elif nesting == 0:
                                post_pop = cur_idx + 1
                                (
                                    (_parsed_9 := self.skip_synthetic_var_cleanup(post_pop)),
                                    (post_pop := _parsed_9[1]),
                                    _parsed_9[0],
                                )[-1]
                                p_ins = (
                                    self._instrs[post_pop] if post_pop < len(self._instrs) else None
                                )
                                if p_ins is not None and p_ins.op_code == OpCodes.JUMP_FORWARD:
                                    target = (
                                        p_ins.target_offset
                                        if p_ins.target_offset is not None
                                        else 0
                                    )
                                    if target > max_end:
                                        max_end = target
                                elif p_ins is not None and p_ins.op_code in (
                                    OpCodes.JUMP_BACKWARD,
                                    OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                ):
                                    target = (
                                        p_ins.target_offset
                                        if p_ins.target_offset is not None
                                        else 0
                                    )
                                    if target < p_ins.offset and target >= s.body_end_offset:
                                        max_end = target
                                        break
                                elif p_ins is not None and p_ins.op_code in (
                                    OpCodes.RETURN_VALUE,
                                    OpCodes.RETURN_CONST,
                                ):
                                    after = p_ins.offset + 2
                                    if after > max_end:
                                        max_end = after
                                    if (
                                        eh_end := except_handler_ends.get(h, _MISSING)
                                    ) is not _MISSING and cur.offset >= eh_end:
                                        break
                                else:
                                    after = (
                                        self._instrs[post_pop - 1].offset
                                        if post_pop > cur_idx + 1
                                        else cur.offset
                                    ) + 2
                                    if after > max_end:
                                        max_end = after
                                    break
                        elif cur.op_code == OpCodes.RERAISE and nesting == 0:
                            if self.is_pep709_cleanup_reraise(cur_idx):
                                handler_end = (
                                    eh_end_scan
                                    if (eh_end_scan := except_handler_ends.get(h, _MISSING))
                                    is not _MISSING
                                    else max_end
                                )
                                region_end = self.find_except_handler_region_end(
                                    h, handler_end, max_end
                                )
                                if region_end > max_end:
                                    max_end = region_end
                                cur_idx += 1
                                continue
                            after = cur.offset + 2
                            if (
                                cur.op_arg == 0
                                and cur_idx + 3 < len(self._instrs)
                                and (self._instrs[cur_idx + 1].op_code == OpCodes.COPY)
                                and (self._instrs[cur_idx + 2].op_code == OpCodes.POP_EXCEPT)
                                and (self._instrs[cur_idx + 3].op_code == OpCodes.RERAISE)
                            ):
                                after = self._instrs[cur_idx + 3].offset + 2
                            if after > max_end:
                                max_end = after
                            break
                        cur_idx += 1
            s.end_offset = max_end
        unclaimed_fins = list(
            _order_by(
                _distinct(
                    _select(
                        _where(
                            user_entries,
                            lambda e: (
                                self.is_finally_handler(e.target)
                                and (not e.target in claimed_targets)
                            ),
                        ),
                        lambda e: e.target,
                    )
                ),
                lambda t: t,
            )
        )
        for f in _iter_values(unclaimed_fins):
            f_depth_entries = list(
                _select(_where(user_entries, lambda e: e.target == f), lambda e: e.depth)
            )
            f_depth = min(_iter_values(f_depth_entries)) if len(f_depth_entries) > 0 else 0
            eligible = []
            for s in _iter_values(scopes):
                if (
                    s.start_offset < f
                    and f - s.start_offset < 600
                    and (not s.finally_handler is not None)
                    and (s.depth == f_depth)
                ):
                    enclosing_h = list(
                        _where(
                            except_targets,
                            lambda h: (
                                h < s.start_offset
                                and (eh_end := except_handler_ends.get(h, _MISSING)) is not _MISSING
                                and (s.start_offset < eh_end)
                            ),
                        )
                    )
                    enc_h = enclosing_h[-1] if enclosing_h else None
                    if enc_h is None:
                        _add(eligible, s)
                    elif (
                        e_end := except_handler_ends.get(enc_h, _MISSING)
                    ) is not _MISSING and f < e_end:
                        _add(eligible, s)
            if len(eligible) == 0:
                eligible = list(
                    _where(
                        scopes,
                        lambda s: (
                            s.start_offset < f
                            and f - s.start_offset < 600
                            and (not s.finally_handler is not None)
                            and (s.depth == f_depth)
                        ),
                    )
                )
            if len(eligible) > 0:
                chosen = _first_required(_order_by_descending(eligible, lambda s: s.start_offset))
                chosen.finally_handler = f
                _add(claimed_targets, f)
                if (f_idx := self._offset_to_index.get(f, _MISSING)) is not _MISSING:
                    if f_idx > 0 and self._instrs[f_idx - 1].op_code == OpCodes.JUMP_FORWARD:
                        target = (
                            _value_10
                            if (_value_10 := self._instrs[f_idx - 1].target_offset) is not None
                            else 0
                        )
                        if target > chosen.end_offset:
                            chosen.end_offset = target
                    else:
                        cur = f_idx
                        while cur < len(self._instrs):
                            if self._instrs[cur].op_code == OpCodes.JUMP_FORWARD:
                                target = (
                                    _value_11
                                    if (_value_11 := self._instrs[cur].target_offset) is not None
                                    else 0
                                )
                                if target > chosen.end_offset:
                                    chosen.end_offset = target
                                break
                            elif self._instrs[cur].op_code == OpCodes.RERAISE:
                                after = self._instrs[cur].offset + 2
                                if after > chosen.end_offset:
                                    chosen.end_offset = after
                                break
                            cur += 1
        for pair in _iter_values(cluster_pairs):
            if (
                _item_field(pair, "inner", 1).finally_handler is not None
                and (
                    fin_h_idx := self._offset_to_index.get(
                        _item_field(pair, "inner", 1).finally_handler, _MISSING
                    )
                )
                is not _MISSING
            ):
                if fin_h_idx > 0 and self._instrs[fin_h_idx - 1].op_code == OpCodes.JUMP_FORWARD:
                    target = (
                        _value_12
                        if (_value_12 := self._instrs[fin_h_idx - 1].target_offset) is not None
                        else 0
                    )
                    if target > _item_field(pair, "inner", 1).end_offset:
                        _item_field(pair, "inner", 1).end_offset = target
            _item_field(pair, "outer", 0).body_end_offset = _item_field(pair, "inner", 1).end_offset
            if (
                _item_field(pair, "outer", 0).body_end_offset
                > _item_field(pair, "outer", 0).end_offset
            ):
                _item_field(pair, "outer", 0).end_offset = _item_field(
                    pair, "outer", 0
                ).body_end_offset
            if len(_item_field(pair, "outer", 0).except_handlers) > 0:
                first_h = _item_field(pair, "outer", 0).except_handlers[0]
                if (h_idx := self._offset_to_index.get(first_h, _MISSING)) is not _MISSING and (
                    body_end_idx := self._offset_to_index.get(
                        _item_field(pair, "outer", 0).body_end_offset, _MISSING
                    )
                ) is not _MISSING:
                    k = h_idx - 1
                    while k > body_end_idx:
                        if (
                            self._instrs[k].op_code == OpCodes.JUMP_FORWARD
                            and (
                                _value_13
                                if (_value_13 := self._instrs[k].target_offset) is not None
                                else 0
                            )
                            > first_h
                            and StructuralDecompiler.has_real_statements_between(
                                self._instrs, body_end_idx, k
                            )
                        ):
                            _item_field(pair, "outer", 0).else_start_offset = _item_field(
                                pair, "outer", 0
                            ).body_end_offset
                            _item_field(pair, "outer", 0).else_end_offset = self._instrs[k].offset
                            break
                        k -= 1
        _remove_all(
            scopes, lambda sc: len(sc.except_handlers) == 0 and (not sc.finally_handler is not None)
        )
        for s in _iter_values(scopes):
            if s.finally_handler is not None and len(s.except_handlers) == 0:
                inner = _first(
                    scopes, lambda other: other.start_offset == s.body_end_offset and other != s
                )
                if inner is not None:
                    if (
                        inner.finally_handler is not None
                        and (if_h_idx := self._offset_to_index.get(inner.finally_handler, _MISSING))
                        is not _MISSING
                    ):
                        if (
                            if_h_idx > 0
                            and self._instrs[if_h_idx - 1].op_code == OpCodes.JUMP_FORWARD
                        ):
                            target = (
                                _value_14
                                if (_value_14 := self._instrs[if_h_idx - 1].target_offset)
                                is not None
                                else 0
                            )
                            if target > inner.end_offset:
                                inner.end_offset = target
                    if inner.end_offset > s.end_offset:
                        s.end_offset = inner.end_offset
        for s in _iter_values(scopes):
            _remove_all(s.except_handlers, lambda h: not self.is_except_handler_entry(h))
        _remove_all(
            scopes, lambda sc: len(sc.except_handlers) == 0 and (not sc.finally_handler is not None)
        )
        self.add_orphan_handler_scopes(scopes, user_entries)
        self._try_scopes.extend(scopes)

    def is_except_handler_entry(self, offset):
        if not (idx := self._offset_to_index.get(offset, _MISSING)) is not _MISSING:
            return False
        if self._instrs[idx].op_code != OpCodes.PUSH_EXC_INFO:
            return False
        if (
            idx + 1 < len(self._instrs)
            and self._instrs[idx + 1].op_code == OpCodes.WITH_EXCEPT_START
        ):
            return False
        if self.is_finally_handler(offset):
            return False
        return True

    def add_orphan_handler_scopes(self, scopes, user_entries):

        def is_covered(off):
            return _any(
                entries,
                lambda e: off >= _item_field(e, "start", 0) and off < _item_field(e, "end", 1),
            )

        entries = self._code.exception_table
        if len(entries) == 0:
            return
        h_idx = 0
        while h_idx < len(self._instrs):
            h_ins = self._instrs[h_idx]
            if h_ins.op_code != OpCodes.PUSH_EXC_INFO:
                h_idx += 1
                continue
            h_offset = h_ins.offset
            if _any(user_entries, lambda e: e.target == h_offset):
                h_idx += 1
                continue
            if not _any(entries, lambda e: _item_field(e, "start", 0) == h_offset):
                h_idx += 1
                continue
            if not self.is_except_dispatch_start(h_idx):
                h_idx += 1
                continue
            try_start_idx = self.find_try_start_index(h_idx, is_covered)
            if try_start_idx < 0:
                h_idx += 1
                continue
            try_start_offset = self._instrs[try_start_idx].offset
            body_start_idx = try_start_idx
            while body_start_idx < h_idx and self._instrs[body_start_idx].op_code == OpCodes.NOP:
                body_start_idx += 1
            if body_start_idx >= h_idx:
                h_idx += 1
                continue
            _remove_all(
                scopes,
                lambda s: (
                    s.start_offset >= try_start_offset
                    and s.start_offset < self._instrs[body_start_idx].offset
                    and (len(s.except_handlers) == 0)
                ),
            )
            if _any(scopes, lambda s: s.start_offset == try_start_offset):
                h_idx += 1
                continue
            scope = TryScope(
                start_offset=try_start_offset,
                body_end_offset=self._instrs[body_start_idx].offset,
                end_offset=self._instrs[body_start_idx].offset,
                depth=0,
            )
            _add(scope.except_handlers, h_offset)
            completion = self.find_handler_completion(h_idx)
            body_jump_idx = -1
            j = body_start_idx
            while j < h_idx:
                if (
                    self._instrs[j].op_code == OpCodes.JUMP_FORWARD
                    and (_value_1 if (_value_1 := self._instrs[j].target_offset) is not None else 0)
                    > h_offset
                ):
                    body_jump_idx = j
                    break
                j += 1
            if body_jump_idx == body_start_idx:
                scope.body_end_offset = self._instrs[body_jump_idx].offset
                else_start = self._instrs[body_jump_idx].target_offset
                if (
                    type(completion.jump_target) is int
                    and (else_end := completion.jump_target) is not None
                    and (else_end > else_start)
                ):
                    scope.else_start_offset = else_start
                    scope.else_end_offset = else_end
                    scope.end_offset = else_end
                else:
                    scope.end_offset = else_start
                _add(scopes, scope)
                h_idx += 1
                continue
            if body_jump_idx > body_start_idx:
                else_start_offset = self._instrs[body_start_idx].offset
                else_end_offset = self._instrs[body_jump_idx].offset
                fin_target = _first(
                    _select(
                        _where(
                            entries,
                            lambda e: (
                                _item_field(e, "start", 0) >= try_start_offset
                                and _item_field(e, "start", 0) <= else_start_offset
                                and (e.target > h_offset)
                                and self.is_finally_handler(e.target)
                            ),
                        ),
                        lambda e: e.target,
                    )
                )
                if fin_target is not None and fin_target > 0:
                    scope.else_start_offset = else_start_offset
                    scope.else_end_offset = else_end_offset
                    scope.finally_handler = fin_target
                    scope.end_offset = self._instrs[body_jump_idx].target_offset
                else:
                    scope.body_end_offset = else_end_offset
                    scope.end_offset = self._instrs[body_jump_idx].target_offset
                _add(scopes, scope)
                h_idx += 1
                continue
            first = self._instrs[body_start_idx]
            if first.op_code in (OpCodes.RETURN_CONST, OpCodes.RETURN_VALUE, OpCodes.RAISE_VARARGS):
                scope.body_end_offset = first.offset
                scope.end_offset = (
                    completion.next_offset if completion.next_offset > 0 else h_offset
                )
                _add(scopes, scope)
                h_idx += 1
                continue
            if (
                type(completion.jump_target) is int
                and (back_target := completion.jump_target) is not None
                and (back_target >= self._instrs[body_start_idx].offset)
                and (back_target < h_offset)
                and ((back_idx := self._offset_to_index.get(back_target, _MISSING)) is not _MISSING)
            ):
                if back_idx > body_start_idx:
                    scope.else_start_offset = self._instrs[body_start_idx].offset
                    scope.else_end_offset = back_target
                scope.end_offset = back_target
                _add(scopes, scope)
                h_idx += 1
                continue
            normal_len = h_idx - body_start_idx
            dup_len = 0
            if completion.next_idx >= 0 and completion.next_idx < len(self._instrs):
                max_ = min(normal_len, len(self._instrs) - completion.next_idx)
                m = max_
                while m >= 1:
                    match_ = True
                    k = 0
                    while k < m:
                        a = self._instrs[h_idx - m + k]
                        b = self._instrs[completion.next_idx + k]
                        if a.op_code != b.op_code or not _same_value(a.arg_val, b.arg_val):
                            match_ = False
                            break
                        k += 1
                    if match_:
                        dup_len = m
                        break
                    m -= 1
            else_end_idx = h_idx - dup_len
            if else_end_idx > body_start_idx:
                scope.else_start_offset = self._instrs[body_start_idx].offset
                scope.else_end_offset = self._instrs[else_end_idx].offset
                scope.end_offset = self._instrs[else_end_idx].offset
            else:
                scope.end_offset = self._instrs[body_start_idx].offset
            _add(scopes, scope)
            h_idx += 1

    def is_except_dispatch_start(self, h_idx):
        k = h_idx + 1
        while k < len(self._instrs):
            op = self._instrs[k].op_code
            if op == OpCodes.CHECK_EXC_MATCH or op == OpCodes.CHECK_EG_MATCH:
                return True
            if op in (
                OpCodes.WITH_EXCEPT_START,
                OpCodes.PUSH_EXC_INFO,
                OpCodes.POP_EXCEPT,
                OpCodes.RERAISE,
            ):
                return False
            k += 1
        return False

    def find_try_start_index(self, h_idx, is_covered):
        h_line = self._instrs[h_idx].starts_line
        k = h_idx - 1
        while k >= 0:
            ins = self._instrs[k]
            if ins.op_code != OpCodes.NOP or is_covered(ins.offset):
                k -= 1
                continue
            if (
                h_line is not None
                and ins.starts_line is not None
                and (ins.starts_line > _item_field(h_line, "value", 1))
            ):
                k -= 1
                continue
            run_start = k
            while (
                run_start > 0
                and self._instrs[run_start - 1].op_code == OpCodes.NOP
                and (not is_covered(self._instrs[run_start - 1].offset))
            ):
                run_start -= 1
            return run_start
            k -= 1
        return -1

    def find_handler_completion(self, h_idx):
        nesting = 0
        k = h_idx + 1
        while k < len(self._instrs):
            op = self._instrs[k].op_code
            if op == OpCodes.PUSH_EXC_INFO:
                nesting += 1
            elif op == OpCodes.POP_EXCEPT:
                if nesting > 0:
                    nesting -= 1
                    k += 1
                    continue
                next_ = k + 1
                while (
                    next_ < len(self._instrs)
                    and self._instrs[next_].op_code == OpCodes.EXTENDED_ARG
                ):
                    next_ += 1
                if next_ >= len(self._instrs):
                    return HandlerCompletion(-1, 0, None)
                if (
                    self._instrs[next_].op_code
                    in (
                        OpCodes.JUMP_FORWARD,
                        OpCodes.JUMP_BACKWARD,
                        OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                    )
                    and (
                        (_subject_2 := self._instrs[next_].target_offset),
                        type(_subject_2) is int and (target := _subject_2) is not None,
                    )[-1]
                ):
                    return HandlerCompletion(next_, self._instrs[next_].offset, target)
                if self._instrs[next_].op_code in (
                    OpCodes.RETURN_VALUE,
                    OpCodes.RETURN_CONST,
                    OpCodes.RERAISE,
                ):
                    return HandlerCompletion(next_, self._instrs[next_].offset, None)
                return HandlerCompletion(next_, self._instrs[next_].offset, None)
            elif op == OpCodes.RERAISE and nesting == 0:
                return HandlerCompletion(-1, 0, None)
            k += 1
        return HandlerCompletion(-1, 0, None)

    @staticmethod
    def body_always_terminates(body):
        return (
            len(body) > 0
            and (
                (_subject_1 := body[-1]),
                isinstance(_subject_1, RaiseStmt) or isinstance(_subject_1, ReturnStmt),
            )[-1]
        )

    def synthesize_dead_else(self, handlers):
        handler_names = list(
            _select(
                _where(
                    _select(handlers, lambda h: h.variable_name), lambda n: not _empty_string(n)
                ),
                lambda n: n,
            )
        )
        if len(handler_names) == 0:
            return []
        first_handler_var_idx = min(
            _iter_values(
                _default_if_empty(
                    _where(
                        _select(handler_names, lambda n: self.index_of_var(n)), lambda i: i >= 0
                    ),
                    -1,
                )
            )
        )
        if first_handler_var_idx < 0:
            return []
        assigned = []
        seen = set()
        for h in _iter_values(handlers):
            for name in _iter_values(StructuralDecompiler.collect_assigned_names(h.body)):
                if not _add(seen, name):
                    continue
                idx = self.index_of_var(name)
                if idx >= 0 and idx < first_handler_var_idx:
                    _add(assigned, (idx, name))
        assigned.sort(
            key=cmp_to_key(
                lambda a, b: _compare_to(_item_field(a, "index", 0), _item_field(b, "index", 0))
            )
        )
        return list(
            _select(
                assigned,
                lambda a: AssignStmt([NameExpr(_item_field(a, "name", 0))], ConstantExpr(None)),
            )
        )

    def index_of_var(self, name):
        i = 0
        while i < len(self._code.var_names):
            if self._code.var_names[i] == name:
                return i
            i += 1
        return -1

    @staticmethod
    def collect_assigned_names(stmts):
        for stmt in _iter_values(stmts):
            if isinstance((a := stmt), AssignStmt):
                for t in _iter_values(a.targets):
                    for n in _iter_values(StructuralDecompiler.collect_target_names(t)):
                        yield n
            elif isinstance((aug := stmt), AugAssignStmt):
                for n in _iter_values(StructuralDecompiler.collect_target_names(aug.target)):
                    yield n
            elif isinstance((f := stmt), ForStmt):
                for n in _iter_values(StructuralDecompiler.collect_target_names(f.target)):
                    yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(f.body)):
                    yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(f.else_body)):
                    yield n
            elif isinstance((w := stmt), WhileStmt):
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(w.body)):
                    yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(w.else_body)):
                    yield n
            elif isinstance((i := stmt), IfStmt):
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(i.then_body)):
                    yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(i.else_body)):
                    yield n
            elif isinstance((t := stmt), TryStmt):
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(t.body)):
                    yield n
                for h in _iter_values(t.handlers):
                    for n in _iter_values(StructuralDecompiler.collect_assigned_names(h.body)):
                        yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(t.else_body)):
                    yield n
                if t.finally_body is not None:
                    for n in _iter_values(
                        StructuralDecompiler.collect_assigned_names(t.finally_body)
                    ):
                        yield n
            elif isinstance((with_node := stmt), WithStmt):
                for item in _iter_values(with_node.items):
                    if item.optional_vars is not None:
                        for n in _iter_values(
                            StructuralDecompiler.collect_target_names(item.optional_vars)
                        ):
                            yield n
                for n in _iter_values(StructuralDecompiler.collect_assigned_names(with_node.body)):
                    yield n
            elif isinstance((m := stmt), MatchStmt):
                for c in _iter_values(m.cases):
                    for n in _iter_values(StructuralDecompiler.collect_assigned_names(c.body)):
                        yield n

    @staticmethod
    def collect_target_names(target):
        if (
            isinstance(target, NameExpr)
            and len(_item_field(target, "name", 0)) > 0
            and ((name := target) is not None)
        ):
            yield _item_field(name, "name", 0)
        elif isinstance((tuple_ := target), TupleExpr):
            for e in _iter_values(tuple_.elements):
                for n in _iter_values(StructuralDecompiler.collect_target_names(e)):
                    yield n
        elif isinstance((list_ := target), ListExpr):
            for e in _iter_values(list_.elements):
                for n in _iter_values(StructuralDecompiler.collect_target_names(e)):
                    yield n
        elif isinstance((star := target), StarredExpr):
            for n in _iter_values(
                StructuralDecompiler.collect_target_names(_item_field(star, "value", 1))
            ):
                yield n

    def is_cleanup_only_entry(self, e):
        if _item_field(e, "end", 1) != e.target:
            return False
        if (
            not (si := self._offset_to_index.get(_item_field(e, "start", 0), _MISSING))
            is not _MISSING
        ):
            return False
        seq = [0] * 3
        n = 0
        q = si
        while q < len(self._instrs) and self._instrs[q].offset < _item_field(e, "end", 1):
            op = self._instrs[q].op_code
            if op == OpCodes.EXTENDED_ARG:
                q += 1
                continue
            if n >= 3:
                return False
            seq[n] = op
            n += 1
            q += 1
        return (
            n == 3
            and seq[0] == OpCodes.COPY
            and (seq[1] == OpCodes.POP_EXCEPT)
            and (seq[2] == OpCodes.RERAISE)
        )

    def is_finally_handler(self, target_offset):
        if not (t_idx := self._offset_to_index.get(target_offset, _MISSING)) is not _MISSING:
            return False
        if t_idx + 1 < len(self._instrs) and self._instrs[t_idx + 1].op_code == OpCodes.POP_TOP:
            return False
        next_handler_offset = min(
            _iter_values(
                _default_if_empty(
                    _where(
                        _select(self._code.exception_table, lambda entry: entry.target),
                        lambda target: target > target_offset,
                    ),
                    2147483647,
                )
            )
        )
        limit = len(self._instrs)
        if (
            next_handler_offset != 2147483647
            and (next_handler_idx := self._offset_to_index.get(next_handler_offset, _MISSING))
            is not _MISSING
        ):
            limit = next_handler_idx
        i = t_idx
        while i < limit:
            if self._instrs[i].op_code == OpCodes.CHECK_EXC_MATCH:
                return False
            if self._instrs[i].op_code == OpCodes.POP_EXCEPT:
                return False
            if self._instrs[i].op_code == OpCodes.RERAISE and self._instrs[i].op_arg == 0:
                return True
            i += 1
        return False

    def decompile(self):
        try:
            stack = SymbolicStack()
            stmts = self.decompile_range(0, len(self._instrs), stack)
            if len(self._code.free_vars) > 0:
                modified = set()
                for ins in _iter_values(self._instrs):
                    if ins.op_code == OpCodes.STORE_DEREF:
                        v_name = (
                            _value_2
                            if (
                                _value_2 := (
                                    str(_subject_1)
                                    if (
                                        _subject_1 := (
                                            ins.arg_repr
                                            if ins.arg_repr is not None
                                            else ins.arg_val
                                        )
                                    )
                                    is not None
                                    else None
                                )
                            )
                            is not None
                            else ""
                        )
                        if v_name in self._code.free_vars:
                            _add(modified, v_name)
                if len(modified) > 0:
                    stmts.insert(0, NonlocalStmt(list(_order_by(modified, lambda v: v))))
            global_modified = set()
            for ins in _iter_values(self._instrs):
                if ins.op_code in (OpCodes.STORE_GLOBAL, OpCodes.DELETE_GLOBAL):
                    name = (
                        str(_subject_3)
                        if (
                            _subject_3 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                    if not _empty_string(name):
                        _add(global_modified, name)
            if len(global_modified) > 0:
                stmts.insert(0, GlobalStmt(list(_order_by(global_modified, lambda v: v))))
            self.preserve_unreachable_else(stmts)
            self.merge_trailing_conditional_return(stmts)
            self.append_unused_name_references(stmts)
            stmts = SyntheticCleanupPass.clean_statements(stmts)
            if self._version >= (3, 14) and (not self._code.flags & 1):
                stmts = self.restore_deferred_annotations(stmts)
            if self._code.flags & 1 and self._code.constants:
                doc = self._code.constants[0]
                has_docstring_flag = self._version < (3, 14) or self._code.flags & 67108864
                if type(doc) is str and has_docstring_flag:
                    stmts.insert(0, ExprStmt(ConstantExpr(doc)))
            return stmts
        except FailFastException as ex:
            raise FailFastException(
                "Error in code object '{}': {}".format(_item_field(self._code, "name", 0), str(ex)),
                ex,
            )
        except Exception as ex:
            raise DecompilationError(
                "Error in code object '{}': {}".format(_item_field(self._code, "name", 0), str(ex)),
                ex,
            )

    def restore_deferred_annotations(self, statements):
        function_name = "__annotate__" if self._code.name == "<module>" else "__annotate_func__"
        if function_name not in self._code.names:
            return statements
        helper = next(
            (
                stmt
                for stmt in statements
                if isinstance(stmt, FunctionDefStmt)
                and stmt.name == function_name
                and (stmt.parameters[:1] == ["format"])
            ),
            None,
        )
        if helper is None:
            return statements
        entries = []

        def collect(body):
            for stmt in body:
                if isinstance(stmt, IfStmt) and (not stmt.else_body):
                    if len(stmt.then_body) == 1 and isinstance(stmt.then_body[0], RaiseStmt):
                        continue
                    if not collect(stmt.then_body):
                        return False
                elif isinstance(stmt, AssignStmt) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if not isinstance(target, SubscriptExpr) or not isinstance(
                        target.target, DictExpr
                    ):
                        return False
                    entries.append((target.index, stmt.value))
                elif isinstance(stmt, ReturnStmt) and isinstance(stmt.value, DictExpr):
                    entries.extend(stmt.value.entries)
                else:
                    return False
            return True

        if (
            not collect(helper.body)
            or not entries
            or any(
                (
                    not isinstance(key, ConstantExpr)
                    or type(key.value) is not str
                    or (not key.value.isidentifier())
                    for key, _ in entries
                )
            )
        ):
            return statements
        annotations = [(NameExpr(key.value), value) for key, value in entries]
        synthetic = NameExpr("__conditional_annotations__")
        result = [
            stmt
            for stmt in statements
            if stmt is not helper
            and (not (isinstance(stmt, ExprStmt) and stmt.expression == synthetic))
            and (not (isinstance(stmt, AssignStmt) and stmt.targets == [synthetic]))
        ]
        pending = []
        insertion = 0
        if (
            result
            and isinstance(result[0], ExprStmt)
            and isinstance(result[0].expression, ConstantExpr)
        ):
            insertion = 1
        for target, annotation in annotations:
            idx = next(
                (
                    i
                    for i in range(insertion, len(result))
                    if isinstance(result[i], AssignStmt) and result[i].targets == [target]
                ),
                None,
            )
            if idx is None:
                pending.append(AnnAssignStmt(target, annotation))
                continue
            assignment = result[idx]
            result[idx : idx + 1] = [*pending, AnnAssignStmt(target, annotation, assignment.value)]
            insertion = idx + len(pending) + 1
            pending.clear()
        result[insertion:insertion] = pending
        return result

    def merge_trailing_conditional_return(self, stmts):
        if self.is_pre312():
            return
        while True:
            if len(stmts) < 2:
                return
            if not (
                isinstance(stmts[-1], ReturnStmt)
                and stmts[-1].from_return_const is False
                and ((last := stmts[-1]) is not None)
            ):
                return
            if not (
                isinstance(stmts[-2], IfStmt)
                and len(stmts[-2].else_body) == 0
                and ((if_stmt := stmts[-2]) is not None)
            ):
                return
            if len(if_stmt.then_body) != 1 or not (
                isinstance(if_stmt.then_body[0], ReturnStmt)
                and if_stmt.then_body[0].from_return_const is False
                and ((then_ret := if_stmt.then_body[0]) is not None)
            ):
                return
            ternary = TernaryExpr(
                if_stmt.condition,
                (
                    _value_1
                    if (_value_1 := _item_field(then_ret, "value", 1)) is not None
                    else ConstantExpr(None)
                ),
                (
                    _value_2
                    if (_value_2 := _item_field(last, "value", 1)) is not None
                    else ConstantExpr(None)
                ),
            )
            _remove_range(stmts, len(stmts) - 2, 2)
            _add(stmts, ReturnStmt(ternary))

    def try_skip_legacy_inline_exit_cleanup(self, idx, end_idx):
        p = self.skip_transparent(idx + 1)
        if (
            p + 4 < end_idx
            and StructuralDecompiler.is_none_constant(self._instrs[p])
            and (self._instrs[p + 1].op_code == OpCodes.LEGACY_DUP_TOP)
            and (self._instrs[p + 2].op_code == OpCodes.LEGACY_DUP_TOP)
            and (self._instrs[p + 3].op_code == OpCodes.CALL)
            and (
                (_value_1 if (_value_1 := self._instrs[p + 3].op_arg) is not None else 0) in (2, 3)
            )
            and (self._instrs[p + 4].op_code == OpCodes.POP_TOP)
        ):
            p = self.skip_transparent(p + 5)
            if p < end_idx and self._instrs[p].op_code == OpCodes.POP_TOP:
                p += 1
            p = self.skip_transparent(p)
        if p < end_idx and self._instrs[p].op_code in (
            OpCodes.JUMP_FORWARD,
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            OpCodes.RETURN_VALUE,
            OpCodes.RETURN_CONST,
        ):
            idx = p
            return (True, idx)
        return (False, idx)

    def try_skip_modern_inline_exit_cleanup(self, idx, end_idx):
        p = idx
        matched = False
        if (
            p + 3 < end_idx
            and StructuralDecompiler.is_none_constant(self._instrs[p])
            and (self._instrs[p + 1].op_code == OpCodes.LEGACY_DUP_TOP)
            and (self._instrs[p + 2].op_code == OpCodes.LEGACY_DUP_TOP)
        ):
            call_idx = p + 3
            while call_idx < end_idx and self._instrs[call_idx].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
                OpCodes.EXTENDED_ARG,
            ):
                call_idx += 1
            if (
                call_idx + 1 < end_idx
                and self._instrs[call_idx].op_code == OpCodes.CALL
                and (
                    (_value_2 if (_value_2 := self._instrs[call_idx].op_arg) is not None else 0)
                    in (2, 3)
                )
                and (self._instrs[call_idx + 1].op_code == OpCodes.POP_TOP)
            ):
                p = self.skip_transparent(call_idx + 2)
                matched = True
        elif (
            p + 2 < end_idx
            and StructuralDecompiler.is_none_constant(self._instrs[p])
            and StructuralDecompiler.is_none_constant(self._instrs[p + 1])
            and StructuralDecompiler.is_none_constant(self._instrs[p + 2])
        ):
            call_idx = p + 3
            while call_idx < end_idx and self._instrs[call_idx].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
                OpCodes.EXTENDED_ARG,
            ):
                call_idx += 1
            if (
                call_idx < end_idx
                and self._instrs[call_idx].op_code == OpCodes.CALL
                and (
                    (_value_5 if (_value_5 := self._instrs[call_idx].op_arg) is not None else 0)
                    in (2, 3)
                )
            ):
                p = self.skip_transparent(call_idx + 1)
                matched = True
        if not matched:
            return (False, idx)
        p = self.skip_transparent(p)
        while p < end_idx and self._instrs[p].op_code == OpCodes.POP_TOP:
            p += 1
            p = self.skip_transparent(p)
        if p < end_idx and self._instrs[p].op_code in (
            OpCodes.JUMP_FORWARD,
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            OpCodes.RETURN_VALUE,
            OpCodes.RETURN_CONST,
        ):
            idx = p
            return (True, idx)
        if (
            p + 1 < end_idx
            and StructuralDecompiler.is_none_constant(self._instrs[p])
            and (self._instrs[p + 1].op_code == OpCodes.RETURN_VALUE)
        ):
            idx = p
            return (True, idx)
        return (False, idx)

    def try_find_legacy_block_pop(self, body_start, limit, prefer_last=False):
        pop_block_idx = 0
        setup_stack = []
        found = -1
        p = body_start
        while p < limit:
            op = self._instrs[p].op_code
            if op in (OpCodes.LEGACY_SETUP_FINALLY, OpCodes.LEGACY_SETUP_WITH):
                setup_stack.append(p)
            elif op == OpCodes.LEGACY_POP_BLOCK:
                if len(setup_stack) == 0:
                    found = p
                    if not prefer_last:
                        pop_block_idx = found
                        return (True, pop_block_idx)
                else:
                    setup_stack.pop()
            p += 1
        pop_block_idx = found
        return (found >= 0, pop_block_idx)

    def try_decompile_legacy_try(self, idx, end_idx):
        try_stmt = None
        extra_stmt = None
        next_idx = 0
        try_stmt = None
        extra_stmt = None
        next_idx = idx
        if self._instrs[idx].op_code != OpCodes.LEGACY_SETUP_FINALLY:
            return (False, try_stmt, extra_stmt, next_idx)
        handler_offset = (
            _value_1 if (_value_1 := self._instrs[idx].target_offset) is not None else -1
        )
        if not (handler_idx := self._offset_to_index.get(handler_offset, _MISSING)) is not _MISSING:
            return (False, try_stmt, extra_stmt, next_idx)
        if handler_idx <= idx or handler_idx >= end_idx:
            return (False, try_stmt, extra_stmt, next_idx)
        (
            (_parsed_3 := self.try_find_legacy_block_pop(idx + 1, handler_idx, prefer_last=True)),
            (pop_block_idx := _parsed_3[1]),
            _parsed_3[0],
        )[-1]
        if pop_block_idx < 0:
            p = handler_idx - 1
            while p > idx:
                if self._instrs[p].op_code == OpCodes.LEGACY_POP_BLOCK:
                    pop_block_idx = p
                    break
                p -= 1
        if pop_block_idx < 0 and (not self.looks_like_except_handler(handler_idx)):
            abrupt_fin_end = handler_idx
            while abrupt_fin_end < end_idx and (
                not self._instrs[abrupt_fin_end].op_code
                in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY)
            ):
                abrupt_fin_end += 1
            if abrupt_fin_end < end_idx:
                abrupt_fin_body = self.decompile_range(handler_idx, abrupt_fin_end, SymbolicStack())
                try_body = self.decompile_range(idx + 1, handler_idx, SymbolicStack())
                try_stmt = TryStmt(try_body, [], [], abrupt_fin_body)
                next_idx = abrupt_fin_end + 1
                return (True, try_stmt, extra_stmt, next_idx)
        if pop_block_idx < 0:
            h_end = handler_idx
            while h_end < end_idx and (
                not self._instrs[h_end].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY)
            ):
                h_end += 1
            if h_end < end_idx:
                h_end += 1
            if (
                (_parsed_4 := self.parse_legacy_except_handlers(handler_idx, h_end)),
                (abrupt_handlers := _parsed_4[1]),
                _parsed_4[0],
            )[-1] and len(abrupt_handlers) > 0:
                abrupt_body = self.decompile_range(idx + 1, handler_idx, SymbolicStack())
                try_stmt = TryStmt(abrupt_body, abrupt_handlers, [])
                next_idx = h_end
                return (True, try_stmt, extra_stmt, next_idx)
            return (False, try_stmt, extra_stmt, next_idx)
        after_pop = pop_block_idx + 1
        while after_pop < handler_idx and self._instrs[after_pop].op_code in (
            OpCodes.NOP,
            OpCodes.EXTENDED_ARG,
        ):
            after_pop += 1
        if after_pop < handler_idx and self._instrs[after_pop].op_code == OpCodes.JUMP_FORWARD:
            normal_target_offset = (
                _value_6 if (_value_6 := self._instrs[after_pop].target_offset) is not None else -1
            )
            if (
                not (normal_target_idx := self._offset_to_index.get(normal_target_offset, _MISSING))
                is not _MISSING
            ):
                return (False, try_stmt, extra_stmt, next_idx)
            if normal_target_idx <= handler_idx or normal_target_idx > end_idx:
                return (False, try_stmt, extra_stmt, next_idx)
            dispatch_end_idx = -1
            q = handler_idx
            while q < end_idx:
                if self._instrs[q].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY):
                    dispatch_end_idx = q
                    break
                q += 1
            if dispatch_end_idx < 0:
                dispatch_end_idx = normal_target_idx
            handler_exit_targets = set()
            handler_scan_end = max(dispatch_end_idx, normal_target_idx)
            q = handler_idx
            while q < handler_scan_end:
                if (
                    self._instrs[q].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (_subject_8 := self._instrs[q].target_offset),
                        type(_subject_8) is int and (exit_target := _subject_8) is not None,
                    )[-1]
                    and (exit_target > normal_target_offset)
                ):
                    _add(handler_exit_targets, exit_target)
                q += 1
            handler_exit_offset = (
                min(_iter_values(handler_exit_targets))
                if len(handler_exit_targets) > 0
                else normal_target_offset
            )
            has_else = len(handler_exit_targets) > 0 and normal_target_offset < handler_exit_offset
            if (
                not (handler_exit_idx := self._offset_to_index.get(handler_exit_offset, _MISSING))
                is not _MISSING
            ):
                return (False, try_stmt, extra_stmt, next_idx)
            else_end_idx = min(handler_exit_idx if has_else else normal_target_idx, end_idx)
            if not (
                (_parsed_9 := self.parse_legacy_except_handlers(handler_idx, normal_target_idx)),
                (handlers := _parsed_9[1]),
                _parsed_9[0],
            )[-1]:
                return (False, try_stmt, extra_stmt, next_idx)
            if len(handlers) == 0:
                return (False, try_stmt, extra_stmt, next_idx)
            body = self.decompile_range(idx + 1, pop_block_idx, SymbolicStack())
            else_body = (
                self.decompile_range(normal_target_idx, else_end_idx, SymbolicStack())
                if has_else
                else []
            )
            try_stmt = TryStmt(body, handlers, else_body)
            next_idx = handler_exit_idx
            return (True, try_stmt, extra_stmt, next_idx)
        if after_pop < handler_idx and self.looks_like_except_handler(handler_idx):
            handler_exit = -1
            fall_handler_returns = False
            q = handler_idx
            while q < end_idx:
                if (
                    self._instrs[q].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (_subject_10 := self._instrs[q].target_offset),
                        type(_subject_10) is int and (exit_target := _subject_10) is not None,
                    )[-1]
                    and (exit_target > self._instrs[handler_idx].offset)
                    and (
                        (exit_idx := self._offset_to_index.get(exit_target, _MISSING))
                        is not _MISSING
                    )
                ):
                    handler_exit = exit_idx
                    break
                if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                    handler_exit = q + 1
                    fall_handler_returns = True
                    break
                if self._instrs[q].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY):
                    break
                q += 1
            if (
                handler_exit > handler_idx
                and (
                    (_parsed_13 := self.parse_legacy_except_handlers(handler_idx, handler_exit)),
                    (fall_handlers := _parsed_13[1]),
                    _parsed_13[0],
                )[-1]
                and (len(fall_handlers) > 0)
            ):
                body_end = pop_block_idx
                if fall_handler_returns and self._instrs[after_pop].op_code == OpCodes.RETURN_VALUE:
                    body_end = after_pop + 1
                try_body = self.decompile_range(idx + 1, body_end, SymbolicStack())
                else_body = (
                    []
                    if fall_handler_returns
                    else self.decompile_range(after_pop, handler_idx, SymbolicStack())
                )
                try_stmt = TryStmt(try_body, fall_handlers, else_body)
                next_idx = handler_exit
                return (True, try_stmt, extra_stmt, next_idx)
        if (
            after_pop < handler_idx
            and self._instrs[after_pop].op_code == OpCodes.LEGACY_CALL_FINALLY
        ):
            end_fin = handler_idx
            while end_fin < end_idx and self._instrs[end_fin].op_code != OpCodes.LEGACY_END_FINALLY:
                end_fin += 1
            if end_fin < end_idx:
                try_stack = SymbolicStack()
                try_body = self.decompile_range(idx + 1, pop_block_idx, try_stack)
                if len(try_stack) > 0:
                    _add(try_body, ReturnStmt(try_stack.pop()))
                call_finally_body = self.decompile_range(handler_idx, end_fin, SymbolicStack())
                try_stmt = TryStmt(try_body, [], [], call_finally_body)
                next_idx = end_fin + 1
                return (True, try_stmt, extra_stmt, next_idx)
        if after_pop >= handler_idx:
            end_fin = handler_idx
            while end_fin < end_idx and self._instrs[end_fin].op_code != OpCodes.LEGACY_END_FINALLY:
                end_fin += 1
            if end_fin < end_idx:
                shared_body = self.decompile_range(handler_idx, end_fin, SymbolicStack())
                shared_try = self.decompile_range(idx + 1, pop_block_idx, SymbolicStack())
                try_stmt = TryStmt(shared_try, [], [], shared_body)
                next_idx = end_fin + 1
                return (True, try_stmt, extra_stmt, next_idx)
        fin_start = after_pop
        fin_end = -1
        fin_next_idx = end_idx
        q = fin_start
        while q < end_idx:
            if (
                self._instrs[q].op_code == OpCodes.JUMP_FORWARD
                and (_value_14 if (_value_14 := self._instrs[q].target_offset) is not None else -1)
                > handler_offset
            ):
                fin_end = q
                if (
                    jump_next_idx := self._offset_to_index.get(
                        (
                            _value_15
                            if (_value_15 := self._instrs[q].target_offset) is not None
                            else -1
                        ),
                        _MISSING,
                    )
                ) is not _MISSING:
                    fin_next_idx = jump_next_idx
                break
            if (
                self._instrs[q].op_code
                in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and (
                    (_subject_17 := self._instrs[q].target_offset),
                    type(_subject_17) is int and (loop_header := _subject_17) is not None,
                )[-1]
                and self.is_active_loop_header(loop_header)
            ):
                fin_end = q
                fin_next_idx = self.find_legacy_handler_resume(handler_idx, end_idx, loop_header)
                break
            if self._instrs[q].op_code in (
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.RAISE_VARARGS,
            ):
                fin_end = q + 1
                fin_next_idx = end_idx
                break
            q += 1
        if fin_end < 0:
            return (False, try_stmt, extra_stmt, next_idx)
        returns_after_finally = (
            fin_end > fin_start and self._instrs[fin_end - 1].op_code == OpCodes.RETURN_VALUE
        )
        fin_body_end = fin_end - 1 if returns_after_finally else fin_end
        try_body_stack = SymbolicStack()
        fin_try_body = self.decompile_range(idx + 1, pop_block_idx, try_body_stack)
        if returns_after_finally and len(try_body_stack) > 0:
            _add(fin_try_body, ReturnStmt(try_body_stack.pop()))
        fin_stack = SymbolicStack()
        fin_body = self.decompile_range(fin_start, fin_body_end, fin_stack)
        handler_returns = False
        q = handler_idx
        while q < end_idx:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                handler_returns = True
                break
            if self._instrs[q].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY):
                break
            q += 1
        trailing_return = None
        if (
            returns_after_finally
            and len(fin_stack) > 0
            and (len(fin_try_body) == 0 or not isinstance(fin_try_body[-1], ReturnStmt))
        ):
            trailing_return = ReturnStmt(fin_stack.pop())
        if (
            not handler_returns
            and len(fin_body) > 0
            and (
                (_subject_21 := fin_body[-1]),
                isinstance(_subject_21, ReturnStmt) and _subject_21.from_return_const is False,
            )[-1]
        ):
            trailing_return = fin_body[-1]
            fin_body.pop(len(fin_body) - 1)
        if len(fin_body) > 0:
            fin_try_body = StructuralDecompiler.remove_inlined_finally_copies(
                fin_try_body, fin_body
            )
        if (
            len(fin_try_body) == 1
            and ((_subject_22 := fin_try_body[0]), isinstance((inner_try := _subject_22), TryStmt))[
                -1
            ]
            and (inner_try.finally_body is None)
        ):
            try_stmt = TryStmt(inner_try.body, inner_try.handlers, inner_try.else_body, fin_body)
        else:
            try_stmt = TryStmt(fin_try_body, [], [], fin_body)
        extra_stmt = trailing_return
        next_idx = fin_next_idx
        return (True, try_stmt, extra_stmt, next_idx)

    def looks_like_except_handler(self, h_idx):
        q = h_idx
        while q < len(self._instrs):
            if self._instrs[q].op_code == OpCodes.LEGACY_JUMP_IF_NOT_EXC_MATCH:
                return True
            if (
                self._instrs[q].op_code == OpCodes.COMPARE_OP
                and (
                    _value_3
                    if (
                        _value_3 := (
                            str(_subject_2)
                            if (
                                _subject_2 := (
                                    _value_1
                                    if (_value_1 := self._instrs[q].arg_repr) is not None
                                    else self._instrs[q].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                == "exception match"
            ):
                return True
            if self._instrs[q].op_code in (
                OpCodes.POP_EXCEPT,
                OpCodes.RERAISE,
                OpCodes.LEGACY_END_FINALLY,
            ):
                return False
            q += 1
        return False

    def parse_legacy_except_handlers(self, h_start, h_end):
        handlers = None
        handlers = []
        h = h_start
        while h < h_end:
            while h < h_end and self._instrs[h].op_code in (OpCodes.NOP, OpCodes.EXTENDED_ARG):
                h += 1
            if h >= h_end:
                break
            if self._instrs[h].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY):
                break
            exc_type = None
            var_name = None
            unpack_start = 0
            next_clause_offset = -1
            if self._instrs[h].op_code == OpCodes.LEGACY_DUP_TOP:
                type_start = h + 1
                check_idx = -1
                q = type_start
                while q < h_end:
                    op = self._instrs[q].op_code
                    if (
                        op == OpCodes.COMPARE_OP
                        and (
                            _value_5
                            if (
                                _value_5 := (
                                    str(_subject_4)
                                    if (
                                        _subject_4 := (
                                            _value_3
                                            if (_value_3 := self._instrs[q].arg_repr) is not None
                                            else self._instrs[q].arg_val
                                        )
                                    )
                                    is not None
                                    else None
                                )
                            )
                            is not None
                            else ""
                        )
                        == "exception match"
                    ):
                        check_idx = q
                        break
                    if op == OpCodes.LEGACY_JUMP_IF_NOT_EXC_MATCH:
                        check_idx = q
                        break
                    if op in (OpCodes.POP_TOP, OpCodes.STORE_FAST):
                        break
                    q += 1
                if check_idx < 0:
                    return (False, handlers)
                exc_type = self.decompile_legacy_expr(type_start, check_idx)
                if exc_type is None:
                    return (False, handlers)
                if self._instrs[check_idx].op_code == OpCodes.COMPARE_OP:
                    jump_pos = check_idx + 1
                    while jump_pos < h_end and self._instrs[jump_pos].op_code in (
                        OpCodes.EXTENDED_ARG,
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        jump_pos += 1
                    if (
                        jump_pos >= h_end
                        or self._instrs[jump_pos].op_code != OpCodes.POP_JUMP_IF_FALSE
                    ):
                        return (False, handlers)
                    next_clause_offset = (
                        _value_7
                        if (_value_7 := self._instrs[jump_pos].target_offset) is not None
                        else -1
                    )
                    unpack_start = jump_pos + 1
                else:
                    next_clause_offset = (
                        _value_8
                        if (_value_8 := self._instrs[check_idx].target_offset) is not None
                        else -1
                    )
                    unpack_start = check_idx + 1
            elif self._instrs[h].op_code == OpCodes.POP_TOP:
                unpack_start = h
            else:
                return (False, handlers)
            q2 = unpack_start
            pop_count = 0
            while q2 < h_end and self._instrs[q2].op_code == OpCodes.POP_TOP and (pop_count < 3):
                pop_count += 1
                q2 += 1
            if pop_count == 0:
                return (False, handlers)
            if (
                pop_count < 3
                and q2 + 1 < h_end
                and (
                    self._instrs[q2].op_code
                    in (OpCodes.STORE_FAST, OpCodes.STORE_NAME, OpCodes.STORE_DEREF)
                )
            ):
                var_name = (
                    str(_subject_11)
                    if (
                        _subject_11 := (
                            _value_10
                            if (_value_10 := self._instrs[q2].arg_repr) is not None
                            else self._instrs[q2].arg_val
                        )
                    )
                    is not None
                    else None
                )
                q2 += 1
                while q2 < h_end and self._instrs[q2].op_code == OpCodes.POP_TOP:
                    q2 += 1
            body_start = q2
            cleanup_setup_idx = -1
            if (
                body_start < h_end
                and self._instrs[body_start].op_code == OpCodes.LEGACY_SETUP_FINALLY
            ):
                cleanup_setup_idx = body_start
                body_start += 1
            body_end = -1
            if (
                cleanup_setup_idx >= 0
                and (
                    (_subject_12 := self._instrs[cleanup_setup_idx].target_offset),
                    type(_subject_12) is int and (cleanup_target := _subject_12) is not None,
                )[-1]
                and (
                    (cleanup_idx := self._offset_to_index.get(cleanup_target, _MISSING))
                    is not _MISSING
                )
                and (cleanup_idx > body_start)
                and (cleanup_idx < h_end)
            ):
                body_end = cleanup_idx
            depth = 0
            q = body_start
            while body_end < 0 and q < h_end:
                op = self._instrs[q].op_code
                if op == OpCodes.LEGACY_SETUP_FINALLY:
                    depth += 1
                elif op == OpCodes.LEGACY_POP_BLOCK:
                    if depth > 0:
                        depth -= 1
                elif op == OpCodes.POP_EXCEPT and depth == 0:
                    body_end = q
                    break
                elif op in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY) and depth == 0:
                    after_end = q + 1
                    while after_end < h_end and self._instrs[after_end].op_code in (
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        after_end += 1
                    if after_end < h_end and self._instrs[after_end].op_code == OpCodes.POP_EXCEPT:
                        q += 1
                        continue
                    break
                q += 1
            if (
                body_end < 0
                and h_end > body_start
                and (self._instrs[h_end - 1].op_code == OpCodes.RERAISE)
            ):
                body_end = h_end - 1
            if body_end < 0:
                return (False, handlers)
            if (
                body_end > body_start
                and body_end + 1 < h_end
                and (self._instrs[body_end - 1].op_code == OpCodes.LEGACY_ROT_FOUR)
                and (self._instrs[body_end].op_code == OpCodes.POP_EXCEPT)
                and (self._instrs[body_end + 1].op_code == OpCodes.RETURN_VALUE)
            ):
                return_stack = SymbolicStack()
                clause_body = self.decompile_range(body_start, body_end - 1, return_stack)
                clause_body.append(ReturnStmt(return_stack.pop()))
            else:
                clause_body = self.decompile_range(body_start, body_end, SymbolicStack())
            _add(handlers, ExceptClause(exc_type, var_name, clause_body))
            if next_clause_offset >= 0:
                if (
                    not (h := self._offset_to_index.get(next_clause_offset, _MISSING))
                    is not _MISSING
                ):
                    return (False, handlers)
            else:
                h = body_end + 1
                while h < h_end and (
                    not self._instrs[h].op_code in (OpCodes.RERAISE, OpCodes.LEGACY_END_FINALLY)
                ):
                    h += 1
        return (len(handlers) > 0, handlers)

    def decompile_legacy_expr(self, start, end):
        expr_stack = SymbolicStack()
        i = start
        while i < end:
            self.execute_instruction(self._instrs[i], expr_stack, None)
            i += 1
        return expr_stack.pop() if len(expr_stack) > 0 else None

    def try_decompile_load_special_with(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if idx + 2 >= len(self._instrs):
            return (False, with_stmt, next_idx)
        if (
            self._instrs[idx].op_code != OpCodes.COPY
            or (_value_1 if (_value_1 := self._instrs[idx].op_arg) is not None else 0) != 1
        ):
            return (False, with_stmt, next_idx)
        p = idx + 1
        while p < len(self._instrs) and self._instrs[p].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            p += 1
        if p >= len(self._instrs) or self._instrs[p].op_code != OpCodes.LOAD_SPECIAL:
            return (False, with_stmt, next_idx)
        special_loads = 0
        while p < len(self._instrs) and self._instrs[p].op_code in (
            OpCodes.LOAD_SPECIAL,
            OpCodes.SWAP,
        ):
            if self._instrs[p].op_code == OpCodes.LOAD_SPECIAL:
                special_loads += 1
            p += 1
        while p < len(self._instrs) and self._instrs[p].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            p += 1
        if (
            p >= len(self._instrs)
            or self._instrs[p].op_code != OpCodes.CALL
            or (_value_5 if (_value_5 := self._instrs[p].op_arg) is not None else 0) != 0
        ):
            return (False, with_stmt, next_idx)
        if special_loads < 2:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        if p + 1 < len(self._instrs) and self._instrs[p + 1].op_code == OpCodes.GET_AWAITABLE:
            return (False, with_stmt, next_idx)
        context = stack.peek()
        body_start = p + 1
        target = None
        if body_start < len(self._instrs) and self._instrs[body_start].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target = NameExpr(
                _value_9
                if (
                    _value_9 := (
                        str(_subject_8)
                        if (
                            _subject_8 := (
                                _value_7
                                if (_value_7 := self._instrs[body_start].arg_repr) is not None
                                else self._instrs[body_start].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            body_start += 1
        cleanup_start = -1
        q = body_start
        while q + 4 < end_idx:
            if (
                StructuralDecompiler.is_none_constant(self._instrs[q])
                and StructuralDecompiler.is_none_constant(self._instrs[q + 1])
                and StructuralDecompiler.is_none_constant(self._instrs[q + 2])
                and (self._instrs[q + 3].op_code == OpCodes.CALL)
                and (
                    (_value_10 if (_value_10 := self._instrs[q + 3].op_arg) is not None else 0) == 3
                )
                and (self._instrs[q + 4].op_code == OpCodes.POP_TOP)
            ):
                cleanup_start = q
                break
            q += 1
        if cleanup_start < 0:
            return (False, with_stmt, next_idx)
        body = self.decompile_range(body_start, cleanup_start, SymbolicStack())
        stack.pop()
        with_stmt = WithStmt([WithItem(context, target)], body)
        next_idx = cleanup_start + 5
        return (True, with_stmt, next_idx)

    @staticmethod
    def is_none_constant(ins):
        return ins.op_code == OpCodes.LOAD_CONST and (ins.arg_val is None or ins.arg_repr == "None")

    def try_decompile_legacy_with(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if self._instrs[idx].op_code != OpCodes.LEGACY_SETUP_WITH:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        context = stack.peek()
        handler_offset = (
            _value_1 if (_value_1 := self._instrs[idx].target_offset) is not None else -1
        )
        limit = (
            handler_idx
            if (handler_idx := self._offset_to_index.get(handler_offset, _MISSING)) is not _MISSING
            else end_idx
        )
        body_start = idx + 1
        while body_start < limit and self._instrs[body_start].op_code in (
            OpCodes.NOP,
            OpCodes.EXTENDED_ARG,
        ):
            body_start += 1
        target = None
        if body_start < limit and self._instrs[body_start].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target = NameExpr(
                _value_6
                if (
                    _value_6 := (
                        str(_subject_5)
                        if (
                            _subject_5 := (
                                _value_4
                                if (_value_4 := self._instrs[body_start].arg_repr) is not None
                                else self._instrs[body_start].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            body_start += 1
        elif body_start < limit and self._instrs[body_start].op_code == OpCodes.POP_TOP:
            body_start += 1
        (
            (_parsed_8 := self.try_find_legacy_block_pop(body_start, limit, prefer_last=True)),
            (pop_block_idx := _parsed_8[1]),
            _parsed_8[0],
        )[-1]
        if pop_block_idx < 0:
            abrupt_body = self.decompile_range(body_start, limit, SymbolicStack())
            pop_exc = limit
            while pop_exc < end_idx and self._instrs[pop_exc].op_code != OpCodes.POP_EXCEPT:
                pop_exc += 1
            after_handler = pop_exc + 1
            while after_handler < end_idx and self._instrs[after_handler].op_code in (
                OpCodes.POP_TOP,
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                after_handler += 1
            stack.pop()
            with_stmt = WithStmt([WithItem(context, target)], abrupt_body)
            next_idx = after_handler
            return (True, with_stmt, next_idx)
        body_stack = SymbolicStack()
        body = self.decompile_range(body_start, pop_block_idx, body_stack)
        body_return_idx = -1
        q = pop_block_idx + 1
        while q < limit:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                body_return_idx = q
                break
            q += 1
        body_has_return = body_return_idx >= 0 and len(body_stack) > 0
        if body_has_return:
            _add(body, ReturnStmt(body_stack.pop()))
        if body_has_return:
            _add(self._handler_stop_indices, limit)
            stack.pop()
            with_stmt = WithStmt([WithItem(context, target)], body)
            next_idx = body_return_idx + 1
            return (True, with_stmt, next_idx)
        next_ = pop_block_idx + 1
        while next_ < end_idx and self._instrs[next_].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
            OpCodes.EXTENDED_ARG,
        ):
            next_ += 1
        if (
            next_ + 4 < len(self._instrs)
            and StructuralDecompiler.is_none_constant(self._instrs[next_])
            and (self._instrs[next_ + 1].op_code == OpCodes.LEGACY_DUP_TOP)
            and (self._instrs[next_ + 2].op_code == OpCodes.LEGACY_DUP_TOP)
            and (self._instrs[next_ + 3].op_code == OpCodes.CALL)
            and (
                (_value_12 if (_value_12 := self._instrs[next_ + 3].op_arg) is not None else 0) == 3
            )
            and (self._instrs[next_ + 4].op_code == OpCodes.POP_TOP)
        ):
            next_ += 5
            while next_ < end_idx and self._instrs[next_].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
                OpCodes.EXTENDED_ARG,
            ):
                next_ += 1
            if next_ < end_idx and self._instrs[next_].op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                jump_target = (
                    _value_15
                    if (_value_15 := self._instrs[next_].target_offset) is not None
                    else -1
                )
                if self.is_active_loop_header(jump_target):
                    _add(self._handler_stop_indices, limit)
                    next_ = limit
                elif jump_target in self._active_loop_exits:
                    pass
                elif (
                    jump_next := self._offset_to_index.get(jump_target, _MISSING)
                ) is not _MISSING:
                    next_ = jump_next
        else:
            end_finally_idx = next_
            while end_finally_idx < end_idx and self._instrs[end_finally_idx].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                end_finally_idx += 1
            if (
                end_finally_idx >= end_idx
                or self._instrs[end_finally_idx].op_code != OpCodes.LEGACY_END_FINALLY
            ):
                return (False, with_stmt, next_idx)
            next_ = end_finally_idx + 1
        stack.pop()
        with_stmt = WithStmt([WithItem(context, target)], body)
        next_idx = next_
        return (True, with_stmt, next_idx)

    def decompile_range(self, start_idx, end_idx, stack):
        stmts = []
        idx = start_idx
        report_position = self._report_position
        while idx < end_idx:
            if report_position is not None:
                report_position(self._code, idx + 1)
            if idx in self._handler_stop_indices:
                break
            ins = self._instrs[idx]
            if (
                ins.op_code == OpCodes.LEGACY_POP_BLOCK
                and (
                    (_parsed_1 := self.try_skip_legacy_inline_exit_cleanup(idx, end_idx)),
                    (idx := _parsed_1[1]),
                    _parsed_1[0],
                )[-1]
            ):
                continue
            if (
                ins.op_code == OpCodes.LOAD_CONST
                and StructuralDecompiler.is_none_constant(ins)
                and (
                    (_parsed_2 := self.try_skip_modern_inline_exit_cleanup(idx, end_idx)),
                    (idx := _parsed_2[1]),
                    _parsed_2[0],
                )[-1]
            ):
                continue
            if ins.op_code in (OpCodes.PUSH_EXC_INFO, OpCodes.POP_EXCEPT, OpCodes.CHECK_EXC_MATCH):
                break
            if (
                (_parsed_3 := self.try_decompile_legacy_try(idx, end_idx)),
                (legacy_try_stmt := _parsed_3[1]),
                (legacy_extra_stmt := _parsed_3[2]),
                (legacy_try_next := _parsed_3[3]),
                _parsed_3[0],
            )[-1]:
                if legacy_try_stmt is not None:
                    _add(stmts, legacy_try_stmt)
                if legacy_extra_stmt is not None:
                    _add(stmts, legacy_extra_stmt)
                idx = legacy_try_next
                continue
            if (
                (_parsed_4 := self.try_decompile_with314(idx, end_idx, stack)),
                (with314_stmt := _parsed_4[1]),
                (with314_next_idx := _parsed_4[2]),
                _parsed_4[0],
            )[-1]:
                _add(stmts, with314_stmt)
                idx = with314_next_idx
                continue
            if (
                (_parsed_5 := self.try_decompile_load_special_with(idx, end_idx, stack)),
                (special_with_stmt := _parsed_5[1]),
                (special_with_next := _parsed_5[2]),
                _parsed_5[0],
            )[-1]:
                if special_with_stmt is not None:
                    _add(stmts, special_with_stmt)
                idx = special_with_next
                continue
            if (
                (_parsed_6 := self.try_decompile_legacy_with(idx, end_idx, stack)),
                (legacy_with_stmt := _parsed_6[1]),
                (legacy_with_next := _parsed_6[2]),
                _parsed_6[0],
            )[-1]:
                if legacy_with_stmt is not None:
                    _add(stmts, legacy_with_stmt)
                idx = legacy_with_next
                continue
            if (
                (_parsed_7 := self.try_decompile_dead_try_finally(idx)),
                (dead_try_stmt := _parsed_7[1]),
                (dead_try_next := _parsed_7[2]),
                _parsed_7[0],
            )[-1]:
                _add(stmts, dead_try_stmt)
                idx = dead_try_next
                continue
            if ins.op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                target_offset = ins.target_offset if ins.target_offset is not None else -1
                if target_offset in self._active_loop_exits:
                    _add(stmts, BreakStmt())
                    idx += 1
                    continue
                if self.is_active_loop_header(target_offset):
                    _add(stmts, ContinueStmt())
                    idx += 1
                    continue
                if (
                    ins.op_code == OpCodes.JUMP_FORWARD
                    and ins.target_offset is not None
                    and (
                        (jump_target_idx := self._offset_to_index.get(ins.target_offset, _MISSING))
                        is not _MISSING
                    )
                    and (jump_target_idx > idx)
                ):
                    if jump_target_idx <= end_idx:
                        idx = jump_target_idx
                        continue
                    break
            if (
                (_parsed_8 := self.try_decompile_import(idx, stmts)),
                (import_next_idx := _parsed_8[1]),
                _parsed_8[0],
            )[-1]:
                idx = import_next_idx
                continue
            if (
                (_parsed_9 := self.try_decompile_with311(idx, end_idx, stack)),
                (with311_stmt := _parsed_9[1]),
                (with311_next := _parsed_9[2]),
                _parsed_9[0],
            )[-1]:
                _add(stmts, with311_stmt)
                idx = with311_next
                continue
            if (
                (_parsed_10 := self.try_decompile_async_with311(idx, end_idx, stack)),
                (async_with311_stmt := _parsed_10[1]),
                (async_with311_next := _parsed_10[2]),
                _parsed_10[0],
            )[-1]:
                _add(stmts, async_with311_stmt)
                idx = async_with311_next
                continue
            if (
                (_parsed_11 := self.try_decompile_legacy_async_with(idx, end_idx, stack)),
                (legacy_async_with_stmt := _parsed_11[1]),
                (legacy_async_with_next := _parsed_11[2]),
                _parsed_11[0],
            )[-1]:
                _add(stmts, legacy_async_with_stmt)
                idx = legacy_async_with_next
                continue
            if (
                (_parsed_12 := self.try_decompile_with_stmt(idx, end_idx, stack)),
                (with_stmt := _parsed_12[1]),
                (with_next_idx := _parsed_12[2]),
                _parsed_12[0],
            )[-1]:
                _add(stmts, with_stmt)
                idx = with_next_idx
                continue
            if (
                (_parsed_13 := self.try_decompile_await_or_yield_from(idx, stack)),
                (await_yield_expr := _parsed_13[1]),
                (await_next_idx := _parsed_13[2]),
                _parsed_13[0],
            )[-1]:
                stack.push(await_yield_expr)
                idx = await_next_idx
                continue
            if (
                (_parsed_14 := self.try_decompile_list_genexpr315(idx, end_idx, stack)),
                (list_gen_next := _parsed_14[1]),
                _parsed_14[0],
            )[-1]:
                idx = list_gen_next
                continue
            if (
                (_parsed_15 := self.try_decompile_seq_or_pattern_legacy(idx, end_idx, stack)),
                (seq_or_stmt := _parsed_15[1]),
                (seq_or_next := _parsed_15[2]),
                _parsed_15[0],
            )[-1]:
                _add(stmts, seq_or_stmt)
                idx = seq_or_next
                continue
            if (
                (_parsed_16 := self.try_decompile_class_or_pattern_legacy(idx, end_idx, stack)),
                (class_or_stmt := _parsed_16[1]),
                (class_or_next := _parsed_16[2]),
                _parsed_16[0],
            )[-1]:
                _add(stmts, class_or_stmt)
                idx = class_or_next
                continue
            if (
                (_parsed_17 := self.try_decompile_or_pattern_legacy(idx, end_idx, stack)),
                (or_match_stmt := _parsed_17[1]),
                (or_match_next := _parsed_17[2]),
                _parsed_17[0],
            )[-1]:
                _add(stmts, or_match_stmt)
                idx = or_match_next
                continue
            if (
                (_parsed_18 := self.try_decompile_match_stmt(idx, end_idx, stack)),
                (match_stmt := _parsed_18[1]),
                (match_next_idx := _parsed_18[2]),
                _parsed_18[0],
            )[-1]:
                _add(stmts, match_stmt)
                idx = match_next_idx
                continue
            if (
                (_parsed_19 := self.try_decompile_inlined_comprehension(idx, stack, stmts)),
                (inlined_comp_next_idx := _parsed_19[1]),
                _parsed_19[0],
            )[-1]:
                idx = inlined_comp_next_idx
                continue
            if (
                (_parsed_20 := self.try_decompile_for_loop(idx, end_idx, stack)),
                (for_stmt := _parsed_20[1]),
                (for_next_idx := _parsed_20[2]),
                _parsed_20[0],
            )[-1]:
                _add(stmts, for_stmt)
                idx = for_next_idx
                continue
            if (
                (_parsed_21 := self.try_decompile_legacy_async_for_loop(idx, end_idx, stack)),
                (legacy_async_for_stmt := _parsed_21[1]),
                (legacy_async_for_next := _parsed_21[2]),
                _parsed_21[0],
            )[-1]:
                _add(stmts, legacy_async_for_stmt)
                idx = legacy_async_for_next
                continue
            if (
                (_parsed_22 := self.try_decompile_async_for_loop(idx, end_idx, stack)),
                (async_for_stmt := _parsed_22[1]),
                (async_for_next_idx := _parsed_22[2]),
                _parsed_22[0],
            )[-1]:
                _add(stmts, async_for_stmt)
                idx = async_for_next_idx
                continue
            if (
                (_parsed_23 := self.try_decompile_while_loop(idx, end_idx, stack)),
                (while_stmt := _parsed_23[1]),
                (while_next_idx := _parsed_23[2]),
                _parsed_23[0],
            )[-1]:
                _add(stmts, while_stmt)
                idx = while_next_idx
                continue
            if (
                (_parsed_24 := self.try_decompile_while_true_loop(idx, end_idx, stack)),
                (while_true_stmt := _parsed_24[1]),
                (while_true_next_idx := _parsed_24[2]),
                _parsed_24[0],
            )[-1]:
                _add(stmts, while_true_stmt)
                idx = while_true_next_idx
                continue
            if (
                (_parsed_25 := self.try_decompile_empty_try_except_else(idx, end_idx, stack)),
                (empty_try_stmt := _parsed_25[1]),
                (empty_try_next_idx := _parsed_25[2]),
                _parsed_25[0],
            )[-1]:
                _add(stmts, empty_try_stmt)
                idx = empty_try_next_idx
                continue
            if (
                (_parsed_26 := self.try_decompile_try_scope(idx, end_idx, stack)),
                (try_stmt := _parsed_26[1]),
                (try_next_idx := _parsed_26[2]),
                _parsed_26[0],
            )[-1]:
                _add(stmts, try_stmt)
                idx = try_next_idx
                continue
            if (
                (_parsed_27 := self.try_decompile_swap_unpack_assign(idx, stmts)),
                (swap_unpack_next := _parsed_27[1]),
                _parsed_27[0],
            )[-1]:
                idx = swap_unpack_next
                continue
            if (
                (_parsed_28 := self.try_decompile_unpack_assign(idx, stack, stmts)),
                (unpack_next_idx := _parsed_28[1]),
                _parsed_28[0],
            )[-1]:
                idx = unpack_next_idx
                continue
            if ins.op_code in (
                OpCodes.SWAP,
                OpCodes.LEGACY_ROT_TWO,
                OpCodes.LEGACY_ROT_THREE,
                OpCodes.LEGACY_ROT_FOUR,
            ):
                assignment, next_idx = self.try_decompile_parallel_store(idx, end_idx, stack)
                if assignment is not None:
                    stmts.append(assignment)
                    idx = next_idx
                    continue
            if (
                (_parsed_29 := self.try_decompile_multi_assign(idx, stack, stmts)),
                (multi_next_idx := _parsed_29[1]),
                _parsed_29[0],
            )[-1]:
                idx = multi_next_idx
                continue
            if (
                (_parsed_30 := self.try_decompile_inlined_finally_return(idx, stack, stmts)),
                (fin_ret_next_idx := _parsed_30[1]),
                _parsed_30[0],
            )[-1]:
                idx = fin_ret_next_idx
                continue
            if (
                (_parsed_31 := self.try_decompile_ternary_expr(idx, end_idx, stack)),
                (ternary_early_next := _parsed_31[1]),
                _parsed_31[0],
            )[-1]:
                idx = ternary_early_next
                continue
            if (
                (_parsed_32 := self.try_decompile_short_circuit_bool_op(idx, end_idx, stack)),
                (idx := _parsed_32[1]),
                _parsed_32[0],
            )[-1]:
                idx += 1
                continue
            if (
                (_parsed_33 := self.try_decompile_assert(idx, stack, stmts)),
                (assert_next_idx := _parsed_33[1]),
                _parsed_33[0],
            )[-1]:
                idx = assert_next_idx
                continue
            if (
                (_parsed_34 := self.try_decompile_chained_assign(idx, stack, stmts)),
                (chain_next_idx := _parsed_34[1]),
                _parsed_34[0],
            )[-1]:
                idx = chain_next_idx
                continue
            if (
                (_parsed_35 := self.try_decompile_walrus(idx, stack)),
                (idx := _parsed_35[1]),
                _parsed_35[0],
            )[-1]:
                continue
            if (
                (_parsed_36 := self.try_decompile_inlined_comprehension(idx, stack, stmts)),
                (comp_next_idx := _parsed_36[1]),
                _parsed_36[0],
            )[-1]:
                idx = comp_next_idx
                continue
            if (
                (
                    _parsed_37 := self.try_decompile_duplicated_tail_conditional(
                        idx, end_idx, stack, stmts
                    )
                ),
                (dup_tail_next_idx := _parsed_37[1]),
                _parsed_37[0],
            )[-1]:
                idx = dup_tail_next_idx
                continue
            if (
                ins.op_code in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                and idx + 1 < end_idx
                and (self._instrs[idx + 1].op_code == OpCodes.NOP)
                and (not self._instrs[idx + 1].starts_line is not None)
            ):
                pad_back_idx = self.skip_transparent(idx + 1)
                if (
                    pad_back_idx < end_idx
                    and self._instrs[pad_back_idx].op_code
                    in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                    and (
                        (_subject_39 := self._instrs[pad_back_idx].target_offset),
                        type(_subject_39) is int and (pad_back_target := _subject_39) is not None,
                    )[-1]
                    and (pad_back_target in self._active_loop_headers)
                ):
                    idx = pad_back_idx + 1
                    continue
            if (
                (_parsed_40 := self.try_decompile_conditional(idx, end_idx, stack)),
                (if_stmt := _parsed_40[1]),
                (if_next_idx := _parsed_40[2]),
                _parsed_40[0],
            )[-1]:
                if if_stmt is not None:
                    _add(stmts, if_stmt)
                idx = if_next_idx
                continue
            if (
                (_parsed_41 := self.try_decompile_chained_comparison(idx, stack)),
                (chained_expr := _parsed_41[1]),
                (cc_next_idx := _parsed_41[2]),
                _parsed_41[0],
            )[-1]:
                stack.push(chained_expr)
                idx = cc_next_idx
                continue
            if (
                (_parsed_42 := self.try_decompile_except_return(idx, stack, stmts)),
                (ret_next_idx := _parsed_42[1]),
                _parsed_42[0],
            )[-1]:
                idx = ret_next_idx
                continue
            try:
                self.execute_instruction(ins, stack, stmts)
            except FailFastException as ex:
                raise FailFastException(
                    "Error executing [{}] at index {}: {}".format(ins, idx, str(ex)), ex
                )
            except Exception as ex:
                raise DecompilationError(
                    "Error executing [{}] at index {}: {}".format(ins, idx, str(ex)), ex
                )
            if ins.op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                next_reachable = -1
                scan = idx + 1
                while scan < end_idx:
                    if self._instrs[scan].is_jump_target and _any(
                        _take(self._instrs, idx + 1),
                        lambda i: (
                            (i.target_offset if i.target_offset is not None else -1)
                            == self._instrs[scan].offset
                        ),
                    ):
                        next_reachable = scan
                        break
                    scan += 1
                if next_reachable == -1:
                    break
                idx = next_reachable
                continue
            idx += 1
        return stmts

    def try_decompile_empty_try_except_else(self, idx, end_idx, stack):
        stmt = None
        next_idx = 0
        stmt = None
        next_idx = idx
        if idx + 2 >= len(self._instrs):
            return (False, stmt, next_idx)
        if self._instrs[idx].op_code != OpCodes.NOP:
            return (False, stmt, next_idx)
        if self._instrs[idx + 1].op_code != OpCodes.JUMP_FORWARD:
            return (False, stmt, next_idx)
        else_offset = (
            _value_1 if (_value_1 := self._instrs[idx + 1].target_offset) is not None else -1
        )
        if else_offset <= self._instrs[idx + 1].offset:
            return (False, stmt, next_idx)
        if (
            not (else_idx := self._offset_to_index.get(else_offset, _MISSING)) is not _MISSING
            or else_idx <= idx + 1
        ):
            return (False, stmt, next_idx)
        handler_idx = idx + 2
        while (
            handler_idx < else_idx
            and handler_idx < len(self._instrs)
            and (self._instrs[handler_idx].op_code in (OpCodes.NOP, OpCodes.EXTENDED_ARG))
        ):
            handler_idx += 1
        if handler_idx >= else_idx or handler_idx >= len(self._instrs):
            return (False, stmt, next_idx)
        if self._instrs[handler_idx].op_code != OpCodes.PUSH_EXC_INFO:
            return (False, stmt, next_idx)
        handler_offset = self._instrs[handler_idx].offset
        if _any(self._code.exception_table, lambda e: e.target == handler_offset):
            return (False, stmt, next_idx)
        has_except_match = False
        exit_offset = -1
        i = handler_idx + 1
        while i < else_idx and i < len(self._instrs):
            if self._instrs[i].op_code in (OpCodes.CHECK_EXC_MATCH, OpCodes.CHECK_EG_MATCH):
                has_except_match = True
            if (
                has_except_match
                and self._instrs[i].op_code == OpCodes.JUMP_FORWARD
                and (i > handler_idx)
                and (self._instrs[i - 1].op_code == OpCodes.POP_EXCEPT)
                and (
                    (_value_4 if (_value_4 := self._instrs[i].target_offset) is not None else -1)
                    >= else_offset
                )
            ):
                exit_offset = (
                    _value_5 if (_value_5 := self._instrs[i].target_offset) is not None else -1
                )
                break
            i += 1
        if not has_except_match or exit_offset < 0:
            return (False, stmt, next_idx)
        if exit_offset not in self._offset_to_index:
            return (False, stmt, next_idx)
        if _any(
            self._try_scopes,
            lambda s: (
                s.start_offset == self._instrs[idx].offset and handler_offset in s.except_handlers
            ),
        ):
            return (False, stmt, next_idx)
        synthetic = TryScope(
            start_offset=self._instrs[idx].offset,
            body_end_offset=else_offset,
            end_offset=exit_offset,
            else_start_offset=else_offset,
            else_end_offset=exit_offset,
            depth=0,
        )
        _add(synthetic.except_handlers, handler_offset)
        _add(self._try_scopes, synthetic)
        if not (
            (_parsed_6 := self.try_decompile_try_scope(idx, end_idx, stack)),
            (stmt := _parsed_6[1]),
            (next_idx := _parsed_6[2]),
            _parsed_6[0],
        )[-1]:
            _remove(self._try_scopes, synthetic)
            return (False, stmt, next_idx)
        return (True, stmt, next_idx)

    def try_decompile_try_scope(self, idx, end_idx, stack):
        try_stmt = None
        next_idx = 0
        try_stmt = None
        next_idx = idx
        ins = self._instrs[idx]
        matching_scopes = list(
            _order_by_descending(
                _where(
                    self._try_scopes,
                    lambda s: s.start_offset == ins.offset and (not s in self._active_scopes),
                ),
                lambda s: s.end_offset - s.start_offset,
            )
        )
        if len(matching_scopes) == 0:
            return (False, try_stmt, next_idx)
        scope = matching_scopes[0]
        _add(self._active_scopes, scope)
        try:
            if (
                not (body_end_idx := self._offset_to_index.get(scope.body_end_offset, _MISSING))
                is not _MISSING
            ):
                return (False, try_stmt, next_idx)
            body_start_idx = idx
            if self._instrs[body_start_idx].op_code == OpCodes.NOP:
                body_start_idx += 1
            body_end_raw_idx = body_end_idx
            while (
                body_end_idx < len(self._instrs)
                and self._instrs[body_end_idx].op_code == OpCodes.EXTENDED_ARG
            ):
                body_end_idx += 1
            if body_end_idx < len(self._instrs) and self._instrs[body_end_idx].op_code in (
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
            ):
                body_end_idx += 1
            if body_end_idx < len(self._instrs) and self._instrs[body_end_idx].op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                body_jump_target = (
                    _value_3
                    if (_value_3 := self._instrs[body_end_idx].target_offset) is not None
                    else -1
                )
                if body_jump_target in self._active_loop_exits or (
                    body_end_raw_idx < body_end_idx and self.is_active_loop_header(body_jump_target)
                ):
                    body_end_idx += 1
            if (
                len(scope.except_handlers) == 0
                and scope.finally_handler is not None
                and scope.finally_handler in self._offset_to_index
                and (body_start_idx < body_end_idx)
                and (body_end_idx < len(self._instrs))
                and (self._instrs[body_start_idx].op_code == OpCodes.JUMP_FORWARD)
                and (
                    (
                        _value_4
                        if (_value_4 := self._instrs[body_start_idx].target_offset) is not None
                        else -1
                    )
                    == self._instrs[body_end_raw_idx].offset
                )
            ):
                scan = body_start_idx + 1
                while scan < body_end_idx and self._instrs[scan].op_code in (
                    OpCodes.NOP,
                    OpCodes.EXTENDED_ARG,
                ):
                    scan += 1
                if (
                    scan < body_end_idx
                    and self._instrs[scan].op_code == OpCodes.PUSH_EXC_INFO
                    and (scan + 1 < body_end_idx)
                    and (
                        self._instrs[scan + 1].op_code
                        in (
                            OpCodes.POP_TOP,
                            OpCodes.BUILD_LIST,
                            OpCodes.LOAD_GLOBAL,
                            OpCodes.LOAD_NAME,
                        )
                    )
                ):
                    handler_off = self._instrs[scan].offset
                    has_exit_jump = False
                    q = scan + 1
                    while q < body_end_idx:
                        if self._instrs[q].op_code == OpCodes.RERAISE:
                            break
                        if self._instrs[q].op_code == OpCodes.JUMP_FORWARD:
                            has_exit_jump = True
                            break
                        q += 1
                    if has_exit_jump and (
                        not _any(self._code.exception_table, lambda e: e.target == handler_off)
                    ):
                        _add(scope.except_handlers, handler_off)
            body_stack = stack.clone()
            try_body = self.decompile_range(body_start_idx, body_end_idx, body_stack)
            handlers = []
            for h_offset in _iter_values(scope.except_handlers):
                if not (h_idx := self._offset_to_index.get(h_offset, _MISSING)) is not _MISSING:
                    continue
                if (
                    h_idx < len(self._instrs)
                    and self._instrs[h_idx].op_code == OpCodes.PUSH_EXC_INFO
                ):
                    h_idx += 1
                while h_idx < len(self._instrs):
                    while (
                        h_idx < len(self._instrs)
                        and self._instrs[h_idx].op_code == OpCodes.EXTENDED_ARG
                    ):
                        h_idx += 1
                    if (
                        self._instrs[h_idx].op_code == OpCodes.BUILD_LIST
                        and h_idx + 1 < len(self._instrs)
                        and (self._instrs[h_idx + 1].op_code == OpCodes.COPY)
                    ):
                        h_idx += 2
                    while (
                        h_idx < len(self._instrs)
                        and self._instrs[h_idx].op_code == OpCodes.EXTENDED_ARG
                    ):
                        h_idx += 1
                    match_idx = h_idx
                    while match_idx < len(self._instrs) and (
                        not self._instrs[match_idx].op_code
                        in (
                            OpCodes.CHECK_EXC_MATCH,
                            OpCodes.CHECK_EG_MATCH,
                            OpCodes.POP_EXCEPT,
                            OpCodes.RERAISE,
                        )
                    ):
                        if (
                            self._instrs[match_idx].op_code == OpCodes.POP_TOP
                            and match_idx == h_idx
                        ):
                            break
                        match_idx += 1
                    if match_idx < len(self._instrs) and self._instrs[match_idx].op_code in (
                        OpCodes.CHECK_EXC_MATCH,
                        OpCodes.CHECK_EG_MATCH,
                    ):
                        is_star = self._instrs[match_idx].op_code == OpCodes.CHECK_EG_MATCH
                        type_stack = SymbolicStack()
                        self.decompile_range(h_idx, match_idx, type_stack)
                        exc_type = (
                            type_stack.pop() if len(type_stack) > 0 else NameExpr("Exception")
                        )
                        h_idx = match_idx + 1
                        while (
                            h_idx < len(self._instrs)
                            and self._instrs[h_idx].op_code == OpCodes.EXTENDED_ARG
                        ):
                            h_idx += 1
                        if (
                            is_star
                            and h_idx < len(self._instrs)
                            and (self._instrs[h_idx].op_code == OpCodes.COPY)
                        ):
                            h_idx += 1
                        while (
                            h_idx < len(self._instrs)
                            and self._instrs[h_idx].op_code == OpCodes.EXTENDED_ARG
                        ):
                            h_idx += 1
                        next_clause_offset = -1
                        if h_idx < len(self._instrs) and self._instrs[h_idx].op_code in (
                            OpCodes.POP_JUMP_IF_FALSE,
                            OpCodes.POP_JUMP_IF_NONE,
                        ):
                            next_clause_offset = (
                                _value_9
                                if (_value_9 := self._instrs[h_idx].target_offset) is not None
                                else -1
                            )
                            h_idx += 1
                        while (
                            h_idx < len(self._instrs)
                            and self._instrs[h_idx].op_code == OpCodes.EXTENDED_ARG
                        ):
                            h_idx += 1
                        while h_idx < len(self._instrs) and self._instrs[h_idx].op_code in (
                            OpCodes.NOP,
                            OpCodes.NOT_TAKEN,
                            OpCodes.EXTENDED_ARG,
                        ):
                            h_idx += 1
                        var_name = None
                        if h_idx < len(self._instrs) and self._instrs[h_idx].op_code in (
                            OpCodes.STORE_NAME,
                            OpCodes.STORE_FAST,
                        ):
                            var_name = (
                                str(_subject_13)
                                if (
                                    _subject_13 := (
                                        _value_12
                                        if (_value_12 := self._instrs[h_idx].arg_repr) is not None
                                        else self._instrs[h_idx].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                            h_idx += 1
                        elif (
                            h_idx < len(self._instrs)
                            and self._instrs[h_idx].op_code == OpCodes.POP_TOP
                        ):
                            h_idx += 1
                        clause_end_idx = h_idx
                        if (
                            is_star
                            and next_clause_offset >= 0
                            and (
                                (
                                    star_limit_idx := self._offset_to_index.get(
                                        next_clause_offset, _MISSING
                                    )
                                )
                                is not _MISSING
                            )
                        ):
                            scan = star_limit_idx - 1
                            while (
                                scan >= h_idx and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                            ):
                                scan -= 1
                            if scan >= h_idx and self._instrs[scan].op_code == OpCodes.JUMP_FORWARD:
                                scan -= 1
                                while (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                ):
                                    scan -= 1
                                if (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.JUMP_FORWARD
                                ):
                                    scan -= 1
                                    while (
                                        scan >= h_idx
                                        and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                    ):
                                        scan -= 1
                                if scan >= h_idx and self._instrs[scan].op_code == OpCodes.POP_TOP:
                                    scan -= 1
                                    while (
                                        scan >= h_idx
                                        and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                    ):
                                        scan -= 1
                                if (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.LIST_APPEND
                                    and (self._instrs[scan].op_arg == 3)
                                ):
                                    scan -= 1
                                    while (
                                        scan >= h_idx
                                        and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                    ):
                                        scan -= 1
                                if (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.JUMP_FORWARD
                                ):
                                    scan -= 1
                                    while (
                                        scan >= h_idx
                                        and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                    ):
                                        scan -= 1
                            clause_end_idx = scan + 1
                        elif (
                            next_clause_offset >= 0
                            and (
                                limit_idx := self._offset_to_index.get(next_clause_offset, _MISSING)
                            )
                            is not _MISSING
                        ):
                            scan = limit_idx - 1
                            while (
                                scan >= h_idx and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                            ):
                                scan -= 1
                            if (
                                scan >= h_idx
                                and self._instrs[scan].op_code == OpCodes.RERAISE
                                and (self._instrs[scan].op_arg == 1)
                            ):
                                scan -= 1
                                while (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                ):
                                    scan -= 1
                            if scan >= h_idx and self._instrs[scan].op_code in (
                                OpCodes.JUMP_FORWARD,
                                OpCodes.RETURN_CONST,
                                OpCodes.RETURN_VALUE,
                            ):
                                exit_idx = scan
                                scan -= 1
                                while (
                                    scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.EXTENDED_ARG
                                ):
                                    scan -= 1
                                clause_end_idx = (
                                    scan
                                    if scan >= h_idx
                                    and self._instrs[scan].op_code == OpCodes.POP_EXCEPT
                                    else exit_idx
                                )
                            else:
                                clause_end_idx = scan + 1
                        else:
                            nesting = 0
                            while clause_end_idx < len(self._instrs):
                                if self._instrs[clause_end_idx].op_code == OpCodes.PUSH_EXC_INFO:
                                    nesting += 1
                                elif self._instrs[clause_end_idx].op_code == OpCodes.POP_EXCEPT:
                                    if nesting == 0:
                                        break
                                    nesting -= 1
                                elif self._instrs[clause_end_idx].op_code == OpCodes.RERAISE:
                                    if nesting == 0:
                                        break
                                clause_end_idx += 1
                        clause_stack = stack.clone()
                        clause_body = self.decompile_range(h_idx, clause_end_idx, clause_stack)
                        clause_body = [
                            *SyntheticCleanupPass.clean_exception_variable(clause_body, var_name)
                        ]
                        if (
                            clause_end_idx < len(self._instrs)
                            and self._instrs[clause_end_idx].op_code == OpCodes.POP_EXCEPT
                        ):
                            after_pop = clause_end_idx + 1
                            limit_scan = (
                                nxt
                                if next_clause_offset >= 0
                                and (nxt := self._offset_to_index.get(next_clause_offset, _MISSING))
                                is not _MISSING
                                else len(self._instrs)
                            )
                            while after_pop < limit_scan:
                                if self._instrs[after_pop].op_code in (
                                    OpCodes.RETURN_VALUE,
                                    OpCodes.RETURN_CONST,
                                ):
                                    if _item_field(self._code, "name", 0) != "<module>":
                                        ret_val = (
                                            clause_stack.pop()
                                            if len(clause_stack) > len(stack)
                                            else (
                                                ConstantExpr(self._instrs[after_pop].arg_val)
                                                if self._instrs[after_pop].op_code
                                                == OpCodes.RETURN_CONST
                                                else None
                                            )
                                        )
                                        _add(clause_body, ReturnStmt(ret_val))
                                    break
                                if self._instrs[after_pop].op_code in (
                                    OpCodes.JUMP_BACKWARD,
                                    OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                ):
                                    target = (
                                        _value_17
                                        if (_value_17 := self._instrs[after_pop].target_offset)
                                        is not None
                                        else -1
                                    )
                                    if self.is_active_loop_header(target):
                                        _add(clause_body, ContinueStmt())
                                        break
                                if self._instrs[after_pop].op_code in (
                                    OpCodes.JUMP_FORWARD,
                                    OpCodes.JUMP_BACKWARD,
                                ):
                                    target = (
                                        _value_19
                                        if (_value_19 := self._instrs[after_pop].target_offset)
                                        is not None
                                        else -1
                                    )
                                    if target in self._active_loop_exits:
                                        _add(clause_body, BreakStmt())
                                        break
                                if self._instrs[after_pop].op_code == OpCodes.RERAISE:
                                    break
                                after_pop += 1
                        else:
                            pop_except_idx = -1
                            pop_depth = 0
                            q = h_idx
                            while q < clause_end_idx and q < len(self._instrs):
                                q_op = self._instrs[q].op_code
                                if q_op == OpCodes.PUSH_EXC_INFO:
                                    pop_depth += 1
                                elif q_op == OpCodes.POP_EXCEPT:
                                    if pop_depth == 0:
                                        pop_except_idx = q
                                        break
                                    pop_depth -= 1
                                elif q_op == OpCodes.RERAISE and pop_depth == 0:
                                    break
                                q += 1
                            if pop_except_idx >= 0:
                                q = pop_except_idx + 1
                                while q < clause_end_idx and q < len(self._instrs):
                                    if self._instrs[q].op_code in (
                                        OpCodes.JUMP_BACKWARD,
                                        OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                    ):
                                        target = (
                                            _value_21
                                            if (_value_21 := self._instrs[q].target_offset)
                                            is not None
                                            else -1
                                        )
                                        if self.is_active_loop_header(target):
                                            _add(clause_body, ContinueStmt())
                                            break
                                    if self._instrs[q].op_code in (
                                        OpCodes.JUMP_FORWARD,
                                        OpCodes.JUMP_BACKWARD,
                                    ):
                                        target = (
                                            _value_23
                                            if (_value_23 := self._instrs[q].target_offset)
                                            is not None
                                            else -1
                                        )
                                        if target in self._active_loop_exits:
                                            _add(clause_body, BreakStmt())
                                            break
                                    if self._instrs[q].op_code == OpCodes.RERAISE:
                                        break
                                    q += 1
                        if not _empty_string(var_name):
                            clause_body = list(
                                SyntheticCleanupPass.clean_exception_variable(clause_body, var_name)
                            )
                        _add(handlers, ExceptClause(exc_type, var_name, clause_body, is_star))
                        if (
                            next_clause_offset >= 0
                            and (
                                jump_next_idx := self._offset_to_index.get(
                                    next_clause_offset, _MISSING
                                )
                            )
                            is not _MISSING
                        ):
                            h_idx = jump_next_idx
                            while (
                                h_idx < len(self._instrs)
                                and self._instrs[h_idx].op_code == OpCodes.POP_TOP
                                and (h_idx + 1 < len(self._instrs))
                                and (
                                    self._instrs[h_idx + 1].op_code
                                    in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                                )
                            ):
                                h_idx += 1
                        else:
                            break
                    elif (
                        h_idx < len(self._instrs) and self._instrs[h_idx].op_code == OpCodes.POP_TOP
                    ):
                        if (
                            h_idx + 1 < len(self._instrs)
                            and self._instrs[h_idx + 1].op_code == OpCodes.LIST_APPEND
                        ):
                            break
                        h_idx += 1
                        clause_end_idx = h_idx
                        nesting = 0
                        while clause_end_idx < len(self._instrs):
                            if self._instrs[clause_end_idx].op_code == OpCodes.PUSH_EXC_INFO:
                                nesting += 1
                            elif self._instrs[clause_end_idx].op_code == OpCodes.POP_EXCEPT:
                                if nesting == 0:
                                    break
                                nesting -= 1
                            elif self._instrs[clause_end_idx].op_code == OpCodes.RERAISE:
                                if nesting == 0:
                                    break
                            clause_end_idx += 1
                        clause_stack = stack.clone()
                        clause_body = self.decompile_range(h_idx, clause_end_idx, clause_stack)
                        _add(handlers, ExceptClause(None, None, clause_body))
                        break
                    else:
                        break
            finally_body = None
            inlined_end_idx = -1
            inlined_return_idx = -1
            has_fin_inner = len(scope.except_handlers) == 0 and _any(
                self._try_scopes,
                lambda other: other.start_offset == scope.body_end_offset and other != scope,
            )
            if (
                has_fin_inner
                and (fin_start_idx := self._offset_to_index.get(scope.body_end_offset, _MISSING))
                is not _MISSING
            ):
                fin_end_idx = (
                    fe_idx
                    if (fe_idx := self._offset_to_index.get(scope.end_offset, _MISSING))
                    is not _MISSING
                    else end_idx
                )
                f_stack = stack.clone()
                finally_body = self.decompile_range(fin_start_idx, fin_end_idx, f_stack)
            elif (
                scope.finally_handler is not None
                and (fin_h_idx := self._offset_to_index.get(scope.finally_handler, _MISSING))
                is not _MISSING
            ):
                inlined_start_idx = -1
                if (
                    scope.else_end_offset is not None
                    and (e_end_idx := self._offset_to_index.get(scope.else_end_offset, _MISSING))
                    is not _MISSING
                ):
                    f_start = e_end_idx
                    if f_start < len(self._instrs) and self._instrs[f_start].op_code == OpCodes.NOP:
                        f_start += 1
                    if (
                        f_start < len(self._instrs)
                        and self._instrs[f_start].op_code == OpCodes.JUMP_FORWARD
                        and (
                            (_subject_25 := self._instrs[f_start].target_offset),
                            type(_subject_25) is int
                            and (j_target_offset := _subject_25) is not None,
                        )[-1]
                        and (
                            (j_target := self._offset_to_index.get(j_target_offset, _MISSING))
                            is not _MISSING
                        )
                    ):
                        f_start = j_target
                    fh_start = fin_h_idx
                    if (
                        fh_start < len(self._instrs)
                        and self._instrs[fh_start].op_code == OpCodes.PUSH_EXC_INFO
                    ):
                        fh_start += 1
                    fh_end = fh_start
                    nesting = 0
                    max_jump = fh_start
                    while fh_end < len(self._instrs):
                        if (
                            (
                                (_subject_26 := self._instrs[fh_end].target_offset),
                                type(_subject_26) is int
                                and (target_offset := _subject_26) is not None,
                            )[-1]
                            and (t_idx := self._offset_to_index.get(target_offset, _MISSING))
                            is not _MISSING
                            and (t_idx > fh_end)
                            and (self._instrs[fh_end].op_code != OpCodes.JUMP_BACKWARD)
                        ):
                            if t_idx > max_jump:
                                max_jump = t_idx
                        if self._instrs[fh_end].op_code == OpCodes.PUSH_EXC_INFO:
                            nesting += 1
                        elif self._instrs[fh_end].op_code == OpCodes.POP_EXCEPT:
                            if nesting > 0:
                                nesting -= 1
                        elif self._instrs[fh_end].op_code in (
                            OpCodes.RERAISE,
                            OpCodes.RETURN_VALUE,
                            OpCodes.RETURN_CONST,
                        ):
                            if nesting == 0 and fh_end >= max_jump:
                                break
                        fh_end += 1
                    fin_len = fh_end - fh_start
                    if fin_len > 0 and f_start + fin_len <= len(self._instrs):
                        match_ = True
                        m = 0
                        while m < fin_len:
                            if f_start + m >= len(self._instrs) or (
                                self._instrs[f_start + m].op_code
                                != self._instrs[fh_start + m].op_code
                                and (
                                    not (
                                        self._instrs[f_start + m].op_code == OpCodes.JUMP_FORWARD
                                        and self._instrs[fh_start + m].op_code == OpCodes.RERAISE
                                    )
                                )
                                and (
                                    not (
                                        m == fin_len - 1
                                        and self._instrs[f_start + m].op_code
                                        in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
                                        and (self._instrs[fh_start + m].op_code == OpCodes.RERAISE)
                                    )
                                )
                            ):
                                match_ = False
                                break
                            m += 1
                        if match_:
                            inlined_start_idx = f_start
                            inlined_end_idx = f_start + fin_len
                            if (
                                fin_len > 0
                                and self._instrs[fh_start + fin_len - 1].op_code == OpCodes.RERAISE
                                and (
                                    self._instrs[inlined_end_idx - 1].op_code
                                    in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
                                )
                            ):
                                inlined_return_idx = inlined_end_idx - 1
                    if inlined_start_idx < 0:
                        f_end = f_start
                        while f_end < len(self._instrs) and f_end < fin_h_idx:
                            if self._instrs[f_end].op_code in (
                                OpCodes.RETURN_VALUE,
                                OpCodes.RETURN_CONST,
                                OpCodes.JUMP_BACKWARD,
                                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                OpCodes.JUMP_FORWARD,
                            ):
                                break
                            f_end += 1
                        if f_end > f_start:
                            inlined_start_idx = f_start
                            inlined_end_idx = f_end
                elif fin_h_idx > 0 and self._instrs[fin_h_idx - 1].op_code == OpCodes.JUMP_FORWARD:
                    jump_target = (
                        _value_31
                        if (_value_31 := self._instrs[fin_h_idx - 1].target_offset) is not None
                        else 0
                    )
                    if jump_target > self._instrs[fin_h_idx].offset:
                        inlined_end_idx = fin_h_idx - 1
                        if len(scope.except_handlers) > 0:
                            min_scan_idx = (
                                b_end
                                if (
                                    b_end := self._offset_to_index.get(
                                        scope.body_end_offset, _MISSING
                                    )
                                )
                                is not _MISSING
                                else 0
                            )
                            j = inlined_end_idx - 1
                            while j >= min_scan_idx:
                                if self._instrs[j].op_code == OpCodes.RERAISE:
                                    inlined_start_idx = j + 1
                                    break
                                j -= 1
                        elif (
                            b_end := self._offset_to_index.get(scope.body_end_offset, _MISSING)
                        ) is not _MISSING:
                            inlined_start_idx = b_end
                        if (
                            inlined_start_idx >= 0
                            and inlined_start_idx < len(self._instrs)
                            and (self._instrs[inlined_start_idx].op_code == OpCodes.NOP)
                        ):
                            inlined_start_idx += 1
                if inlined_start_idx < 0 and fin_h_idx < len(self._instrs):
                    fh_start = fin_h_idx
                    if (
                        fh_start < len(self._instrs)
                        and self._instrs[fh_start].op_code == OpCodes.PUSH_EXC_INFO
                    ):
                        fh_start += 1
                    fh_end = fh_start
                    nesting = 0
                    max_jump = fh_start
                    while fh_end < len(self._instrs):
                        if (
                            (
                                (_subject_32 := self._instrs[fh_end].target_offset),
                                type(_subject_32) is int
                                and (target_offset := _subject_32) is not None,
                            )[-1]
                            and (t_idx := self._offset_to_index.get(target_offset, _MISSING))
                            is not _MISSING
                            and (t_idx > fh_end)
                            and (self._instrs[fh_end].op_code != OpCodes.JUMP_BACKWARD)
                        ):
                            if t_idx > max_jump:
                                max_jump = t_idx
                        if self._instrs[fh_end].op_code == OpCodes.PUSH_EXC_INFO:
                            nesting += 1
                        elif self._instrs[fh_end].op_code == OpCodes.POP_EXCEPT:
                            if nesting > 0:
                                nesting -= 1
                        elif self._instrs[fh_end].op_code in (
                            OpCodes.RERAISE,
                            OpCodes.RETURN_VALUE,
                            OpCodes.RETURN_CONST,
                        ):
                            if nesting == 0 and fh_end >= max_jump:
                                break
                        fh_end += 1
                    fin_len = fh_end - fh_start
                    if fin_len > 0:
                        min_scan = (
                            b_end
                            if (b_end := self._offset_to_index.get(scope.body_end_offset, _MISSING))
                            is not _MISSING
                            else 0
                        )
                        max_scan = min(fin_h_idx - 1, len(self._instrs) - fin_len)
                        scan = max_scan
                        while scan >= min_scan:
                            if scan + fin_len > fin_h_idx:
                                scan -= 1
                                continue
                            match_ = True
                            m = 0
                            while m < fin_len:
                                a = self._instrs[scan + m]
                                b = self._instrs[fh_start + m]
                                if (
                                    a.op_code != b.op_code
                                    and (
                                        not (
                                            a.op_code == OpCodes.JUMP_FORWARD
                                            and b.op_code == OpCodes.RERAISE
                                        )
                                    )
                                    and (
                                        not (
                                            m == fin_len - 1
                                            and a.op_code
                                            in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
                                            and (b.op_code == OpCodes.RERAISE)
                                        )
                                    )
                                ):
                                    match_ = False
                                    break
                                if not a.op_code in (
                                    OpCodes.JUMP_FORWARD,
                                    OpCodes.JUMP_BACKWARD,
                                    OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                                ) and (not _same_value(a.arg_val, b.arg_val)):
                                    match_ = False
                                    break
                                m += 1
                            if match_:
                                inlined_start_idx = scan
                                inlined_end_idx = scan + fin_len
                                if self._instrs[
                                    fh_start + fin_len - 1
                                ].op_code == OpCodes.RERAISE and self._instrs[
                                    inlined_end_idx - 1
                                ].op_code in (
                                    OpCodes.RETURN_VALUE,
                                    OpCodes.RETURN_CONST,
                                ):
                                    inlined_return_idx = inlined_end_idx - 1
                                break
                            scan -= 1
                if inlined_start_idx >= 0 and inlined_end_idx > inlined_start_idx:
                    f_stack = stack.clone()
                    finally_range_end = (
                        inlined_return_idx if inlined_return_idx >= 0 else inlined_end_idx
                    )
                    finally_body = self.decompile_range(
                        inlined_start_idx, finally_range_end, f_stack
                    )
                else:
                    f_idx = fin_h_idx
                    if (
                        f_idx < len(self._instrs)
                        and self._instrs[f_idx].op_code == OpCodes.PUSH_EXC_INFO
                    ):
                        f_idx += 1
                    f_end_idx = f_idx
                    nesting = 0
                    max_jump = f_idx
                    while f_end_idx < len(self._instrs):
                        if (
                            (
                                (_subject_35 := self._instrs[f_end_idx].target_offset),
                                type(_subject_35) is int
                                and (target_offset := _subject_35) is not None,
                            )[-1]
                            and (t_idx := self._offset_to_index.get(target_offset, _MISSING))
                            is not _MISSING
                            and (t_idx > f_end_idx)
                            and (self._instrs[f_end_idx].op_code != OpCodes.JUMP_BACKWARD)
                        ):
                            if t_idx > max_jump:
                                max_jump = t_idx
                        if self._instrs[f_end_idx].op_code == OpCodes.PUSH_EXC_INFO:
                            nesting += 1
                        elif self._instrs[f_end_idx].op_code == OpCodes.POP_EXCEPT:
                            if nesting > 0:
                                nesting -= 1
                        elif self._instrs[f_end_idx].op_code == OpCodes.RERAISE:
                            if nesting == 0 and f_end_idx >= max_jump:
                                break
                        f_end_idx += 1
                    f_stack = stack.clone()
                    finally_body = self.decompile_range(f_idx, f_end_idx, f_stack)
            trailing_return_after_finally = (
                finally_body is not None
                and inlined_end_idx >= 0
                and (inlined_end_idx < len(self._instrs))
                and (
                    self._instrs[inlined_end_idx].op_code
                    in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
                )
            )
            if inlined_return_idx >= 0:
                ret_ins = self._instrs[inlined_return_idx]
                if ret_ins.op_code == OpCodes.RETURN_CONST:
                    _add(try_body, ReturnStmt(ConstantExpr(ret_ins.arg_val)))
                elif len(body_stack) > 0:
                    _add(try_body, ReturnStmt(body_stack.pop()))
            elif trailing_return_after_finally and len(body_stack) > len(stack):
                ret_ins = self._instrs[inlined_end_idx]
                if ret_ins.op_code == OpCodes.RETURN_CONST:
                    _add(try_body, ReturnStmt(ConstantExpr(ret_ins.arg_val)))
                else:
                    _add(try_body, ReturnStmt(body_stack.pop()))
                inlined_return_idx = inlined_end_idx
            try_return_epilogue = False
            if (
                not scope.finally_handler is not None
                and len(scope.except_handlers) > 0
                and (scope.else_start_offset == scope.body_end_offset)
                and (len(body_stack) > len(stack))
                and (scope.else_end_offset is not None)
                and (
                    (epilogue_end_idx := self._offset_to_index.get(scope.else_end_offset, _MISSING))
                    is not _MISSING
                )
            ):
                saw_pop_except = False
                q = body_end_idx
                while q < epilogue_end_idx and q < len(self._instrs):
                    if self._instrs[q].op_code == OpCodes.POP_EXCEPT:
                        saw_pop_except = True
                    elif saw_pop_except and self._instrs[q].op_code in (
                        OpCodes.RETURN_VALUE,
                        OpCodes.RETURN_CONST,
                    ):
                        try_return_epilogue = True
                        break
                    q += 1
            if try_return_epilogue:
                _add(try_body, ReturnStmt(body_stack.pop()))
            else_body = []
            if (
                not try_return_epilogue
                and scope.else_start_offset is not None
                and (scope.else_end_offset is not None)
                and (
                    (else_start_idx := self._offset_to_index.get(scope.else_start_offset, _MISSING))
                    is not _MISSING
                )
                and (
                    (else_end_idx := self._offset_to_index.get(scope.else_end_offset, _MISSING))
                    is not _MISSING
                )
            ):
                else_stack = stack.clone()
                else_body = self.decompile_range(else_start_idx, else_end_idx, else_stack)
            if (
                scope.finally_handler is not None
                and inlined_end_idx >= 0
                and (inlined_end_idx < len(self._instrs))
                and (
                    self._instrs[inlined_end_idx].op_code
                    in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
                )
                and (len(body_stack) > len(stack))
            ):
                _add(try_body, ReturnStmt(body_stack.pop()))
            if (
                not try_return_epilogue
                and len(else_body) == 0
                and (len(handlers) > 0)
                and StructuralDecompiler.body_always_terminates(try_body)
            ):
                else_body = self.synthesize_dead_else(handlers)
            if (
                not (scope_end_idx := self._offset_to_index.get(scope.end_offset, _MISSING))
                is not _MISSING
            ):
                scope_end_idx = end_idx
            if len(handlers) == 0 and finally_body is None:
                if len(scope.except_handlers) > 0:
                    _add(handlers, ExceptClause(None, None, []))
                else:
                    return (False, try_stmt, next_idx)
            if (
                finally_body is not None
                and len(finally_body) > 0
                and (len(self._active_loop_headers) > 0)
            ):
                self.replace_inlined_break(try_body, finally_body, body_start_idx, body_end_idx)
            if (
                not try_return_epilogue
                and len(else_body) == 0
                and (len(handlers) > 0)
                and (len(try_body) > 0)
                and (
                    (_subject_39 := try_body[-1]),
                    isinstance(_subject_39, RaiseStmt)
                    or isinstance(_subject_39, ReturnStmt)
                    or isinstance(_subject_39, BreakStmt)
                    or isinstance(_subject_39, ContinueStmt),
                )[-1]
            ):
                for h in _iter_values(handlers):
                    if _empty_string(h.variable_name):
                        continue
                    hv_idx = self.index_in_var_names(h.variable_name)
                    if hv_idx < 0:
                        continue
                    pick = None
                    pick_idx = 2147483647
                    for name in _iter_values(StructuralDecompiler.simple_assigned_names(h.body)):
                        n_idx = self.index_in_var_names(name)
                        if n_idx >= 0 and n_idx < hv_idx and (n_idx < pick_idx):
                            pick = name
                            pick_idx = n_idx
                    if pick is not None:
                        else_body = [AssignStmt([NameExpr(pick)], ConstantExpr(None))]
                        break
            if finally_body is not None and len(finally_body) > 0:
                try_body = StructuralDecompiler.remove_inlined_finally_copies(
                    try_body, finally_body
                )
            try_stmt = TryStmt(try_body, handlers, else_body, finally_body)
            if (
                has_fin_inner
                and (fe_next_idx := self._offset_to_index.get(scope.end_offset, _MISSING))
                is not _MISSING
            ):
                next_idx = fe_next_idx
            elif (
                scope.finally_handler is not None
                and (fin_h_idx := self._offset_to_index.get(scope.finally_handler, _MISSING))
                is not _MISSING
            ):
                if inlined_return_idx >= 0:
                    next_idx = min(self.find_finally_handler_end_idx(fin_h_idx), end_idx)
                elif inlined_end_idx >= 0 and inlined_end_idx < fin_h_idx:
                    if self._instrs[inlined_end_idx].op_code in (
                        OpCodes.RETURN_VALUE,
                        OpCodes.RETURN_CONST,
                    ):
                        next_idx = inlined_end_idx
                    elif self._instrs[inlined_end_idx].op_code in (
                        OpCodes.JUMP_BACKWARD,
                        OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                    ):
                        next_idx = inlined_end_idx + 1
                    elif self._instrs[inlined_end_idx].op_code == OpCodes.JUMP_FORWARD:
                        next_idx = (
                            j_idx
                            if (
                                j_idx := self._offset_to_index.get(
                                    (
                                        _value_42
                                        if (
                                            _value_42 := self._instrs[inlined_end_idx].target_offset
                                        )
                                        is not None
                                        else 0
                                    ),
                                    _MISSING,
                                )
                            )
                            is not _MISSING
                            else scope_end_idx
                        )
                    else:
                        next_idx = inlined_end_idx
                elif fin_h_idx > 0 and self._instrs[fin_h_idx - 1].op_code == OpCodes.JUMP_FORWARD:
                    next_idx = (
                        j_idx
                        if (
                            j_idx := self._offset_to_index.get(
                                (
                                    _value_43
                                    if (_value_43 := self._instrs[fin_h_idx - 1].target_offset)
                                    is not None
                                    else 0
                                ),
                                _MISSING,
                            )
                        )
                        is not _MISSING
                        else scope_end_idx
                    )
                elif (
                    fin_h_idx > 0
                    and self._instrs[fin_h_idx - 1].op_code == OpCodes.RAISE_VARARGS
                    and (self._instrs[fin_h_idx - 1].offset > scope.body_end_offset)
                ):
                    next_idx = fin_h_idx - 1
                else:
                    cur_n = min(scope_end_idx, end_idx)
                    if (
                        finally_body is not None
                        and len(finally_body) > 0
                        and (
                            (b_end := self._offset_to_index.get(scope.body_end_offset, _MISSING))
                            is not _MISSING
                        )
                        and (b_end < len(self._instrs))
                    ):
                        f_start = b_end
                        while (
                            f_start < len(self._instrs)
                            and self._instrs[f_start].op_code == OpCodes.NOP
                        ):
                            f_start += 1
                        h_start = fin_h_idx + 1
                        while (
                            h_start < len(self._instrs)
                            and self._instrs[h_start].op_code == OpCodes.NOP
                        ):
                            h_start += 1
                        match_count = 0
                        while (
                            f_start + match_count < len(self._instrs)
                            and h_start + match_count < len(self._instrs)
                            and (self._instrs[h_start + match_count].op_code != OpCodes.RERAISE)
                            and (
                                self._instrs[f_start + match_count].op_code
                                == self._instrs[h_start + match_count].op_code
                            )
                        ):
                            match_count += 1
                        if (
                            match_count > 0
                            and h_start + match_count < len(self._instrs)
                            and (self._instrs[h_start + match_count].op_code == OpCodes.RERAISE)
                        ):
                            matched_end = f_start + match_count
                            is_pep709_restore = (
                                matched_end >= 1
                                and self._instrs[matched_end - 1].op_code
                                in (OpCodes.STORE_FAST, OpCodes.STORE_NAME, OpCodes.STORE_DEREF)
                                and (
                                    (
                                        _value_47
                                        if (
                                            _value_47 := (
                                                str(_subject_46)
                                                if (
                                                    _subject_46 := (
                                                        _value_45
                                                        if (
                                                            _value_45 := self._instrs[
                                                                matched_end - 1
                                                            ].arg_repr
                                                        )
                                                        is not None
                                                        else self._instrs[matched_end - 1].arg_val
                                                    )
                                                )
                                                is not None
                                                else None
                                            )
                                        )
                                        is not None
                                        else ""
                                    )
                                    in self._cleared_fast_vars
                                )
                            )
                            if not is_pep709_restore:
                                cur_n = matched_end
                    next_idx = cur_n
            elif (
                scope.post_else_resume_offset is not None
                and (
                    post_else_idx := self._offset_to_index.get(
                        scope.post_else_resume_offset, _MISSING
                    )
                )
                is not _MISSING
                and (post_else_idx > idx)
                and (post_else_idx < len(self._instrs))
            ):
                next_idx = min(post_else_idx, end_idx)
            else:
                next_idx = min(scope_end_idx, end_idx)
            return (True, try_stmt, next_idx)
        finally:
            _remove(self._active_scopes, scope)

    def skip_transparent(self, idx):
        while (
            idx < len(self._instrs)
            and self._instrs[idx].op_code in (OpCodes.NOP, OpCodes.EXTENDED_ARG, OpCodes.NOT_TAKEN)
            and (not self._instrs[idx].starts_line is not None)
        ):
            idx += 1
        return idx

    def matches_instruction_sequence(self, start_a, start_b, count):
        if start_a < 0 or start_b < 0:
            return False
        a = start_a
        b = start_b
        i = 0
        while i < count:
            a = self.skip_transparent(a)
            b = self.skip_transparent(b)
            if a >= len(self._instrs) or b >= len(self._instrs):
                return False
            ins_a = self._instrs[a]
            ins_b = self._instrs[b]
            if ins_a.op_code != ins_b.op_code:
                return False
            if ins_a.op_arg != ins_b.op_arg:
                return False
            if not _same_value(ins_a.arg_val, ins_b.arg_val):
                return False
            a += 1
            b += 1
            i += 1
        return True

    @staticmethod
    def are_nodes_equal(a, b):
        if a is b:
            return True
        if a is None or b is None:
            return False
        if type(a) != type(b):
            return False
        if isinstance((ca := a), ConstantExpr) and isinstance((cb := b), ConstantExpr):
            return _same_value(_item_field(ca, "value", 1), _item_field(cb, "value", 1))
        if isinstance((na := a), NameExpr) and isinstance((nb := b), NameExpr):
            return _item_field(na, "name", 0) == _item_field(nb, "name", 0)
        if isinstance((ua := a), UnaryExpr) and isinstance((ub := b), UnaryExpr):
            return _item_field(ua, "op", 0) == _item_field(
                ub, "op", 0
            ) and StructuralDecompiler.are_nodes_equal(ua.operand, ub.operand)
        if isinstance((ba := a), BinaryExpr) and isinstance((bb := b), BinaryExpr):
            return (
                _item_field(ba, "op", 0) == _item_field(bb, "op", 0)
                and StructuralDecompiler.are_nodes_equal(ba.left, bb.left)
                and StructuralDecompiler.are_nodes_equal(
                    _item_field(ba, "right", 1), _item_field(bb, "right", 1)
                )
            )
        if isinstance((ta := a), TupleExpr) and isinstance((tb := b), TupleExpr):
            return len(ta.elements) == len(tb.elements) and _all(
                _zip(ta.elements, tb.elements),
                lambda p: StructuralDecompiler.are_nodes_equal(
                    _item_field(p, "first", 0), _item_field(p, "second", 1)
                ),
            )
        if isinstance((ra := a), ReturnStmt) and isinstance((rb := b), ReturnStmt):
            return StructuralDecompiler.are_nodes_equal(
                _item_field(ra, "value", 1), _item_field(rb, "value", 1)
            )
        if isinstance((aa := a), AugAssignStmt) and isinstance((ab := b), AugAssignStmt):
            return (
                _item_field(aa, "op", 0) == _item_field(ab, "op", 0)
                and StructuralDecompiler.are_nodes_equal(aa.target, ab.target)
                and StructuralDecompiler.are_nodes_equal(
                    _item_field(aa, "value", 1), _item_field(ab, "value", 1)
                )
            )
        if isinstance((asg_a := a), AssignStmt) and isinstance((asg_b := b), AssignStmt):
            return (
                StructuralDecompiler.are_nodes_equal(
                    _item_field(asg_a, "value", 1), _item_field(asg_b, "value", 1)
                )
                and len(asg_a.targets) == len(asg_b.targets)
                and _all(
                    _zip(asg_a.targets, asg_b.targets),
                    lambda p: StructuralDecompiler.are_nodes_equal(
                        _item_field(p, "first", 0), _item_field(p, "second", 1)
                    ),
                )
            )
        if isinstance((ea := a), ExprStmt) and isinstance((eb := b), ExprStmt):
            return StructuralDecompiler.are_nodes_equal(ea.expression, eb.expression)
        if isinstance((ca := a), CallExpr) and isinstance((cb := b), CallExpr):
            return (
                StructuralDecompiler.are_nodes_equal(ca.function, cb.function)
                and len(ca.arguments) == len(cb.arguments)
                and _all(
                    _zip(ca.arguments, cb.arguments),
                    lambda p: StructuralDecompiler.are_nodes_equal(
                        _item_field(p, "first", 0), _item_field(p, "second", 1)
                    ),
                )
            )
        if isinstance((cmpa := a), CompareExpr) and isinstance((cmpb := b), CompareExpr):
            return (
                StructuralDecompiler.are_nodes_equal(cmpa.left, cmpb.left)
                and len(cmpa.comparators) == len(cmpb.comparators)
                and _all(
                    _zip(cmpa.comparators, cmpb.comparators),
                    lambda p: (
                        _item_field(_item_field(p, "first", 0), "op", 0)
                        == _item_field(_item_field(p, "second", 1), "op", 0)
                        and StructuralDecompiler.are_nodes_equal(
                            _item_field(_item_field(p, "first", 0), "right", 1),
                            _item_field(_item_field(p, "second", 1), "right", 1),
                        )
                    ),
                )
            )
        if isinstance(a, BreakStmt):
            return True
        if isinstance(a, ContinueStmt):
            return True
        if isinstance(a, PassStmt):
            return True
        return _same_value(a, b)

    @staticmethod
    def check_and_replace_trailing_break(block, finally_body):
        if len(block) > 0 and isinstance(block[-1], ReturnStmt):
            fin_count = len(finally_body)
            if len(block) >= fin_count + 1:
                match_ = True
                j = 0
                while j < fin_count:
                    if not StructuralDecompiler.are_nodes_equal(
                        block[len(block) - 1 - fin_count + j], finally_body[j]
                    ):
                        match_ = False
                        break
                    j += 1
                if match_:
                    _remove_range(block, len(block) - 1 - fin_count, fin_count + 1)
                    _add(block, BreakStmt())
                    return True
        return False

    def replace_inlined_break(self, stmts, finally_body, body_start_idx, body_end_idx):
        replaced = False
        i = 0
        while i < len(stmts):
            if ((_subject_1 := stmts[i]), isinstance((if_stmt := _subject_1), IfStmt))[-1]:
                then_list = list(if_stmt.then_body)
                if StructuralDecompiler.check_and_replace_trailing_break(then_list, finally_body):
                    stmts[i] = IfStmt(if_stmt.condition, then_list, if_stmt.else_body)
                    replaced = True
                else_list = list(if_stmt.else_body)
                if StructuralDecompiler.check_and_replace_trailing_break(else_list, finally_body):
                    stmts[i] = IfStmt(if_stmt.condition, if_stmt.then_body, else_list)
                    replaced = True
            i += 1
        if replaced:
            loop_exit = self._active_loop_exits[-1] if len(self._active_loop_exits) > 0 else -1
            if (
                loop_exit >= 0
                and (le_idx := self._offset_to_index.get(loop_exit, _MISSING)) is not _MISSING
            ):
                end_ret = -1
                scan = le_idx
                while scan < len(self._instrs):
                    if self._instrs[scan].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                        end_ret = scan
                        break
                    scan += 1
                if end_ret >= 0:
                    scan = body_end_idx - 1
                    while scan >= body_start_idx:
                        if self._instrs[scan].op_code in (
                            OpCodes.RETURN_VALUE,
                            OpCodes.RETURN_CONST,
                        ):
                            match_len = 0
                            while (
                                scan - match_len >= body_start_idx and end_ret - match_len >= le_idx
                            ):
                                ins_a = self._instrs[scan - match_len]
                                ins_b = self._instrs[end_ret - match_len]
                                if (
                                    ins_a.op_code != ins_b.op_code
                                    or ins_a.op_arg != ins_b.op_arg
                                    or (not _same_value(ins_a.arg_val, ins_b.arg_val))
                                ):
                                    break
                                match_len += 1
                            if match_len > 0:
                                self._last_inlined_exit_start = scan - match_len + 1
                                self._last_inlined_exit_length = match_len
                            break
                        scan -= 1
        return replaced

    def try_decompile_for_loop(self, idx, end_idx, stack):
        for_stmt = None
        next_idx = 0
        for_stmt = None
        next_idx = idx
        ins = self._instrs[idx]
        if ins.op_code != OpCodes.FOR_ITER:
            return (False, for_stmt, next_idx)
        target_offset = ins.target_offset if ins.target_offset is not None else 0
        if not (loop_end_idx := self._offset_to_index.get(target_offset, _MISSING)) is not _MISSING:
            return (False, for_stmt, next_idx)
        if len(stack) == 0:
            return (False, for_stmt, next_idx)
        iterator = stack.pop()
        body_start_idx = idx + 1
        while (
            body_start_idx < end_idx
            and self._instrs[body_start_idx].op_code == OpCodes.EXTENDED_ARG
        ):
            body_start_idx += 1
        target = NameExpr("__pycrune_ph_item")
        t_scan = body_start_idx
        if (
            (_parsed_1 := self.try_parse_target(t_scan)),
            (t_scan := _parsed_1[1]),
            (parsed_target := _parsed_1[2]),
            _parsed_1[0],
        )[-1] and t_scan <= loop_end_idx:
            target = parsed_target
            body_start_idx = t_scan
        else:
            target_stack = SymbolicStack()
            target_stack.push(NameExpr("__pycrune_ph_for_target"))
            target_stmts = []
            t_idx = body_start_idx
            while t_idx < loop_end_idx:
                t_ins = self._instrs[t_idx]
                self.execute_instruction(t_ins, target_stack, target_stmts)
                t_idx += 1
                if len(target_stack) == 0 and len(target_stmts) > 0:
                    break
            if (
                len(target_stmts) == 1
                and (
                    (_subject_2 := target_stmts[0]),
                    isinstance((asgn1 := _subject_2), AssignStmt),
                )[-1]
            ):
                target = asgn1.targets[0]
                body_start_idx = t_idx
            elif len(target_stmts) > 1:
                target = TupleExpr(
                    list(_select(_of_type(target_stmts, AssignStmt), lambda a: a.targets[0]))
                )
                body_start_idx = t_idx
            elif body_start_idx < end_idx and self._instrs[body_start_idx].op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                target_name = (
                    _value_6
                    if (
                        _value_6 := (
                            str(_subject_5)
                            if (
                                _subject_5 := (
                                    _value_4
                                    if (_value_4 := self._instrs[body_start_idx].arg_repr)
                                    is not None
                                    else self._instrs[body_start_idx].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else "__pycrune_ph_item"
                )
                target = NameExpr(target_name)
                body_start_idx += 1
        loop_body_end = loop_end_idx
        if loop_body_end > 0 and self._instrs[loop_body_end - 1].op_code in (
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
        ):
            loop_body_end -= 1
        max_break_target = -1
        nested_break_target = -1
        scan = body_start_idx
        while scan < loop_end_idx:
            scan_ins = self._instrs[scan]
            if scan_ins.op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                t = scan_ins.target_offset if scan_ins.target_offset is not None else -1
                if t > target_offset and t > max_break_target:
                    max_break_target = t
                elif (
                    scan_ins.op_code in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                    and t >= 0
                    and (t < ins.offset)
                    and self.is_active_loop_header(t)
                    and (t > nested_break_target)
                ):
                    nested_break_target = t
            scan += 1
        loop_exit = (
            nested_break_target
            if nested_break_target > 0
            else max_break_target
            if max_break_target > target_offset
            else target_offset
        )
        self._active_loop_headers.append(ins.offset)
        self._active_loop_exits.append(loop_exit)
        body = None
        try:
            loop_stack = stack.clone()
            body = self.decompile_range(body_start_idx, loop_body_end, loop_stack)
        finally:
            self._active_loop_headers.pop()
            self._active_loop_exits.pop()
        body = self.try_merge_continue_guards(body, body_start_idx, loop_body_end)
        else_body = []
        next_idx = loop_end_idx
        if next_idx < end_idx and self._instrs[next_idx].op_code == OpCodes.END_FOR:
            next_idx += 1
        if nested_break_target > 0:
            latch_idx = -1
            scan = next_idx
            while scan < len(self._instrs):
                s_ins = self._instrs[scan]
                if (
                    s_ins.op_code in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                    and s_ins.target_offset == nested_break_target
                ):
                    latch_idx = scan
                    break
                scan += 1
            if latch_idx > next_idx:
                else_stack = stack.clone()
                else_body = self.decompile_range(next_idx, latch_idx, else_stack)
                next_idx = latch_idx
        elif (
            max_break_target > target_offset
            and (else_end_idx := self._offset_to_index.get(max_break_target, _MISSING))
            is not _MISSING
            and (next_idx < else_end_idx)
        ):
            else_stack = stack.clone()
            else_body = self.decompile_range(next_idx, else_end_idx, else_stack)
            next_idx = else_end_idx
        elif (
            self._pending_break_exit_start >= body_start_idx
            and self._pending_break_exit_start < loop_end_idx
            and (self._pending_break_exit_length > 0)
        ):
            match_idx = -1
            scan = next_idx
            while scan + self._pending_break_exit_length <= len(self._instrs):
                if self.matches_instruction_sequence(
                    self._pending_break_exit_start, scan, self._pending_break_exit_length
                ):
                    match_idx = scan
                    break
                scan += 1
            if match_idx > next_idx:
                else_stack = stack.clone()
                else_body = self.decompile_range(next_idx, match_idx, else_stack)
                next_idx = match_idx
            self._pending_break_exit_start = -1
            self._pending_break_exit_length = -1
        elif self._last_inlined_exit_start >= 0 and self._last_inlined_exit_length > 0:
            match_idx = -1
            scan = next_idx
            while scan + self._last_inlined_exit_length <= len(self._instrs):
                if self.matches_instruction_sequence(
                    self._last_inlined_exit_start, scan, self._last_inlined_exit_length
                ):
                    match_idx = scan
                    break
                scan += 1
            if match_idx > next_idx:
                else_stack = stack.clone()
                else_body = self.decompile_range(next_idx, match_idx, else_stack)
                next_idx = match_idx
            self._last_inlined_exit_start = -1
            self._last_inlined_exit_length = -1
        for_stmt = ForStmt(target, iterator, body, else_body)
        return (True, for_stmt, next_idx)

    def try_merge_continue_guards(self, body, start_idx, end_idx):
        if len(body) < 3 or start_idx >= end_idx or end_idx > len(self._instrs):
            return body
        boundary_count = 0
        for e in _iter_values(self._code.exception_table):
            if not e.last_i:
                continue
            if (
                _item_field(e, "end", 1) <= self._instrs[start_idx].offset
                or _item_field(e, "end", 1) > self._instrs[end_idx - 1].offset + 2
            ):
                continue
            if (
                not (b_idx := self._offset_to_index.get(_item_field(e, "end", 1), _MISSING))
                is not _MISSING
            ):
                continue
            b_idx = self.skip_transparent(b_idx)
            if b_idx < len(self._instrs) and self._instrs[b_idx].op_code in (
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                boundary_count += 1
        if boundary_count < 2:
            return body
        chain_start = -1
        chain_end = -1
        i = 0
        while i < len(body):
            if (
                (
                    (_subject_2 := body[i]),
                    isinstance(_subject_2, IfStmt)
                    and len(_subject_2.else_body) == 0
                    and ((s := _subject_2) is not None),
                )[-1]
                and len(s.then_body) == 1
                and isinstance(s.then_body[0], ContinueStmt)
            ):
                if chain_start < 0:
                    chain_start = i
            elif chain_start >= 0:
                chain_end = i
                break
            i += 1
        if chain_start < 0:
            return body
        if chain_end < 0:
            chain_end = len(body)
        if chain_end - chain_start < 2:
            return body
        merged = None
        i = chain_start
        while i < chain_end:
            cond = self.invert_condition(body[i].condition)
            merged = cond if merged is None else self.combine_and(merged, cond)
            i += 1
        rest = _get_range(body, chain_end, len(body) - chain_end)
        if len(rest) == 0 or not isinstance(rest[-1], ContinueStmt):
            _add(rest, ContinueStmt())
        result = _get_range(body, 0, chain_start)
        _add(result, IfStmt(merged, rest, []))
        return result

    @staticmethod
    def remove_trailing_continue(body):
        if len(body) > 0 and isinstance(body[-1], ContinueStmt):
            body.pop(len(body) - 1)

    def find_legacy_handler_resume(self, handler_idx, end_idx, target_offset):
        h = handler_idx
        while h < end_idx:
            if (
                self._instrs[h].op_code
                in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT, OpCodes.JUMP_FORWARD)
                and self._instrs[h].target_offset == target_offset
            ):
                return h + 1
            if self._instrs[h].op_code == OpCodes.END_ASYNC_FOR:
                break
            h += 1
        return end_idx

    def find_legacy_loop_body_end(self, body_start, end_async_idx, header_offset):
        handler_limit = end_async_idx
        k = body_start
        while k < end_async_idx:
            op = self._instrs[k].op_code
            if not op in (OpCodes.LEGACY_SETUP_FINALLY, OpCodes.LEGACY_SETUP_WITH):
                k += 1
                continue
            if not (
                type(self._instrs[k].target_offset) is int
                and (target := self._instrs[k].target_offset) is not None
            ):
                k += 1
                continue
            if not (target_idx := self._offset_to_index.get(target, _MISSING)) is not _MISSING:
                k += 1
                continue
            if target_idx <= k or target_idx >= end_async_idx:
                k += 1
                continue
            if self._instrs[target_idx].op_code == OpCodes.END_ASYNC_FOR:
                k += 1
                continue
            if target_idx < handler_limit:
                handler_limit = target_idx
            k += 1
        k = handler_limit - 1
        while k >= body_start:
            if (
                self._instrs[k].op_code
                in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and self._instrs[k].target_offset == header_offset
            ):
                return k
            k -= 1
        return end_async_idx

    def try_decompile_legacy_async_for_loop(self, idx, end_idx, stack):
        async_for_stmt = None
        next_idx = 0
        async_for_stmt = None
        next_idx = idx
        if idx + 2 >= end_idx:
            return (False, async_for_stmt, next_idx)
        if self._instrs[idx].op_code != OpCodes.GET_AITER:
            return (False, async_for_stmt, next_idx)
        setup_idx = idx + 1
        while setup_idx < end_idx and self._instrs[setup_idx].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            setup_idx += 1
        if setup_idx >= end_idx or self._instrs[setup_idx].op_code != OpCodes.LEGACY_SETUP_FINALLY:
            return (False, async_for_stmt, next_idx)
        next_op_idx = setup_idx + 1
        while next_op_idx < end_idx and self._instrs[next_op_idx].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            next_op_idx += 1
        if next_op_idx >= end_idx or self._instrs[next_op_idx].op_code != OpCodes.GET_ANEXT:
            return (False, async_for_stmt, next_idx)
        if len(stack) == 0:
            return (False, async_for_stmt, next_idx)
        iterator = stack.peek()
        header_offset = self._instrs[setup_idx].offset
        bind_idx = next_op_idx + 1
        while bind_idx < end_idx and self._instrs[bind_idx].op_code in (
            OpCodes.LOAD_CONST,
            OpCodes.LEGACY_YIELD_FROM,
            OpCodes.LEGACY_POP_BLOCK,
            OpCodes.NOP,
            OpCodes.EXTENDED_ARG,
            OpCodes.NOT_TAKEN,
        ):
            bind_idx += 1
        if bind_idx >= end_idx or not self._instrs[bind_idx].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            return (False, async_for_stmt, next_idx)
        target_name = (
            str(_subject_5)
            if (
                _subject_5 := (
                    _value_4
                    if (_value_4 := self._instrs[bind_idx].arg_repr) is not None
                    else self._instrs[bind_idx].arg_val
                )
            )
            is not None
            else None
        )
        if _empty_string(target_name):
            raise FailFastException("async for target name missing")
        target = NameExpr(target_name)
        bind_idx += 1
        if (
            not (
                type(self._instrs[setup_idx].target_offset) is int
                and (end_async_offset := self._instrs[setup_idx].target_offset) is not None
            )
            or not (end_async_idx := self._offset_to_index.get(end_async_offset, _MISSING))
            is not _MISSING
            or end_async_idx >= end_idx
            or (self._instrs[end_async_idx].op_code != OpCodes.END_ASYNC_FOR)
        ):
            return (False, async_for_stmt, next_idx)
        loop_body_end = self.find_legacy_loop_body_end(bind_idx, end_async_idx, header_offset)
        parse_end = loop_body_end
        k = bind_idx
        while k < end_async_idx:
            if not self._instrs[k].op_code in (
                OpCodes.LEGACY_SETUP_FINALLY,
                OpCodes.LEGACY_SETUP_WITH,
            ):
                k += 1
                continue
            if not (
                type(self._instrs[k].target_offset) is int
                and (handler_offset := self._instrs[k].target_offset) is not None
            ):
                k += 1
                continue
            if (
                not (handler_idx := self._offset_to_index.get(handler_offset, _MISSING))
                is not _MISSING
            ):
                k += 1
                continue
            if handler_idx <= loop_body_end or handler_idx >= end_async_idx:
                k += 1
                continue
            if self._instrs[handler_idx].op_code == OpCodes.END_ASYNC_FOR:
                k += 1
                continue
            parse_end = end_async_idx
            break
            k += 1
        self._active_loop_headers.append(header_offset)
        body = None
        try:
            body = self.decompile_range(bind_idx, parse_end, stack.clone())
            StructuralDecompiler.remove_trailing_continue(body)
        finally:
            self._active_loop_headers.pop()
        stack.pop()
        async_for_stmt = ForStmt(target, iterator, body, [], is_async=True)
        next_idx = end_async_idx + 1
        return (True, async_for_stmt, next_idx)

    def try_decompile_async_for_loop(self, idx, end_idx, stack):
        async_for_stmt = None
        next_idx = 0
        async_for_stmt = None
        next_idx = idx
        ins = self._instrs[idx]
        if (
            ins.op_code != OpCodes.GET_AITER
            or idx + 1 >= end_idx
            or self._instrs[idx + 1].op_code != OpCodes.GET_ANEXT
        ):
            return (False, async_for_stmt, next_idx)
        if len(stack) == 0:
            return (False, async_for_stmt, next_idx)
        iterator = stack.pop()
        send_idx = idx + 1
        while send_idx < end_idx and (
            not self._instrs[send_idx].op_code in (OpCodes.SEND, OpCodes.END_SEND)
        ):
            send_idx += 1
        if send_idx >= end_idx:
            return (False, async_for_stmt, next_idx)
        bind_idx = 0
        if (
            self._instrs[send_idx].op_code == OpCodes.SEND
            and (
                (_subject_1 := self._instrs[send_idx].target_offset),
                type(_subject_1) is int and (send_target := _subject_1) is not None,
            )[-1]
            and (
                (send_target_idx := self._offset_to_index.get(send_target, _MISSING))
                is not _MISSING
            )
        ):
            bind_idx = send_target_idx
            if bind_idx < end_idx and self._instrs[bind_idx].op_code == OpCodes.END_SEND:
                bind_idx += 1
        else:
            while send_idx < end_idx and self._instrs[send_idx].op_code != OpCodes.END_SEND:
                send_idx += 1
            if send_idx >= end_idx:
                return (False, async_for_stmt, next_idx)
            bind_idx = send_idx + 1
        target = None
        target_stack = SymbolicStack()
        target_stack.push(NameExpr("__pycrune_ph_for_target"))
        target_stmts = []
        t_idx = bind_idx
        while t_idx < end_idx:
            t_ins = self._instrs[t_idx]
            self.execute_instruction(t_ins, target_stack, target_stmts)
            t_idx += 1
            if len(target_stack) == 0 and len(target_stmts) > 0:
                break
        if (
            len(target_stmts) == 1
            and ((_subject_2 := target_stmts[0]), isinstance((asgn1 := _subject_2), AssignStmt))[-1]
        ):
            target = asgn1.targets[0]
            bind_idx = t_idx
        elif len(target_stmts) > 1:
            target = TupleExpr(
                list(_select(_of_type(target_stmts, AssignStmt), lambda a: a.targets[0]))
            )
            bind_idx = t_idx
        elif bind_idx < end_idx and self._instrs[bind_idx].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target_name = (
                str(_subject_5)
                if (
                    _subject_5 := (
                        _value_4
                        if (_value_4 := self._instrs[bind_idx].arg_repr) is not None
                        else self._instrs[bind_idx].arg_val
                    )
                )
                is not None
                else None
            )
            if _empty_string(target_name):
                return (False, async_for_stmt, next_idx)
            target = NameExpr(target_name)
            bind_idx += 1
        if target is None:
            return (False, async_for_stmt, next_idx)
        send_offset = self._instrs[send_idx].offset
        end_async_idx = -1
        for entry in _iter_values(self._code.exception_table):
            if send_offset < _item_field(entry, "start", 0) or send_offset >= _item_field(
                entry, "end", 1
            ):
                continue
            if (
                not (handler_idx := self._offset_to_index.get(entry.target, _MISSING))
                is not _MISSING
            ):
                continue
            if handler_idx >= end_idx:
                continue
            if self._instrs[handler_idx].op_code != OpCodes.END_ASYNC_FOR:
                continue
            end_async_idx = handler_idx
            break
        if end_async_idx < 0:
            scan = bind_idx
            while scan < end_idx:
                if self._instrs[scan].op_code == OpCodes.END_ASYNC_FOR:
                    end_async_idx = scan
                    break
                scan += 1
        if end_async_idx < 0:
            return (False, async_for_stmt, next_idx)
        loop_body_end = end_async_idx
        scan_limit = end_async_idx
        k = bind_idx
        while k < end_async_idx:
            if self._instrs[k].op_code == OpCodes.PUSH_EXC_INFO:
                scan_limit = k
                break
            k += 1
        k = scan_limit - 1
        while k >= bind_idx:
            if (
                self._instrs[k].op_code
                in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and self._instrs[k].target_offset == self._instrs[idx + 1].offset
            ):
                loop_body_end = k
                break
            k -= 1
        loop_stack = stack.clone()
        self._active_loop_headers.append(self._instrs[idx + 1].offset)
        body = None
        try:
            body = self.decompile_range(bind_idx, loop_body_end, loop_stack)
            StructuralDecompiler.remove_trailing_continue(body)
        finally:
            self._active_loop_headers.pop()
        next_idx = end_async_idx + 1
        async_for_stmt = ForStmt(target, iterator, body, [], is_async=True)
        return (True, async_for_stmt, next_idx)

    def try_decompile_while_loop(self, idx, end_idx, stack):
        while_stmt = None
        next_idx = 0
        while_stmt = None
        next_idx = idx
        ins = self._instrs[idx]
        if not ins.op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
        ):
            return (False, while_stmt, next_idx)
        exit_target = ins.target_offset if ins.target_offset is not None else 0
        if (
            not (exit_idx := self._offset_to_index.get(exit_target, _MISSING)) is not _MISSING
            or exit_idx <= idx
        ):
            return (False, while_stmt, next_idx)
        is_rotated = False
        latch_jump_idx = -1
        body_start_offset = -1
        legacy_cond_latch_idx = -1
        scan = exit_idx - 1
        while scan > idx:
            scan_ins = self._instrs[scan]
            if scan_ins.op_code in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT):
                t = scan_ins.target_offset if scan_ins.target_offset is not None else -1
                if self.is_active_loop_header(t):
                    scan -= 1
                    continue
                if t >= self._instrs[idx + 1].offset and t < scan_ins.offset:
                    if (target_idx := self._offset_to_index.get(t, _MISSING)) is not _MISSING:
                        while (
                            target_idx < len(self._instrs)
                            and self._instrs[target_idx].op_code == OpCodes.EXTENDED_ARG
                        ):
                            target_idx += 1
                        if (
                            target_idx < len(self._instrs)
                            and self._instrs[target_idx].op_code == OpCodes.FOR_ITER
                        ):
                            scan -= 1
                            continue
                    if scan < exit_idx - 5:
                        scan -= 1
                        continue
                    valid_rotated_cond = True
                    if (bs := self._offset_to_index.get(t, _MISSING)) is not _MISSING:
                        c = idx + 1
                        while c < bs:
                            if self._instrs[c].op_code in (
                                OpCodes.POP_JUMP_IF_FALSE,
                                OpCodes.POP_JUMP_IF_TRUE,
                                OpCodes.POP_JUMP_IF_NONE,
                                OpCodes.POP_JUMP_IF_NOT_NONE,
                            ):
                                if self._instrs[c].target_offset != exit_target:
                                    valid_rotated_cond = False
                                    break
                            if self._instrs[c].op_code in (OpCodes.FOR_ITER, OpCodes.PUSH_EXC_INFO):
                                valid_rotated_cond = False
                                break
                            c += 1
                    if not valid_rotated_cond:
                        scan -= 1
                        continue
                    is_rotated = True
                    latch_jump_idx = scan
                    body_start_offset = t
                    break
                elif t <= self._instrs[idx].offset and t > (
                    self._active_loop_headers[-1] if len(self._active_loop_headers) > 0 else -1
                ):
                    latch_jump_idx = scan
                    break
            elif (
                scan_ins.op_code in (OpCodes.POP_JUMP_IF_TRUE, OpCodes.POP_JUMP_IF_FALSE)
                and type(scan_ins.target_offset) is int
                and ((cond_back_target := scan_ins.target_offset) is not None)
                and (cond_back_target >= self._instrs[idx + 1].offset)
                and (cond_back_target < scan_ins.offset)
                and (cond_back_target != exit_target)
            ):
                legacy_cond_latch_idx = scan
                break
            scan -= 1
        if latch_jump_idx == -1 and legacy_cond_latch_idx >= 0:
            top_cond_start = idx
            while top_cond_start > 0 and StructuralDecompiler.is_condition_expression_opcode(
                self._instrs[top_cond_start - 1].op_code
            ):
                top_cond_start -= 1
            cond_len = idx - top_cond_start
            bottom_cond_start = legacy_cond_latch_idx - cond_len
            if cond_len > 0 and bottom_cond_start > idx + 1:
                same_condition = True
                k = 0
                while k < cond_len:
                    if not StructuralDecompiler.same_condition_instruction(
                        self._instrs[top_cond_start + k], self._instrs[bottom_cond_start + k]
                    ):
                        same_condition = False
                        break
                    k += 1
                if same_condition:
                    latch_jump_idx = bottom_cond_start
        if latch_jump_idx == -1:
            return (False, while_stmt, next_idx)
        body_start_idx = (
            bs_idx
            if is_rotated
            and (bs_idx := self._offset_to_index.get(body_start_offset, _MISSING)) is not _MISSING
            else idx + 1
        )
        body_end_idx = latch_jump_idx
        if is_rotated and body_start_idx > idx + 1:
            valid_rotated = True
            c = idx + 1
            while c < body_start_idx:
                if self._instrs[c].op_code in (
                    OpCodes.POP_JUMP_IF_FALSE,
                    OpCodes.POP_JUMP_IF_TRUE,
                    OpCodes.POP_JUMP_IF_NONE,
                    OpCodes.POP_JUMP_IF_NOT_NONE,
                ):
                    if self._instrs[c].target_offset != exit_target:
                        valid_rotated = False
                        break
                c += 1
            if not valid_rotated:
                return (False, while_stmt, next_idx)
        cond = stack.pop()
        if ins.op_code == OpCodes.POP_JUMP_IF_NONE:
            cond = CompareExpr(cond, [("is not", ConstantExpr(None))])
        elif ins.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
            cond = CompareExpr(cond, [("is", ConstantExpr(None))])
        elif ins.op_code == OpCodes.POP_JUMP_IF_TRUE:
            cond = UnaryExpr("not", cond)
        if is_rotated and body_start_idx > idx + 1:
            c_idx = idx
            while c_idx < body_start_idx:
                branch_ins = self._instrs[c_idx]
                if branch_ins.op_code in (
                    OpCodes.POP_JUMP_IF_FALSE,
                    OpCodes.POP_JUMP_IF_TRUE,
                    OpCodes.POP_JUMP_IF_NONE,
                    OpCodes.POP_JUMP_IF_NOT_NONE,
                ):
                    next_branch_idx = c_idx + 1
                    while next_branch_idx < body_start_idx and (
                        not self._instrs[next_branch_idx].op_code
                        in (
                            OpCodes.POP_JUMP_IF_FALSE,
                            OpCodes.POP_JUMP_IF_TRUE,
                            OpCodes.POP_JUMP_IF_NONE,
                            OpCodes.POP_JUMP_IF_NOT_NONE,
                        )
                    ):
                        next_branch_idx += 1
                    if next_branch_idx < body_start_idx:
                        next_cond_stack = SymbolicStack()
                        self.decompile_range(c_idx + 1, next_branch_idx, next_cond_stack)
                        if len(next_cond_stack) > 0:
                            next_cond = next_cond_stack.pop()
                            op = "and" if branch_ins.op_code == OpCodes.POP_JUMP_IF_FALSE else "or"
                            cond = BinaryExpr(cond, op, next_cond)
                        c_idx = next_branch_idx
                    else:
                        break
                else:
                    c_idx += 1
        if is_rotated:
            top_cond_start_idx = idx
            cond_line = self._instrs[idx].starts_line
            if cond_line is not None:
                while (
                    top_cond_start_idx > 0
                    and self._instrs[top_cond_start_idx - 1].starts_line == cond_line
                ):
                    top_cond_start_idx -= 1
            top_cond_len = body_start_idx - top_cond_start_idx
            latch_cond_end = latch_jump_idx
            while (
                latch_cond_end > body_start_idx
                and self._instrs[latch_cond_end - 1].op_code == OpCodes.EXTENDED_ARG
            ):
                latch_cond_end -= 1
            if top_cond_len > 0 and latch_cond_end - top_cond_len >= body_start_idx:
                body_end_idx = latch_cond_end - top_cond_len
        max_break_target = -1
        scan = body_start_idx
        while scan < latch_jump_idx:
            scan_ins = self._instrs[scan]
            if scan_ins.op_code in (OpCodes.JUMP_FORWARD, OpCodes.JUMP_BACKWARD):
                t = scan_ins.target_offset if scan_ins.target_offset is not None else -1
                if t > exit_target and t > max_break_target:
                    max_break_target = t
            scan += 1
        loop_cond_offset = self._instrs[idx].offset
        scan_back = idx - 1
        while scan_back >= 0:
            if self._instrs[scan_back + 1].is_jump_target:
                loop_cond_offset = self._instrs[scan_back + 1].offset
                break
            scan_back -= 1
        loop_exit = max_break_target if max_break_target > exit_target else exit_target
        self._active_loop_headers.append(loop_cond_offset)
        self._active_loop_headers.append(self._instrs[idx].offset)
        if is_rotated and body_start_offset >= 0:
            self._active_loop_headers.append(body_start_offset)
        self._active_loop_exits.append(loop_exit)
        body = None
        try:
            loop_stack = stack.clone()
            body = self.decompile_range(body_start_idx, body_end_idx, loop_stack)
        finally:
            if is_rotated and body_start_offset >= 0:
                self._active_loop_headers.pop()
            self._active_loop_headers.pop()
            self._active_loop_headers.pop()
            self._active_loop_exits.pop()
        else_body = []
        next_idx = exit_idx
        if (
            max_break_target > exit_target
            and (else_end_idx := self._offset_to_index.get(max_break_target, _MISSING))
            is not _MISSING
            and (next_idx < else_end_idx)
        ):
            else_stack = stack.clone()
            else_body = self.decompile_range(next_idx, else_end_idx, else_stack)
            next_idx = else_end_idx
        elif (
            self._pending_break_exit_start >= body_start_idx
            and self._pending_break_exit_start < latch_jump_idx
            and (self._pending_break_exit_length > 0)
        ):
            match_idx = -1
            scan = next_idx
            while scan + self._pending_break_exit_length <= len(self._instrs):
                if self.matches_instruction_sequence(
                    self._pending_break_exit_start, scan, self._pending_break_exit_length
                ):
                    match_idx = scan
                    break
                scan += 1
            if match_idx > next_idx:
                else_stack = stack.clone()
                else_body = self.decompile_range(next_idx, match_idx, else_stack)
                next_idx = match_idx
            self._pending_break_exit_start = -1
            self._pending_break_exit_length = -1
        elif self._last_inlined_exit_start >= 0 and self._last_inlined_exit_length > 0:
            match_idx = -1
            scan = next_idx
            while scan + self._last_inlined_exit_length <= len(self._instrs):
                if self.matches_instruction_sequence(
                    self._last_inlined_exit_start, scan, self._last_inlined_exit_length
                ):
                    match_idx = scan
                    break
                scan += 1
            if match_idx > next_idx:
                else_stack = stack.clone()
                else_body = self.decompile_range(next_idx, match_idx, else_stack)
                next_idx = match_idx
            self._last_inlined_exit_start = -1
            self._last_inlined_exit_length = -1
        if (
            len(body) > 0
            and (
                (_subject_4 := body[-1]),
                isinstance(_subject_4, IfStmt)
                and len(_subject_4.else_body) == 0
                and ((latch_if := _subject_4) is not None),
            )[-1]
            and (len(latch_if.then_body) == 1)
            and isinstance(latch_if.then_body[0], ContinueStmt)
            and StructuralDecompiler.are_nodes_equal(latch_if.condition, cond)
        ):
            body.pop(len(body) - 1)
        while_stmt = WhileStmt(cond, body, else_body)
        return (True, while_stmt, next_idx)

    def try_decompile_while_true_loop(self, idx, end_idx, stack):
        while_stmt = None
        next_idx = 0
        while_stmt = None
        next_idx = idx
        ins = self._instrs[idx]
        if not ins.is_jump_target:
            return (False, while_stmt, next_idx)
        if self.is_active_loop_header(ins.offset):
            return (False, while_stmt, next_idx)
        header_idx = idx
        while (
            header_idx < len(self._instrs)
            and self._instrs[header_idx].op_code == OpCodes.EXTENDED_ARG
        ):
            header_idx += 1
        header_op = (
            self._instrs[header_idx].op_code
            if header_idx < len(self._instrs)
            else OpCodes.EXTENDED_ARG
        )
        if header_op in (
            OpCodes.FOR_ITER,
            OpCodes.GET_AITER,
            OpCodes.GET_ANEXT,
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
        ):
            return (False, while_stmt, next_idx)
        latch_idx = -1
        scan = idx + 1
        while scan < end_idx:
            s_ins = self._instrs[scan]
            if (
                s_ins.op_code in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and s_ins.target_offset == ins.offset
            ):
                if scan > 0 and self._instrs[scan - 1].op_code == OpCodes.POP_EXCEPT:
                    scan += 1
                    continue
                is_exc_handler = False
                b = scan
                while b >= idx:
                    if self._instrs[b].op_code == OpCodes.PUSH_EXC_INFO:
                        is_exc_handler = True
                        break
                    b -= 1
                if is_exc_handler:
                    scan += 1
                    continue
                latch_idx = scan
                break
            scan += 1
        if latch_idx == -1:
            return (False, while_stmt, next_idx)
        c = idx
        while c < latch_idx:
            c_ins = self._instrs[c]
            if c_ins.op_code in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE):
                target = c_ins.target_offset if c_ins.target_offset is not None else -1
                if target >= self._instrs[latch_idx].offset:
                    return (False, while_stmt, next_idx)
                break
            if c_ins.op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
                OpCodes.STORE_ATTR,
                OpCodes.STORE_SUBSCR,
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.RAISE_VARARGS,
            ):
                prev = self._instrs[c - 1] if c > idx else None
                is_walrus = (
                    prev.op_code == OpCodes.COPY
                    and (prev.op_arg if prev.op_arg is not None else 0) == 1
                    or prev.op_code in (OpCodes.LEGACY_DUP_TOP, OpCodes.LEGACY_DUP_TOP_TWO)
                )
                if is_walrus:
                    c += 1
                    continue
                break
            c += 1
        max_break_target = -1
        scan = idx
        while scan < latch_idx:
            s_ins = self._instrs[scan]
            if s_ins.op_code in (OpCodes.JUMP_FORWARD, OpCodes.JUMP_BACKWARD):
                t = s_ins.target_offset if s_ins.target_offset is not None else -1
                if t > self._instrs[latch_idx].offset and t > max_break_target:
                    max_break_target = t
            scan += 1
        self._active_loop_headers.append(ins.offset)
        if max_break_target > 0:
            self._active_loop_exits.append(max_break_target)
        body = None
        try:
            loop_stack = stack.clone()
            body = self.decompile_range(idx, latch_idx, loop_stack)
        finally:
            self._active_loop_headers.pop()
            if max_break_target > 0:
                self._active_loop_exits.pop()
        if (
            max_break_target > 0
            and (break_idx := self._offset_to_index.get(max_break_target, _MISSING)) is not _MISSING
        ):
            next_idx = break_idx
        else:
            next_idx = latch_idx + 1
        while_stmt = WhileStmt(ConstantExpr(True), body, [])
        return (True, while_stmt, next_idx)

    def try_extract_match_guard(self, start_idx, limit):
        guard = None
        body_idx = 0

        def valid_guard_target(t):
            return (
                t == fail_off
                or t == body_off
                or t in jump_start_to_index
                or (t == 2147483647)
                or (t >= guard_region_start and t < body_off)
            )

        guard = None
        body_idx = start_idx
        jumps = []
        jump_start_to_index = {}
        cond_stack = SymbolicStack()
        i = start_idx
        while i < limit:
            ins = self._instrs[i]
            if ins.op_code in (
                OpCodes.POP_JUMP_IF_FALSE,
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_NONE,
                OpCodes.POP_JUMP_IF_NOT_NONE,
            ):
                if not (
                    (_parsed_1 := self.try_get_branch_info(ins, cond_stack)),
                    (cond := _parsed_1[1]),
                    (jit := _parsed_1[2]),
                    (target := _parsed_1[3]),
                    _parsed_1[0],
                )[-1]:
                    break
                fall = self._instrs[i + 1].offset if i + 1 < limit else 2147483647
                _add(
                    jumps,
                    ConditionJump(
                        instr_index=i,
                        condition=cond,
                        true_offset=target if jit else fall,
                        false_offset=fall if jit else target,
                    ),
                )
                jump_start_to_index[ins.offset] = len(jumps) - 1
                i += 1
                continue
            if ins.op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_DEREF,
                OpCodes.STORE_GLOBAL,
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.POP_TOP,
                OpCodes.RAISE_VARARGS,
                OpCodes.RERAISE,
                OpCodes.NOP,
            ):
                break
            try:
                self.execute_instruction(ins, cond_stack, None)
            except Exception as ex:
                if not not isinstance(ex, FailFastException):
                    raise
                break
            i += 1
        if len(jumps) == 0:
            return (False, guard, body_idx)
        k = 0
        while k + 1 < len(jumps):
            fall_idx = jumps[k].instr_index + 1
            if fall_idx < len(self._instrs):
                fall_off = self._instrs[fall_idx].offset
                if not fall_off in jump_start_to_index:
                    jump_start_to_index[fall_off] = k + 1
            k += 1
        body_i = jumps[-1].instr_index + 1
        body_off = self._instrs[body_i].offset if body_i < len(self._instrs) else 2147483647
        fail_off = -1
        for j in _iter_values(jumps):
            for t in _iter_values([j.true_offset, j.false_offset]):
                if (
                    t != body_off
                    and (not t in jump_start_to_index)
                    and (t != 2147483647)
                    and (t > fail_off)
                ):
                    fail_off = t
        if fail_off < 0:
            return (False, guard, body_idx)
        guard_region_start = (
            self._instrs[start_idx].offset if start_idx < len(self._instrs) else 2147483647
        )
        for j in _iter_values(jumps):
            if not valid_guard_target(j.true_offset):
                return (False, guard, body_idx)
            if not valid_guard_target(j.false_offset):
                return (False, guard, body_idx)
        guard = self.reconstruct_condition(body_off, fail_off, jumps, jump_start_to_index)
        body_idx = body_i
        return (True, guard, body_idx)

    @staticmethod
    def require_pattern_name(ins):
        return (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else _raise(
                PatParseException(
                    "missing name for {} at offset {}".format(ins.op_name, ins.offset)
                )
            )
        )

    @staticmethod
    def pat_clone_slot(s):
        return PatSlot(
            pat=_item_field(s, "pat", 1),
            name=_item_field(s, "name", 0),
            is_capture=s.is_capture,
            is_star=s.is_star,
            is_pseudo=s.is_pseudo,
            is_keys=s.is_keys,
        )

    @staticmethod
    def pat_pop(c):
        if len(c.stack) == 0:
            raise PatParseException()
        s = c.stack[-1]
        c.stack.pop(len(c.stack) - 1)
        return s

    @staticmethod
    def pat_swap(c, n):
        if n < 2 or n > len(c.stack):
            raise PatParseException("swap {} on depth {}".format(n, len(c.stack)))
        c.stack[-1], c.stack[-n] = (c.stack[-n], c.stack[-1])

    @staticmethod
    def is_pat_rotate(op):
        return op in (
            OpCodes.LEGACY_ROT_TWO,
            OpCodes.LEGACY_ROT_THREE,
            OpCodes.LEGACY_ROT_FOUR,
            OpCodes.LEGACY_ROT_N,
        )

    @staticmethod
    def pat_rotate_count(ins):
        if ins.op_code == OpCodes.LEGACY_ROT_TWO:
            return 2
        if ins.op_code == OpCodes.LEGACY_ROT_THREE:
            return 3
        if ins.op_code == OpCodes.LEGACY_ROT_FOUR:
            return 4
        if ins.op_code == OpCodes.LEGACY_ROT_N:
            return max(2, ins.op_arg if ins.op_arg is not None else 2)
        return 2

    @staticmethod
    def pat_rotate(stack, n):
        if n < 2 or n > len(stack):
            raise PatParseException("rotate {} on depth {}".format(n, len(stack)))
        top = stack[-1]
        stack.pop(len(stack) - 1)
        stack.insert(len(stack) - (n - 1), top)

    @staticmethod
    def pat_copy(c, n):
        if n < 1 or n > len(c.stack):
            raise PatParseException()
        _add(c.stack, StructuralDecompiler.pat_clone_slot(c.stack[-n]))

    @staticmethod
    def is_pat_store_op(op):
        return op in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_DEREF,
            OpCodes.STORE_GLOBAL,
        )

    @staticmethod
    def pat_store_name(ins):
        return (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else "__pycrune_ph_x"
        )

    @staticmethod
    def pat_cmp_index(ins):
        repr_ = (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if repr_.startswith("bool(") and repr_.endswith(")"):
            repr_ = repr_[5:-1]
        if repr_ == "<":
            return 0
        if repr_ == "<=":
            return 1
        if repr_ == "==":
            return 2
        if repr_ == "!=":
            return 3
        if repr_ == ">":
            return 4
        if repr_ == ">=":
            return 5
        return (ins.op_arg if ins.op_arg is not None else 0) >> 4

    @staticmethod
    def pat_tuple_items(v):
        if v is None:
            return
        if type(v) is str:
            yield v
            return
        if isinstance((en := v), Iterable):
            for o in _iter_values(en):
                yield o

    def pat_note_fail(self, c, ins):
        if type(ins.target_offset) is int and (t := ins.target_offset) is not None:
            _add(c.fail_targets, t)

    def pat_next(self, i):
        while i < len(self._instrs) and self._instrs[i].op_code in (
            OpCodes.EXTENDED_ARG,
            OpCodes.NOT_TAKEN,
        ):
            i += 1
        return i

    def pat_consume_swaps(self, c, limit):
        c.i = self.pat_next(c.i)
        while c.i < limit and self._instrs[c.i].op_code == OpCodes.SWAP:
            StructuralDecompiler.pat_swap(
                c, _value_1 if (_value_1 := self._instrs[c.i].op_arg) is not None else 0
            )
            c.i = self.pat_next(c.i + 1)

    def resolve_pat2(self, p):
        if isinstance((v := p), Pat2Val):
            return MatchValuePattern(_item_field(v, "value", 1))
        if isinstance(p, Pat2Wild):
            return MatchWildcardPattern()
        if isinstance((cap := p), Pat2Cap):
            return self.resolve_pat2_cap(cap.slot)
        if isinstance((st := p), Pat2Star):
            return MatchStarPattern(
                _value_1 if (_value_1 := _item_field(st.slot, "name", 0)) is not None else "_"
            )
        if isinstance((sq := p), Pat2Seq):
            return MatchSequencePattern(list(_select(sq.items, self.resolve_pat2)), sq.is_list)
        if isinstance((mp := p), Pat2Map):
            return MatchMappingPattern(
                list(
                    _select(
                        mp.entries,
                        lambda e: (
                            _item_field(e, "key", 0),
                            self.resolve_pat2(_item_field(e, "pat", 1)),
                        ),
                    )
                ),
                None if mp.rest is None else self.resolve_pat2(mp.rest),
            )
        if isinstance((cl := p), Pat2Class):
            return MatchClassPattern(
                cl.cls, list(_select(_item_field(cl, "args", 1), self.resolve_pat2))
            )
        if isinstance((kw := p), Pat2Kw):
            return MatchKeywordPattern(
                _item_field(kw, "attr", 0), self.resolve_pat2(_item_field(kw, "pat", 1))
            )
        if isinstance((alternatives := p), Pat2Or):
            return MatchOrPattern(
                list(
                    _select(
                        alternatives.alts, lambda a: self.resolve_pat2(_item_field(a, "pat", 1))
                    )
                )
            )
        return MatchWildcardPattern()

    def resolve_pat2_cap(self, s):
        if _item_field(s, "name", 0) is None:
            return (
                MatchWildcardPattern()
                if _item_field(s, "pat", 1) is None
                else self.resolve_pat2(_item_field(s, "pat", 1))
            )
        return MatchAsPattern(
            (
                None
                if _item_field(s, "pat", 1) is None
                else self.resolve_pat2(_item_field(s, "pat", 1))
            ),
            _item_field(s, "name", 0),
        )

    def propagate_or_names(self, p):
        if isinstance((alternatives := p), Pat2Or):
            first = alternatives.alts[0].caps
            ai = 1
            while ai < len(alternatives.alts):
                caps = alternatives.alts[ai].caps
                n = min(len(first), len(caps))
                i = 0
                while i < n:
                    if (
                        _item_field(caps[i], "name", 0) is None
                        and _item_field(first[i], "name", 0) is not None
                    ):
                        caps[i].name = _item_field(first[i], "name", 0)
                        caps[i].is_capture = True
                    i += 1
                ai += 1
            for a in _iter_values(alternatives.alts):
                self.propagate_or_names(_item_field(a, "pat", 1))
        elif isinstance((sq := p), Pat2Seq):
            for x in _iter_values(sq.items):
                self.propagate_or_names(x)
        elif isinstance((mp := p), Pat2Map):
            for e in _iter_values(mp.entries):
                self.propagate_or_names(_item_field(e, "pat", 1))
            if mp.rest is not None:
                self.propagate_or_names(mp.rest)
        elif isinstance((cl := p), Pat2Class):
            for x in _iter_values(_item_field(cl, "args", 1)):
                self.propagate_or_names(x)
        elif isinstance((kw := p), Pat2Kw):
            self.propagate_or_names(_item_field(kw, "pat", 1))
        elif isinstance((cap := p), Pat2Cap):
            if _item_field(cap.slot, "pat", 1) is not None:
                self.propagate_or_names(_item_field(cap.slot, "pat", 1))

    @staticmethod
    def pat_caps_above(c, depth):
        caps = []
        i = len(c.stack) - 1
        while i >= depth:
            _add(caps, c.stack[i])
            i -= 1
        return caps

    @staticmethod
    def pat_index_of_max(c):
        best = -1
        best_val = -2147483648
        for t in _iter_values(c.fail_targets):
            if t > best_val:
                best_val = t
                best = t
        return best

    @staticmethod
    def pat_tail_matches(a, a_start, a_end, b, b_start, b_end):
        if a_end - a_start != b_end - b_start:
            return False
        k = 0
        while k < a_end - a_start:
            x = a[a_start + k]
            y = b[b_start + k]
            if x.op_code != y.op_code:
                return False
            if x.op_arg != y.op_arg and (
                not x.op_code
                in (
                    OpCodes.JUMP_FORWARD,
                    OpCodes.JUMP_BACKWARD,
                    OpCodes.POP_JUMP_IF_FALSE,
                    OpCodes.POP_JUMP_IF_TRUE,
                    OpCodes.POP_JUMP_IF_NONE,
                    OpCodes.POP_JUMP_IF_NOT_NONE,
                )
            ):
                return False
            if (
                type(x.arg_val) is str
                and (sx := x.arg_val) is not None
                and (type(y.arg_val) is str)
                and ((sy := y.arg_val) is not None)
                and (sx != sy)
            ):
                return False
            k += 1
        return True

    @staticmethod
    def is_pat_copy(ins):
        return ins.op_code == OpCodes.LEGACY_DUP_TOP or (
            ins.op_code == OpCodes.COPY and ins.op_arg == 1
        )

    def parse_pat(self, c, slot, limit):
        c.i = self.pat_next(c.i)
        if (
            c.i < limit
            and StructuralDecompiler.is_pat_rotate(self._instrs[c.i].op_code)
            and (len(c.stack) > 0)
            and (not c.stack[-1] is slot)
        ):
            trial = list(c.stack)
            p = c.i
            try:
                while p < limit and StructuralDecompiler.is_pat_rotate(self._instrs[p].op_code):
                    StructuralDecompiler.pat_rotate(
                        trial, StructuralDecompiler.pat_rotate_count(self._instrs[p])
                    )
                    p = self.pat_next(p + 1)
            except PatParseException:
                trial = None
            if trial is not None and len(trial) > 0 and (trial[-1] is slot):
                c.stack.clear()
                c.stack.extend(trial)
                c.i = p
        while c.i < limit and self._instrs[c.i].op_code == OpCodes.SWAP:
            expect = _value_1 if (_value_1 := self._instrs[c.i].op_arg) is not None else 0
            while (
                c.i < limit
                and self._instrs[c.i].op_code == OpCodes.SWAP
                and (
                    (_value_2 if (_value_2 := self._instrs[c.i].op_arg) is not None else 0)
                    == expect
                )
                and (expect >= 2)
            ):
                StructuralDecompiler.pat_swap(c, expect)
                expect -= 1
                c.i = self.pat_next(c.i + 1)
            if len(c.stack) == 0 or not c.stack[-1] is slot:
                slot.is_capture = True
                return Pat2Cap(slot)
        if len(c.stack) == 0 or not c.stack[-1] is slot:
            slot.is_capture = True
            return Pat2Cap(slot)
        c.i = self.pat_next(c.i)
        if c.i >= limit:
            raise PatParseException()
        ins = self._instrs[c.i]
        if self.is_pat_copy(ins):
            return self.parse_copy_pat(c, slot, limit)
        if ins.op_code == OpCodes.POP_TOP:
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
            return Pat2Wild()
        if StructuralDecompiler.is_pat_store_op(ins.op_code):
            name = StructuralDecompiler.pat_store_name(ins)
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
            slot.is_capture = True
            slot.name = name
            return Pat2Cap(slot)
        if ins.op_code == OpCodes.MATCH_SEQUENCE:
            return self.parse_seq_pat(c, slot, limit)
        if ins.op_code == OpCodes.MATCH_MAPPING:
            return self.parse_map_pat(c, slot, limit)
        if ins.op_code in (OpCodes.POP_JUMP_IF_NONE, OpCodes.POP_JUMP_IF_NOT_NONE):
            StructuralDecompiler.pat_pop(c)
            self.pat_note_fail(c, ins)
            c.i = self.pat_next(c.i + 1)
            return Pat2Val(ConstantExpr(None))
        if ins.op_code in (
            OpCodes.LOAD_CONST,
            OpCodes.LOAD_GLOBAL,
            OpCodes.LOAD_NAME,
            OpCodes.LOAD_FAST,
            OpCodes.LOAD_DEREF,
        ):
            return self.parse_value_or_class_pat(c, slot, limit)
        slot.is_capture = True
        return Pat2Cap(slot)

    def parse_copy_pat(self, c, slot, limit):
        base_stack = list(c.stack)
        fail_start = len(c.fail_targets)
        copy = StructuralDecompiler.pat_clone_slot(slot)
        _add(c.stack, copy)
        c.i = self.pat_next(c.i + 1)
        sub = self.parse_pat(c, copy, limit)
        c.i = self.pat_next(c.i)
        while c.i < limit and self._instrs[c.i].op_code == OpCodes.SWAP:
            StructuralDecompiler.pat_swap(
                c, _value_1 if (_value_1 := self._instrs[c.i].op_arg) is not None else 0
            )
            c.i = self.pat_next(c.i + 1)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.JUMP_FORWARD:
            return self.parse_or_pat(c, slot, base_stack, sub, limit, fail_start, True)
        alt_fails = []
        i = fail_start
        while i < len(c.fail_targets):
            _add(alt_fails, c.fail_targets[i])
            i += 1
        if (
            len(alt_fails) > 0
            and (
                (
                    _parsed_2 := self.try_parse_duplicated_tail_or(
                        c, slot, base_stack, sub, limit, alt_fails
                    )
                ),
                (or_pat := _parsed_2[1]),
                _parsed_2[0],
            )[-1]
        ):
            return or_pat
        slot.is_capture = True
        slot.pat = sub
        return Pat2Cap(slot)

    def try_parse_duplicated_tail_or(self, c, slot, base_stack, first_alt, limit, first_fails):
        or_pat = None
        or_pat = None
        first_fail_offset = max(_iter_values(first_fails))
        if (
            not (first_fail_idx := self._offset_to_index.get(first_fail_offset, _MISSING))
            is not _MISSING
        ):
            return (False, or_pat)
        tail_start = c.i
        if first_fail_idx <= tail_start:
            return (False, or_pat)
        alts = [Pat2Alt(first_alt, StructuralDecompiler.pat_caps_above(c, len(base_stack)))]
        boundaries = []
        probe = first_fail_idx
        while True:
            p = probe
            while p < limit and self._instrs[p].op_code == OpCodes.POP_TOP:
                p += 1
            if p >= limit or not self.is_pat_copy(self._instrs[p]):
                break
            tmp = PatCtx(i=p)
            for s in _iter_values(base_stack):
                _add(tmp.stack, s)
            copy = StructuralDecompiler.pat_clone_slot(slot)
            _add(tmp.stack, copy)
            tmp.i = self.pat_next(tmp.i + 1)
            alt = None
            try:
                alt = self.parse_pat(tmp, copy, limit)
                while tmp.i < limit and self._instrs[tmp.i].op_code == OpCodes.SWAP:
                    StructuralDecompiler.pat_swap(
                        tmp, _value_2 if (_value_2 := self._instrs[tmp.i].op_arg) is not None else 0
                    )
                    tmp.i = self.pat_next(tmp.i + 1)
            except PatParseException:
                return (False, or_pat)
            fail_offset = StructuralDecompiler.pat_index_of_max(tmp)
            if (
                fail_offset < 0
                or not (fail_idx := self._offset_to_index.get(fail_offset, _MISSING))
                is not _MISSING
                or fail_idx <= tmp.i
            ):
                break
            if not StructuralDecompiler.pat_tail_matches(
                self._instrs, tail_start, first_fail_idx, self._instrs, tmp.i, fail_idx
            ):
                break
            _add(alts, Pat2Alt(alt, StructuralDecompiler.pat_caps_above(tmp, len(base_stack))))
            _add(boundaries, fail_idx)
            probe = fail_idx
        if len(alts) < 2:
            return (False, or_pat)
        cap_count = len(alts[0].caps)
        if _any(alts, lambda a: len(a.caps) != cap_count):
            return (False, or_pat)
        for b in _iter_values(boundaries):
            _add(c.fail_targets, self._instrs[b].offset)
        or_pat = Pat2Or(alts)
        return (True, or_pat)

    def parse_or_pat(self, c, slot, base_stack, first_alt, limit, fail_start, shared_end):
        first_fail_offset = (
            max(_iter_values(_skip(c.fail_targets, fail_start)))
            if len(c.fail_targets) > fail_start
            else -1
        )
        if (
            first_fail_offset < 0
            or not (probe := self._offset_to_index.get(first_fail_offset, _MISSING)) is not _MISSING
        ):
            raise PatParseException()
        end_offset = _value_1 if (_value_1 := self._instrs[c.i].target_offset) is not None else -1
        if not (end_idx := self._offset_to_index.get(end_offset, _MISSING)) is not _MISSING:
            raise PatParseException()
        alts = [Pat2Alt(first_alt, StructuralDecompiler.pat_caps_above(c, len(base_stack)))]
        boundaries = []
        loop = 0
        while (loop := (loop + 1)) - 1 < 256:
            p = probe
            while p < limit and self._instrs[p].op_code == OpCodes.POP_TOP:
                p += 1
            if p >= limit:
                break
            if self._instrs[p].op_code == OpCodes.JUMP_FORWARD:
                if (
                    (_subject_2 := self._instrs[p].target_offset),
                    type(_subject_2) is int and (nm := _subject_2) is not None,
                )[-1]:
                    _add(boundaries, nm)
                break
            if not self.is_pat_copy(self._instrs[p]):
                break
            tmp = PatCtx(i=p)
            for s in _iter_values(base_stack):
                _add(tmp.stack, s)
            copy = StructuralDecompiler.pat_clone_slot(slot)
            _add(tmp.stack, copy)
            tmp.i = self.pat_next(tmp.i + 1)
            alt = None
            try:
                alt = self.parse_pat(tmp, copy, limit)
                while tmp.i < limit and self._instrs[tmp.i].op_code == OpCodes.SWAP:
                    StructuralDecompiler.pat_swap(
                        tmp, _value_4 if (_value_4 := self._instrs[tmp.i].op_arg) is not None else 0
                    )
                    tmp.i = self.pat_next(tmp.i + 1)
            except PatParseException:
                raise PatParseException()
            _add(alts, Pat2Alt(alt, StructuralDecompiler.pat_caps_above(tmp, len(base_stack))))
            alt_fail_offset = StructuralDecompiler.pat_index_of_max(tmp)
            if tmp.i < limit and self._instrs[tmp.i].op_code == OpCodes.JUMP_FORWARD:
                if (
                    _value_5 if (_value_5 := self._instrs[tmp.i].target_offset) is not None else -1
                ) != end_offset:
                    raise PatParseException()
                if (
                    alt_fail_offset < 0
                    or not (probe := self._offset_to_index.get(alt_fail_offset, _MISSING))
                    is not _MISSING
                ):
                    raise PatParseException()
            else:
                if alt_fail_offset < 0:
                    break
                if (
                    not (probe := self._offset_to_index.get(alt_fail_offset, _MISSING))
                    is not _MISSING
                ):
                    raise PatParseException()
        cap_count = len(alts[0].caps)
        if _any(alts, lambda a: len(a.caps) != cap_count):
            raise PatParseException()
        c.i = end_idx
        for b in _iter_values(boundaries):
            _add(c.fail_targets, b)
        return Pat2Or(alts)

    def parse_seq_pat(self, c, slot, limit):
        c.i = self.pat_next(c.i)
        self.pat_note_fail(c, self._instrs[c.i])
        c.i = self.pat_next(c.i + 1)
        flag = PatSlot(is_pseudo=True)
        _add(c.stack, flag)
        if c.i < limit and self._instrs[c.i].op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
        ):
            self.pat_note_fail(c, self._instrs[c.i])
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
        size = -1
        has_star = False
        if (
            c.i + 3 < limit
            and self._instrs[c.i].op_code == OpCodes.GET_LEN
            and (self._instrs[c.i + 1].op_code == OpCodes.LOAD_CONST)
            and (self._instrs[c.i + 2].op_code == OpCodes.COMPARE_OP)
            and (
                self._instrs[c.i + 3].op_code
                in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
            )
        ):
            n = _cast_int(
                _value_3 if (_value_3 := self._instrs[c.i + 1].arg_val) is not None else 0
            )
            cmp = StructuralDecompiler.pat_cmp_index(self._instrs[c.i + 2])
            if cmp == 2:
                size = n
                has_star = False
            elif cmp == 5:
                size = n + 1
                has_star = True
            else:
                raise PatParseException()
            self.pat_note_fail(c, self._instrs[c.i + 3])
            c.i = self.pat_next(c.i + 4)
        if len(c.stack) == 0 or not c.stack[-1] is slot:
            raise PatParseException()
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
            c.i = self.pat_next(c.i + 1)
            StructuralDecompiler.pat_pop(c)
            n = size if size >= 0 else 1
            items = []
            i = 0
            while i < n:
                if has_star and i == n - 1:
                    st = PatSlot(is_capture=True, is_star=True)
                    _add(items, Pat2Star(st))
                else:
                    _add(items, Pat2Wild())
                i += 1
            return Pat2Seq(items, True)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.UNPACK_SEQUENCE:
            n = _value_4 if (_value_4 := self._instrs[c.i].op_arg) is not None else 0
            c.i = self.pat_next(c.i + 1)
            StructuralDecompiler.pat_pop(c)
            elems = []
            i = 0
            while i < n:
                _add(elems, PatSlot())
                i += 1
            i = n - 1
            while i >= 0:
                _add(c.stack, elems[i])
                i -= 1
            items = []
            for e in _iter_values(elems):
                _add(items, self.parse_pat(c, e, limit))
            return Pat2Seq(items, True)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.UNPACK_EX:
            arg = _value_5 if (_value_5 := self._instrs[c.i].op_arg) is not None else 0
            c.i = self.pat_next(c.i + 1)
            lead = arg & 255
            trail = arg >> 8
            n = lead + 1 + trail
            StructuralDecompiler.pat_pop(c)
            elems = []
            i = 0
            while i < n:
                _add(elems, PatSlot())
                i += 1
            i = n - 1
            while i >= 0:
                _add(c.stack, elems[i])
                i -= 1
            items = []
            i = 0
            while i < n:
                if i == lead:
                    elems[i].is_capture = True
                    elems[i].is_star = True
                    _add(items, Pat2Star(elems[i]))
                else:
                    _add(items, self.parse_pat(c, elems[i], limit))
                i += 1
            return Pat2Seq(items, True)
        if (
            c.i < limit
            and self._instrs[c.i].op_code == OpCodes.COPY
            and ((_value_6 if (_value_6 := self._instrs[c.i].op_arg) is not None else 0) == 1)
        ):
            if size < 0:
                raise PatParseException()
            pats = [None] * size
            guard = 0
            while (
                c.i < limit
                and self._instrs[c.i].op_code == OpCodes.COPY
                and ((_value_7 if (_value_7 := self._instrs[c.i].op_arg) is not None else 0) == 1)
                and ((guard := (guard + 1)) - 1 < 64)
            ):
                c.i = self.pat_next(c.i + 1)
                StructuralDecompiler.pat_copy(c, 1)
                idx_pseudo = PatSlot(is_pseudo=True)
                pos = 0
                if c.i < limit and self._instrs[c.i].op_code == OpCodes.LOAD_CONST:
                    pos = _cast_int(
                        _value_8 if (_value_8 := self._instrs[c.i].arg_val) is not None else 0
                    )
                    c.i = self.pat_next(c.i + 1)
                    _add(c.stack, idx_pseudo)
                elif (
                    c.i + 2 < limit
                    and self._instrs[c.i].op_code == OpCodes.GET_LEN
                    and (self._instrs[c.i + 1].op_code == OpCodes.LOAD_CONST)
                    and (self._instrs[c.i + 2].op_code == OpCodes.BINARY_OP)
                ):
                    k = _cast_int(
                        _value_9 if (_value_9 := self._instrs[c.i + 1].arg_val) is not None else 0
                    )
                    pos = size - k
                    c.i = self.pat_next(c.i + 3)
                    _add(c.stack, idx_pseudo)
                else:
                    raise PatParseException()
                if c.i >= limit or self._instrs[c.i].op_code != OpCodes.BINARY_SUBSCR:
                    raise PatParseException()
                c.i = self.pat_next(c.i + 1)
                StructuralDecompiler.pat_pop(c)
                StructuralDecompiler.pat_pop(c)
                if pos < 0 or pos >= size:
                    raise PatParseException()
                elem = PatSlot()
                _add(c.stack, elem)
                pats[pos] = self.parse_pat(c, elem, limit)
            self.pat_consume_swaps(c, limit)
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
                c.i = self.pat_next(c.i + 1)
                StructuralDecompiler.pat_pop(c)
            star_pos = -1
            i = 0
            while i < size:
                if pats[i] is None:
                    if star_pos < 0:
                        star_pos = i
                i += 1
            if star_pos < 0:
                raise PatParseException()
            items = []
            i = 0
            while i < size:
                if i == star_pos:
                    st = PatSlot(is_capture=True, is_star=True)
                    _add(items, Pat2Star(st))
                else:
                    _add(items, _value_10 if (_value_10 := pats[i]) is not None else Pat2Wild())
                i += 1
            return Pat2Seq(items, True)
        raise PatParseException()

    @staticmethod
    def assign_pat_names(p, names, index):
        if (
            isinstance((cap := p), Pat2Cap)
            and _item_field(cap.slot, "name", 0) is None
            and (_item_field(cap.slot, "pat", 1) is None)
            and (not cap.slot.is_star)
        ):
            if index < len(names):
                cap.slot.name = names[(index := (index + 1)) - 1]
        elif isinstance((seq := p), Pat2Seq):
            for item in _iter_values(seq.items):
                (
                    (_parsed_2 := StructuralDecompiler.assign_pat_names(item, names, index)),
                    (index := _parsed_2[1]),
                    _parsed_2[0],
                )[-1]
        elif isinstance((map_ := p), Pat2Map):
            for _, pat in _iter_values(map_.entries):
                (
                    (_parsed_4 := StructuralDecompiler.assign_pat_names(pat, names, index)),
                    (index := _parsed_4[1]),
                    _parsed_4[0],
                )[-1]
            if map_.rest is not None:
                (
                    (_parsed_6 := StructuralDecompiler.assign_pat_names(map_.rest, names, index)),
                    (index := _parsed_6[1]),
                    _parsed_6[0],
                )[-1]
        elif isinstance((cls := p), Pat2Class):
            for arg in _iter_values(_item_field(cls, "args", 1)):
                (
                    (_parsed_8 := StructuralDecompiler.assign_pat_names(arg, names, index)),
                    (index := _parsed_8[1]),
                    _parsed_8[0],
                )[-1]
        elif isinstance((kw := p), Pat2Kw):
            (
                (
                    _parsed_10 := StructuralDecompiler.assign_pat_names(
                        _item_field(kw, "pat", 1), names, index
                    )
                ),
                (index := _parsed_10[1]),
                _parsed_10[0],
            )[-1]
        elif isinstance((alternatives := p), Pat2Or):
            for alt in _iter_values(alternatives.alts):
                (
                    (
                        _parsed_12 := StructuralDecompiler.assign_pat_names(
                            _item_field(alt, "pat", 1), names, index
                        )
                    ),
                    (index := _parsed_12[1]),
                    _parsed_12[0],
                )[-1]
        return (None, index)

    def parse_map_pat(self, c, slot, limit):
        c.i = self.pat_next(c.i)
        self.pat_note_fail(c, self._instrs[c.i])
        c.i = self.pat_next(c.i + 1)
        _add(c.stack, PatSlot(is_pseudo=True))
        if c.i < limit and self._instrs[c.i].op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
        ):
            self.pat_note_fail(c, self._instrs[c.i])
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
        if (
            c.i + 3 < limit
            and self._instrs[c.i].op_code == OpCodes.GET_LEN
            and (self._instrs[c.i + 1].op_code == OpCodes.LOAD_CONST)
            and (self._instrs[c.i + 2].op_code == OpCodes.COMPARE_OP)
            and (
                self._instrs[c.i + 3].op_code
                in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
            )
        ):
            self.pat_note_fail(c, self._instrs[c.i + 3])
            c.i = self.pat_next(c.i + 4)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
            if len(c.stack) == 0 or not c.stack[-1] is slot:
                raise PatParseException()
            StructuralDecompiler.pat_pop(c)
            return Pat2Map([], None)
        keys = []
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.LOAD_CONST:
            for k in _iter_values(StructuralDecompiler.pat_tuple_items(self._instrs[c.i].arg_val)):
                _add(keys, k)
            _add(c.stack, PatSlot(is_pseudo=True))
            c.i = self.pat_next(c.i + 1)
        elif c.i < limit and self._instrs[c.i].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            while c.i < limit and self._instrs[c.i].op_code in (
                OpCodes.LOAD_GLOBAL,
                OpCodes.LOAD_NAME,
            ):
                name = (
                    _value_8
                    if (
                        _value_8 := (
                            str(_subject_7)
                            if (
                                _subject_7 := (
                                    _value_6
                                    if (_value_6 := self._instrs[c.i].arg_repr) is not None
                                    else self._instrs[c.i].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                _add(keys, name)
                c.i = self.pat_next(c.i + 1)
                while c.i < limit and self._instrs[c.i].op_code == OpCodes.LOAD_ATTR:
                    c.i = self.pat_next(c.i + 1)
            if c.i >= limit or self._instrs[c.i].op_code != OpCodes.BUILD_TUPLE:
                raise PatParseException()
            c.i = self.pat_next(c.i + 1)
            _add(c.stack, PatSlot(is_pseudo=True))
        else:
            raise PatParseException()
        if c.i >= limit or self._instrs[c.i].op_code != OpCodes.MATCH_KEYS:
            raise PatParseException()
        c.i = self.pat_next(c.i + 1)
        keys_slot = StructuralDecompiler.pat_pop(c)
        keys_slot.is_pseudo = False
        keys_slot.is_keys = True
        _add(c.stack, keys_slot)
        vals_slot = PatSlot()
        _add(c.stack, vals_slot)
        if (
            c.i < limit
            and self._instrs[c.i].op_code == OpCodes.COPY
            and ((_value_9 if (_value_9 := self._instrs[c.i].op_arg) is not None else 0) == 1)
        ):
            StructuralDecompiler.pat_copy(c, 1)
            c.i = self.pat_next(c.i + 1)
        if c.i < limit and self._instrs[c.i].op_code in (
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_FALSE,
        ):
            self.pat_note_fail(c, self._instrs[c.i])
            StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
        if (
            c.i + 1 < limit
            and self._instrs[c.i].op_code == OpCodes.POP_TOP
            and (self._instrs[c.i + 1].op_code == OpCodes.LEGACY_COPY_DICT_WITHOUT_KEYS)
        ):
            c.i = self.pat_next(c.i + 2)
            StructuralDecompiler.pat_pop(c)
            rest_slot = PatSlot(is_capture=True, is_star=True)
            _add(c.stack, rest_slot)
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.LEGACY_ROT_TWO:
                StructuralDecompiler.pat_rotate(c.stack, 2)
                c.i = self.pat_next(c.i + 1)
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
                StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
            if c.i < limit and StructuralDecompiler.is_pat_store_op(self._instrs[c.i].op_code):
                rest_slot.name = StructuralDecompiler.pat_store_name(self._instrs[c.i])
                StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
            return Pat2Map([], Pat2Star(rest_slot))
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.LEGACY_DUP_TOP:
            map_pats = []
            while c.i < limit and self._instrs[c.i].op_code == OpCodes.LEGACY_DUP_TOP:
                if (
                    c.i + 2 >= limit
                    or self._instrs[c.i + 1].op_code != OpCodes.LOAD_CONST
                    or self._instrs[c.i + 2].op_code != OpCodes.BINARY_SUBSCR
                ):
                    raise PatParseException()
                c.i = self.pat_next(c.i + 3)
                elem_slot = PatSlot()
                tmp = PatCtx(i=c.i, depth=c.depth + 1)
                _add(tmp.stack, elem_slot)
                p = None
                try:
                    p = self.parse_pat(tmp, elem_slot, limit)
                except PatParseException:
                    p = Pat2Cap(elem_slot)
                c.i = tmp.i
                while c.i < limit and StructuralDecompiler.is_pat_rotate(self._instrs[c.i].op_code):
                    c.i = self.pat_next(c.i + 1)
                _add(map_pats, p)
            if len(map_pats) != len(keys):
                raise PatParseException()
            map_stores = []
            map_rest = None
            rest_scan = c.i
            pop_run = 0
            while rest_scan < limit and self._instrs[rest_scan].op_code == OpCodes.POP_TOP:
                pop_run += 1
                rest_scan += 1
            has_rest = (
                rest_scan < limit
                and self._instrs[rest_scan].op_code == OpCodes.LEGACY_COPY_DICT_WITHOUT_KEYS
            )
            if has_rest and pop_run > 1 and (c.depth > 0):
                inner_entries = []
                i = 0
                while i < len(map_pats):
                    _add(inner_entries, (ConstantExpr(keys[i]), map_pats[i]))
                    i += 1
                return Pat2Map(inner_entries, None)
            if has_rest:
                c.i = self.pat_next(rest_scan + 1)
                StructuralDecompiler.pat_pop(c)
                map_rest = PatSlot(is_capture=True, is_star=True)
                _add(c.stack, map_rest)
                while c.i < limit and StructuralDecompiler.is_pat_rotate(self._instrs[c.i].op_code):
                    c.i = self.pat_next(c.i + 1)
                if c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
                    StructuralDecompiler.pat_pop(c)
                    c.i = self.pat_next(c.i + 1)
            else:
                while c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
                    c.i = self.pat_next(c.i + 1)
            while c.i < limit and StructuralDecompiler.is_pat_store_op(self._instrs[c.i].op_code):
                _add(map_stores, StructuralDecompiler.pat_store_name(self._instrs[c.i]))
                c.i = self.pat_next(c.i + 1)
            name_at = 0
            for map_pat in _iter_values(map_pats):
                (
                    (
                        _parsed_12 := StructuralDecompiler.assign_pat_names(
                            map_pat, map_stores, name_at
                        )
                    ),
                    (name_at := _parsed_12[1]),
                    _parsed_12[0],
                )[-1]
            if map_rest is not None and name_at < len(map_stores):
                map_rest.name = map_stores[(name_at := (name_at + 1)) - 1]
            map_entries = []
            i = 0
            while i < len(map_pats):
                _add(map_entries, (ConstantExpr(keys[i]), map_pats[i]))
                i += 1
            return Pat2Map(map_entries, None if map_rest is None else Pat2Star(map_rest))
        if c.i >= limit or self._instrs[c.i].op_code != OpCodes.UNPACK_SEQUENCE:
            raise PatParseException()
        n = _value_13 if (_value_13 := self._instrs[c.i].op_arg) is not None else 0
        c.i = self.pat_next(c.i + 1)
        StructuralDecompiler.pat_pop(c)
        elems = []
        i = 0
        while i < n:
            _add(elems, PatSlot())
            i += 1
        i = n - 1
        while i >= 0:
            _add(c.stack, elems[i])
            i -= 1
        entries = []
        i = 0
        while i < n:
            p = self.parse_pat(c, elems[i], limit)
            if i >= len(keys):
                raise PatParseException()
            key = ConstantExpr(keys[i])
            _add(entries, (key, p))
            i += 1
        self.pat_consume_swaps(c, limit)
        rest = None
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.BUILD_MAP:
            c.i = self.pat_next(c.i + 1)
            dict_slot = PatSlot()
            _add(c.stack, dict_slot)
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.SWAP:
                StructuralDecompiler.pat_swap(
                    c, _value_14 if (_value_14 := self._instrs[c.i].op_arg) is not None else 0
                )
                c.i = self.pat_next(c.i + 1)
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.DICT_UPDATE:
                StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
            copies = n
            if c.i < limit and self._instrs[c.i].op_code == OpCodes.UNPACK_SEQUENCE:
                copies = _value_15 if (_value_15 := self._instrs[c.i].op_arg) is not None else 0
                c.i = self.pat_next(c.i + 1)
                StructuralDecompiler.pat_pop(c)
                i = 0
                while i < copies:
                    _add(c.stack, PatSlot(is_pseudo=True))
                    i += 1
            d = 0
            while d < copies:
                if c.i < limit and self._instrs[c.i].op_code == OpCodes.COPY:
                    StructuralDecompiler.pat_copy(
                        c, _value_16 if (_value_16 := self._instrs[c.i].op_arg) is not None else 1
                    )
                    c.i = self.pat_next(c.i + 1)
                if c.i < limit and self._instrs[c.i].op_code == OpCodes.SWAP:
                    StructuralDecompiler.pat_swap(
                        c, _value_17 if (_value_17 := self._instrs[c.i].op_arg) is not None else 0
                    )
                    c.i = self.pat_next(c.i + 1)
                if c.i < limit and self._instrs[c.i].op_code == OpCodes.DELETE_SUBSCR:
                    StructuralDecompiler.pat_pop(c)
                    StructuralDecompiler.pat_pop(c)
                    c.i = self.pat_next(c.i + 1)
                else:
                    raise PatParseException()
                d += 1
            if len(c.stack) == 0 or not c.stack[-1] is dict_slot:
                raise PatParseException()
            dict_slot.is_capture = True
            dict_slot.is_star = True
            rest = Pat2Star(dict_slot)
        else:
            while (
                c.i < limit
                and self._instrs[c.i].op_code == OpCodes.POP_TOP
                and (len(c.stack) > 0)
                and (c.stack[-1] is keys_slot or c.stack[-1] is slot)
            ):
                StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
        return Pat2Map(entries, rest)

    def pat_read_dotted(self, c, limit):
        if c.i >= limit or not self._instrs[c.i].op_code in (
            OpCodes.LOAD_GLOBAL,
            OpCodes.LOAD_NAME,
        ):
            raise PatParseException()
        ins = self._instrs[c.i]
        expr = NameExpr(
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else _raise(PatParseException("pattern class name missing"))
        )
        c.i = self.pat_next(c.i + 1)
        while c.i < limit and self._instrs[c.i].op_code == OpCodes.LOAD_ATTR:
            a = self._instrs[c.i]
            expr = AttributeExpr(
                expr,
                (
                    _value_4
                    if (
                        _value_4 := (
                            str(_subject_3)
                            if (_subject_3 := (a.arg_repr if a.arg_repr is not None else a.arg_val))
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                ),
            )
            c.i = self.pat_next(c.i + 1)
        return expr

    def parse_value_or_class_pat(self, c, slot, limit):
        c.i = self.pat_next(c.i)
        if self._instrs[c.i].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            save = c.i
            expr = self.pat_read_dotted(c, limit)
            if (
                c.i + 1 < limit
                and self._instrs[c.i].op_code == OpCodes.LOAD_CONST
                and (self._instrs[c.i + 1].op_code == OpCodes.MATCH_CLASS)
            ):
                return self.parse_class_pat(c, slot, expr, limit)
            c.i = save
        value = None
        if self._instrs[c.i].op_code == OpCodes.LOAD_CONST:
            value = ConstantExpr(self._instrs[c.i].arg_val)
            c.i = self.pat_next(c.i + 1)
        elif self._instrs[c.i].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            value = self.pat_read_dotted(c, limit)
        else:
            raise PatParseException()
        _add(c.stack, PatSlot(is_pseudo=True))
        if c.i >= limit or not self._instrs[c.i].op_code in (OpCodes.COMPARE_OP, OpCodes.IS_OP):
            raise PatParseException()
        c.i = self.pat_next(c.i + 1)
        StructuralDecompiler.pat_pop(c)
        StructuralDecompiler.pat_pop(c)
        if c.i >= limit or not self._instrs[c.i].op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
        ):
            raise PatParseException()
        self.pat_note_fail(c, self._instrs[c.i])
        c.i = self.pat_next(c.i + 1)
        return Pat2Val(value)

    def parse_class_pat(self, c, slot, cls, limit):
        c.i = self.pat_next(c.i)
        _add(c.stack, PatSlot(is_pseudo=True))
        if c.i >= limit or self._instrs[c.i].op_code != OpCodes.LOAD_CONST:
            raise PatParseException()
        attr_names = list(
            _select(
                StructuralDecompiler.pat_tuple_items(self._instrs[c.i].arg_val),
                lambda o: (
                    _value_1
                    if (_value_1 := (str(o) if o is not None else None)) is not None
                    else ""
                ),
            )
        )
        _add(c.stack, PatSlot(is_pseudo=True))
        c.i = self.pat_next(c.i + 1)
        if c.i >= limit or self._instrs[c.i].op_code != OpCodes.MATCH_CLASS:
            raise PatParseException()
        nargs = _value_2 if (_value_2 := self._instrs[c.i].op_arg) is not None else 0
        c.i = self.pat_next(c.i + 1)
        StructuralDecompiler.pat_pop(c)
        StructuralDecompiler.pat_pop(c)
        StructuralDecompiler.pat_pop(c)
        attrs_slot = PatSlot()
        _add(c.stack, attrs_slot)
        had_copy = (
            c.i < limit
            and self._instrs[c.i].op_code == OpCodes.COPY
            and ((_value_3 if (_value_3 := self._instrs[c.i].op_arg) is not None else 0) == 1)
        )
        if had_copy:
            StructuralDecompiler.pat_copy(c, 1)
            c.i = self.pat_next(c.i + 1)
        if c.i < limit and self._instrs[c.i].op_code in (
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_FALSE,
        ):
            self.pat_note_fail(c, self._instrs[c.i])
            if had_copy:
                StructuralDecompiler.pat_pop(c)
            c.i = self.pat_next(c.i + 1)
        total = nargs + len(attr_names)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.UNPACK_SEQUENCE:
            total = _value_5 if (_value_5 := self._instrs[c.i].op_arg) is not None else total
            c.i = self.pat_next(c.i + 1)
            StructuralDecompiler.pat_pop(c)
            elems = []
            i = 0
            while i < total:
                _add(elems, PatSlot())
                i += 1
            i = total - 1
            while i >= 0:
                _add(c.stack, elems[i])
                i -= 1
            args = []
            i = 0
            while i < total:
                p = self.parse_pat(c, elems[i], limit)
                if i >= nargs and i - nargs < len(attr_names):
                    p = Pat2Kw(attr_names[i - nargs], p)
                _add(args, p)
                i += 1
            return Pat2Class(cls, args)
        if c.i < limit and self._instrs[c.i].op_code == OpCodes.LEGACY_DUP_TOP:
            pats = []
            while c.i < limit and self._instrs[c.i].op_code == OpCodes.LEGACY_DUP_TOP:
                if (
                    c.i + 2 >= limit
                    or self._instrs[c.i + 1].op_code != OpCodes.LOAD_CONST
                    or self._instrs[c.i + 2].op_code != OpCodes.BINARY_SUBSCR
                ):
                    raise PatParseException()
                c.i = self.pat_next(c.i + 3)
                elem_slot = PatSlot()
                tmp = PatCtx(i=c.i, depth=c.depth + 1)
                _add(tmp.stack, elem_slot)
                p = None
                try:
                    p = self.parse_pat(tmp, elem_slot, limit)
                except PatParseException:
                    p = Pat2Cap(elem_slot)
                c.i = tmp.i
                while c.i < limit and StructuralDecompiler.is_pat_rotate(self._instrs[c.i].op_code):
                    c.i = self.pat_next(c.i + 1)
                _add(pats, p)
            while c.i < limit and self._instrs[c.i].op_code == OpCodes.POP_TOP:
                c.i = self.pat_next(c.i + 1)
            store_names = []
            while c.i < limit and StructuralDecompiler.is_pat_store_op(self._instrs[c.i].op_code):
                _add(store_names, StructuralDecompiler.pat_store_name(self._instrs[c.i]))
                c.i = self.pat_next(c.i + 1)
            name_idx = 0
            for p in _iter_values(pats):
                if (
                    isinstance((cap := p), Pat2Cap)
                    and _item_field(cap.slot, "name", 0) is None
                    and (_item_field(cap.slot, "pat", 1) is None)
                    and (not cap.slot.is_star)
                    and (name_idx < len(store_names))
                ):
                    cap.slot.name = store_names[(name_idx := (name_idx + 1)) - 1]
            if len(pats) == total:
                args = []
                i = 0
                while i < len(pats):
                    p = pats[i]
                    if i >= nargs and i - nargs < len(attr_names):
                        p = Pat2Kw(attr_names[i - nargs], p)
                    _add(args, p)
                    i += 1
                return Pat2Class(cls, args)
            raise PatParseException()
        if total != 0:
            raise PatParseException()
        return self.try_wrap_class_as_capture(c, Pat2Class(cls, []), limit)

    def try_wrap_class_as_capture(self, c, pattern, limit):
        if c.i >= limit or self._instrs[c.i].op_code != OpCodes.POP_TOP:
            return pattern
        save = c.i
        c.i = self.pat_next(c.i + 1)
        if c.i < limit and StructuralDecompiler.is_pat_store_op(self._instrs[c.i].op_code):
            slot = PatSlot(name=StructuralDecompiler.pat_store_name(self._instrs[c.i]), pat=pattern)
            c.i = self.pat_next(c.i + 1)
            return Pat2Cap(slot)
        c.i = save
        return pattern

    def try_run_pat_cleanup(self, c, limit):
        c.i = self.pat_next(c.i)
        guard = 0
        while c.i < limit and len(c.stack) > 0 and ((guard := (guard + 1)) - 1 < 4096):
            ins = self._instrs[c.i]
            if StructuralDecompiler.is_pat_rotate(ins.op_code):
                StructuralDecompiler.pat_rotate(c.stack, StructuralDecompiler.pat_rotate_count(ins))
                c.i = self.pat_next(c.i + 1)
                continue
            if ins.op_code == OpCodes.SWAP:
                StructuralDecompiler.pat_swap(c, ins.op_arg if ins.op_arg is not None else 0)
                c.i = self.pat_next(c.i + 1)
                continue
            if ins.op_code == OpCodes.COPY:
                StructuralDecompiler.pat_copy(c, ins.op_arg if ins.op_arg is not None else 1)
                c.i = self.pat_next(c.i + 1)
                continue
            if ins.op_code == OpCodes.POP_TOP:
                s = StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
                if s.is_capture and _item_field(s, "name", 0) is None and (not s.is_star):
                    _add(c.pending_popped, s)
                continue
            if StructuralDecompiler.is_pat_store_op(ins.op_code):
                s = StructuralDecompiler.pat_pop(c)
                c.i = self.pat_next(c.i + 1)
                name = StructuralDecompiler.pat_store_name(ins)
                if s.is_capture and _item_field(s, "name", 0) is None:
                    s.name = name
                elif len(c.pending_popped) > 0:
                    idx = _find_last_index(
                        c.pending_popped, lambda p: _item_field(p, "pat", 1) is not None
                    )
                    if idx < 0:
                        idx = len(c.pending_popped) - 1
                    pend = c.pending_popped[idx]
                    c.pending_popped.pop(idx)
                    pend.name = name
                else:
                    return False
                continue
            break
        for s in _iter_values(c.stack):
            if s.is_capture or s.is_star:
                return False
        return True

    def try_decompile_guard_expr(self, start_idx, end_idx, fail_off):
        guard = None
        body_start = 0
        guard = None
        body_start = start_idx
        last_jump = -1
        i = start_idx
        while i < end_idx and i < len(self._instrs):
            if (
                self._instrs[i].op_code in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                and (_value_2 if (_value_2 := self._instrs[i].target_offset) is not None else -1)
                >= fail_off
            ):
                last_jump = i
            i += 1
        if last_jump < 0:
            return (False, guard, body_start)
        g_stack = SymbolicStack()
        try:
            self.decompile_range(start_idx, last_jump, g_stack)
        except Exception:
            return (False, guard, body_start)
        if len(g_stack) == 0:
            return (False, guard, body_start)
        guard = g_stack.pop()
        body_start = last_jump + 1
        while body_start < end_idx and self._instrs[body_start].op_code == OpCodes.POP_TOP:
            body_start += 1
        return (True, guard, body_start)

    def pat_jumps_to_fail(self, start_idx, end_idx, fail_off):
        i = start_idx
        while i < end_idx and i < len(self._instrs):
            if self._instrs[i].op_code in (
                OpCodes.POP_JUMP_IF_FALSE,
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_NONE,
                OpCodes.POP_JUMP_IF_NOT_NONE,
            ):
                if (
                    _value_2 if (_value_2 := self._instrs[i].target_offset) is not None else -1
                ) >= fail_off:
                    return True
            i += 1
        return False

    def looks_like_match_header(self, idx, end_idx):
        if idx + 1 >= end_idx:
            return False
        next_op = self._instrs[idx + 1].op_code
        return (
            next_op
            in (OpCodes.MATCH_SEQUENCE, OpCodes.MATCH_MAPPING, OpCodes.MATCH_KEYS, OpCodes.COPY)
            or (
                next_op in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                and idx + 3 < end_idx
                and (self._instrs[idx + 3].op_code == OpCodes.MATCH_CLASS)
            )
            or (
                next_op in (OpCodes.LOAD_CONST, OpCodes.LOAD_NAME, OpCodes.LOAD_GLOBAL)
                and idx + 2 < end_idx
                and (self._instrs[idx + 2].op_code in (OpCodes.IS_OP, OpCodes.COMPARE_OP))
            )
        )

    def try_parse_case_v2(self, start_idx, end_idx):
        if not self.is_pat_copy(self._instrs[start_idx]):
            return self.try_parse_case_attempt_v2(start_idx, end_idx, False)

        # A copy can preserve the case subject or start an OR-pattern alternative.
        ordinary = self.try_parse_case_attempt_v2(start_idx, end_idx, True)
        pattern_copy = self.try_parse_case_attempt_v2(start_idx, end_idx, False)
        pattern = pattern_copy[1]
        if isinstance(pattern, MatchOrPattern) and len(pattern.patterns) > 1:
            return pattern_copy
        return ordinary if ordinary[0] else pattern_copy

    def try_parse_case_attempt_v2(self, start_idx, end_idx, has_case_copy):
        pattern = None
        after_pat = 0
        boundaries = None
        next_case_idx = 0
        pattern = None
        after_pat = start_idx
        boundaries = []
        next_case_idx = start_idx
        c = PatCtx(i=start_idx, case_start=start_idx)
        subject = PatSlot()
        _add(c.stack, subject)
        if has_case_copy:
            _add(c.stack, StructuralDecompiler.pat_clone_slot(subject))
            c.i = self.pat_next(c.i + 1)
        pat = None
        try:
            pat = self.parse_pat(c, c.stack[-1], end_idx)
            if not self.try_run_pat_cleanup(c, end_idx):
                return (False, pattern, after_pat, boundaries, next_case_idx)
        except Exception as ex:
            if not (
                isinstance(ex, PatParseException)
                or isinstance(ex, DecompilationError)
                or isinstance(ex, IndexError)
                or isinstance(ex, ValueError)
            ):
                raise
            return (False, pattern, after_pat, boundaries, next_case_idx)
        for s in _iter_values(c.pending_popped):
            if _item_field(s, "name", 0) is None and _item_field(s, "pat", 1) is not None:
                return (False, pattern, after_pat, boundaries, next_case_idx)
        self.propagate_or_names(pat)
        bounds = []
        for off in _iter_values(c.fail_targets):
            if (bi := self._offset_to_index.get(off, _MISSING)) is not _MISSING:
                _add(bounds, bi)
        bounds.sort()
        boundaries = bounds
        max_off = max(_iter_values(c.fail_targets)) if len(c.fail_targets) > 0 else -1
        if (
            max_off < 0
            or not (next_idx := self._offset_to_index.get(max_off, _MISSING)) is not _MISSING
        ):
            next_case_idx = end_idx
        else:
            p = next_idx
            guard = 0
            while p < end_idx and (guard := (guard + 1)) - 1 < 64:
                if self._instrs[p].op_code == OpCodes.POP_TOP:
                    p += 1
                    continue
                if (
                    self._instrs[p].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (_subject_1 := self._instrs[p].target_offset),
                        type(_subject_1) is int and (t := _subject_1) is not None,
                    )[-1]
                    and ((j := self._offset_to_index.get(t, _MISSING)) is not _MISSING)
                ):
                    p = j
                    while p < end_idx and self._instrs[p].op_code == OpCodes.POP_TOP:
                        p += 1
                break
            next_case_idx = p
        pattern = self.resolve_pat2(pat)
        after_pat = c.i
        return (True, pattern, after_pat, boundaries, next_case_idx)

    def try_decompile_match_stmt_v2(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if len(stack) == 0:
            return (False, match_stmt, next_idx)
        if idx >= end_idx:
            return (False, match_stmt, next_idx)
        ins = self._instrs[idx]
        starts_with_copy = (
            ins.op_code == OpCodes.COPY
            and (ins.op_arg if ins.op_arg is not None else 0) == 1
            or ins.op_code == OpCodes.LEGACY_DUP_TOP
        )
        if starts_with_copy:
            if not self.looks_like_match_header(idx, end_idx):
                return (False, match_stmt, next_idx)
        else:
            op = ins.op_code
            is_class_header = False
            if op in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
                p = idx + 1
                while p < end_idx and self._instrs[p].op_code == OpCodes.LOAD_ATTR:
                    p += 1
                is_class_header = (
                    p + 1 < end_idx
                    and self._instrs[p].op_code == OpCodes.LOAD_CONST
                    and (self._instrs[p + 1].op_code == OpCodes.MATCH_CLASS)
                )
            if not op in (OpCodes.MATCH_SEQUENCE, OpCodes.MATCH_MAPPING) and (not is_class_header):
                return (False, match_stmt, next_idx)
        subject = stack.peek()
        cases = []
        cur = idx
        match_exit_offset = -1
        first = True
        loops = 0
        while cur < end_idx and (loops := (loops + 1)) - 1 < 256:
            if self._instrs[cur].op_code == OpCodes.NOP:
                limit = end_idx
                if (
                    match_exit_offset >= 0
                    and (me_idx := self._offset_to_index.get(match_exit_offset, _MISSING))
                    is not _MISSING
                ):
                    limit = min(end_idx, me_idx)
                def_body = self.decompile_range(cur + 1, limit, stack.clone())
                _add(cases, MatchCase(MatchWildcardPattern(), None, def_body))
                cur = limit
                break
            if not (
                (_parsed_1 := self.try_parse_case_v2(cur, end_idx)),
                (pat := _parsed_1[1]),
                (after_pat := _parsed_1[2]),
                (boundaries := _parsed_1[3]),
                (next_case_idx := _parsed_1[4]),
                _parsed_1[0],
            )[-1]:
                if first:
                    return (False, match_stmt, next_idx)
                break
            first = False
            body_start = after_pat
            body_end = next_case_idx
            for b in _iter_values(boundaries):
                if b >= after_pat and b < body_end:
                    body_end = b
                    break
            guard = None
            fail_off = (
                self._instrs[boundaries[-1]].offset
                if len(boundaries) > 0
                else (
                    self._instrs[next_case_idx].offset
                    if next_case_idx < len(self._instrs)
                    else 2147483647
                )
            )
            if (
                body_start < body_end
                and (
                    (_parsed_2 := self.try_extract_match_guard(body_start, body_end)),
                    (g := _parsed_2[1]),
                    (g_body_idx := _parsed_2[2]),
                    _parsed_2[0],
                )[-1]
                and (g is not None)
                and self.pat_jumps_to_fail(body_start, g_body_idx, fail_off)
            ):
                guard = g
                body_start = g_body_idx
                if body_start < body_end and self._instrs[body_start].op_code == OpCodes.POP_TOP:
                    body_start += 1
            elif (
                body_start < body_end
                and (
                    (_parsed_3 := self.try_decompile_guard_expr(body_start, body_end, fail_off)),
                    (g2 := _parsed_3[1]),
                    (g2_body := _parsed_3[2]),
                    _parsed_3[0],
                )[-1]
            ):
                guard = g2
                body_start = g2_body
                if body_start < body_end and self._instrs[body_start].op_code == OpCodes.POP_TOP:
                    body_start += 1
            while body_end > body_start and self._instrs[body_end - 1].op_code in (
                OpCodes.NOP,
                OpCodes.POP_TOP,
            ):
                body_end -= 1
            if body_end > body_start and self._instrs[body_end - 1].op_code == OpCodes.JUMP_FORWARD:
                jt = (
                    _value_5
                    if (_value_5 := self._instrs[body_end - 1].target_offset) is not None
                    else -1
                )
                if jt >= 0 and (not jt in self._active_loop_exits):
                    if match_exit_offset < 0 or jt < match_exit_offset:
                        match_exit_offset = jt
                    body_end -= 1
            body = self.decompile_range(body_start, body_end, stack.clone())
            if (
                len(body) > 0
                and isinstance((subject_name := subject), NameExpr)
                and (
                    (_subject_6 := body[0]),
                    isinstance(_subject_6, ExprStmt)
                    and isinstance((body_name := _subject_6.expression), NameExpr),
                )[-1]
                and (_item_field(body_name, "name", 0) == _item_field(subject_name, "name", 0))
            ):
                body.pop(0)
            _add(cases, MatchCase(pat, guard, body))
            if next_case_idx <= cur:
                break
            cur = next_case_idx
        if len(cases) == 0:
            return (False, match_stmt, next_idx)
        stack.pop()
        match_stmt = MatchStmt(subject, cases)
        next_idx = (
            exit_idx
            if match_exit_offset >= 0
            and (exit_idx := self._offset_to_index.get(match_exit_offset, _MISSING)) is not _MISSING
            else cur
        )
        if next_idx < idx:
            next_idx = end_idx
        return (True, match_stmt, next_idx)

    def try_decompile_list_genexpr315(self, idx, end_idx, stack):
        next_idx = 0
        next_idx = idx
        if idx + 4 >= end_idx:
            return (False, next_idx)
        if not self._instrs[idx].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            return (False, next_idx)
        list_name = (
            _value_3
            if (
                _value_3 := (
                    str(_subject_2)
                    if (
                        _subject_2 := (
                            _value_1
                            if (_value_1 := self._instrs[idx].arg_repr) is not None
                            else self._instrs[idx].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if list_name != "list":
            return (False, next_idx)
        if (
            self._instrs[idx + 1].op_code != OpCodes.COPY
            or (_value_4 if (_value_4 := self._instrs[idx + 1].op_arg) is not None else 0) != 1
        ):
            return (False, next_idx)
        if self._instrs[idx + 2].op_code != OpCodes.LOAD_CONST:
            return (False, next_idx)
        const_name = (
            _value_7
            if (
                _value_7 := (
                    str(_subject_6)
                    if (
                        _subject_6 := (
                            _value_5
                            if (_value_5 := self._instrs[idx + 2].arg_repr) is not None
                            else self._instrs[idx + 2].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if const_name != "list":
            return (False, next_idx)
        if (
            self._instrs[idx + 3].op_code != OpCodes.IS_OP
            or (_value_8 if (_value_8 := self._instrs[idx + 3].op_arg) is not None else 0) != 0
        ):
            return (False, next_idx)
        if self._instrs[idx + 4].op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, next_idx)
        cur = idx + 5
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOT_TAKEN, OpCodes.NOP):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.POP_TOP:
            return (False, next_idx)
        cur += 1
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOT_TAKEN, OpCodes.NOP):
            cur += 1
        if (
            cur >= end_idx
            or self._instrs[cur].op_code != OpCodes.BUILD_LIST
            or (_value_11 if (_value_11 := self._instrs[cur].op_arg) is not None else 0) != 0
        ):
            return (False, next_idx)
        gen_start = cur + 1
        for_iter_idx = gen_start
        while for_iter_idx < end_idx and self._instrs[for_iter_idx].op_code != OpCodes.FOR_ITER:
            for_iter_idx += 1
        if for_iter_idx >= end_idx:
            return (False, next_idx)
        push_null_idx = for_iter_idx
        while push_null_idx > gen_start and self._instrs[push_null_idx - 1].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            push_null_idx -= 1
        if (
            push_null_idx > gen_start
            and self._instrs[push_null_idx - 1].op_code == OpCodes.PUSH_NULL
        ):
            push_null_idx -= 1
        gen_stack = SymbolicStack()
        try:
            self.decompile_range(gen_start, push_null_idx, gen_stack)
        except Exception:
            return (False, next_idx)
        if len(gen_stack) != 1 or not isinstance((gen_expr := gen_stack.peek()), GeneratorExpr):
            return (False, next_idx)
        if (
            not (
                type(self._instrs[for_iter_idx].target_offset) is int
                and (loop_exit_target := self._instrs[for_iter_idx].target_offset) is not None
            )
            or not (after_loop := self._offset_to_index.get(loop_exit_target, _MISSING))
            is not _MISSING
        ):
            return (False, next_idx)
        while after_loop < end_idx and (
            self._instrs[after_loop].op_name in ("END_FOR", "POP_ITER")
            or self._instrs[after_loop].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN)
        ):
            after_loop += 1
        if (
            not (
                type(self._instrs[idx + 4].target_offset) is int
                and (fallback_offset := self._instrs[idx + 4].target_offset) is not None
            )
            or not (fallback_idx := self._offset_to_index.get(fallback_offset, _MISSING))
            is not _MISSING
        ):
            return (False, next_idx)
        if (
            after_loop < end_idx
            and self._instrs[after_loop].op_code == OpCodes.JUMP_FORWARD
            and (
                (_subject_15 := self._instrs[after_loop].target_offset),
                type(_subject_15) is int and (join_offset := _subject_15) is not None,
            )[-1]
            and ((join_idx := self._offset_to_index.get(join_offset, _MISSING)) is not _MISSING)
        ):
            fallback_start = after_loop + 1
            while fallback_start < end_idx and self._instrs[fallback_start].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                fallback_start += 1
            if fallback_idx != fallback_start:
                return (False, next_idx)
            next_idx = join_idx
        else:
            if fallback_idx < after_loop:
                return (False, next_idx)
            next_idx = after_loop
        stack.push(CallExpr(NameExpr("list"), [gen_expr], []))
        return (True, next_idx)

    def try_decompile_any_or_case_legacy(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if (
            (_parsed_1 := self.try_decompile_or_pattern_legacy(idx, end_idx, stack)),
            (value_stmt := _parsed_1[1]),
            (value_next := _parsed_1[2]),
            _parsed_1[0],
        )[-1] and isinstance((vm := value_stmt), MatchStmt):
            match_stmt = vm
            next_idx = value_next
            return (True, match_stmt, next_idx)
        class_stack = stack.clone()
        if (
            (_parsed_2 := self.try_decompile_class_or_pattern_legacy(idx, end_idx, class_stack)),
            (class_stmt := _parsed_2[1]),
            (class_next := _parsed_2[2]),
            _parsed_2[0],
        )[-1] and isinstance((cm := class_stmt), MatchStmt):
            match_stmt = cm
            next_idx = class_next
            return (True, match_stmt, next_idx)
        seq_stack = stack.clone()
        if (
            (_parsed_3 := self.try_decompile_seq_or_pattern_legacy(idx, end_idx, seq_stack)),
            (seq_stmt := _parsed_3[1]),
            (seq_next := _parsed_3[2]),
            _parsed_3[0],
        )[-1] and isinstance((sm := seq_stmt), MatchStmt):
            match_stmt = sm
            next_idx = seq_next
            return (True, match_stmt, next_idx)
        return (False, match_stmt, next_idx)

    def try_decompile_seq_or_pattern_legacy(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if idx + 6 >= end_idx or len(stack) == 0:
            return (False, match_stmt, next_idx)
        if (
            self._instrs[idx].op_code != OpCodes.LEGACY_DUP_TOP
            or self._instrs[idx + 1].op_code != OpCodes.MATCH_SEQUENCE
        ):
            return (False, match_stmt, next_idx)
        subject = stack.peek()
        alternatives = []
        cur = idx
        fail_idx = -1
        merge_idx = -1
        guard = 0
        while (guard := (guard + 1)) - 1 < 64:
            if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LEGACY_DUP_TOP:
                return (False, match_stmt, next_idx)
            if cur + 1 >= end_idx or self._instrs[cur + 1].op_code != OpCodes.MATCH_SEQUENCE:
                return (False, match_stmt, next_idx)
            if (
                cur + 5 >= end_idx
                or self._instrs[cur + 2].op_code != OpCodes.POP_JUMP_IF_FALSE
                or self._instrs[cur + 3].op_code != OpCodes.GET_LEN
                or (self._instrs[cur + 4].op_code != OpCodes.LOAD_CONST)
                or (self._instrs[cur + 5].op_code != OpCodes.COMPARE_OP)
            ):
                return (False, match_stmt, next_idx)
            n = _cast_int(
                _value_1 if (_value_1 := self._instrs[cur + 4].arg_val) is not None else 0
            )
            if n != 1:
                return (False, match_stmt, next_idx)
            jump_idx = cur + 6
            while jump_idx < end_idx and self._instrs[jump_idx].op_code in (
                OpCodes.EXTENDED_ARG,
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                jump_idx += 1
            if jump_idx >= end_idx or self._instrs[jump_idx].op_code != OpCodes.POP_JUMP_IF_FALSE:
                return (False, match_stmt, next_idx)
            fail_target = (
                _value_3 if (_value_3 := self._instrs[jump_idx].target_offset) is not None else -1
            )
            if (
                jump_idx + 1 >= end_idx
                or self._instrs[jump_idx + 1].op_code != OpCodes.UNPACK_SEQUENCE
            ):
                return (False, match_stmt, next_idx)
            if (
                jump_idx + 2 >= end_idx
                or self._instrs[jump_idx + 2].op_code != OpCodes.JUMP_FORWARD
            ):
                return (False, match_stmt, next_idx)
            if (
                not (
                    type(self._instrs[jump_idx + 2].target_offset) is int
                    and (merge_target := self._instrs[jump_idx + 2].target_offset) is not None
                )
                or not (merge_idx := self._offset_to_index.get(merge_target, _MISSING))
                is not _MISSING
            ):
                return (False, match_stmt, next_idx)
            if not (next_alt := self._offset_to_index.get(fail_target, _MISSING)) is not _MISSING:
                return (False, match_stmt, next_idx)
            _add(alternatives, n)
            if (
                next_alt < end_idx
                and self._instrs[next_alt].op_code == OpCodes.POP_TOP
                and (next_alt + 2 < end_idx)
                and (self._instrs[next_alt + 1].op_code == OpCodes.LEGACY_DUP_TOP)
                and (self._instrs[next_alt + 2].op_code == OpCodes.MATCH_SEQUENCE)
            ):
                cur = next_alt + 1
                continue
            fail_idx = next_alt
            break
        if fail_idx < 0 or merge_idx < 0:
            return (False, match_stmt, next_idx)
        match_end_idx = fail_idx
        if fail_idx < end_idx and self._instrs[fail_idx].op_code == OpCodes.POP_TOP:
            after_fail = fail_idx + 1
            if after_fail < end_idx and self._instrs[after_fail].op_code == OpCodes.POP_TOP:
                after_fail += 1
            if (
                after_fail < end_idx
                and self._instrs[after_fail].op_code == OpCodes.JUMP_FORWARD
                and (
                    (_subject_4 := self._instrs[after_fail].target_offset),
                    type(_subject_4) is int and (end_target := _subject_4) is not None,
                )[-1]
                and ((end_idx2 := self._offset_to_index.get(end_target, _MISSING)) is not _MISSING)
            ):
                match_end_idx = end_idx2
        body_cur = merge_idx
        if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.LEGACY_ROT_TWO:
            body_cur += 1
        if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.POP_TOP:
            body_cur += 1
        capture = None
        if body_cur < end_idx and StructuralDecompiler.is_pat_store_op(
            self._instrs[body_cur].op_code
        ):
            capture = StructuralDecompiler.pat_store_name(self._instrs[body_cur])
            body_cur += 1
        if capture is None:
            return (False, match_stmt, next_idx)
        body_end = body_cur
        while body_end < match_end_idx and (
            not self._instrs[body_end].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
        ):
            body_end += 1
        if body_end < match_end_idx:
            body_end += 1
        case_body = self.decompile_range(body_cur, body_end, stack.clone())
        seq_patterns = list(
            _select(
                alternatives, lambda _: MatchSequencePattern([MatchAsPattern(None, capture)], True)
            )
        )
        pat = seq_patterns[0] if len(seq_patterns) == 1 else MatchOrPattern(seq_patterns)
        cases = [MatchCase(pat, None, case_body)]
        after_match = match_end_idx
        if match_end_idx < end_idx and self._instrs[match_end_idx].op_code == OpCodes.POP_TOP:
            after_match = match_end_idx + 1
        if after_match < end_idx and self._instrs[after_match].op_code == OpCodes.NOP:
            def_start = after_match + 1
            def_end = def_start
            while def_end < end_idx and (
                not self._instrs[def_end].op_code
                in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST, OpCodes.RAISE_VARARGS)
            ):
                def_end += 1
            if def_end < end_idx:
                def_end += 1
            _add(
                cases,
                MatchCase(
                    MatchWildcardPattern(),
                    None,
                    self.decompile_range(def_start, def_end, stack.clone()),
                ),
            )
            after_match = def_end
        stack.pop()
        match_stmt = MatchStmt(subject, cases)
        next_idx = after_match
        return (True, match_stmt, next_idx)

    def try_decompile_class_or_pattern_legacy(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if idx + 6 >= end_idx or len(stack) == 0:
            return (False, match_stmt, next_idx)
        if (
            self._instrs[idx].op_code != OpCodes.LEGACY_DUP_TOP
            or self._instrs[idx + 1].op_code != OpCodes.LEGACY_DUP_TOP
        ):
            return (False, match_stmt, next_idx)
        if not self._instrs[idx + 2].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            return (False, match_stmt, next_idx)
        if (
            self._instrs[idx + 3].op_code != OpCodes.LOAD_CONST
            or self._instrs[idx + 4].op_code != OpCodes.MATCH_CLASS
            or self._instrs[idx + 5].op_code != OpCodes.POP_JUMP_IF_FALSE
        ):
            return (False, match_stmt, next_idx)
        subject = stack.peek()
        alternatives = []
        cur = idx
        while cur < end_idx and self._instrs[cur].op_code == OpCodes.LEGACY_DUP_TOP:
            cur += 1
        fail_idx = -1
        merge_idx = -1
        guard = 0
        while (guard := (guard + 1)) - 1 < 64:
            if cur < end_idx and self._instrs[cur].op_code == OpCodes.POP_TOP:
                cur += 1
            if cur < end_idx and self._instrs[cur].op_code == OpCodes.LEGACY_DUP_TOP:
                cur += 1
            if cur >= end_idx or not self._instrs[cur].op_code in (
                OpCodes.LOAD_GLOBAL,
                OpCodes.LOAD_NAME,
            ):
                return (False, match_stmt, next_idx)
            cls_expr = NameExpr(
                _value_3
                if (
                    _value_3 := (
                        str(_subject_2)
                        if (
                            _subject_2 := (
                                _value_1
                                if (_value_1 := self._instrs[cur].arg_repr) is not None
                                else self._instrs[cur].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else _raise(PatParseException("pattern class name missing"))
            )
            cur += 1
            while cur < end_idx and self._instrs[cur].op_code == OpCodes.LOAD_ATTR:
                cls_expr = AttributeExpr(
                    cls_expr,
                    (
                        _value_6
                        if (
                            _value_6 := (
                                str(_subject_5)
                                if (
                                    _subject_5 := (
                                        _value_4
                                        if (_value_4 := self._instrs[cur].arg_repr) is not None
                                        else self._instrs[cur].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else ""
                    ),
                )
                cur += 1
            if (
                cur + 4 >= end_idx
                or self._instrs[cur].op_code != OpCodes.LOAD_CONST
                or self._instrs[cur + 1].op_code != OpCodes.MATCH_CLASS
                or (self._instrs[cur + 2].op_code != OpCodes.POP_JUMP_IF_FALSE)
                or (self._instrs[cur + 3].op_code != OpCodes.POP_TOP)
                or (self._instrs[cur + 4].op_code != OpCodes.JUMP_FORWARD)
            ):
                return (False, match_stmt, next_idx)
            match_arg_count = (
                _value_7 if (_value_7 := self._instrs[cur + 1].op_arg) is not None else 0
            )
            if match_arg_count != 0:
                return (False, match_stmt, next_idx)
            _add(alternatives, MatchClassPattern(cls_expr, []))
            if (
                not (
                    type(self._instrs[cur + 4].target_offset) is int
                    and (merge_target := self._instrs[cur + 4].target_offset) is not None
                )
                or not (merge_idx := self._offset_to_index.get(merge_target, _MISSING))
                is not _MISSING
            ):
                return (False, match_stmt, next_idx)
            if (
                not (
                    type(self._instrs[cur + 2].target_offset) is int
                    and (false_target := self._instrs[cur + 2].target_offset) is not None
                )
                or not (next_alt_idx := self._offset_to_index.get(false_target, _MISSING))
                is not _MISSING
            ):
                return (False, match_stmt, next_idx)
            if (
                next_alt_idx >= end_idx
                or self._instrs[next_alt_idx].op_code != OpCodes.POP_TOP
                or next_alt_idx + 1 >= end_idx
                or (self._instrs[next_alt_idx + 1].op_code != OpCodes.LEGACY_DUP_TOP)
            ):
                fail_idx = next_alt_idx
                break
            cur = next_alt_idx + 1
        if fail_idx < 0 or merge_idx < 0:
            return (False, match_stmt, next_idx)
        match_end_idx = fail_idx
        if fail_idx < end_idx and self._instrs[fail_idx].op_code == OpCodes.POP_TOP:
            after_fail = fail_idx + 1
            if after_fail < end_idx and self._instrs[after_fail].op_code == OpCodes.POP_TOP:
                after_fail += 1
            if (
                after_fail < end_idx
                and self._instrs[after_fail].op_code == OpCodes.JUMP_FORWARD
                and (
                    (_subject_8 := self._instrs[after_fail].target_offset),
                    type(_subject_8) is int and (end_target := _subject_8) is not None,
                )[-1]
                and ((end_idx2 := self._offset_to_index.get(end_target, _MISSING)) is not _MISSING)
            ):
                match_end_idx = end_idx2
        body_cur = merge_idx
        if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.POP_TOP:
            body_cur += 1
        capture = None
        if body_cur < end_idx and StructuralDecompiler.is_pat_store_op(
            self._instrs[body_cur].op_code
        ):
            capture = StructuralDecompiler.pat_store_name(self._instrs[body_cur])
            body_cur += 1
        case_guard = None
        if (
            body_cur < end_idx
            and (
                (_parsed_9 := self.try_extract_match_guard(body_cur, match_end_idx)),
                (g := _parsed_9[1]),
                (g_body_idx := _parsed_9[2]),
                _parsed_9[0],
            )[-1]
            and (g is not None)
        ):
            case_guard = g
            body_cur = g_body_idx
            if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.POP_TOP:
                body_cur += 1
        body_end = body_cur
        while body_end < match_end_idx and (
            not self._instrs[body_end].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
        ):
            body_end += 1
        if body_end < match_end_idx:
            body_end += 1
        case_body = self.decompile_range(body_cur, body_end, stack.clone())
        pat = alternatives[0] if len(alternatives) == 1 else MatchOrPattern(alternatives)
        if capture is not None:
            pat = MatchAsPattern(pat, capture)
        cases = [MatchCase(pat, case_guard, case_body)]
        after_match = match_end_idx
        if match_end_idx < end_idx and self._instrs[match_end_idx].op_code == OpCodes.POP_TOP:
            after_match = match_end_idx + 1
        if after_match < end_idx and self._instrs[after_match].op_code == OpCodes.NOP:
            def_start = after_match + 1
            def_end = def_start
            while def_end < end_idx and (
                not self._instrs[def_end].op_code
                in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST, OpCodes.RAISE_VARARGS)
            ):
                def_end += 1
            if def_end < end_idx:
                def_end += 1
            _add(
                cases,
                MatchCase(
                    MatchWildcardPattern(),
                    None,
                    self.decompile_range(def_start, def_end, stack.clone()),
                ),
            )
            after_match = def_end
        stack.pop()
        match_stmt = MatchStmt(subject, cases)
        next_idx = after_match
        return (True, match_stmt, next_idx)

    def try_decompile_or_pattern_legacy(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if idx + 3 >= end_idx or len(stack) == 0:
            return (False, match_stmt, next_idx)
        subject = stack.peek()
        if (
            self._instrs[idx].op_code != OpCodes.LEGACY_DUP_TOP
            or self._instrs[idx + 1].op_code != OpCodes.LEGACY_DUP_TOP
        ):
            return (False, match_stmt, next_idx)
        if self._instrs[idx + 2].op_code != OpCodes.LOAD_CONST:
            return (False, match_stmt, next_idx)
        if not self._instrs[idx + 3].op_code in (OpCodes.COMPARE_OP, OpCodes.IS_OP):
            return (False, match_stmt, next_idx)
        if idx + 4 >= end_idx or self._instrs[idx + 4].op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, match_stmt, next_idx)
        if idx + 5 >= end_idx or self._instrs[idx + 5].op_code != OpCodes.JUMP_FORWARD:
            return (False, match_stmt, next_idx)
        alternatives = []
        cur = idx
        while cur < end_idx and self._instrs[cur].op_code == OpCodes.LEGACY_DUP_TOP:
            cur += 1
        fail_idx = -1
        merge_idx = -1
        guard_limit = 0
        while (guard_limit := (guard_limit + 1)) - 1 < 64:
            if (
                cur + 4 >= end_idx
                or self._instrs[cur].op_code != OpCodes.LOAD_CONST
                or (not self._instrs[cur + 1].op_code in (OpCodes.COMPARE_OP, OpCodes.IS_OP))
                or (self._instrs[cur + 2].op_code != OpCodes.POP_JUMP_IF_FALSE)
                or (self._instrs[cur + 3].op_code != OpCodes.JUMP_FORWARD)
            ):
                return (False, match_stmt, next_idx)
            _add(alternatives, MatchValuePattern(ConstantExpr(self._instrs[cur].arg_val)))
            if (
                not (
                    type(self._instrs[cur + 3].target_offset) is int
                    and (merge_target := self._instrs[cur + 3].target_offset) is not None
                )
                or not (merge_idx := self._offset_to_index.get(merge_target, _MISSING))
                is not _MISSING
            ):
                return (False, match_stmt, next_idx)
            if (
                not (
                    type(self._instrs[cur + 2].target_offset) is int
                    and (false_target := self._instrs[cur + 2].target_offset) is not None
                )
                or not (next_alt_idx := self._offset_to_index.get(false_target, _MISSING))
                is not _MISSING
            ):
                return (False, match_stmt, next_idx)
            if (
                next_alt_idx >= end_idx
                or self._instrs[next_alt_idx].op_code != OpCodes.LEGACY_DUP_TOP
            ):
                fail_idx = next_alt_idx
                break
            cur = next_alt_idx + 1
        if fail_idx < 0 or merge_idx < 0:
            return (False, match_stmt, next_idx)
        match_end_idx = fail_idx
        if fail_idx < end_idx and self._instrs[fail_idx].op_code == OpCodes.POP_TOP:
            after_fail = fail_idx + 1
            if (
                after_fail < end_idx
                and self._instrs[after_fail].op_code == OpCodes.JUMP_FORWARD
                and (
                    (_subject_1 := self._instrs[after_fail].target_offset),
                    type(_subject_1) is int and (end_target := _subject_1) is not None,
                )[-1]
                and ((end_idx2 := self._offset_to_index.get(end_target, _MISSING)) is not _MISSING)
            ):
                match_end_idx = end_idx2
        body_cur = merge_idx
        if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.POP_TOP:
            body_cur += 1
        capture = None
        if body_cur < end_idx and StructuralDecompiler.is_pat_store_op(
            self._instrs[body_cur].op_code
        ):
            capture = StructuralDecompiler.pat_store_name(self._instrs[body_cur])
            body_cur += 1
        guard = None
        if (
            body_cur < end_idx
            and (
                (_parsed_2 := self.try_extract_match_guard(body_cur, match_end_idx)),
                (g := _parsed_2[1]),
                (g_body_idx := _parsed_2[2]),
                _parsed_2[0],
            )[-1]
            and (g is not None)
        ):
            guard = g
            body_cur = g_body_idx
            if body_cur < end_idx and self._instrs[body_cur].op_code == OpCodes.POP_TOP:
                body_cur += 1
        body_end = body_cur
        while body_end < match_end_idx and (
            not self._instrs[body_end].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
        ):
            body_end += 1
        if body_end < match_end_idx:
            body_end += 1
        else:
            body_end = match_end_idx
        case_body = self.decompile_range(body_cur, body_end, stack.clone())
        pat = alternatives[0] if len(alternatives) == 1 else MatchOrPattern(alternatives)
        if capture is not None:
            pat = MatchAsPattern(pat, capture)
        cases = [MatchCase(pat, guard, case_body)]
        after_match = match_end_idx
        if match_end_idx < end_idx and self._instrs[match_end_idx].op_code == OpCodes.POP_TOP:
            after_match = match_end_idx + 1
        if after_match < end_idx and self._instrs[after_match].op_code == OpCodes.NOP:
            def_start = after_match + 1
            def_end = def_start
            while def_end < end_idx and (
                not self._instrs[def_end].op_code
                in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST, OpCodes.RAISE_VARARGS)
            ):
                def_end += 1
            if def_end < end_idx:
                def_end += 1
            _add(
                cases,
                MatchCase(
                    MatchWildcardPattern(),
                    None,
                    self.decompile_range(def_start, def_end, stack.clone()),
                ),
            )
            after_match = def_end
        if after_match < end_idx and self._instrs[after_match].op_code == OpCodes.LEGACY_DUP_TOP:
            next_stack = stack.clone()
            if (
                (
                    _parsed_3 := self.try_decompile_any_or_case_legacy(
                        after_match, end_idx, next_stack
                    )
                ),
                (next_match := _parsed_3[1]),
                (next_after := _parsed_3[2]),
                _parsed_3[0],
            )[-1]:
                cases.extend(next_match.cases)
                after_match = next_after
        stack.pop()
        match_stmt = MatchStmt(subject, cases)
        next_idx = after_match
        return (True, match_stmt, next_idx)

    def looks_like_match_start(self, idx, end_idx):
        if idx >= end_idx:
            return False
        ins = self._instrs[idx]
        if (
            ins.op_code == OpCodes.COPY
            and (ins.op_arg if ins.op_arg is not None else 0) == 1
            or ins.op_code == OpCodes.LEGACY_DUP_TOP
        ):
            return self.looks_like_match_header(idx, end_idx)
        if ins.op_code in (OpCodes.MATCH_SEQUENCE, OpCodes.MATCH_MAPPING):
            return True
        if ins.op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            p = idx + 1
            while p < end_idx and self._instrs[p].op_code == OpCodes.LOAD_ATTR:
                p += 1
            return (
                p + 1 < end_idx
                and self._instrs[p].op_code == OpCodes.LOAD_CONST
                and (self._instrs[p + 1].op_code == OpCodes.MATCH_CLASS)
            )
        return False

    def try_decompile_match_stmt(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if not self.looks_like_match_start(idx, end_idx):
            return (False, match_stmt, next_idx)
        if self.match_shape_is_advanced(idx, end_idx):
            advanced_stack = stack.clone()
            if (
                (_parsed_1 := self.try_decompile_match_stmt_v2(idx, end_idx, stack)),
                (match_stmt := _parsed_1[1]),
                (next_idx := _parsed_1[2]),
                _parsed_1[0],
            )[-1]:
                return (True, match_stmt, next_idx)
            stack.restore_from(advanced_stack)
            return (
                (
                    (_parsed_2 := self.try_decompile_match_stmt_legacy(idx, end_idx, stack)),
                    (match_stmt := _parsed_2[1]),
                    (next_idx := _parsed_2[2]),
                    _parsed_2[0],
                )[-1],
                match_stmt,
                next_idx,
            )
        saved = stack.clone()
        try:
            if (
                (_parsed_3 := self.try_decompile_match_stmt_legacy(idx, end_idx, stack)),
                (match_stmt := _parsed_3[1]),
                (next_idx := _parsed_3[2]),
                _parsed_3[0],
            )[-1]:
                return (True, match_stmt, next_idx)
        except Exception as ex:
            if not not isinstance(ex, FailFastException):
                raise
            pass
        stack.restore_from(saved)
        if not self.has_real_match_opcode(idx, end_idx):
            match_stmt = None
            next_idx = idx
            return (False, match_stmt, next_idx)
        try:
            return (
                (
                    (_parsed_4 := self.try_decompile_match_stmt_v2(idx, end_idx, stack)),
                    (match_stmt := _parsed_4[1]),
                    (next_idx := _parsed_4[2]),
                    _parsed_4[0],
                )[-1],
                match_stmt,
                next_idx,
            )
        except Exception as ex:
            if not not isinstance(ex, FailFastException):
                raise
            match_stmt = None
            next_idx = idx
            return (False, match_stmt, next_idx)

    def has_real_match_opcode(self, idx, end_idx):
        i = idx
        while i < end_idx:
            if self._instrs[i].op_code in (
                OpCodes.MATCH_SEQUENCE,
                OpCodes.MATCH_MAPPING,
                OpCodes.MATCH_KEYS,
                OpCodes.MATCH_CLASS,
            ):
                return True
            i += 1
        return False

    def match_shape_is_advanced(self, idx, end_idx):
        try:
            if not (
                (_parsed_1 := self.try_parse_case_v2(idx, end_idx)),
                (pat := _parsed_1[1]),
                _parsed_1[0],
            )[-1]:
                return False
            return StructuralDecompiler.pat_node_is_advanced(pat, 0)
        except Exception:
            return False

    @staticmethod
    def pat_binds_name(p):
        if isinstance((a := p), MatchAsPattern):
            return (
                _item_field(a, "name", 0) is not None
                and _item_field(a, "name", 0) != "_"
                or (
                    _item_field(a, "pattern", 1) is not None
                    and StructuralDecompiler.pat_binds_name(_item_field(a, "pattern", 1))
                )
            )
        if isinstance((st := p), MatchStarPattern):
            return _item_field(st, "name", 0) != "_"
        if isinstance((s := p), MatchSequencePattern):
            return _any(s.patterns, StructuralDecompiler.pat_binds_name)
        if isinstance((m := p), MatchMappingPattern):
            return m.rest is not None or _any(
                m.entries,
                lambda e: StructuralDecompiler.pat_binds_name(_item_field(e, "pattern", 1)),
            )
        if isinstance((c := p), MatchClassPattern):
            return _any(c.patterns, StructuralDecompiler.pat_binds_name)
        if isinstance((k := p), MatchKeywordPattern):
            return StructuralDecompiler.pat_binds_name(_item_field(k, "pattern", 1))
        if isinstance((o := p), MatchOrPattern):
            return _any(o.patterns, StructuralDecompiler.pat_binds_name)
        return False

    @staticmethod
    def pat_node_is_advanced(p, depth):
        if isinstance(p, MatchStarPattern):
            return True
        if isinstance((m := p), MatchMappingPattern):
            return (
                m.rest is not None
                or _any(
                    m.entries,
                    lambda e: StructuralDecompiler.pat_node_is_advanced(
                        _item_field(e, "pattern", 1), depth + 1
                    ),
                )
                or depth > 0
            )
        if isinstance((s := p), MatchSequencePattern):
            return _any(
                s.patterns, lambda x: StructuralDecompiler.pat_node_is_advanced(x, depth + 1)
            ) or _any(
                s.patterns,
                lambda x: (
                    isinstance(x, MatchSequencePattern)
                    or isinstance(x, MatchClassPattern)
                    or isinstance(x, MatchMappingPattern)
                ),
            )
        if isinstance((c := p), MatchClassPattern):
            return _any(
                c.patterns, lambda x: StructuralDecompiler.pat_node_is_advanced(x, depth + 1)
            ) or _any(
                c.patterns,
                lambda x: (
                    isinstance(x, MatchClassPattern)
                    or isinstance(x, MatchSequencePattern)
                    or isinstance(x, MatchMappingPattern)
                ),
            )
        if isinstance((o := p), MatchOrPattern):
            return _any(o.patterns, StructuralDecompiler.pat_binds_name) or _any(
                o.patterns,
                lambda x: (
                    isinstance(x, MatchAsPattern)
                    or isinstance(x, MatchStarPattern)
                    or StructuralDecompiler.pat_node_is_advanced(x, depth + 1)
                ),
            )
        if isinstance((a := p), MatchAsPattern):
            return (
                (_subject_1 := _item_field(a, "pattern", 1)),
                (
                    True
                    if isinstance(_subject_1, MatchOrPattern)
                    or isinstance(_subject_1, MatchSequencePattern)
                    or isinstance(_subject_1, MatchMappingPattern)
                    else (
                        _any(
                            c.patterns,
                            lambda x: StructuralDecompiler.pat_node_is_advanced(x, depth + 1),
                        )
                        if isinstance((c := _subject_1), MatchClassPattern)
                        else (
                            False
                            if _subject_1 is None
                            else StructuralDecompiler.pat_node_is_advanced(
                                _item_field(a, "pattern", 1), depth + 1
                            )
                        )
                    )
                ),
            )[-1]
        return False

    def try_decompile_match_stmt_legacy(self, idx, end_idx, stack):
        match_stmt = None
        next_idx = 0
        match_stmt = None
        next_idx = idx
        if len(stack) == 0:
            return (False, match_stmt, next_idx)
        ins = self._instrs[idx]
        if ins.op_code != OpCodes.COPY or (ins.op_arg if ins.op_arg is not None else 0) != 1:
            return (False, match_stmt, next_idx)
        if idx + 1 >= end_idx:
            return (False, match_stmt, next_idx)
        next_op = self._instrs[idx + 1].op_code
        is_match_header = (
            next_op
            in (OpCodes.MATCH_SEQUENCE, OpCodes.MATCH_MAPPING, OpCodes.MATCH_KEYS, OpCodes.COPY)
            or (
                next_op in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                and idx + 3 < end_idx
                and (self._instrs[idx + 3].op_code == OpCodes.MATCH_CLASS)
            )
            or (
                next_op in (OpCodes.LOAD_CONST, OpCodes.LOAD_NAME, OpCodes.LOAD_GLOBAL)
                and idx + 2 < end_idx
                and (self._instrs[idx + 2].op_code in (OpCodes.IS_OP, OpCodes.COMPARE_OP))
            )
        )
        if not is_match_header:
            return (False, match_stmt, next_idx)
        subject = stack.pop()
        cases = []
        cur_idx = idx
        match_exit_offset = -1
        last_case_end_idx = -1
        while cur_idx < end_idx:
            loop_start_idx = cur_idx
            while cur_idx < end_idx and self._instrs[cur_idx].op_code == OpCodes.POP_TOP:
                cur_idx += 1
            if cur_idx >= end_idx:
                break
            cur_ins = self._instrs[cur_idx]
            if cur_ins.op_code == OpCodes.NOP:
                body_limit = (
                    min(end_idx, me_wild)
                    if match_exit_offset >= 0
                    and (me_wild := self._offset_to_index.get(match_exit_offset, _MISSING))
                    is not _MISSING
                    else end_idx
                )
                body = self.decompile_range(cur_idx + 1, body_limit, stack.clone())
                _add(cases, MatchCase(MatchWildcardPattern(), None, body))
                last_case_end_idx = body_limit
                break
            if (
                self.is_pat_copy(cur_ins)
                and cur_idx + 2 < end_idx
                and self._instrs[cur_idx + 1].op_code == OpCodes.LOAD_CONST
                and self._instrs[cur_idx + 2].op_code in (OpCodes.COMPARE_OP, OpCodes.IS_OP)
            ):
                _, pattern, _, _, _ = self.try_parse_case_attempt_v2(cur_idx, end_idx, False)
                if isinstance(pattern, MatchOrPattern) and len(pattern.patterns) > 1:
                    tail_stack = stack.clone()
                    tail_stack.push(subject)
                    ok, tail, after_tail = self.try_decompile_match_stmt_v2(
                        cur_idx, end_idx, tail_stack
                    )
                    if ok:
                        cases.extend(tail.cases)
                        return True, MatchStmt(subject, cases), after_tail
            if (
                cur_ins.op_code == OpCodes.COPY
                and cur_idx + 1 < end_idx
                and (self._instrs[cur_idx + 1].op_code == OpCodes.COPY)
            ):
                or_patterns = []
                or_idx = cur_idx
                body_start_idx = -1
                body_end_idx = -1
                next_case_offset = -1
                while or_idx < end_idx and self._instrs[or_idx].op_code == OpCodes.COPY:
                    if (
                        or_idx + 3 < end_idx
                        and self._instrs[or_idx + 1].op_code == OpCodes.LOAD_CONST
                        and (self._instrs[or_idx + 2].op_code == OpCodes.COMPARE_OP)
                        and (self._instrs[or_idx + 3].op_code == OpCodes.POP_JUMP_IF_FALSE)
                    ):
                        val = self._instrs[or_idx + 1].arg_val
                        _add(or_patterns, MatchValuePattern(ConstantExpr(val)))
                        jump_target = (
                            _value_2
                            if (_value_2 := self._instrs[or_idx + 3].target_offset) is not None
                            else 0
                        )
                        next_case_offset = jump_target
                        if body_start_idx == -1:
                            s = or_idx + 4
                            while s < end_idx and self._instrs[s].op_code == OpCodes.POP_TOP:
                                s += 1
                            body_start_idx = s
                            e = s
                            while e < end_idx and (
                                not self._instrs[e].op_code
                                in (OpCodes.RETURN_CONST, OpCodes.RETURN_VALUE)
                            ):
                                e += 1
                            if e < end_idx:
                                e += 1
                            body_end_idx = e
                        if (
                            (j_idx := self._offset_to_index.get(jump_target, _MISSING))
                            is not _MISSING
                            and j_idx < end_idx
                            and (self._instrs[j_idx].op_code == OpCodes.COPY)
                        ):
                            or_idx = j_idx
                        else:
                            break
                    elif (
                        or_idx + 4 < end_idx
                        and self._instrs[or_idx + 1].op_code == OpCodes.COPY
                        and (self._instrs[or_idx + 2].op_code == OpCodes.LOAD_CONST)
                        and (self._instrs[or_idx + 3].op_code == OpCodes.COMPARE_OP)
                        and (self._instrs[or_idx + 4].op_code == OpCodes.POP_JUMP_IF_FALSE)
                    ):
                        val = self._instrs[or_idx + 2].arg_val
                        _add(or_patterns, MatchValuePattern(ConstantExpr(val)))
                        jump_target = (
                            _value_3
                            if (_value_3 := self._instrs[or_idx + 4].target_offset) is not None
                            else 0
                        )
                        next_case_offset = jump_target
                        if body_start_idx == -1:
                            s = or_idx + 5
                            while s < end_idx and self._instrs[s].op_code == OpCodes.POP_TOP:
                                s += 1
                            body_start_idx = s
                            e = s
                            while e < end_idx and (
                                not self._instrs[e].op_code
                                in (OpCodes.RETURN_CONST, OpCodes.RETURN_VALUE)
                            ):
                                e += 1
                            if e < end_idx:
                                e += 1
                            body_end_idx = e
                        if (
                            (j_idx := self._offset_to_index.get(jump_target, _MISSING))
                            is not _MISSING
                            and j_idx < end_idx
                            and (self._instrs[j_idx].op_code == OpCodes.COPY)
                        ):
                            or_idx = j_idx
                        else:
                            break
                    else:
                        break
                if len(or_patterns) > 0:
                    pat = or_patterns[0] if len(or_patterns) == 1 else MatchOrPattern(or_patterns)
                    body = (
                        self.decompile_range(body_start_idx, body_end_idx, stack.clone())
                        if body_start_idx >= 0 and body_end_idx >= body_start_idx
                        else []
                    )
                    _add(cases, MatchCase(pat, None, body))
                    if body_end_idx >= 0:
                        last_case_end_idx = body_end_idx
                    if (
                        next_case_offset >= 0
                        and (nxt_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                    ):
                        cur_idx = max(loop_start_idx + 1, nxt_idx)
                    else:
                        break
                    continue
            if (
                cur_ins.op_code == OpCodes.COPY
                and cur_idx + 3 < end_idx
                and (
                    (_subject_4 := self._instrs[cur_idx + 1].op_code),
                    _subject_4 == OpCodes.LOAD_CONST
                    or _subject_4 == OpCodes.LOAD_NAME
                    or _subject_4 == OpCodes.LOAD_GLOBAL,
                )[-1]
                and (
                    (_subject_5 := self._instrs[cur_idx + 2].op_code),
                    _subject_5 == OpCodes.IS_OP or _subject_5 == OpCodes.COMPARE_OP,
                )[-1]
                and (
                    self._instrs[cur_idx + 3].op_code
                    in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                )
            ):
                val = self._instrs[cur_idx + 1].arg_val
                fail_target = (
                    _value_7
                    if (_value_7 := self._instrs[cur_idx + 3].target_offset) is not None
                    else 0
                )
                body_start = cur_idx + 4
                while body_start < end_idx and self._instrs[body_start].op_code == OpCodes.POP_TOP:
                    body_start += 1
                body_end = (
                    f_idx
                    if (f_idx := self._offset_to_index.get(fail_target, _MISSING)) is not _MISSING
                    else end_idx
                )
                if (
                    body_end > body_start
                    and self._instrs[body_end - 1].op_code == OpCodes.JUMP_FORWARD
                ):
                    j_target = (
                        _value_8
                        if (_value_8 := self._instrs[body_end - 1].target_offset) is not None
                        else -1
                    )
                    if j_target >= fail_target and (not j_target in self._active_loop_exits):
                        match_exit_offset = (
                            j_target if match_exit_offset < 0 else min(match_exit_offset, j_target)
                        )
                        body_end -= 1
                pat = MatchValuePattern(ConstantExpr(val))
                body = self.decompile_range(body_start, body_end, stack.clone())
                _add(cases, MatchCase(pat, None, body))
                last_case_end_idx = body_end
                if (
                    next_case_idx := self._offset_to_index.get(fail_target, _MISSING)
                ) is not _MISSING:
                    cur_idx = next_case_idx
                else:
                    break
                continue
            if (
                cur_ins.op_code in (OpCodes.LOAD_CONST, OpCodes.LOAD_NAME, OpCodes.LOAD_GLOBAL)
                and cur_idx + 2 < end_idx
                and (self._instrs[cur_idx + 1].op_code in (OpCodes.IS_OP, OpCodes.COMPARE_OP))
                and (
                    self._instrs[cur_idx + 2].op_code
                    in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                )
            ):
                val = cur_ins.arg_val
                fail_target = (
                    _value_11
                    if (_value_11 := self._instrs[cur_idx + 2].target_offset) is not None
                    else 0
                )
                body_start = cur_idx + 3
                while body_start < end_idx and self._instrs[body_start].op_code == OpCodes.POP_TOP:
                    body_start += 1
                body_end = (
                    f_idx
                    if (f_idx := self._offset_to_index.get(fail_target, _MISSING)) is not _MISSING
                    else end_idx
                )
                if (
                    body_end > body_start
                    and self._instrs[body_end - 1].op_code == OpCodes.JUMP_FORWARD
                ):
                    j_target = (
                        _value_12
                        if (_value_12 := self._instrs[body_end - 1].target_offset) is not None
                        else -1
                    )
                    if j_target >= fail_target and (not j_target in self._active_loop_exits):
                        match_exit_offset = (
                            j_target if match_exit_offset < 0 else min(match_exit_offset, j_target)
                        )
                        body_end -= 1
                pat = MatchValuePattern(ConstantExpr(val))
                body = self.decompile_range(body_start, body_end, stack.clone())
                _add(cases, MatchCase(pat, None, body))
                last_case_end_idx = body_end
                if (
                    next_case_idx := self._offset_to_index.get(fail_target, _MISSING)
                ) is not _MISSING:
                    cur_idx = next_case_idx
                else:
                    break
                if (
                    match_exit_offset >= 0
                    and (me := self._offset_to_index.get(match_exit_offset, _MISSING))
                    is not _MISSING
                    and (cur_idx >= me)
                ):
                    break
                continue
            if cur_ins.op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                var_name = (
                    _value_14
                    if (
                        _value_14 := (
                            str(_subject_13)
                            if (
                                _subject_13 := (
                                    cur_ins.arg_repr
                                    if cur_ins.arg_repr is not None
                                    else cur_ins.arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else "__pycrune_ph_x"
                )
                pat = MatchAsPattern(None, var_name)
                if (
                    (
                        _parsed_15 := self.try_decompile_conditional(
                            cur_idx + 1, end_idx, stack.clone()
                        )
                    ),
                    (cond_stmt := _parsed_15[1]),
                    (cond_next_idx := _parsed_15[2]),
                    _parsed_15[0],
                )[-1] and isinstance((if_guard := cond_stmt), IfStmt):
                    _add(cases, MatchCase(pat, if_guard.condition, if_guard.then_body))
                    cur_idx = cond_next_idx
                    continue
                else:
                    body_limit = (
                        min(end_idx, me_as)
                        if match_exit_offset >= 0
                        and (me_as := self._offset_to_index.get(match_exit_offset, _MISSING))
                        is not _MISSING
                        else end_idx
                    )
                    body = self.decompile_range(cur_idx + 1, body_limit, stack.clone())
                    _add(cases, MatchCase(pat, None, body))
                    last_case_end_idx = body_limit
                    break
            c_scan = cur_idx
            while (
                c_scan < end_idx
                and self._instrs[c_scan].op_code == OpCodes.COPY
                and (
                    (_value_16 if (_value_16 := self._instrs[c_scan].op_arg) is not None else 0)
                    == 1
                )
            ):
                c_scan += 1
            is_class_pattern = (
                c_scan + 2 < end_idx
                and self._instrs[c_scan].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                and (self._instrs[c_scan + 2].op_code == OpCodes.MATCH_CLASS)
            )
            if is_class_pattern:
                cls_idx = c_scan
                kw_idx = c_scan + 1
                match_class_idx = c_scan + 2
                or_alts = []
                alt_cur = cur_idx
                common_success_offset = -1
                next_case_offset = -1
                while alt_cur < end_idx:
                    scan_a = alt_cur
                    while (
                        scan_a < end_idx
                        and self._instrs[scan_a].op_code == OpCodes.COPY
                        and (
                            (
                                _value_18
                                if (_value_18 := self._instrs[scan_a].op_arg) is not None
                                else 0
                            )
                            == 1
                        )
                    ):
                        scan_a += 1
                    if (
                        scan_a + 2 >= end_idx
                        or not self._instrs[scan_a].op_code
                        in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                        or self._instrs[scan_a + 2].op_code != OpCodes.MATCH_CLASS
                    ):
                        break
                    alt_cls = (
                        _value_21
                        if (
                            _value_21 := (
                                str(_subject_20)
                                if (
                                    _subject_20 := (
                                        _value_19
                                        if (_value_19 := self._instrs[scan_a].arg_repr) is not None
                                        else self._instrs[scan_a].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else _raise(PatParseException("pattern class name missing"))
                    )
                    alt_count = (
                        _value_22
                        if (_value_22 := self._instrs[scan_a + 2].op_arg) is not None
                        else 0
                    )
                    _add(or_alts, (alt_cls, alt_count))
                    j_idx = scan_a + 3
                    while j_idx < end_idx and (
                        not self._instrs[j_idx].op_code
                        in (OpCodes.POP_JUMP_IF_NONE, OpCodes.POP_JUMP_IF_FALSE)
                    ):
                        j_idx += 1
                    if j_idx >= end_idx:
                        break
                    alt_fail_target = (
                        _value_23
                        if (_value_23 := self._instrs[j_idx].target_offset) is not None
                        else 0
                    )
                    after_j = j_idx + 1
                    while (
                        after_j < end_idx
                        and self._instrs[after_j].op_code == OpCodes.UNPACK_SEQUENCE
                    ):
                        after_j += 1
                    if after_j < end_idx and self._instrs[after_j].op_code == OpCodes.JUMP_FORWARD:
                        succ_target = (
                            _value_24
                            if (_value_24 := self._instrs[after_j].target_offset) is not None
                            else 0
                        )
                        if common_success_offset == -1:
                            common_success_offset = succ_target
                        if (
                            f_idx := self._offset_to_index.get(alt_fail_target, _MISSING)
                        ) is not _MISSING:
                            f_pop = f_idx
                            while (
                                f_pop < end_idx and self._instrs[f_pop].op_code == OpCodes.POP_TOP
                            ):
                                f_pop += 1
                            if f_pop < end_idx and self._instrs[f_pop].op_code == OpCodes.COPY:
                                alt_cur = f_pop
                                continue
                            elif (
                                f_pop < end_idx
                                and self._instrs[f_pop].op_code == OpCodes.JUMP_FORWARD
                            ):
                                next_case_offset = (
                                    _value_25
                                    if (_value_25 := self._instrs[f_pop].target_offset) is not None
                                    else 0
                                )
                                break
                            elif f_pop < end_idx:
                                next_case_offset = self._instrs[f_pop].offset
                                break
                    else:
                        next_case_offset = alt_fail_target
                        break
                    break
                if (
                    len(or_alts) > 1
                    and common_success_offset >= 0
                    and (
                        (comm_idx := self._offset_to_index.get(common_success_offset, _MISSING))
                        is not _MISSING
                    )
                ):
                    p_idx = comm_idx
                    bound_vars = []
                    while p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.POP_TOP:
                        p_idx += 1
                    while p_idx < end_idx and self._instrs[p_idx].op_code in (
                        OpCodes.STORE_FAST,
                        OpCodes.STORE_NAME,
                    ):
                        v_name = (
                            _value_29
                            if (
                                _value_29 := (
                                    str(_subject_28)
                                    if (
                                        _subject_28 := (
                                            _value_27
                                            if (_value_27 := self._instrs[p_idx].arg_repr)
                                            is not None
                                            else self._instrs[p_idx].arg_val
                                        )
                                    )
                                    is not None
                                    else None
                                )
                            )
                            is not None
                            else "__pycrune_ph_x"
                        )
                        _add(bound_vars, v_name)
                        p_idx += 1
                    while p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.POP_TOP:
                        p_idx += 1
                    body_start_idx = p_idx
                    body_end_idx = (
                        n_idx
                        if (n_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                        else end_idx
                    )
                    while body_end_idx > body_start_idx and self._instrs[
                        body_end_idx - 1
                    ].op_code in (OpCodes.NOP, OpCodes.POP_TOP):
                        body_end_idx -= 1
                    if (
                        body_end_idx > body_start_idx
                        and self._instrs[body_end_idx - 1].op_code == OpCodes.JUMP_FORWARD
                    ):
                        match_exit_offset = max(
                            match_exit_offset,
                            (
                                _value_31
                                if (_value_31 := self._instrs[body_end_idx - 1].target_offset)
                                is not None
                                else -1
                            ),
                        )
                        body_end_idx -= 1
                    pattern_list = []
                    for alt in _iter_values(or_alts):
                        class_args = list(
                            _select(
                                _take(bound_vars, _item_field(alt, "arg_count", 1)),
                                lambda v: MatchAsPattern(None, v),
                            )
                        )
                        _add(
                            pattern_list,
                            MatchClassPattern(
                                NameExpr(_item_field(alt, "cls_name", 0)), class_args
                            ),
                        )
                    pat = (
                        pattern_list[0] if len(pattern_list) == 1 else MatchOrPattern(pattern_list)
                    )
                    if len(bound_vars) > 0 and _all(
                        or_alts, lambda a: _item_field(a, "arg_count", 1) == 0
                    ):
                        pat = MatchAsPattern(pat, bound_vars[0])
                    guard = None
                    body = None
                    if (
                        (_parsed_32 := self.try_extract_match_guard(body_start_idx, body_end_idx)),
                        (extracted_guard := _parsed_32[1]),
                        (extracted_body_idx := _parsed_32[2]),
                        _parsed_32[0],
                    )[-1] and extracted_guard is not None:
                        guard = extracted_guard
                        body = self.decompile_range(extracted_body_idx, body_end_idx, stack.clone())
                    elif (
                        (
                            _parsed_33 := self.try_decompile_conditional(
                                body_start_idx, body_end_idx, stack.clone()
                            )
                        ),
                        (cond_stmt := _parsed_33[1]),
                        (cond_next_idx := _parsed_33[2]),
                        _parsed_33[0],
                    )[-1] and isinstance((if_guard := cond_stmt), IfStmt):
                        guard = if_guard.condition
                        body = [*if_guard.then_body]
                    else:
                        body = self.decompile_range(body_start_idx, body_end_idx, stack.clone())
                    _add(cases, MatchCase(pat, guard, body))
                    if body_end_idx >= 0:
                        last_case_end_idx = body_end_idx
                    if (
                        next_case_offset >= 0
                        and (nxt_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                    ):
                        cur_idx = max(loop_start_idx + 1, nxt_idx)
                    else:
                        break
                    continue
                else:
                    cls_name = (
                        _value_36
                        if (
                            _value_36 := (
                                str(_subject_35)
                                if (
                                    _subject_35 := (
                                        _value_34
                                        if (_value_34 := self._instrs[cls_idx].arg_repr) is not None
                                        else self._instrs[cls_idx].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else _raise(PatParseException("pattern class name missing"))
                    )
                    kw_names = (
                        _value_40
                        if (
                            _value_40 := list(
                                _select(
                                    _subject_39,
                                    lambda o: (
                                        _value_37
                                        if (_value_37 := (str(o) if o is not None else None))
                                        is not None
                                        else ""
                                    ),
                                )
                                if (
                                    _subject_39 := (
                                        _value_38
                                        if isinstance(
                                            (_value_38 := self._instrs[kw_idx].arg_val),
                                            (list, tuple, set, frozenset),
                                        )
                                        else None
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else []
                    )
                    pos_args_count = (
                        _value_41
                        if (_value_41 := self._instrs[match_class_idx].op_arg) is not None
                        else 0
                    )
                    total_expected_args = pos_args_count + len(kw_names)
                    jump_idx = match_class_idx + 1
                    while jump_idx < end_idx and (
                        not self._instrs[jump_idx].op_code
                        in (OpCodes.POP_JUMP_IF_NONE, OpCodes.POP_JUMP_IF_FALSE)
                    ):
                        jump_idx += 1
                    next_case_offset = (
                        (
                            _value_42
                            if (_value_42 := self._instrs[jump_idx].target_offset) is not None
                            else 0
                        )
                        if jump_idx < end_idx
                        else 0
                    )
                    p_idx = jump_idx + 1
                    if p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.UNPACK_SEQUENCE:
                        p_idx += 1
                    class_args = []
                    ai = 0
                    while ai < total_expected_args and p_idx < end_idx:
                        attr_name = None if ai < pos_args_count else kw_names[ai - pos_args_count]
                        arg_pat = None
                        if (
                            p_idx + 2 < end_idx
                            and (
                                self._instrs[p_idx].op_code
                                in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                                or (
                                    self._instrs[p_idx].op_code == OpCodes.COPY
                                    and self._instrs[p_idx + 1].op_code
                                    in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                                )
                            )
                            and (
                                self._instrs[p_idx + 2].op_code == OpCodes.MATCH_CLASS
                                or (
                                    p_idx + 3 < end_idx
                                    and self._instrs[p_idx + 3].op_code == OpCodes.MATCH_CLASS
                                )
                            )
                        ):
                            arg_pat = (
                                (_parsed_45 := self.parse_class_pattern_args(p_idx, end_idx)),
                                (p_idx := _parsed_45[1]),
                                _parsed_45[0],
                            )[-1]
                        elif (
                            self._instrs[p_idx].op_code in (OpCodes.MATCH_SEQUENCE, OpCodes.SWAP)
                            and p_idx + 1 < end_idx
                            and (
                                self._instrs[p_idx].op_code == OpCodes.MATCH_SEQUENCE
                                or self._instrs[p_idx + 1].op_code == OpCodes.MATCH_SEQUENCE
                            )
                        ):
                            seq_start = (
                                p_idx
                                if self._instrs[p_idx].op_code == OpCodes.MATCH_SEQUENCE
                                else p_idx + 1
                            )
                            s_scan = seq_start + 1
                            seq_stores = []
                            while s_scan < end_idx and (
                                not self._instrs[s_scan].op_code
                                in (
                                    OpCodes.RETURN_CONST,
                                    OpCodes.RETURN_VALUE,
                                    OpCodes.LOAD_FAST,
                                    OpCodes.LOAD_GLOBAL,
                                    OpCodes.LOAD_NAME,
                                )
                            ):
                                if self._instrs[s_scan].op_code in (
                                    OpCodes.STORE_FAST,
                                    OpCodes.STORE_NAME,
                                ):
                                    sn = (
                                        str(_subject_49)
                                        if (
                                            _subject_49 := (
                                                _value_48
                                                if (_value_48 := self._instrs[s_scan].arg_repr)
                                                is not None
                                                else self._instrs[s_scan].arg_val
                                            )
                                        )
                                        is not None
                                        else None
                                    )
                                    if not _empty_string(sn) and (not sn in seq_stores):
                                        _add(seq_stores, sn)
                                s_scan += 1
                            if len(seq_stores) >= 2:
                                arg_pat = MatchSequencePattern(
                                    [
                                        MatchAsPattern(None, seq_stores[1]),
                                        MatchStarPattern("_"),
                                        MatchAsPattern(None, seq_stores[0]),
                                    ],
                                    True,
                                )
                            elif len(seq_stores) == 1:
                                arg_pat = MatchSequencePattern(
                                    [MatchAsPattern(None, seq_stores[0])], True
                                )
                            else:
                                arg_pat = MatchWildcardPattern()
                            p_idx = s_scan
                        elif (
                            self._instrs[p_idx].op_code in (OpCodes.MATCH_MAPPING, OpCodes.SWAP)
                            and p_idx + 1 < end_idx
                            and (
                                self._instrs[p_idx].op_code == OpCodes.MATCH_MAPPING
                                or self._instrs[p_idx + 1].op_code == OpCodes.MATCH_MAPPING
                            )
                        ):
                            map_start = (
                                p_idx
                                if self._instrs[p_idx].op_code == OpCodes.MATCH_MAPPING
                                else p_idx + 1
                            )
                            map_keys = _empty()
                            mk_scan = map_start
                            while (
                                mk_scan < end_idx
                                and self._instrs[mk_scan].op_code != OpCodes.MATCH_KEYS
                            ):
                                if self._instrs[mk_scan].op_code == OpCodes.LOAD_CONST:
                                    v = self._instrs[mk_scan].arg_val
                                    if isinstance((arr := v), list):
                                        map_keys = arr
                                    elif isinstance((en := v), Iterable) and (not type(v) is str):
                                        map_keys = list(_iter_values(en))
                                mk_scan += 1
                            map_stores = []
                            m_scan = mk_scan
                            while m_scan < end_idx and (
                                not self._instrs[m_scan].op_code
                                in (
                                    OpCodes.RETURN_CONST,
                                    OpCodes.RETURN_VALUE,
                                    OpCodes.LOAD_FAST,
                                    OpCodes.LOAD_GLOBAL,
                                )
                            ):
                                if self._instrs[m_scan].op_code in (
                                    OpCodes.STORE_FAST,
                                    OpCodes.STORE_NAME,
                                ):
                                    sn = (
                                        str(_subject_53)
                                        if (
                                            _subject_53 := (
                                                _value_52
                                                if (_value_52 := self._instrs[m_scan].arg_repr)
                                                is not None
                                                else self._instrs[m_scan].arg_val
                                            )
                                        )
                                        is not None
                                        else None
                                    )
                                    if not _empty_string(sn) and (not sn in map_stores):
                                        _add(map_stores, sn)
                                m_scan += 1
                            rest_var = None
                            val_var = None
                            other_var = None
                            if len(map_stores) >= 3:
                                rest_var = map_stores[0]
                                other_var = map_stores[1]
                                val_var = map_stores[2]
                            elif len(map_stores) == 2:
                                rest_var = map_stores[0]
                                val_var = map_stores[1]
                            elif len(map_stores) == 1:
                                val_var = map_stores[0]
                            if other_var is not None:
                                if len(class_args) > 0 and isinstance(
                                    class_args[-1], MatchWildcardPattern
                                ):
                                    class_args[-1] = MatchAsPattern(None, other_var)
                                elif len(class_args) == 0:
                                    _add(class_args, MatchAsPattern(None, other_var))
                                    ai += 1
                            map_entries = []
                            if len(map_keys) > 0:
                                map_key = (
                                    _value_54
                                    if (_value_54 := map_keys[0]) is not None
                                    else _raise(PatParseException("mapping key missing"))
                                )
                                value_pattern = (
                                    MatchAsPattern(None, val_var)
                                    if val_var is not None
                                    else MatchWildcardPattern()
                                )
                                _add(map_entries, (ConstantExpr(map_key), value_pattern))
                            map_rest = MatchStarPattern(rest_var) if rest_var is not None else None
                            arg_pat = MatchMappingPattern(map_entries, map_rest)
                            p_idx = m_scan
                        elif self._instrs[p_idx].op_code == OpCodes.LOAD_CONST:
                            c_val = self._instrs[p_idx].arg_val
                            arg_pat = MatchValuePattern(ConstantExpr(c_val))
                            p_idx += 1
                            while p_idx < end_idx and self._instrs[p_idx].op_code in (
                                OpCodes.COMPARE_OP,
                                OpCodes.POP_JUMP_IF_FALSE,
                            ):
                                p_idx += 1
                        elif self._instrs[p_idx].op_code in (
                            OpCodes.STORE_FAST,
                            OpCodes.STORE_NAME,
                        ):
                            v_name = (
                                _value_59
                                if (
                                    _value_59 := (
                                        str(_subject_58)
                                        if (
                                            _subject_58 := (
                                                _value_57
                                                if (_value_57 := self._instrs[p_idx].arg_repr)
                                                is not None
                                                else self._instrs[p_idx].arg_val
                                            )
                                        )
                                        is not None
                                        else None
                                    )
                                )
                                is not None
                                else "__pycrune_ph_x"
                            )
                            arg_pat = MatchAsPattern(None, v_name)
                            p_idx += 1
                        elif self._instrs[p_idx].op_code == OpCodes.POP_TOP:
                            arg_pat = MatchWildcardPattern()
                            p_idx += 1
                        else:
                            arg_pat = MatchWildcardPattern()
                            p_idx += 1
                        if not _empty_string(attr_name):
                            arg_pat = MatchKeywordPattern(attr_name, arg_pat)
                        _add(class_args, arg_pat)
                        ai += 1
                    pattern_stores = []
                    while p_idx < end_idx:
                        if self._instrs[p_idx].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                            _add(
                                pattern_stores,
                                (
                                    _value_63
                                    if (
                                        _value_63 := (
                                            str(_subject_62)
                                            if (
                                                _subject_62 := (
                                                    _value_61
                                                    if (_value_61 := self._instrs[p_idx].arg_repr)
                                                    is not None
                                                    else self._instrs[p_idx].arg_val
                                                )
                                            )
                                            is not None
                                            else None
                                        )
                                    )
                                    is not None
                                    else "__pycrune_ph_x"
                                ),
                            )
                            p_idx += 1
                        elif self._instrs[p_idx].op_code in (OpCodes.POP_TOP, OpCodes.SWAP):
                            p_idx += 1
                        else:
                            break
                    as_capture = None
                    if len(pattern_stores) > 0:
                        s_idx = 0
                        ai = 0
                        while ai < len(class_args) and s_idx < len(pattern_stores):
                            arg = class_args[ai]
                            if isinstance((kw_pat := arg), MatchKeywordPattern) and isinstance(
                                _item_field(kw_pat, "pattern", 1), MatchWildcardPattern
                            ):
                                class_args[ai] = MatchKeywordPattern(
                                    kw_pat.attribute,
                                    MatchAsPattern(
                                        None, pattern_stores[(s_idx := (s_idx + 1)) - 1]
                                    ),
                                )
                            elif isinstance(arg, MatchWildcardPattern):
                                class_args[ai] = MatchAsPattern(
                                    None, pattern_stores[(s_idx := (s_idx + 1)) - 1]
                                )
                            ai += 1
                        if s_idx < len(pattern_stores):
                            as_capture = pattern_stores[s_idx]
                    guard = None
                    cur_case_limit = (
                        nco_idx
                        if (nco_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                        else end_idx
                    )
                    guard_start = p_idx
                    guard_end = -1
                    g = cur_case_limit - 1
                    while g >= guard_start:
                        if self._instrs[g].op_code in (
                            OpCodes.POP_JUMP_IF_FALSE,
                            OpCodes.POP_JUMP_IF_TRUE,
                        ):
                            tgt = (
                                _value_66
                                if (_value_66 := self._instrs[g].target_offset) is not None
                                else -1
                            )
                            if tgt >= next_case_offset:
                                guard_end = g
                                next_case_offset = tgt
                                break
                        g -= 1
                    if guard_end > guard_start:
                        g_stack = SymbolicStack()
                        self.decompile_range(guard_start, guard_end, g_stack)
                        if len(g_stack) > 0:
                            guard = g_stack.pop()
                        p_idx = guard_end + 1
                        while p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.POP_TOP:
                            p_idx += 1
                    body_start_idx = p_idx
                    body_end_idx = (
                        n_idx
                        if (n_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                        else end_idx
                    )
                    while body_end_idx > body_start_idx and self._instrs[
                        body_end_idx - 1
                    ].op_code in (OpCodes.NOP, OpCodes.POP_TOP):
                        body_end_idx -= 1
                    if (
                        body_end_idx > body_start_idx
                        and self._instrs[body_end_idx - 1].op_code == OpCodes.JUMP_FORWARD
                    ):
                        match_exit_offset = max(
                            match_exit_offset,
                            (
                                _value_68
                                if (_value_68 := self._instrs[body_end_idx - 1].target_offset)
                                is not None
                                else -1
                            ),
                        )
                        body_end_idx -= 1
                    body = self.decompile_range(body_start_idx, body_end_idx, SymbolicStack())
                    class_pattern = MatchClassPattern(NameExpr(cls_name), class_args)
                    if as_capture is not None:
                        class_pattern = MatchAsPattern(class_pattern, as_capture)
                    _add(cases, MatchCase(class_pattern, guard, body))
                    if body_end_idx >= 0:
                        last_case_end_idx = body_end_idx
                    if (
                        next_case_offset >= 0
                        and (nxt_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                        is not _MISSING
                    ):
                        cur_idx = max(loop_start_idx + 1, nxt_idx)
                    else:
                        break
                    continue
            if cur_ins.op_code == OpCodes.MATCH_SEQUENCE or (
                cur_ins.op_code == OpCodes.COPY
                and cur_idx + 1 < end_idx
                and (self._instrs[cur_idx + 1].op_code == OpCodes.MATCH_SEQUENCE)
            ):
                seq_start_idx = (
                    cur_idx if cur_ins.op_code == OpCodes.MATCH_SEQUENCE else cur_idx + 1
                )
                fail_targets = []
                unpack_idx = -1
                is_unpack_ex = False
                unpack_arg = 0
                s = seq_start_idx
                while s < end_idx:
                    if (
                        (_subject_69 := self._instrs[s].target_offset),
                        type(_subject_69) is int and (to1 := _subject_69) is not None,
                    )[-1] and self._instrs[s].op_code in (
                        OpCodes.POP_JUMP_IF_FALSE,
                        OpCodes.POP_JUMP_IF_NONE,
                    ):
                        _add(fail_targets, to1)
                    if self._instrs[s].op_code == OpCodes.UNPACK_SEQUENCE:
                        unpack_idx = s
                        unpack_arg = (
                            _value_71 if (_value_71 := self._instrs[s].op_arg) is not None else 0
                        )
                        break
                    if self._instrs[s].op_code == OpCodes.UNPACK_EX:
                        unpack_idx = s
                        is_unpack_ex = True
                        unpack_arg = (
                            _value_72 if (_value_72 := self._instrs[s].op_arg) is not None else 0
                        )
                        break
                    s += 1
                if unpack_idx < 0:
                    fail_target = fail_targets[0] if len(fail_targets) > 0 else 0
                    body_limit = (
                        ft_idx
                        if (ft_idx := self._offset_to_index.get(fail_target, _MISSING))
                        is not _MISSING
                        else end_idx
                    )
                    saw_first = False
                    saw_last = False
                    stores = []
                    last_store = -1
                    scan_stores = False
                    s = seq_start_idx
                    while s < body_limit:
                        if (
                            self._instrs[s].op_code == OpCodes.LOAD_CONST
                            and s + 1 < body_limit
                            and (self._instrs[s + 1].op_code == OpCodes.BINARY_SUBSCR)
                            and StructuralDecompiler.is_integer_value(self._instrs[s].arg_val)
                            and (StructuralDecompiler.to_big_integer(self._instrs[s].arg_val) == 0)
                        ):
                            saw_first = True
                        if (
                            self._instrs[s].op_code == OpCodes.GET_LEN
                            and s + 3 < body_limit
                            and (self._instrs[s + 1].op_code == OpCodes.LOAD_CONST)
                            and (self._instrs[s + 2].op_code == OpCodes.BINARY_OP)
                            and (self._instrs[s + 3].op_code == OpCodes.BINARY_SUBSCR)
                        ):
                            saw_last = True
                            scan_stores = True
                            s += 3
                            s += 1
                            continue
                        if not scan_stores:
                            s += 1
                            continue
                        if self._instrs[s].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                            store_name = (
                                str(_subject_75)
                                if (
                                    _subject_75 := (
                                        _value_74
                                        if (_value_74 := self._instrs[s].arg_repr) is not None
                                        else self._instrs[s].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                            if not _empty_string(store_name):
                                _add(stores, store_name)
                            last_store = s
                            s += 1
                            continue
                        if self._instrs[s].op_code in (
                            OpCodes.POP_TOP,
                            OpCodes.SWAP,
                            OpCodes.NOP,
                            OpCodes.NOT_TAKEN,
                        ):
                            s += 1
                            continue
                        break
                        s += 1
                    if saw_first and saw_last and (len(stores) == 2) and (last_store >= 0):
                        pat = MatchSequencePattern(
                            [
                                MatchAsPattern(None, stores[1]),
                                MatchStarPattern("_"),
                                MatchAsPattern(None, stores[0]),
                            ],
                            True,
                        )
                        body_start = last_store + 1
                        while body_start < body_limit and self._instrs[body_start].op_code in (
                            OpCodes.POP_TOP,
                            OpCodes.SWAP,
                        ):
                            body_start += 1
                        seq_guard = None
                        seq_body = None
                        if (
                            (_parsed_78 := self.try_extract_match_guard(body_start, end_idx)),
                            (g_expr2 := _parsed_78[1]),
                            (g_body_idx2 := _parsed_78[2]),
                            _parsed_78[0],
                        )[-1]:
                            seq_guard = g_expr2
                            b_limit = body_limit
                            while b_limit > g_body_idx2 and self._instrs[b_limit - 1].op_code in (
                                OpCodes.NOP,
                                OpCodes.POP_TOP,
                            ):
                                b_limit -= 1
                            if (
                                b_limit > g_body_idx2
                                and self._instrs[b_limit - 1].op_code == OpCodes.JUMP_FORWARD
                            ):
                                b_limit -= 1
                            seq_body = self.decompile_range(g_body_idx2, b_limit, stack.clone())
                        elif (
                            (
                                _parsed_80 := self.try_decompile_conditional(
                                    body_start, body_limit, stack.clone()
                                )
                            ),
                            (cond_stmt := _parsed_80[1]),
                            _parsed_80[0],
                        )[-1] and isinstance((if_guard := cond_stmt), IfStmt):
                            seq_guard = if_guard.condition
                            seq_body = [*if_guard.then_body]
                        else:
                            seq_body = self.decompile_range(body_start, body_limit, stack.clone())
                        _add(cases, MatchCase(pat, seq_guard, seq_body))
                        last_case_end_idx = body_limit
                        if (
                            fail_target >= 0
                            and (nxt_case_idx := self._offset_to_index.get(fail_target, _MISSING))
                            is not _MISSING
                        ):
                            cur_idx = max(loop_start_idx + 1, nxt_case_idx)
                            continue
                    cur_idx = loop_start_idx + 1
                    continue
                before_count = unpack_arg & 255 if is_unpack_ex else unpack_arg
                after_count = unpack_arg >> 8 if is_unpack_ex else 0
                seq_len = before_count + after_count + 1 if is_unpack_ex else unpack_arg
                star_idx = before_count if is_unpack_ex else -1
                scan_idx = unpack_idx + 1
                last_store_idx = -1
                last_pop_top = -1
                elems_seen = 0
                k = scan_idx
                while k < end_idx:
                    ins_k = self._instrs[k]
                    if (
                        ins_k.op_code == OpCodes.LOAD_CONST
                        and k + 2 < end_idx
                        and (self._instrs[k + 1].op_code == OpCodes.COMPARE_OP)
                        and (
                            self._instrs[k + 2].op_code
                            in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_NONE)
                        )
                    ):
                        if (
                            (_subject_82 := self._instrs[k + 2].target_offset),
                            type(_subject_82) is int and (to_const := _subject_82) is not None,
                        )[-1]:
                            _add(fail_targets, to_const)
                        elems_seen += 1
                        k += 3
                        continue
                    if ins_k.op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
                        scan_m = k + 1
                        while scan_m < end_idx and (
                            not self._instrs[scan_m].op_code
                            in (OpCodes.MATCH_CLASS, OpCodes.STORE_FAST, OpCodes.STORE_NAME)
                        ):
                            scan_m += 1
                        if scan_m < end_idx and self._instrs[scan_m].op_code == OpCodes.MATCH_CLASS:
                            j_scan = scan_m + 1
                            while j_scan < end_idx and (
                                not self._instrs[j_scan].op_code
                                in (OpCodes.POP_JUMP_IF_NONE, OpCodes.POP_JUMP_IF_FALSE)
                            ):
                                j_scan += 1
                            if (
                                j_scan < end_idx
                                and (
                                    (_subject_83 := self._instrs[j_scan].target_offset),
                                    type(_subject_83) is int
                                    and (to_class := _subject_83) is not None,
                                )[-1]
                            ):
                                _add(fail_targets, to_class)
                            k = j_scan + 1
                            continue
                    if ins_k.op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                        last_store_idx = k
                        elems_seen += 1
                        k += 1
                        continue
                    if ins_k.op_code == OpCodes.POP_TOP and elems_seen < seq_len:
                        elems_seen += 1
                        k += 1
                        continue
                    if elems_seen >= seq_len:
                        if ins_k.op_code == OpCodes.POP_TOP:
                            last_pop_top = k
                            break
                        if (
                            type(ins_k.target_offset) is int
                            and (to_guard := ins_k.target_offset) is not None
                            and (
                                ins_k.op_code
                                in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                            )
                        ):
                            _add(fail_targets, to_guard)
                            if k + 1 < end_idx and self._instrs[k + 1].op_code == OpCodes.POP_TOP:
                                last_pop_top = k + 1
                                break
                        if seq_start_idx == cur_idx:
                            break
                    k += 1
                next_case_offset = (
                    _value_85
                    if (_value_85 := self._instrs[seq_start_idx + 1].target_offset) is not None
                    else (
                        (
                            _value_84
                            if (_value_84 := self._instrs[seq_start_idx + 2].target_offset)
                            is not None
                            else 0
                        )
                        if seq_start_idx + 2 < end_idx
                        else 0
                    )
                )
                body_end_idx = (
                    be_idx
                    if (be_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                    is not _MISSING
                    else end_idx
                )
                while body_end_idx > scan_idx and self._instrs[body_end_idx - 1].op_code in (
                    OpCodes.NOP,
                    OpCodes.POP_TOP,
                ):
                    body_end_idx -= 1
                if (
                    body_end_idx > scan_idx
                    and self._instrs[body_end_idx - 1].op_code == OpCodes.JUMP_FORWARD
                ):
                    match_exit_offset = max(
                        match_exit_offset,
                        (
                            _value_87
                            if (_value_87 := self._instrs[body_end_idx - 1].target_offset)
                            is not None
                            else -1
                        ),
                    )
                    body_end_idx -= 1
                min_guard_scan = max(scan_idx, last_store_idx)
                guard_jump_idx = -1
                g = body_end_idx - 1
                while g > min_guard_scan:
                    if self._instrs[g].op_code in (
                        OpCodes.POP_JUMP_IF_FALSE,
                        OpCodes.POP_JUMP_IF_TRUE,
                    ):
                        tgt = (
                            _value_89
                            if (_value_89 := self._instrs[g].target_offset) is not None
                            else -1
                        )
                        if tgt >= next_case_offset:
                            guard_jump_idx = g
                            break
                    g -= 1
                guard = None
                body_start_idx = 0
                guard_start = -1
                if guard_jump_idx >= 0:
                    last_store_before_guard = scan_idx - 1
                    s = scan_idx
                    while s < guard_jump_idx:
                        if self._instrs[s].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                            if (
                                s > 0
                                and self._instrs[s - 1].op_code == OpCodes.COPY
                                and (
                                    (
                                        _value_91
                                        if (_value_91 := self._instrs[s - 1].op_arg) is not None
                                        else 0
                                    )
                                    == 1
                                )
                            ):
                                s += 1
                                continue
                            last_store_before_guard = s
                        s += 1
                    guard_start = last_store_before_guard + 1
                    while guard_start < guard_jump_idx and self._instrs[guard_start].op_code in (
                        OpCodes.POP_TOP,
                        OpCodes.SWAP,
                    ):
                        guard_start += 1
                    extracted_chain = False
                    if guard_start >= 0 and guard_start < guard_jump_idx:
                        first_jump = -1
                        fj = guard_start
                        while fj <= guard_jump_idx:
                            if self._instrs[fj].op_code in (
                                OpCodes.POP_JUMP_IF_FALSE,
                                OpCodes.POP_JUMP_IF_TRUE,
                            ):
                                first_jump = fj
                                break
                            fj += 1
                        if first_jump >= 0:
                            first_target = (
                                _value_94
                                if (_value_94 := self._instrs[first_jump].target_offset) is not None
                                else -1
                            )
                            forward = first_target > self._instrs[first_jump].offset
                            leaves_pattern = first_target >= next_case_offset
                            if (
                                forward
                                and leaves_pattern
                                and (
                                    (
                                        _parsed_95 := self.try_extract_match_guard(
                                            guard_start, body_end_idx
                                        )
                                    ),
                                    (chain_guard := _parsed_95[1]),
                                    (chain_body_idx := _parsed_95[2]),
                                    _parsed_95[0],
                                )[-1]
                                and (chain_guard is not None)
                            ):
                                guard = chain_guard
                                body_start_idx = chain_body_idx
                                while (
                                    body_start_idx < body_end_idx
                                    and self._instrs[body_start_idx].op_code == OpCodes.POP_TOP
                                ):
                                    body_start_idx += 1
                                extracted_chain = True
                    if not extracted_chain:
                        if guard_jump_idx > guard_start:
                            g_stack = SymbolicStack()
                            self.decompile_range(guard_start, guard_jump_idx, g_stack)
                            if len(g_stack) > 0:
                                guard = g_stack.pop()
                        body_start_idx = guard_jump_idx + 1
                        while (
                            body_start_idx < body_end_idx
                            and self._instrs[body_start_idx].op_code == OpCodes.POP_TOP
                        ):
                            body_start_idx += 1
                else:
                    body_start_idx = (
                        last_pop_top + 1
                        if last_pop_top >= 0
                        else last_store_idx + 1
                        if last_store_idx >= 0
                        else scan_idx
                    )
                final_patterns = []
                elem_idx = 0
                p = scan_idx
                p_limit = guard_start if guard is not None and guard_start > 0 else body_start_idx
                has_nested_mapping = False
                s = scan_idx
                while s < p_limit:
                    if self._instrs[s].op_code == OpCodes.MATCH_MAPPING:
                        has_nested_mapping = True
                        break
                    s += 1
                if has_nested_mapping and seq_len == 2:
                    or_values = []
                    as_var = None
                    inner_map_pat = None
                    s = scan_idx
                    while s < p_limit:
                        if (
                            self._instrs[s].op_code == OpCodes.LOAD_CONST
                            and s + 1 < p_limit
                            and (self._instrs[s + 1].op_code == OpCodes.COMPARE_OP)
                            and (s == 0 or self._instrs[s - 1].op_code != OpCodes.GET_LEN)
                        ):
                            _add(
                                or_values, MatchValuePattern(ConstantExpr(self._instrs[s].arg_val))
                            )
                            s += 2
                            continue
                        if self._instrs[s].op_code == OpCodes.MATCH_MAPPING:
                            mk_idx = s
                            map_keys = _empty()
                            while (
                                mk_idx < p_limit
                                and self._instrs[mk_idx].op_code != OpCodes.MATCH_KEYS
                            ):
                                if self._instrs[mk_idx].op_code == OpCodes.LOAD_CONST and (
                                    mk_idx == 0
                                    or self._instrs[mk_idx - 1].op_code != OpCodes.GET_LEN
                                ):
                                    v = self._instrs[mk_idx].arg_val
                                    if isinstance((arr := v), list):
                                        map_keys = arr
                                    elif isinstance((en := v), Iterable) and (not type(v) is str):
                                        map_keys = list(_iter_values(en))
                                mk_idx += 1
                            inner_stores = []
                            st = mk_idx
                            while st < p_limit:
                                if self._instrs[st].op_code in (
                                    OpCodes.STORE_FAST,
                                    OpCodes.STORE_NAME,
                                ):
                                    _add(
                                        inner_stores,
                                        (
                                            _value_99
                                            if (
                                                _value_99 := (
                                                    str(_subject_98)
                                                    if (
                                                        _subject_98 := (
                                                            _value_97
                                                            if (
                                                                _value_97 := self._instrs[
                                                                    st
                                                                ].arg_repr
                                                            )
                                                            is not None
                                                            else self._instrs[st].arg_val
                                                        )
                                                    )
                                                    is not None
                                                    else None
                                                )
                                            )
                                            is not None
                                            else ""
                                        ),
                                    )
                                st += 1
                            map_entries = []
                            mi = 0
                            while mi < len(map_keys):
                                v_name = inner_stores[mi] if mi < len(inner_stores) else "v"
                                _add(
                                    map_entries,
                                    (ConstantExpr(map_keys[mi]), MatchAsPattern(None, v_name)),
                                )
                                mi += 1
                            inner_map_pat = MatchMappingPattern(map_entries, None)
                            if len(inner_stores) > len(map_keys):
                                as_var = inner_stores[-1]
                            break
                        s += 1
                    first_pat = (
                        MatchOrPattern(or_values)
                        if len(or_values) > 1
                        else or_values[0]
                        if len(or_values) == 1
                        else MatchWildcardPattern()
                    )
                    if not _empty_string(as_var):
                        first_pat = MatchAsPattern(first_pat, as_var)
                    _add(final_patterns, first_pat)
                    if inner_map_pat is not None:
                        _add(final_patterns, inner_map_pat)
                else:
                    class_list = []
                    sc = p
                    while sc < p_limit:
                        if self._instrs[sc].op_code == OpCodes.MATCH_CLASS:
                            c_name = "object"
                            b = sc - 1
                            while b >= p:
                                if self._instrs[b].op_code in (
                                    OpCodes.LOAD_GLOBAL,
                                    OpCodes.LOAD_NAME,
                                ):
                                    c_name = (
                                        _value_103
                                        if (
                                            _value_103 := (
                                                str(_subject_102)
                                                if (
                                                    _subject_102 := (
                                                        _value_101
                                                        if (_value_101 := self._instrs[b].arg_repr)
                                                        is not None
                                                        else self._instrs[b].arg_val
                                                    )
                                                )
                                                is not None
                                                else None
                                            )
                                        )
                                        is not None
                                        else _raise(PatParseException("pattern class name missing"))
                                    )
                                    break
                                b -= 1
                            _add(class_list, c_name)
                        sc += 1
                    seq_stores = []
                    sc = p
                    while sc < p_limit:
                        if self._instrs[sc].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                            s_name = (
                                _value_107
                                if (
                                    _value_107 := (
                                        str(_subject_106)
                                        if (
                                            _subject_106 := (
                                                _value_105
                                                if (_value_105 := self._instrs[sc].arg_repr)
                                                is not None
                                                else self._instrs[sc].arg_val
                                            )
                                        )
                                        is not None
                                        else None
                                    )
                                )
                                is not None
                                else ""
                            )
                            if not _empty_string(s_name) and (not s_name in seq_stores):
                                _add(seq_stores, s_name)
                        sc += 1
                    if (
                        len(class_list) > 0
                        and len(class_list) == seq_len
                        and (len(seq_stores) == seq_len)
                    ):
                        i = 0
                        while i < seq_len:
                            _add(
                                final_patterns,
                                MatchClassPattern(
                                    NameExpr(class_list[i]),
                                    [MatchAsPattern(None, seq_stores[seq_len - 1 - i])],
                                ),
                            )
                            i += 1
                        elem_idx = seq_len
                        p = p_limit
                    while p < p_limit and elem_idx < seq_len:
                        if self._instrs[p].op_code == OpCodes.SWAP:
                            p += 1
                            continue
                        if self._instrs[p].op_code == OpCodes.POP_TOP:
                            if elem_idx == star_idx:
                                _add(final_patterns, MatchStarPattern("_"))
                            else:
                                _add(final_patterns, MatchWildcardPattern())
                            elem_idx += 1
                            p += 1
                            continue
                        if self._instrs[p].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                            v_name = (
                                _value_111
                                if (
                                    _value_111 := (
                                        str(_subject_110)
                                        if (
                                            _subject_110 := (
                                                _value_109
                                                if (_value_109 := self._instrs[p].arg_repr)
                                                is not None
                                                else self._instrs[p].arg_val
                                            )
                                        )
                                        is not None
                                        else None
                                    )
                                )
                                is not None
                                else "__pycrune_ph_x"
                            )
                            if elem_idx == star_idx:
                                _add(final_patterns, MatchStarPattern(v_name))
                            else:
                                _add(final_patterns, MatchAsPattern(None, v_name))
                            elem_idx += 1
                            p += 1
                            continue
                        if (
                            self._instrs[p].op_code == OpCodes.LOAD_CONST
                            and p + 1 < p_limit
                            and (self._instrs[p + 1].op_code == OpCodes.COMPARE_OP)
                            and (p == 0 or self._instrs[p - 1].op_code != OpCodes.GET_LEN)
                        ):
                            _add(
                                final_patterns,
                                MatchValuePattern(ConstantExpr(self._instrs[p].arg_val)),
                            )
                            elem_idx += 1
                            p += 2
                            while p < p_limit and self._instrs[p].op_code in (
                                OpCodes.POP_JUMP_IF_FALSE,
                                OpCodes.POP_JUMP_IF_NONE,
                            ):
                                p += 1
                            continue
                        p_check = p
                        while p_check < p_limit and self._instrs[p_check].op_code == OpCodes.COPY:
                            p_check += 1
                        if (
                            p_check + 2 < p_limit
                            and self._instrs[p_check].op_code
                            in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                            and (self._instrs[p_check + 2].op_code == OpCodes.MATCH_CLASS)
                        ):
                            or_class_pats = []
                            common_succ = -1
                            cur_alt = p
                            while cur_alt < p_limit:
                                a_scan = cur_alt
                                while (
                                    a_scan < p_limit
                                    and self._instrs[a_scan].op_code == OpCodes.COPY
                                ):
                                    a_scan += 1
                                if (
                                    a_scan + 2 >= p_limit
                                    or not self._instrs[a_scan].op_code
                                    in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                                    or self._instrs[a_scan + 2].op_code != OpCodes.MATCH_CLASS
                                ):
                                    break
                                p_arg = a_scan
                                parsed_pat = (
                                    (_parsed_114 := self.parse_class_pattern_args(p_arg, p_limit)),
                                    (p_arg := _parsed_114[1]),
                                    _parsed_114[0],
                                )[-1]
                                _add(or_class_pats, parsed_pat)
                                if (
                                    p_arg < p_limit
                                    and self._instrs[p_arg].op_code == OpCodes.JUMP_FORWARD
                                ):
                                    s_tgt = (
                                        _value_115
                                        if (_value_115 := self._instrs[p_arg].target_offset)
                                        is not None
                                        else 0
                                    )
                                    if common_succ == -1:
                                        common_succ = s_tgt
                                    p_arg += 1
                                while (
                                    p_arg < p_limit
                                    and self._instrs[p_arg].op_code == OpCodes.POP_TOP
                                ):
                                    p_arg += 1
                                if (
                                    p_arg < p_limit
                                    and self._instrs[p_arg].op_code == OpCodes.COPY
                                    and (common_succ >= 0)
                                ):
                                    cur_alt = p_arg
                                    continue
                                cur_alt = p_arg
                                break
                            if (
                                len(or_class_pats) > 1
                                and common_succ >= 0
                                and (
                                    (cs_idx := self._offset_to_index.get(common_succ, _MISSING))
                                    is not _MISSING
                                )
                            ):
                                _add(final_patterns, MatchOrPattern(or_class_pats))
                                elem_idx += 1
                                p = cs_idx
                                while p < p_limit and self._instrs[p].op_code == OpCodes.POP_TOP:
                                    p += 1
                                continue
                            elif len(or_class_pats) == 1:
                                _add(final_patterns, or_class_pats[0])
                                elem_idx += 1
                                p = cur_alt
                                while p < p_limit and self._instrs[p].op_code in (
                                    OpCodes.POP_TOP,
                                    OpCodes.COPY,
                                ):
                                    p += 1
                                continue
                        p += 1
                    while elem_idx < seq_len:
                        if elem_idx == star_idx:
                            _add(final_patterns, MatchStarPattern("args"))
                        else:
                            _add(final_patterns, MatchWildcardPattern())
                        elem_idx += 1
                body = self.decompile_range(body_start_idx, body_end_idx, stack.clone())
                _add(cases, MatchCase(MatchSequencePattern(final_patterns, True), guard, body))
                if body_end_idx >= 0:
                    last_case_end_idx = body_end_idx
                if (
                    next_case_offset >= 0
                    and (nxt_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                    is not _MISSING
                ):
                    cur_idx = max(loop_start_idx + 1, nxt_idx)
                else:
                    break
                continue
            if cur_ins.op_code == OpCodes.MATCH_MAPPING or (
                cur_ins.op_code == OpCodes.COPY
                and cur_idx + 1 < end_idx
                and (self._instrs[cur_idx + 1].op_code == OpCodes.MATCH_MAPPING)
            ):

                def count_value_checks_before(target):
                    n = 0
                    sc = m_idx
                    while sc < target and sc < body_end_idx:
                        if (
                            self._instrs[sc].op_code == OpCodes.COMPARE_OP
                            and sc + 1 < body_end_idx
                            and (
                                (_subject_1 := self._instrs[sc + 1].op_code),
                                _subject_1 == OpCodes.POP_JUMP_IF_FALSE
                                or _subject_1 == OpCodes.POP_JUMP_IF_TRUE,
                            )[-1]
                            and (
                                (
                                    _value_2
                                    if (_value_2 := self._instrs[sc].op_arg) is not None
                                    else 0
                                )
                                == 40
                            )
                        ):
                            n += 1
                        sc += 1
                    return n

                map_start = cur_idx if cur_ins.op_code == OpCodes.MATCH_MAPPING else cur_idx + 1
                next_case_offset = (
                    _value_117
                    if (_value_117 := self._instrs[map_start + 1].target_offset) is not None
                    else 0
                )
                keys_tuple = _empty()
                keys_idx = map_start
                while keys_idx < end_idx and self._instrs[keys_idx].op_code != OpCodes.MATCH_KEYS:
                    if self._instrs[keys_idx].op_code == OpCodes.LOAD_CONST:
                        v = self._instrs[keys_idx].arg_val
                        if isinstance((arr := v), list):
                            keys_tuple = arr
                        elif isinstance((en := v), Iterable) and (not type(v) is str):
                            keys_tuple = list(_iter_values(en))
                    keys_idx += 1
                entries = []
                m_idx = keys_idx + 1
                while m_idx < end_idx and self._instrs[m_idx].op_code in (
                    OpCodes.COPY,
                    OpCodes.POP_JUMP_IF_NONE,
                    OpCodes.UNPACK_SEQUENCE,
                ):
                    m_idx += 1
                body_end_idx = (
                    nxt_case_idx
                    if (nxt_case_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                    is not _MISSING
                    else end_idx
                )
                pattern_check_limit = body_end_idx
                sc = m_idx
                while sc < body_end_idx:
                    if self._instrs[sc].op_code in (
                        OpCodes.BUILD_MAP,
                        OpCodes.STORE_FAST,
                        OpCodes.STORE_NAME,
                    ):
                        pattern_check_limit = sc
                        break
                    sc += 1
                stores_in_case = []
                sc = m_idx
                while sc < body_end_idx:
                    if self._instrs[sc].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                        if (
                            sc > 0
                            and self._instrs[sc - 1].op_code == OpCodes.COPY
                            and (
                                (
                                    _value_121
                                    if (_value_121 := self._instrs[sc - 1].op_arg) is not None
                                    else 0
                                )
                                == 1
                            )
                        ):
                            sc += 1
                            continue
                        sn = (
                            str(_subject_123)
                            if (
                                _subject_123 := (
                                    _value_122
                                    if (_value_122 := self._instrs[sc].arg_repr) is not None
                                    else self._instrs[sc].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                        if not _empty_string(sn) and (not sn in stores_in_case):
                            _add(stores_in_case, sn)
                    sc += 1
                const_checks = []
                sc = m_idx
                while sc < pattern_check_limit:
                    if (
                        self._instrs[sc].op_code == OpCodes.LOAD_CONST
                        and sc + 1 < body_end_idx
                        and (self._instrs[sc + 1].op_code == OpCodes.COMPARE_OP)
                        and (sc == 0 or self._instrs[sc - 1].op_code != OpCodes.GET_LEN)
                    ):
                        _add(const_checks, self._instrs[sc].arg_val)
                    sc += 1
                build_map_count = 0
                sc = m_idx
                while sc < body_end_idx:
                    if self._instrs[sc].op_code == OpCodes.BUILD_MAP:
                        build_map_count += 1
                    sc += 1
                rest_pattern = None
                used_stores = set()
                has_nested_map = False
                inner_map_keys = _empty()
                sc = m_idx
                while sc < body_end_idx:
                    if self._instrs[sc].op_code == OpCodes.MATCH_MAPPING:
                        has_nested_map = True
                        ik_idx = sc
                        while (
                            ik_idx < body_end_idx
                            and self._instrs[ik_idx].op_code != OpCodes.MATCH_KEYS
                        ):
                            if self._instrs[ik_idx].op_code == OpCodes.LOAD_CONST and (
                                ik_idx == 0 or self._instrs[ik_idx - 1].op_code != OpCodes.GET_LEN
                            ):
                                v = self._instrs[ik_idx].arg_val
                                if isinstance((arr := v), list):
                                    inner_map_keys = arr
                                elif isinstance((en := v), Iterable) and (not type(v) is str):
                                    inner_map_keys = list(_iter_values(en))
                            ik_idx += 1
                        break
                    sc += 1
                has_nested_seq = False
                seq_unpack_ex = False
                seq_counts = 0
                seq_count = 0
                sc = m_idx
                while sc < body_end_idx:
                    if self._instrs[sc].op_code == OpCodes.MATCH_SEQUENCE:
                        has_nested_seq = True
                        u = sc + 1
                        while u < body_end_idx:
                            if self._instrs[u].op_code == OpCodes.UNPACK_EX:
                                seq_unpack_ex = True
                                seq_counts = (
                                    _value_124
                                    if (_value_124 := self._instrs[u].op_arg) is not None
                                    else 0
                                )
                                break
                            if self._instrs[u].op_code == OpCodes.UNPACK_SEQUENCE:
                                seq_count = (
                                    _value_125
                                    if (_value_125 := self._instrs[u].op_arg) is not None
                                    else 0
                                )
                                break
                            u += 1
                        break
                    sc += 1
                inner_rest_pattern = None
                if has_nested_map:
                    if build_map_count >= 2 and len(stores_in_case) >= 3:
                        inner_rest_pattern = MatchStarPattern(stores_in_case[-1])
                        _add(used_stores, stores_in_case[-1])
                        rest_pattern = MatchStarPattern(stores_in_case[0])
                        _add(used_stores, stores_in_case[0])
                    elif build_map_count == 1 and len(stores_in_case) > 0:
                        remaining_rest = stores_in_case[-1]
                        inner_rest_pattern = MatchStarPattern(remaining_rest)
                        _add(used_stores, remaining_rest)
                elif build_map_count >= 1 and len(stores_in_case) > 0:
                    rest_pattern = MatchStarPattern(stores_in_case[0])
                    _add(used_stores, stores_in_case[0])
                class_checks = []
                sc = m_idx
                while sc < pattern_check_limit:
                    if self._instrs[sc].op_code == OpCodes.MATCH_CLASS:
                        cls_name = "object"
                        b = sc - 1
                        while b >= m_idx:
                            if self._instrs[b].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
                                cls_name = (
                                    _value_129
                                    if (
                                        _value_129 := (
                                            str(_subject_128)
                                            if (
                                                _subject_128 := (
                                                    _value_127
                                                    if (_value_127 := self._instrs[b].arg_repr)
                                                    is not None
                                                    else self._instrs[b].arg_val
                                                )
                                            )
                                            is not None
                                            else None
                                        )
                                    )
                                    is not None
                                    else _raise(PatParseException("pattern class name missing"))
                                )
                                break
                            b -= 1
                        arg_var = ""
                        st = sc + 1
                        while st < body_end_idx:
                            if self._instrs[st].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                                s_name = (
                                    _value_133
                                    if (
                                        _value_133 := (
                                            str(_subject_132)
                                            if (
                                                _subject_132 := (
                                                    _value_131
                                                    if (_value_131 := self._instrs[st].arg_repr)
                                                    is not None
                                                    else self._instrs[st].arg_val
                                                )
                                            )
                                            is not None
                                            else None
                                        )
                                    )
                                    is not None
                                    else ""
                                )
                                if not s_name in used_stores:
                                    arg_var = s_name
                                    break
                            st += 1
                        _add(
                            class_checks,
                            (
                                cls_name,
                                (
                                    _value_134
                                    if (_value_134 := self._instrs[sc].op_arg) is not None
                                    else 0
                                ),
                                arg_var,
                            ),
                        )
                    sc += 1
                seq_pos = -1
                map_pos = -1
                sc = m_idx
                while sc < body_end_idx:
                    if seq_pos < 0 and self._instrs[sc].op_code == OpCodes.MATCH_SEQUENCE:
                        seq_pos = count_value_checks_before(sc)
                    if map_pos < 0 and self._instrs[sc].op_code == OpCodes.MATCH_MAPPING:
                        map_pos = count_value_checks_before(sc)
                    sc += 1
                key_patterns = {}
                const_idx = 0
                class_idx = 0
                key_idx = -1
                for k_obj in _iter_values(keys_tuple):
                    key_idx += 1
                    k_str = (
                        _value_135
                        if (_value_135 := (str(k_obj) if k_obj is not None else None)) is not None
                        else ""
                    )
                    if (
                        has_nested_map
                        and len(inner_map_keys) > 0
                        and (not k_str in key_patterns)
                        and (map_pos < 0 or key_idx == map_pos)
                        and (
                            not has_nested_seq
                            or seq_pos >= 0
                            or const_idx < len(const_checks)
                            or (len(keys_tuple) == 1)
                        )
                    ):
                        inner_entries = []
                        for ik_obj in _iter_values(inner_map_keys):
                            ik_str = (
                                _value_136
                                if (_value_136 := (str(ik_obj) if ik_obj is not None else None))
                                is not None
                                else ""
                            )
                            if const_idx < len(const_checks):
                                _add(
                                    inner_entries,
                                    (
                                        ConstantExpr(ik_str),
                                        MatchValuePattern(
                                            ConstantExpr(
                                                const_checks[(const_idx := (const_idx + 1)) - 1]
                                            )
                                        ),
                                    ),
                                )
                            else:
                                matching_store = (
                                    _value_137
                                    if (
                                        _value_137 := _first(
                                            stores_in_case, lambda s: not s in used_stores
                                        )
                                    )
                                    is not None
                                    else "__pycrune_ph_token"
                                )
                                _add(used_stores, matching_store)
                                _add(
                                    inner_entries,
                                    (ConstantExpr(ik_str), MatchAsPattern(None, matching_store)),
                                )
                        key_patterns[k_str] = MatchMappingPattern(inner_entries, inner_rest_pattern)
                    elif (
                        has_nested_seq
                        and (not k_str in key_patterns)
                        and (seq_pos < 0 or key_idx == seq_pos)
                    ):
                        seq_stores = list(_where(stores_in_case, lambda s: not s in used_stores))
                        if seq_unpack_ex:
                            leading = seq_counts & 255
                            trailing = seq_counts >> 8
                            seq_pats = []
                            s_idx = 0
                            if leading > 0 and s_idx < len(seq_stores):
                                if len(class_checks) > 0:
                                    cls = class_checks[0]
                                    cls_args = []
                                    ci = 0
                                    while ci < _item_field(cls, "args", 1) and s_idx < len(
                                        seq_stores
                                    ):
                                        _add(cls_args, MatchAsPattern(None, seq_stores[s_idx]))
                                        _add(used_stores, seq_stores[(s_idx := (s_idx + 1)) - 1])
                                        ci += 1
                                    _add(
                                        seq_pats,
                                        MatchClassPattern(
                                            NameExpr(_item_field(cls, "name", 0)), cls_args
                                        ),
                                    )
                                else:
                                    _add(seq_pats, MatchAsPattern(None, seq_stores[s_idx]))
                                    _add(used_stores, seq_stores[(s_idx := (s_idx + 1)) - 1])
                            if s_idx < len(seq_stores):
                                _add(seq_pats, MatchStarPattern(seq_stores[s_idx]))
                                _add(used_stores, seq_stores[(s_idx := (s_idx + 1)) - 1])
                            if trailing > 0 and s_idx < len(seq_stores):
                                _add(seq_pats, MatchAsPattern(None, seq_stores[s_idx]))
                                _add(used_stores, seq_stores[(s_idx := (s_idx + 1)) - 1])
                            key_patterns[k_str] = MatchSequencePattern(seq_pats, False)
                        else:
                            seq_pats = []
                            si = 0
                            while si < seq_count and si < len(seq_stores):
                                _add(seq_pats, MatchAsPattern(None, seq_stores[si]))
                                _add(used_stores, seq_stores[si])
                                si += 1
                            key_patterns[k_str] = MatchSequencePattern(seq_pats, True)
                        has_nested_seq = False
                    elif const_idx < len(const_checks):
                        remaining_keys = _count(
                            _where(
                                keys_tuple,
                                lambda k: (
                                    not (
                                        _value_138
                                        if (_value_138 := (str(k) if k is not None else None))
                                        is not None
                                        else ""
                                    )
                                    in key_patterns
                                ),
                            )
                        )
                        has_store_for_other_key = len(stores_in_case) > 0 and _any(
                            keys_tuple,
                            lambda k: (
                                (
                                    _value_139
                                    if (_value_139 := (str(k) if k is not None else None))
                                    is not None
                                    else ""
                                )
                                != k_str
                            ),
                        )
                        if (remaining_keys == 1 or has_store_for_other_key) and len(
                            const_checks
                        ) - const_idx > 1:
                            or_list = []
                            while const_idx < len(const_checks):
                                _add(
                                    or_list,
                                    MatchValuePattern(
                                        ConstantExpr(
                                            const_checks[(const_idx := (const_idx + 1)) - 1]
                                        )
                                    ),
                                )
                            key_patterns[k_str] = MatchOrPattern(or_list)
                        else:
                            key_patterns[k_str] = MatchValuePattern(
                                ConstantExpr(const_checks[(const_idx := (const_idx + 1)) - 1])
                            )
                for k_obj in _iter_values(keys_tuple):
                    k_str = (
                        _value_140
                        if (_value_140 := (str(k_obj) if k_obj is not None else None)) is not None
                        else ""
                    )
                    if k_str in key_patterns:
                        continue
                    matching_direct_store = _first(
                        stores_in_case,
                        lambda s: not s in used_stores and (s == k_str or k_str in s or s in k_str),
                    )
                    if (
                        matching_direct_store is not None
                        and len(class_checks) > 0
                        and (len(keys_tuple) > 2)
                    ):
                        _add(used_stores, matching_direct_store)
                        key_patterns[k_str] = MatchAsPattern(None, matching_direct_store)
                        continue
                    if class_idx < len(class_checks):
                        cls = class_checks[(class_idx := (class_idx + 1)) - 1]
                        arg_var = (
                            _item_field(cls, "arg_var", 2)
                            if not _empty_string(_item_field(cls, "arg_var", 2))
                            else (
                                _value_141
                                if (
                                    _value_141 := _first(
                                        stores_in_case, lambda s: not s in used_stores
                                    )
                                )
                                is not None
                                else "__pycrune_ph_x"
                            )
                        )
                        _add(used_stores, arg_var)
                        key_patterns[k_str] = MatchClassPattern(
                            NameExpr(_item_field(cls, "name", 0)), [MatchAsPattern(None, arg_var)]
                        )
                        continue
                    direct_store = (
                        _value_142
                        if (_value_142 := _first(stores_in_case, lambda s: not s in used_stores))
                        is not None
                        else k_str
                    )
                    _add(used_stores, direct_store)
                    key_patterns[k_str] = MatchAsPattern(None, direct_store)
                for k_obj in _iter_values(keys_tuple):
                    k_str = (
                        _value_143
                        if (_value_143 := (str(k_obj) if k_obj is not None else None)) is not None
                        else ""
                    )
                    if (pat := key_patterns.get(k_str, _MISSING)) is not _MISSING:
                        _add(entries, (ConstantExpr(k_str), pat))
                guard_end = -1
                g = body_end_idx - 1
                while g >= m_idx:
                    if self._instrs[g].op_code in (
                        OpCodes.POP_JUMP_IF_FALSE,
                        OpCodes.POP_JUMP_IF_TRUE,
                    ):
                        tgt = (
                            _value_145
                            if (_value_145 := self._instrs[g].target_offset) is not None
                            else -1
                        )
                        if tgt >= next_case_offset:
                            guard_end = g
                            break
                    g -= 1
                guard = None
                g_start = m_idx
                s = m_idx
                while s < (guard_end if guard_end > 0 else body_end_idx):
                    if self._instrs[s].op_code in (
                        OpCodes.STORE_FAST,
                        OpCodes.STORE_NAME,
                        OpCodes.DELETE_SUBSCR,
                    ):
                        if (
                            s > 0
                            and self._instrs[s - 1].op_code == OpCodes.COPY
                            and (
                                (
                                    _value_147
                                    if (_value_147 := self._instrs[s - 1].op_arg) is not None
                                    else 0
                                )
                                == 1
                            )
                        ):
                            s += 1
                            continue
                        g_start = s + 1
                    s += 1
                while g_start < guard_end and self._instrs[g_start].op_code in (
                    OpCodes.POP_TOP,
                    OpCodes.SWAP,
                ):
                    g_start += 1
                if guard_end > g_start:
                    g_stack = SymbolicStack()
                    self.decompile_range(g_start, guard_end, g_stack)
                    if len(g_stack) > 0:
                        guard = g_stack.pop()
                    m_idx = guard_end + 1
                elif guard_end > 0:
                    m_idx = guard_end + 1
                else:
                    m_idx = g_start
                while m_idx < body_end_idx and self._instrs[m_idx].op_code == OpCodes.POP_TOP:
                    m_idx += 1
                body_start_idx = m_idx
                while body_end_idx > body_start_idx and self._instrs[body_end_idx - 1].op_code in (
                    OpCodes.NOP,
                    OpCodes.POP_TOP,
                ):
                    body_end_idx -= 1
                if (
                    body_end_idx > body_start_idx
                    and self._instrs[body_end_idx - 1].op_code == OpCodes.JUMP_FORWARD
                ):
                    match_exit_offset = max(
                        match_exit_offset,
                        (
                            _value_150
                            if (_value_150 := self._instrs[body_end_idx - 1].target_offset)
                            is not None
                            else -1
                        ),
                    )
                    body_end_idx -= 1
                body = self.decompile_range(body_start_idx, body_end_idx, SymbolicStack())
                _add(cases, MatchCase(MatchMappingPattern(entries, rest_pattern), guard, body))
                if body_end_idx >= 0:
                    last_case_end_idx = body_end_idx
                if (
                    next_case_offset >= 0
                    and (nxt_idx := self._offset_to_index.get(next_case_offset, _MISSING))
                    is not _MISSING
                ):
                    cur_idx = max(loop_start_idx + 1, nxt_idx)
                else:
                    break
                continue
            cur_idx += 1
        if len(cases) == 0:
            stack.push(subject)
            return (False, match_stmt, next_idx)
        match_stmt = MatchStmt(subject, cases)
        if (
            match_exit_offset >= 0
            and (me_idx := self._offset_to_index.get(match_exit_offset, _MISSING)) is not _MISSING
        ):
            next_idx = me_idx
        else:
            next_idx = min(last_case_end_idx, end_idx) if last_case_end_idx >= 0 else end_idx
        return (True, match_stmt, next_idx)

    def parse_class_pattern_args(self, p_idx, end_idx):
        cls_name = (
            (
                _value_3
                if (
                    _value_3 := (
                        str(_subject_2)
                        if (
                            _subject_2 := (
                                _value_1
                                if (_value_1 := self._instrs[p_idx + 1].arg_repr) is not None
                                else self._instrs[p_idx + 1].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else _raise(PatParseException("pattern class name missing"))
            )
            if self._instrs[p_idx].op_code == OpCodes.COPY
            else (
                _value_6
                if (
                    _value_6 := (
                        str(_subject_5)
                        if (
                            _subject_5 := (
                                _value_4
                                if (_value_4 := self._instrs[p_idx].arg_repr) is not None
                                else self._instrs[p_idx].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else _raise(PatParseException("pattern class name missing"))
            )
        )
        match_idx = p_idx
        while match_idx < end_idx and self._instrs[match_idx].op_code != OpCodes.MATCH_CLASS:
            match_idx += 1
        if match_idx >= end_idx:
            return (MatchWildcardPattern(), p_idx)
        pos_args_count = _value_7 if (_value_7 := self._instrs[match_idx].op_arg) is not None else 0
        kw_idx = match_idx - 1
        kw_names = (
            _value_11
            if (
                _value_11 := list(
                    _select(
                        _subject_10,
                        lambda o: (
                            _value_8
                            if (_value_8 := (str(o) if o is not None else None)) is not None
                            else ""
                        ),
                    )
                    if (
                        _subject_10 := (
                            _value_9
                            if isinstance(
                                (_value_9 := self._instrs[kw_idx].arg_val),
                                (list, tuple, set, frozenset),
                            )
                            else None
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else []
        )
        total_expected_args = pos_args_count + len(kw_names)
        jump_idx = match_idx + 1
        while jump_idx < end_idx and (
            not self._instrs[jump_idx].op_code
            in (OpCodes.POP_JUMP_IF_NONE, OpCodes.POP_JUMP_IF_FALSE)
        ):
            jump_idx += 1
        p_idx = jump_idx + 1
        if p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.UNPACK_SEQUENCE:
            p_idx += 1
        class_args = []
        ai = 0
        while ai < total_expected_args and p_idx < end_idx:
            attr_name = None if ai < pos_args_count else kw_names[ai - pos_args_count]
            arg_pat = None
            if (
                p_idx + 2 < end_idx
                and (
                    self._instrs[p_idx].op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                    or (
                        self._instrs[p_idx].op_code == OpCodes.COPY
                        and self._instrs[p_idx + 1].op_code
                        in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME)
                    )
                )
                and (
                    self._instrs[p_idx + 2].op_code == OpCodes.MATCH_CLASS
                    or (
                        p_idx + 3 < end_idx
                        and self._instrs[p_idx + 3].op_code == OpCodes.MATCH_CLASS
                    )
                )
            ):
                arg_pat = (
                    (_parsed_14 := self.parse_class_pattern_args(p_idx, end_idx)),
                    (p_idx := _parsed_14[1]),
                    _parsed_14[0],
                )[-1]
            elif self._instrs[p_idx].op_code == OpCodes.LOAD_CONST:
                c_val = self._instrs[p_idx].arg_val
                arg_pat = MatchValuePattern(ConstantExpr(c_val))
                p_idx += 1
                while p_idx < end_idx and self._instrs[p_idx].op_code in (
                    OpCodes.COMPARE_OP,
                    OpCodes.POP_JUMP_IF_FALSE,
                ):
                    p_idx += 1
            elif self._instrs[p_idx].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME):
                v_name = (
                    _value_19
                    if (
                        _value_19 := (
                            str(_subject_18)
                            if (
                                _subject_18 := (
                                    _value_17
                                    if (_value_17 := self._instrs[p_idx].arg_repr) is not None
                                    else self._instrs[p_idx].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else "__pycrune_ph_x"
                )
                arg_pat = MatchAsPattern(None, v_name)
                p_idx += 1
            elif self._instrs[p_idx].op_code == OpCodes.POP_TOP:
                arg_pat = MatchWildcardPattern()
                p_idx += 1
            else:
                arg_pat = MatchWildcardPattern()
                p_idx += 1
            if not _empty_string(attr_name):
                arg_pat = MatchKeywordPattern(attr_name, arg_pat)
            _add(class_args, arg_pat)
            ai += 1
        while p_idx < end_idx and self._instrs[p_idx].op_code == OpCodes.UNPACK_SEQUENCE:
            p_idx += 1
        class_pat = MatchClassPattern(NameExpr(cls_name), class_args)
        if (
            total_expected_args == 0
            and p_idx < end_idx
            and (self._instrs[p_idx].op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME))
        ):
            as_name = (
                _value_23
                if (
                    _value_23 := (
                        str(_subject_22)
                        if (
                            _subject_22 := (
                                _value_21
                                if (_value_21 := self._instrs[p_idx].arg_repr) is not None
                                else self._instrs[p_idx].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else "__pycrune_ph_x"
            )
            p_idx += 1
            class_pat = MatchAsPattern(class_pat, as_name)
        return (class_pat, p_idx)

    def resolve_jump_target(self, offset):
        seen = set()
        while offset not in seen and offset in self._offset_to_index:
            seen.add(offset)
            ins = self._instrs[self._offset_to_index[offset]]
            if ins.op_code != OpCodes.JUMP_FORWARD or ins.target_offset is None:
                return offset
            offset = ins.target_offset
        if offset in seen:
            raise FailFastException(f"Cyclic forward jump at offset {offset}")
        return offset

    def is_offset_protected(self, offset):
        for e in _iter_values(self._code.exception_table):
            if e.encloses(offset):
                return True
        return False

    def try_decompile_dead_try_finally(self, idx):
        stmt = None
        next_idx = 0

        def take(expected):
            nonlocal p
            p = self.skip_transparent(p)
            if p >= len(self._instrs) or self._instrs[p].op_code != expected:
                return -1
            p += 1
            return p - 1

        stmt = None
        next_idx = idx
        p = idx
        saw_nop = False
        while p < len(self._instrs) and self._instrs[p].op_code in (
            OpCodes.NOP,
            OpCodes.EXTENDED_ARG,
            OpCodes.NOT_TAKEN,
        ):
            if self._instrs[p].op_code == OpCodes.NOP:
                saw_nop = True
            p += 1
        if not saw_nop:
            return (False, stmt, next_idx)
        ret1 = take(OpCodes.RETURN_CONST)
        push = take(OpCodes.PUSH_EXC_INFO)
        pop = take(OpCodes.POP_TOP)
        pop_ex = take(OpCodes.POP_EXCEPT)
        ret2 = take(OpCodes.RETURN_CONST)
        copy = take(OpCodes.COPY)
        pop_ex2 = take(OpCodes.POP_EXCEPT)
        rer = take(OpCodes.RERAISE)
        if (
            ret1 < 0
            or push < 0
            or pop < 0
            or (pop_ex < 0)
            or (ret2 < 0)
            or (copy < 0)
            or (pop_ex2 < 0)
            or (rer < 0)
        ):
            return (False, stmt, next_idx)
        if not _same_value(self._instrs[ret1].arg_val, self._instrs[ret2].arg_val):
            return (False, stmt, next_idx)
        push_off = self._instrs[push].offset
        pop_off = self._instrs[pop].offset
        copy_off = self._instrs[copy].offset
        has_cleanup = False
        for e in _iter_values(self._code.exception_table):
            if (
                _item_field(e, "start", 0) == push_off
                and _item_field(e, "end", 1) == pop_off + 2
                and (e.target == copy_off)
                and e.last_i
                and (e.depth == 1)
            ):
                has_cleanup = True
                break
        if not has_cleanup:
            return (False, stmt, next_idx)
        stmt = TryStmt(
            [ReturnStmt(None)], [], [], [ReturnStmt(ConstantExpr(self._instrs[ret1].arg_val))]
        )
        next_idx = rer + 1
        return (True, stmt, next_idx)

    def append_unused_name_references(self, stmts):
        if len(stmts) == 0 or not StructuralDecompiler.always_terminates(stmts[-1]):
            return
        used = set()
        for ins in _iter_values(self._instrs):
            if type(ins.arg_val) is str and (s := ins.arg_val) is not None:
                _add(used, s)
        for name in _iter_values(self._code.names):
            if not name in used and (not _empty_string(name)):
                _add(stmts, ExprStmt(NameExpr(name)))

    @staticmethod
    def always_terminates(stmt):
        if isinstance(stmt, ReturnStmt):
            return True
        if isinstance(stmt, RaiseStmt):
            return True
        if (
            isinstance((t := stmt), TryStmt)
            and t.finally_body is not None
            and (len(t.finally_body) > 0)
            and ((fin := t.finally_body) is not None)
        ):
            return StructuralDecompiler.always_terminates(fin[-1])
        if isinstance((i := stmt), IfStmt) and len(i.else_body) > 0:
            return (
                len(i.then_body) > 0
                and StructuralDecompiler.always_terminates(i.then_body[-1])
                and StructuralDecompiler.always_terminates(i.else_body[-1])
            )
        return False

    @staticmethod
    def has_statement_opcode_between(instrs, start_idx, end_idx):
        i = start_idx
        while i < end_idx and i < len(instrs):
            op = instrs[i].op_code
            if op in (
                OpCodes.POP_TOP,
                OpCodes.STORE_NAME,
                OpCodes.STORE_FAST,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
                OpCodes.STORE_SUBSCR,
                OpCodes.STORE_ATTR,
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.JUMP_FORWARD,
                OpCodes.RAISE_VARARGS,
                OpCodes.RERAISE,
                OpCodes.DELETE_NAME,
                OpCodes.DELETE_FAST,
                OpCodes.DELETE_GLOBAL,
                OpCodes.DELETE_SUBSCR,
                OpCodes.DELETE_ATTR,
            ):
                return True
            i += 1
        return False

    def find_finally_handler_end_idx(self, fin_h_idx):
        fh_start = fin_h_idx
        if fh_start < len(self._instrs) and self._instrs[fh_start].op_code == OpCodes.PUSH_EXC_INFO:
            fh_start += 1
        fh_end = fh_start
        nesting = 0
        while fh_end < len(self._instrs):
            op = self._instrs[fh_end].op_code
            if op == OpCodes.PUSH_EXC_INFO:
                nesting += 1
            elif op == OpCodes.POP_EXCEPT:
                if nesting > 0:
                    nesting -= 1
            elif op in (OpCodes.RERAISE, OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                if nesting == 0:
                    break
            fh_end += 1
        if (
            fh_end < len(self._instrs)
            and self._instrs[fh_end].op_code == OpCodes.RERAISE
            and ((_value_1 if (_value_1 := self._instrs[fh_end].op_arg) is not None else 0) == 0)
            and (fh_end + 3 < len(self._instrs))
            and (self._instrs[fh_end + 1].op_code == OpCodes.COPY)
            and (self._instrs[fh_end + 2].op_code == OpCodes.POP_EXCEPT)
            and (self._instrs[fh_end + 3].op_code == OpCodes.RERAISE)
        ):
            return fh_end + 4
        return fh_end + 1

    def index_in_var_names(self, name):
        i = 0
        while i < len(self._code.var_names):
            if self._code.var_names[i] == name:
                return i
            i += 1
        return -1

    @staticmethod
    def simple_assigned_names(stmts):
        for s in _iter_values(stmts):
            if isinstance((a := s), AssignStmt):
                for t in _iter_values(a.targets):
                    if isinstance((n := t), NameExpr):
                        yield _item_field(n, "name", 0)
            elif isinstance((aa := s), AugAssignStmt) and isinstance((an := aa.target), NameExpr):
                yield _item_field(an, "name", 0)
            elif isinstance((f := s), ForStmt) and isinstance((fn := f.target), NameExpr):
                yield _item_field(fn, "name", 0)
            elif isinstance((i := s), IfStmt):
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(i.then_body)):
                    yield n
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(i.else_body)):
                    yield n
            elif isinstance((w := s), WhileStmt):
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(w.body)):
                    yield n
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(w.else_body)):
                    yield n
            elif isinstance((t := s), TryStmt):
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(t.body)):
                    yield n
                for h in _iter_values(t.handlers):
                    for n in _iter_values(StructuralDecompiler.simple_assigned_names(h.body)):
                        yield n
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(t.else_body)):
                    yield n
                if t.finally_body is not None:
                    for n in _iter_values(
                        StructuralDecompiler.simple_assigned_names(t.finally_body)
                    ):
                        yield n
            elif isinstance((wi := s), WithStmt):
                for n in _iter_values(StructuralDecompiler.simple_assigned_names(wi.body)):
                    yield n

    @staticmethod
    def has_real_statements_between(instrs, start_idx, end_idx):
        i = start_idx
        while i < end_idx and i < len(instrs):
            op = instrs[i].op_code
            if not op in (
                OpCodes.NOP,
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                OpCodes.EXTENDED_ARG,
            ):
                return True
            i += 1
        return False

    def try_get_branch_info(self, ins, stack):
        cond = None
        jump_if_true = False
        target = 0
        cond = None
        jump_if_true = False
        target = self.resolve_jump_target(ins.target_offset if ins.target_offset is not None else 0)
        if len(stack) == 0:
            return (False, cond, jump_if_true, target)
        if ins.op_code == OpCodes.POP_JUMP_IF_TRUE:
            cond = stack.pop()
            jump_if_true = True
            return (True, cond, jump_if_true, target)
        elif ins.op_code == OpCodes.POP_JUMP_IF_FALSE:
            cond = stack.pop()
            jump_if_true = False
            return (True, cond, jump_if_true, target)
        elif ins.op_code == OpCodes.POP_JUMP_IF_NONE:
            cond = CompareExpr(stack.pop(), [("is not", NameExpr("None"))])
            jump_if_true = False
            return (True, cond, jump_if_true, target)
        elif ins.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
            cond = CompareExpr(stack.pop(), [("is", NameExpr("None"))])
            jump_if_true = False
            return (True, cond, jump_if_true, target)
        else:
            return (False, cond, jump_if_true, target)

    def is311(self):
        version = self._code.python_version
        return not _empty_string(version) and version.startswith("3.11")

    def is_at_least311(self):
        version = self._code.python_version
        if _empty_string(version):
            return False
        parts = _split(version, ".")
        return (
            len(parts) >= 2
            and (major := _parse_int(parts[0])) is not None
            and ((minor := _parse_int(parts[1])) is not None)
            and (major > 3 or (major == 3 and minor >= 11))
        )

    def is310_or311(self):
        version = self._code.python_version
        return version.startswith("3.10") or version.startswith("3.11")

    def used_const_indices(self):
        if self._used_const_indices is not None:
            return self._used_const_indices
        used = set()
        for ins in _iter_values(self._instrs):
            if (
                ins.op_code == OpCodes.LOAD_CONST
                and ins.op_name == "LOAD_CONST"
                and (type(ins.op_arg) is int)
                and ((idx := ins.op_arg) is not None)
            ):
                _add(used, idx)
            elif (
                ins.op_code == OpCodes.KW_NAMES
                and type(ins.op_arg) is int
                and ((kw_idx := ins.op_arg) is not None)
            ):
                _add(used, kw_idx)
            elif (
                ins.op_code == OpCodes.RETURN_CONST
                and ins.op_name == "RETURN_CONST"
                and (type(ins.op_arg) is int)
                and ((ret_idx := ins.op_arg) is not None)
            ):
                _add(used, ret_idx)
        self._used_const_indices = used
        return used

    def first_const_load_positions(self):
        if self._first_const_load_positions is not None:
            return self._first_const_load_positions
        first = {}
        i = 0
        while i < len(self._instrs):
            ins = self._instrs[i]
            if (
                ins.op_code == OpCodes.LOAD_CONST
                and ins.op_name == "LOAD_CONST"
                and (type(ins.op_arg) is int)
                and ((idx := ins.op_arg) is not None)
            ):
                _try_add(first, idx, i)
            i += 1
        self._first_const_load_positions = first
        return first

    def is_out_of_order_constant(self, value_index, load_position):
        if value_index < 0 or load_position < 0:
            return False
        first = self.first_const_load_positions()
        u = 0
        while u < value_index:
            if StructuralDecompiler.is_compiler_generated_constant(self._code.constants[u]):
                u += 1
                continue
            if (later := first.get(u, _MISSING)) is not _MISSING and later > load_position:
                return True
            u += 1
        return False

    @staticmethod
    def is_compiler_generated_constant(value):
        if isinstance(value, SliceValue):
            return True
        if type(value) is str and (s := value) is not None and s.startswith("<code:"):
            return True
        if isinstance((sequence := value), Iterable) and (not type(value) is str):
            for item in _iter_values(sequence):
                if isinstance(item, SliceValue):
                    return True
        return False

    def is_constant_inserted_late(self, constant_index):
        first = self.first_const_load_positions()
        if not (position := first.get(constant_index, _MISSING)) is not _MISSING:
            return False
        higher = constant_index + 1
        while higher < len(self._code.constants):
            if (
                higher_position := first.get(higher, _MISSING)
            ) is not _MISSING and higher_position < position:
                return True
            higher += 1
        return False

    def try_reconstruct_folded_constant(self, ins):
        expr = None
        expr = None
        if not self.is_at_least314() or ins.arg_val is None:
            return (False, expr)
        consts = self._code.constants
        if len(consts) == 0:
            return (False, expr)
        load_position = (
            pos if (pos := self._offset_to_index.get(ins.offset, _MISSING)) is not _MISSING else -1
        )
        value_index = (
            idx
            if ins.op_name == "LOAD_CONST"
            and type(ins.op_arg) is int
            and ((idx := ins.op_arg) is not None)
            else -1
        )
        out_of_order = self.is_out_of_order_constant(value_index, load_position)
        if out_of_order:
            if (
                not self._folded_operand_used
                and (not 0 in self.used_const_indices())
                and (not consts[0] is None)
                and (
                    (
                        _parsed_1 := StructuralDecompiler.try_build_fold_with_operand(
                            consts[0], ins.arg_val
                        )
                    ),
                    (operand_fold := _parsed_1[1]),
                    _parsed_1[0],
                )[-1]
                and StructuralDecompiler.is_delayed_fold_expr(operand_fold)
            ):
                expr = operand_fold
                self._folded_operand_used = True
                return (True, expr)
            return (
                (
                    (_parsed_2 := StructuralDecompiler.try_build_delayed_fold(ins.arg_val)),
                    (expr := _parsed_2[1]),
                    _parsed_2[0],
                )[-1],
                expr,
            )
        if (
            not self._folded_operand_used
            and (not 0 in self.used_const_indices())
            and (not consts[0] is None)
            and (
                (
                    _parsed_3 := StructuralDecompiler.try_build_fold_with_operand(
                        consts[0], ins.arg_val
                    )
                ),
                (expr := _parsed_3[1]),
                _parsed_3[0],
            )[-1]
        ):
            self._folded_operand_used = True
            return (True, expr)
        if (
            StructuralDecompiler.is_integer_value(ins.arg_val)
            and StructuralDecompiler.to_big_integer(ins.arg_val) == 2
        ):
            i = 0
            while i < len(consts):
                if (
                    not (type(consts[i]) is bool and (bool_constant := consts[i]) is not None)
                    or not bool_constant
                ):
                    i += 1
                    continue
                if i in self._late_fold_constants_used:
                    i += 1
                    continue
                if not self.is_constant_inserted_late(i):
                    i += 1
                    continue
                expr = BinaryExpr(ConstantExpr(True), "+", ConstantExpr(True))
                _add(self._late_fold_constants_used, i)
                return (True, expr)
                i += 1
        return (False, expr)

    @staticmethod
    def try_build_delayed_fold(value):
        expr = None
        expr = None
        if StructuralDecompiler.is_integer_value(value):
            v = StructuralDecompiler.to_big_integer(value)
            if v > -1000000000 and v < 1000000000:
                return (False, expr)
            if (
                (_parsed_1 := StructuralDecompiler.try_log2_big(v)),
                (shift := _parsed_1[1]),
                _parsed_1[0],
            )[-1]:
                expr = BinaryExpr(ConstantExpr(1), "<<", ConstantExpr(shift))
                return (True, expr)
            b = 10
            while b >= 2:
                if (
                    (_parsed_2 := StructuralDecompiler.try_perfect_power(v, b)),
                    (exponent := _parsed_2[1]),
                    _parsed_2[0],
                )[-1]:
                    expr = BinaryExpr(ConstantExpr(b), "**", ConstantExpr(exponent))
                    return (True, expr)
                b -= 1
            expr = BinaryExpr(ConstantExpr(1), "+", StructuralDecompiler.integer_constant(v - 1))
            return (True, expr)
        if StructuralDecompiler.is_float_value(value):
            return (
                (
                    (_parsed_3 := StructuralDecompiler.try_build_delayed_float_fold(float(value))),
                    (expr := _parsed_3[1]),
                    _parsed_3[0],
                )[-1],
                expr,
            )
        return (False, expr)

    @staticmethod
    def try_build_delayed_float_fold(value):
        expr = None
        expr = None
        if not math.isfinite(value):
            return (False, expr)
        factors = [3.0, 2.0, 4.0, 0.5, 5.0, 7.0, 10.0, 0.25, 1.5, 2.5]
        for divisor in _iter_values(factors):
            if divisor == 0:
                continue
            dividend = value * divisor
            if math.isfinite(dividend) and dividend / divisor == value:
                expr = BinaryExpr(ConstantExpr(dividend), "/", ConstantExpr(divisor))
                return (True, expr)
        for factor in _iter_values(factors):
            if not math.isfinite(factor) or factor == 0 or factor == 1.0:
                continue
            other = value / factor
            if math.isfinite(other) and other != 1.0 and (factor * other == value):
                expr = BinaryExpr(ConstantExpr(factor), "*", ConstantExpr(other))
                return (True, expr)
        for addend in _iter_values(factors):
            other = value - addend
            if math.isfinite(other) and other != 0.0 and (addend + other == value):
                expr = BinaryExpr(ConstantExpr(addend), "+", ConstantExpr(other))
                return (True, expr)
        return (False, expr)

    @staticmethod
    def is_delayed_fold_expr(expr):
        if isinstance(expr, BinaryExpr) and _item_field(expr, "op", 0) in ("<<", "**"):
            return True
        if (
            isinstance(expr, BinaryExpr)
            and _item_field(expr, "op", 0) in ("+", "-")
            and ((b := expr) is not None)
        ):
            return not StructuralDecompiler.is_constant_zero(b.left) and (
                not StructuralDecompiler.is_constant_zero(_item_field(b, "right", 1))
            )
        if (
            isinstance(expr, BinaryExpr)
            and _item_field(expr, "op", 0) == "*"
            and ((b := expr) is not None)
        ):
            return (
                not StructuralDecompiler.is_constant_zero(b.left)
                and (not StructuralDecompiler.is_constant_zero(_item_field(b, "right", 1)))
                and (not StructuralDecompiler.is_constant_one(b.left))
                and (not StructuralDecompiler.is_constant_one(_item_field(b, "right", 1)))
            )
        if (
            isinstance(expr, BinaryExpr)
            and _item_field(expr, "op", 0) == "/"
            and ((b := expr) is not None)
        ):
            return not StructuralDecompiler.is_constant_zero(_item_field(b, "right", 1)) and (
                not StructuralDecompiler.is_constant_one(_item_field(b, "right", 1))
            )
        return False

    @staticmethod
    def is_constant_zero(expr):
        return isinstance(
            (constant := expr), ConstantExpr
        ) and StructuralDecompiler.is_numeric_zero(_item_field(constant, "value", 1))

    @staticmethod
    def is_constant_one(expr):
        return isinstance((constant := expr), ConstantExpr) and StructuralDecompiler.is_numeric_one(
            _item_field(constant, "value", 1)
        )

    @staticmethod
    def is_numeric_zero(value):
        if type(value) is int and (l := value) is not None:
            return l == 0
        if type(value) is int and (i := value) is not None:
            return i == 0
        if type(value) is int and (ui := value) is not None:
            return ui == 0
        if type(value) is int and (ul := value) is not None:
            return ul == 0
        if type(value) is int and (s := value) is not None:
            return s == 0
        if type(value) is int and (us := value) is not None:
            return us == 0
        if type(value) is int and (b := value) is not None:
            return b == 0
        if type(value) is int and (sb := value) is not None:
            return sb == 0
        if type(value) is int and (bi := value) is not None:
            return bi == 0
        if type(value) is float and (d := value) is not None:
            return d == 0.0
        if type(value) is float and (f := value) is not None:
            return f == 0.0
        if type(value) is complex and (c := value) is not None:
            return c == 0j
        return False

    @staticmethod
    def is_numeric_one(value):
        if type(value) is int and (l := value) is not None:
            return l == 1
        if type(value) is int and (i := value) is not None:
            return i == 1
        if type(value) is int and (ui := value) is not None:
            return ui == 1
        if type(value) is int and (ul := value) is not None:
            return ul == 1
        if type(value) is int and (s := value) is not None:
            return s == 1
        if type(value) is int and (us := value) is not None:
            return us == 1
        if type(value) is int and (b := value) is not None:
            return b == 1
        if type(value) is int and (sb := value) is not None:
            return sb == 1
        if type(value) is int and (bi := value) is not None:
            return bi == 1
        if type(value) is float and (d := value) is not None:
            return d == 1.0
        if type(value) is float and (f := value) is not None:
            return f == 1.0
        if type(value) is complex and (c := value) is not None:
            return c == 1 + 0j
        return False

    @staticmethod
    def try_build_fold_with_operand(operand, value):
        expr = None
        expr = None
        operand_expr = ConstantExpr(operand)
        if type(operand) is bool and (b := operand) is not None:
            if b:
                if StructuralDecompiler.is_integer_value(value):
                    v = StructuralDecompiler.to_big_integer(value)
                    if v == 2:
                        expr = BinaryExpr(ConstantExpr(True), "+", ConstantExpr(True))
                        return (True, expr)
                    expr = BinaryExpr(operand_expr, "*", StructuralDecompiler.integer_constant(v))
                    return (True, expr)
                if StructuralDecompiler.is_float_value(
                    value
                ) or StructuralDecompiler.is_complex_value(value):
                    expr = BinaryExpr(operand_expr, "*", ConstantExpr(value))
                    return (True, expr)
            elif (
                StructuralDecompiler.is_integer_value(value)
                and StructuralDecompiler.to_big_integer(value) == 0
            ):
                expr = BinaryExpr(operand_expr, "*", ConstantExpr(0))
                return (True, expr)
            return (False, expr)
        if StructuralDecompiler.is_integer_value(operand):
            a = StructuralDecompiler.to_big_integer(operand)
            if StructuralDecompiler.is_integer_value(value):
                v = StructuralDecompiler.to_big_integer(value)
                if (
                    a == 1
                    and (
                        (_parsed_1 := StructuralDecompiler.try_log2_big(v)),
                        (shift := _parsed_1[1]),
                        _parsed_1[0],
                    )[-1]
                ):
                    expr = BinaryExpr(ConstantExpr(1), "<<", ConstantExpr(shift))
                    return (True, expr)
                if (
                    a >= 2
                    and (
                        (_parsed_2 := StructuralDecompiler.try_perfect_power(v, a)),
                        (exponent := _parsed_2[1]),
                        _parsed_2[0],
                    )[-1]
                ):
                    expr = BinaryExpr(operand_expr, "**", ConstantExpr(exponent))
                    return (True, expr)
                if a != 0 and v % a == 0:
                    expr = BinaryExpr(
                        operand_expr, "*", StructuralDecompiler.integer_constant(v // a)
                    )
                    return (True, expr)
                expr = BinaryExpr(operand_expr, "+", StructuralDecompiler.integer_constant(v - a))
                return (True, expr)
            if StructuralDecompiler.is_float_value(value):
                av = float(a)
                if (
                    (
                        _parsed_3 := StructuralDecompiler.try_build_float_fold(
                            operand_expr, av, float(value)
                        )
                    ),
                    (expr := _parsed_3[1]),
                    _parsed_3[0],
                )[-1]:
                    return (True, expr)
            if StructuralDecompiler.is_complex_value(value):
                av = complex(float(a), 0)
                if (
                    (
                        _parsed_4 := StructuralDecompiler.try_build_complex_fold(
                            operand_expr, av, value
                        )
                    ),
                    (expr := _parsed_4[1]),
                    _parsed_4[0],
                )[-1]:
                    return (True, expr)
            return (False, expr)
        if StructuralDecompiler.is_float_value(operand) and StructuralDecompiler.is_float_value(
            value
        ):
            if (
                (
                    _parsed_5 := StructuralDecompiler.try_build_float_fold(
                        operand_expr, float(operand), float(value)
                    )
                ),
                (expr := _parsed_5[1]),
                _parsed_5[0],
            )[-1]:
                return (True, expr)
            return (False, expr)
        if StructuralDecompiler.is_complex_value(operand) and StructuralDecompiler.is_complex_value(
            value
        ):
            if (
                (
                    _parsed_6 := StructuralDecompiler.try_build_complex_fold(
                        operand_expr, operand, value
                    )
                ),
                (expr := _parsed_6[1]),
                _parsed_6[0],
            )[-1]:
                return (True, expr)
            return (False, expr)
        return (False, expr)

    @staticmethod
    def try_build_float_fold(operand_expr, a, v):
        expr = None
        expr = None
        if a != 0 and math.isfinite(a):
            scaled = v / a
            if a * scaled == v:
                expr = BinaryExpr(operand_expr, "*", ConstantExpr(scaled))
                return (True, expr)
            one = a / a
            if one == 1.0 and 1.0 * v == v:
                expr = BinaryExpr(
                    BinaryExpr(operand_expr, "/", ConstantExpr(a)), "*", ConstantExpr(v)
                )
                return (True, expr)
        delta = v - a
        if a + delta == v:
            expr = BinaryExpr(operand_expr, "+", ConstantExpr(delta))
            return (True, expr)
        back = a - v
        if a - back == v:
            expr = BinaryExpr(operand_expr, "-", ConstantExpr(back))
            return (True, expr)
        return (False, expr)

    @staticmethod
    def try_build_complex_fold(operand_expr, a, v):
        expr = None
        expr = None
        if a != 0j and math.isfinite(a.real) and math.isfinite(a.imag):
            scaled = v / a
            if a * scaled == v:
                expr = BinaryExpr(operand_expr, "*", ConstantExpr(scaled))
                return (True, expr)
            one = a / a
            if one == 1 + 0j and (1 + 0j) * v == v:
                expr = BinaryExpr(
                    BinaryExpr(operand_expr, "/", ConstantExpr(a)), "*", ConstantExpr(v)
                )
                return (True, expr)
        delta = v - a
        if a + delta == v:
            expr = BinaryExpr(operand_expr, "+", ConstantExpr(delta))
            return (True, expr)
        back = a - v
        if a - back == v:
            expr = BinaryExpr(operand_expr, "-", ConstantExpr(back))
            return (True, expr)
        return (False, expr)

    @staticmethod
    def is_integer_value(value):
        return (
            type(value) is int
            or type(value) is int
            or type(value) is int
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
        )

    @staticmethod
    def is_float_value(value):
        return type(value) is float or type(value) is float

    @staticmethod
    def is_complex_value(value):
        return type(value) is complex

    @staticmethod
    def to_big_integer(value):
        if type(value) is int and (bi := value) is not None:
            return bi
        if type(value) is int and (l := value) is not None:
            return l
        if type(value) is int and (i := value) is not None:
            return i
        if type(value) is int and (ui := value) is not None:
            return ui
        if type(value) is int and (ul := value) is not None:
            return ul
        if type(value) is int and (sh := value) is not None:
            return sh
        if type(value) is int and (ush := value) is not None:
            return ush
        if type(value) is int and (by := value) is not None:
            return by
        if type(value) is int and (sb := value) is not None:
            return sb
        return 0

    @staticmethod
    def integer_constant(value):
        return (
            ConstantExpr(_cast_int(value))
            if value >= -9223372036854775808 and value <= 9223372036854775807
            else ConstantExpr(value)
        )

    @staticmethod
    def try_perfect_power(value, base_value):
        exponent = 0
        exponent = 0
        if value <= 1 or base_value <= 1:
            return (False, exponent)
        power = base_value
        e = 2
        while e <= 4096:
            power *= base_value
            if power == value:
                exponent = e
                return (True, exponent)
            if power > value:
                return (False, exponent)
            e += 1
        return (False, exponent)

    @staticmethod
    def try_log2_big(value):
        shift = 0
        shift = 0
        if value <= 0:
            return (False, shift)
        while value > 1:
            if not value % 2 == 0:
                return (False, shift)
            value >>= 1
            shift += 1
        return (value == 1, shift)

    def preserve_unreachable_else(self, statements):

        def rewrite(list_):
            nonlocal constant_cursor
            i = 0
            while i < len(list_):
                stmt = list_[i]
                if isinstance((try_stmt := stmt), TryStmt):
                    handlers = try_stmt.handlers
                    h = 0
                    while h < len(handlers):
                        if (
                            (_subject_1 := handlers[h].body),
                            isinstance((body_list := _subject_1), list),
                        )[-1]:
                            rewrite(body_list)
                        h += 1
                    if isinstance((else_list := try_stmt.else_body), list):
                        rewrite(else_list)
                    if isinstance((finally_list := try_stmt.finally_body), list):
                        rewrite(finally_list)
                    if (
                        len(try_stmt.else_body) == 0
                        and try_stmt.finally_body is not None
                        and (len(handlers) > 0)
                        and (len(try_stmt.body) > 0)
                        and (
                            (_subject_2 := try_stmt.body[-1]),
                            isinstance(_subject_2, RaiseStmt)
                            or isinstance(_subject_2, ReturnStmt)
                            or isinstance(_subject_2, BreakStmt)
                            or isinstance(_subject_2, ContinueStmt),
                        )[-1]
                    ):
                        is_star = _any(handlers, lambda h: h.is_star)
                        if not is_star and (not self.is310_or311()):
                            i += 1
                            continue
                        constants = unused_constants
                        names = unused_names if is_star else []
                        if len(constants) > 0 and callee is None:
                            i += 1
                            continue
                        dead_else = []
                        for name in _iter_values(names):
                            _add(dead_else, ExprStmt(NameExpr(name)))
                        if constant_cursor < len(
                            constants
                        ) and StructuralDecompiler.contains_call_with_constant(
                            try_stmt.finally_body
                        ):
                            value = constants[(constant_cursor := (constant_cursor + 1)) - 1]
                            _add(
                                dead_else,
                                ExprStmt(CallExpr(NameExpr(callee), [ConstantExpr(value)], [])),
                            )
                        if len(dead_else) > 0:
                            list_[i] = replace(try_stmt, else_body=dead_else)
                    if isinstance((try_list := try_stmt.body), list):
                        rewrite(try_list)
                elif isinstance((with_stmt := stmt), WithStmt) and isinstance(
                    (with_body := with_stmt.body), list
                ):
                    rewrite(with_body)
                elif isinstance((if_stmt := stmt), IfStmt):
                    rewrite(list(if_stmt.then_body))
                    rewrite(list(if_stmt.else_body))
                elif isinstance((for_stmt := stmt), ForStmt):
                    rewrite(list(for_stmt.body))
                    rewrite(list(for_stmt.else_body))
                elif isinstance((while_stmt := stmt), WhileStmt):
                    rewrite(list(while_stmt.body))
                    rewrite(list(while_stmt.else_body))
                i += 1

        used_constants = self.used_const_indices()
        unused_constants = list(
            _select(
                _where(
                    _select_indexed(self._code.constants, lambda value, index: (value, index)),
                    lambda x: (
                        not _item_field(x, "index", 1) in used_constants
                        and StructuralDecompiler.is_preservable_dead_constant(
                            _item_field(x, "value", 0)
                        )
                    ),
                ),
                lambda x: _item_field(x, "value", 0),
            )
        )
        used_names = self.used_name_set()
        unused_names = list(
            _where(
                self._code.names,
                lambda name: (
                    not name in used_names and StructuralDecompiler.is_valid_python_identifier(name)
                ),
            )
        )
        if len(unused_constants) == 0 and len(unused_names) == 0:
            return
        callee = (
            _value_1
            if (_value_1 := _first(unused_names)) is not None
            else self.find_dead_else_callee()
        )
        if len(unused_constants) > 0 and callee is None:
            return
        constant_cursor = 0
        rewrite(statements)
        return

    def used_name_set(self):

        def add_name(set_, value):
            if value.startswith("NULL|self + "):
                value = value[12:]
            elif value.startswith("NULL + "):
                value = value[7:]
            if value.endswith(" + NULL|self"):
                value = value[:-12]
            elif value.endswith(" + NULL"):
                value = value[:-7]
            if not _empty_string(value):
                _add(set_, value)

        used = set()
        for ins in _iter_values(self._instrs):
            if type(ins.arg_val) is str and (value := ins.arg_val) is not None:
                add_name(used, value)
            if type(ins.arg_repr) is str and (repr_ := ins.arg_repr) is not None:
                add_name(used, repr_)
        return used

    def find_dead_else_callee(self):
        for name in _iter_values(self._code.names):
            if StructuralDecompiler.is_valid_python_identifier(name):
                return name
        for name in _iter_values(self._code.var_names):
            if StructuralDecompiler.is_valid_python_identifier(name):
                return name
        return None

    @staticmethod
    def is_preservable_dead_constant(value):
        if value is None:
            return False
        if type(value) is str and (s := value) is not None:
            return not s.startswith("<code:")
        if type(value) is bool:
            return True
        if (
            type(value) is int
            or type(value) is int
            or type(value) is int
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
            or (type(value) is int)
        ):
            return True
        if type(value) is int:
            return True
        if type(value) is float or type(value) is float:
            return True
        if type(value) is complex:
            return True
        return False

    @staticmethod
    def contains_call_with_constant(statements):
        for stmt in _iter_values(statements):
            if StructuralDecompiler.statement_contains_call_with_constant(stmt):
                return True
        return False

    @staticmethod
    def statement_contains_call_with_constant(stmt):
        if isinstance((e := stmt), ExprStmt):
            return StructuralDecompiler.expr_contains_call_with_constant(e.expression)
        if isinstance((a := stmt), AssignStmt):
            return StructuralDecompiler.expr_contains_call_with_constant(
                _item_field(a, "value", 1)
            ) or _any(a.targets, StructuralDecompiler.expr_contains_call_with_constant)
        if isinstance((a := stmt), AugAssignStmt):
            return StructuralDecompiler.expr_contains_call_with_constant(
                _item_field(a, "value", 1)
            ) or StructuralDecompiler.expr_contains_call_with_constant(a.target)
        if isinstance((r := stmt), ReturnStmt):
            return _item_field(
                r, "value", 1
            ) is not None and StructuralDecompiler.expr_contains_call_with_constant(
                _item_field(r, "value", 1)
            )
        if isinstance((i := stmt), IfStmt):
            return StructuralDecompiler.contains_call_with_constant(
                i.then_body
            ) or StructuralDecompiler.contains_call_with_constant(i.else_body)
        if isinstance((f := stmt), ForStmt):
            return StructuralDecompiler.contains_call_with_constant(
                f.body
            ) or StructuralDecompiler.contains_call_with_constant(f.else_body)
        if isinstance((w := stmt), WhileStmt):
            return StructuralDecompiler.contains_call_with_constant(
                w.body
            ) or StructuralDecompiler.contains_call_with_constant(w.else_body)
        if isinstance((w := stmt), WithStmt):
            return _any(
                w.items,
                lambda item: StructuralDecompiler.expr_contains_call_with_constant(
                    item.context_expr
                ),
            ) or StructuralDecompiler.contains_call_with_constant(w.body)
        if isinstance((t := stmt), TryStmt):
            return (
                StructuralDecompiler.contains_call_with_constant(t.body)
                or _any(
                    t.handlers, lambda h: StructuralDecompiler.contains_call_with_constant(h.body)
                )
                or StructuralDecompiler.contains_call_with_constant(t.else_body)
                or (
                    t.finally_body is not None
                    and StructuralDecompiler.contains_call_with_constant(t.finally_body)
                )
            )
        return False

    @staticmethod
    def expr_contains_call_with_constant(expr):
        if isinstance((c := expr), CallExpr):
            return (
                _any(c.arguments, lambda arg: isinstance(arg, ConstantExpr))
                or _any(
                    c.keyword_arguments,
                    lambda kw: isinstance(_item_field(kw, "value", 1), ConstantExpr),
                )
                or StructuralDecompiler.expr_contains_call_with_constant(c.function)
                or _any(c.arguments, StructuralDecompiler.expr_contains_call_with_constant)
            )
        if isinstance((b := expr), BinaryExpr):
            return StructuralDecompiler.expr_contains_call_with_constant(
                b.left
            ) or StructuralDecompiler.expr_contains_call_with_constant(_item_field(b, "right", 1))
        if isinstance((u := expr), UnaryExpr):
            return StructuralDecompiler.expr_contains_call_with_constant(u.operand)
        if isinstance((a := expr), AttributeExpr):
            return StructuralDecompiler.expr_contains_call_with_constant(a.target)
        if isinstance((s := expr), SubscriptExpr):
            return StructuralDecompiler.expr_contains_call_with_constant(
                s.target
            ) or StructuralDecompiler.expr_contains_call_with_constant(_item_field(s, "index", 0))
        if isinstance((t := expr), TupleExpr):
            return _any(t.elements, StructuralDecompiler.expr_contains_call_with_constant)
        if isinstance((l := expr), ListExpr):
            return _any(l.elements, StructuralDecompiler.expr_contains_call_with_constant)
        if isinstance((d := expr), DictExpr):
            return _any(
                d.entries,
                lambda e: (
                    StructuralDecompiler.expr_contains_call_with_constant(
                        _item_field(e, "value", 1)
                    )
                    or (
                        _item_field(e, "key", 0) is not None
                        and StructuralDecompiler.expr_contains_call_with_constant(
                            _item_field(e, "key", 0)
                        )
                    )
                ),
            )
        return False

    @staticmethod
    def is_valid_python_identifier(name):
        if _empty_string(name):
            return False
        if name.startswith("__pycrune_ph_"):
            return False
        first = name[0]
        if not (str.isalpha(first) or first == "_"):
            return False
        i = 1
        while i < len(name):
            c = name[i]
            if not (str.isalnum(c) or c == "_"):
                return False
            i += 1
        return not name in StructuralDecompiler.python_keywords

    @staticmethod
    def remove_inlined_finally_copies(body, finally_body):
        if len(finally_body) == 0:
            return list(body)
        finally_text = PythonEmitter.emit(finally_body)
        return StructuralDecompiler.process_branch(body, finally_body, finally_text)

    @staticmethod
    def process_branch(branch, finally_body, finally_text):
        result = list(branch)
        i = 0
        while i < len(result):
            result[i] = StructuralDecompiler.process_statement(
                result[i], finally_body, finally_text
            )
            i += 1
        a = 0
        while a < len(result):
            if not (
                isinstance(result[a], ReturnStmt)
                or isinstance(result[a], RaiseStmt)
                or isinstance(result[a], BreakStmt)
                or isinstance(result[a], ContinueStmt)
            ):
                a += 1
                continue
            count = len(finally_body)
            if a < count:
                a += 1
                continue
            start = a - count
            candidate = _get_range(result, start, count)
            if PythonEmitter.emit(candidate) != finally_text:
                a += 1
                continue
            _remove_range(result, start, count)
            a -= count
            a += 1
        return result

    @staticmethod
    def process_statement(statement, finally_body, finally_text):
        if isinstance((s := statement), IfStmt):
            return replace(
                s,
                then_body=StructuralDecompiler.process_branch(
                    s.then_body, finally_body, finally_text
                ),
                else_body=StructuralDecompiler.process_branch(
                    s.else_body, finally_body, finally_text
                ),
            )
        if isinstance((s := statement), ForStmt):
            return replace(
                s,
                body=StructuralDecompiler.process_branch(s.body, finally_body, finally_text),
                else_body=StructuralDecompiler.process_branch(
                    s.else_body, finally_body, finally_text
                ),
            )
        if isinstance((s := statement), WhileStmt):
            return replace(
                s,
                body=StructuralDecompiler.process_branch(s.body, finally_body, finally_text),
                else_body=StructuralDecompiler.process_branch(
                    s.else_body, finally_body, finally_text
                ),
            )
        if isinstance((s := statement), WithStmt):
            return replace(
                s, body=StructuralDecompiler.process_branch(s.body, finally_body, finally_text)
            )
        if isinstance((s := statement), TryStmt):
            return replace(
                s,
                body=StructuralDecompiler.process_branch(s.body, finally_body, finally_text),
                handlers=list(
                    _select(
                        s.handlers,
                        lambda h: replace(
                            h,
                            body=StructuralDecompiler.process_branch(
                                h.body, finally_body, finally_text
                            ),
                        ),
                    )
                ),
                else_body=StructuralDecompiler.process_branch(
                    s.else_body, finally_body, finally_text
                ),
            )
        return statement

    def is_at_least314(self):
        version = self._code.python_version
        if _empty_string(version):
            return False
        parts = _split(version, ".")
        return (
            len(parts) >= 2
            and (major := _parse_int(parts[0])) is not None
            and ((minor := _parse_int(parts[1])) is not None)
            and (major > 3 or (major == 3 and minor >= 14))
        )

    def is_followed_only_by_handler_code(self, ins):
        next_handler_offset = min(
            _iter_values(
                _default_if_empty(
                    _where(
                        _select(self._code.exception_table, lambda entry: entry.target),
                        lambda target: target > ins.offset,
                    ),
                    2147483647,
                )
            )
        )
        if not (index := self._offset_to_index.get(ins.offset, _MISSING)) is not _MISSING:
            return False
        i = index + 1
        while i < len(self._instrs):
            if self._instrs[i].offset >= next_handler_offset:
                break
            if self._instrs[i].op_code in (OpCodes.NOP, OpCodes.EXTENDED_ARG, OpCodes.NOT_TAKEN):
                i += 1
                continue
            return False
            i += 1
        return True

    def is_targeted_by_conditional_jump(self, offset):
        for ins in _iter_values(self._instrs):
            if ins.target_offset != offset:
                continue
            if ins.op_code in (
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_FALSE,
                OpCodes.POP_JUMP_IF_NONE,
                OpCodes.POP_JUMP_IF_NOT_NONE,
            ):
                return True
        return False

    def is_pre312(self):
        version = self._code.python_version
        if _empty_string(version):
            return False
        parts = _split(version, ".")
        return (
            len(parts) >= 2
            and (major := _parse_int(parts[0])) is not None
            and ((minor := _parse_int(parts[1])) is not None)
            and (major < 3 or (major == 3 and minor < 12))
        )

    def try_decompile_short_circuit_bool_op(self, idx, end_idx, stack):
        if idx + 2 >= len(self._instrs):
            return (False, idx)
        ins0 = self._instrs[idx]
        if (
            self.is_pre312()
            and ins0.op_code in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
            and (len(stack) > 0)
        ):
            bare_target_offset = self.resolve_jump_target(
                ins0.target_offset if ins0.target_offset is not None else 0
            )
            if (
                bare_target_idx := self._offset_to_index.get(bare_target_offset, _MISSING)
            ) is not _MISSING and bare_target_idx > idx + 1:
                nested_copy = -1
                q = idx + 1
                while q + 1 < bare_target_idx:
                    if (
                        self._instrs[q].op_code != OpCodes.COPY
                        or (_value_1 if (_value_1 := self._instrs[q].op_arg) is not None else 0)
                        != 1
                    ):
                        q += 1
                        continue
                    qj = q + 1
                    while qj < bare_target_idx and self._instrs[qj].op_code in (
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        qj += 1
                    if qj < bare_target_idx and self._instrs[qj].op_code in (
                        OpCodes.POP_JUMP_IF_TRUE,
                        OpCodes.POP_JUMP_IF_FALSE,
                    ):
                        nested_copy = q
                        break
                    q += 1
                if nested_copy > idx + 1:
                    bare_left = stack.pop()
                    bare_right_stack = SymbolicStack()
                    self.decompile_range(idx + 1, nested_copy, bare_right_stack)
                    if len(bare_right_stack) > 0:
                        bare_right = bare_right_stack.pop()
                        bare_op = "and" if ins0.op_code == OpCodes.POP_JUMP_IF_FALSE else "or"
                        stack.push(BoolOpExpr(bare_op, [bare_left, bare_right]))
                        idx = nested_copy - 1
                        return (True, idx)
        jump_pos = idx + 1
        while jump_pos < len(self._instrs) and self._instrs[jump_pos].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            jump_pos += 1
        if jump_pos + 1 >= len(self._instrs):
            return (False, idx)
        pop_pos = jump_pos + 1
        while pop_pos < len(self._instrs) and self._instrs[pop_pos].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            pop_pos += 1
        if pop_pos >= len(self._instrs):
            return (False, idx)
        ins1 = self._instrs[jump_pos]
        ins2 = self._instrs[pop_pos]
        if (
            ins0.op_code == OpCodes.COPY
            and (ins0.op_arg if ins0.op_arg is not None else 0) == 1
            and (ins1.op_code in (OpCodes.POP_JUMP_IF_TRUE, OpCodes.POP_JUMP_IF_FALSE))
            and (ins2.op_code == OpCodes.POP_TOP)
        ):
            if len(stack) == 0:
                return (False, idx)
            op = "or" if ins1.op_code == OpCodes.POP_JUMP_IF_TRUE else "and"
            left = stack.pop()
            target_offset = self.resolve_jump_target(
                ins1.target_offset if ins1.target_offset is not None else 0
            )
            target_idx = self._offset_to_index.get(target_offset)
            if target_idx is None:
                return (False, idx)
            right_end = target_idx
            q = pop_pos + 1
            while q + 1 < target_idx:
                if (
                    self._instrs[q].op_code == OpCodes.COPY
                    and (_value_6 if (_value_6 := self._instrs[q].op_arg) is not None else 0) == 1
                ):
                    qj = q + 1
                    while qj < target_idx and self._instrs[qj].op_code in (
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        qj += 1
                    if (
                        qj < target_idx
                        and self._instrs[qj].op_code
                        in (OpCodes.POP_JUMP_IF_TRUE, OpCodes.POP_JUMP_IF_FALSE)
                        and (
                            (
                                _value_9
                                if (_value_9 := self._instrs[qj].target_offset) is not None
                                else 0
                            )
                            > (ins1.target_offset if ins1.target_offset is not None else 0)
                        )
                    ):
                        right_end = q
                        break
                q += 1
            if target_idx > end_idx:
                if (
                    end_idx >= len(self._instrs)
                    or self._instrs[end_idx].op_code != OpCodes.JUMP_FORWARD
                    or self.resolve_jump_target(
                        _value_10
                        if (_value_10 := self._instrs[end_idx].target_offset) is not None
                        else 0
                    )
                    != target_offset
                ):
                    return (False, idx)
                right_end = end_idx
            right_stack = SymbolicStack()
            self.decompile_range(pop_pos + 1, right_end, right_stack)
            if len(right_stack) > 0:
                right = right_stack.pop()
                values = []
                if isinstance((b_left := left), BoolOpExpr) and _item_field(b_left, "op", 0) == op:
                    values.extend(b_left.values)
                else:
                    _add(values, left)
                if (
                    isinstance((b_right := right), BoolOpExpr)
                    and _item_field(b_right, "op", 0) == op
                ):
                    values.extend(b_right.values)
                else:
                    _add(values, right)
                stack.push(BoolOpExpr(op, values))
                idx = right_end - 1
                return (True, idx)
        return (False, idx)

    def try_reduce_boolean_logic(self, idx, end_idx, stack):
        ins = self._instrs[idx]
        if not (
            (_parsed_1 := self.try_get_branch_info(ins, stack)),
            (cond1 := _parsed_1[1]),
            (jump_if_true := _parsed_1[2]),
            (target := _parsed_1[3]),
            _parsed_1[0],
        )[-1]:
            return (False, idx)
        if jump_if_true:
            if (
                (or_target_idx := self._offset_to_index.get(target, _MISSING)) is not _MISSING
                and or_target_idx > idx + 1
                and (or_target_idx <= end_idx)
            ):
                prev_ins = self._instrs[or_target_idx - 1]
                if prev_ins.op_code in (
                    OpCodes.POP_JUMP_IF_FALSE,
                    OpCodes.POP_JUMP_IF_NONE,
                    OpCodes.POP_JUMP_IF_NOT_NONE,
                ):
                    try:
                        inner_stack = stack.clone()
                        self.decompile_range(idx + 1, or_target_idx - 1, inner_stack)
                        if len(inner_stack) > len(stack):
                            cond2 = inner_stack.pop()
                            stack.push(BoolOpExpr("or", [cond1, cond2]))
                            idx = or_target_idx - 1
                            return (True, idx)
                    except Exception as ex:
                        if not not isinstance(ex, FailFastException):
                            raise
                        pass
        elif (
            false_target_idx := self._offset_to_index.get(target, _MISSING)
        ) is not _MISSING and false_target_idx <= end_idx:
            next_jump_idx = idx + 1
            while next_jump_idx < false_target_idx and (
                not OpCodes.is_jump(self._instrs[next_jump_idx].op_code)
            ):
                if OpCodes.is_terminator(self._instrs[next_jump_idx].op_code):
                    break
                next_jump_idx += 1
            if (
                next_jump_idx < false_target_idx
                and self._instrs[next_jump_idx].target_offset == target
            ):
                try:
                    inner_stack = stack.clone()
                    self.decompile_range(idx + 1, next_jump_idx, inner_stack)
                    if len(inner_stack) > len(stack):
                        cond2 = inner_stack.pop()
                        stack.push(BoolOpExpr("and", [cond1, cond2]))
                        idx = next_jump_idx
                        return (True, idx)
                except Exception as ex:
                    if not not isinstance(ex, FailFastException):
                        raise
                    pass
        stack.push(cond1)
        return (False, idx)

    def try_match_break_path(self, start_idx, end_idx):
        marker_idx = 0
        exit_start = 0
        exit_len = 0
        marker_idx = -1
        exit_start = -1
        exit_len = 0
        if len(self._active_loop_headers) == 0:
            return (False, marker_idx, exit_start, exit_len)
        if start_idx < 0 or end_idx > len(self._instrs) or start_idx >= end_idx:
            return (False, marker_idx, exit_start, exit_len)
        term = -1
        i = end_idx - 1
        while i >= start_idx:
            if self._instrs[i].op_code in (
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.RAISE_VARARGS,
            ):
                term = i
                break
            if self._instrs[i].op_code != OpCodes.NOP:
                break
            i -= 1
        if term < 0:
            return (False, marker_idx, exit_start, exit_len)
        m = term
        while m - 1 >= start_idx and self._instrs[m - 1].op_code in (
            OpCodes.LOAD_FAST,
            OpCodes.LOAD_CONST,
            OpCodes.LOAD_GLOBAL,
            OpCodes.LOAD_NAME,
            OpCodes.LOAD_DEREF,
            OpCodes.LOAD_ATTR,
            OpCodes.BUILD_TUPLE,
            OpCodes.BUILD_LIST,
            OpCodes.BUILD_MAP,
            OpCodes.BUILD_STRING,
            OpCodes.FORMAT_VALUE,
            OpCodes.RETURN_VALUE,
        ):
            m -= 1
        if m - 1 < start_idx:
            return (False, marker_idx, exit_start, exit_len)
        marker = self._instrs[m - 1].op_code
        if not marker in (OpCodes.POP_TOP, OpCodes.NOP):
            return (False, marker_idx, exit_start, exit_len)
        i = start_idx
        while i < m - 1:
            if self._instrs[i].op_code == OpCodes.NOP:
                return (False, marker_idx, exit_start, exit_len)
            i += 1
        if (
            marker == OpCodes.POP_TOP
            and m - 2 >= 0
            and (self._instrs[m - 1].starts_line is not None)
            and (self._instrs[m - 2].starts_line == self._instrs[m - 1].starts_line)
        ):
            return (False, marker_idx, exit_start, exit_len)
        len_ = term - m + 1
        s = end_idx
        while s + len_ <= len(self._instrs):
            if not self.matches_instruction_sequence(m, s, len_):
                s += 1
                continue
            marker_idx = m - 1
            exit_start = m
            exit_len = len_
            self._pending_break_exit_start = m
            self._pending_break_exit_length = len_
            return (True, marker_idx, exit_start, exit_len)
            s += 1
        return (False, marker_idx, exit_start, exit_len)

    def try_decompile_duplicated_tail_conditional(self, idx, end_idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if idx + 1 >= len(self._instrs):
            return (False, next_idx)
        if self.is_pre312():
            return (False, next_idx)
        ins = self._instrs[idx]
        if not ins.op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
        ):
            return (False, next_idx)
        if len(stack) == 0:
            return (False, next_idx)
        base_stack = stack.clone()
        if not (
            (_parsed_1 := self.try_get_branch_info(ins, base_stack)),
            (condition := _parsed_1[1]),
            (jump_if_true := _parsed_1[2]),
            (target_offset := _parsed_1[3]),
            _parsed_1[0],
        )[-1]:
            return (False, next_idx)
        fall_offset = self._instrs[idx + 1].offset
        t0 = target_offset if jump_if_true else fall_offset
        f0 = fall_offset if jump_if_true else target_offset
        then_start_offset = min(t0, f0)
        else_start_offset = max(t0, f0)
        if then_start_offset == else_start_offset:
            return (False, next_idx)
        if (
            not (then_start_idx := self._offset_to_index.get(then_start_offset, _MISSING))
            is not _MISSING
        ):
            return (False, next_idx)
        if (
            not (else_start_idx := self._offset_to_index.get(else_start_offset, _MISSING))
            is not _MISSING
        ):
            return (False, next_idx)
        if then_start_idx <= idx or else_start_idx <= then_start_idx:
            return (False, next_idx)
        then_arm_end = else_start_idx
        else_arm_end = -1
        k = else_start_idx
        while k < end_idx and k < len(self._instrs):
            op = self._instrs[k].op_code
            if op in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST, OpCodes.RAISE_VARARGS):
                else_arm_end = k + 1
                break
            if op in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                return (False, next_idx)
            k += 1
        if else_arm_end < 0:
            return (False, next_idx)
        max_suffix = min(then_arm_end - then_start_idx, else_arm_end - else_start_idx)
        lcs = 0
        len_ = 1
        while len_ <= max_suffix:
            a = self._instrs[then_arm_end - len_]
            b = self._instrs[else_arm_end - len_]
            if a.op_code != b.op_code:
                break
            if a.target_offset is not None or b.target_offset is not None:
                break
            if not _same_value(a.arg_val, b.arg_val):
                break
            lcs = len_
            len_ += 1
        if lcs == 0:
            return (False, next_idx)
        then_prefix_end = then_arm_end - lcs
        else_prefix_end = else_arm_end - lcs
        if then_prefix_end <= then_start_idx or else_prefix_end <= else_start_idx:
            return (False, next_idx)
        base_count = len(base_stack)
        then_stack = base_stack.clone()
        else_stack = base_stack.clone()
        then_body = None
        else_body = None
        try:
            then_body = self.decompile_range(then_start_idx, then_prefix_end, then_stack)
            else_body = self.decompile_range(else_start_idx, else_prefix_end, else_stack)
        except Exception as ex:
            if not not isinstance(ex, FailFastException):
                raise
            return (False, next_idx)
        if len(then_body) != 0 or len(then_stack) != base_count + 1:
            return (False, next_idx)
        if len(else_body) != 0 or len(else_stack) != base_count + 1:
            return (False, next_idx)
        commit_stack = stack.clone()
        if not ((_parsed_2 := self.try_get_branch_info(ins, commit_stack)), _parsed_2[0])[-1]:
            return (False, next_idx)
        commit_stack.push(TernaryExpr(condition, then_stack.pop(), else_stack.pop()))
        suffix_stmts = None
        try:
            suffix_stmts = self.decompile_range(then_prefix_end, then_arm_end, commit_stack)
        except Exception as ex:
            if not not isinstance(ex, FailFastException):
                raise
            return (False, next_idx)
        ((_parsed_4 := self.try_get_branch_info(ins, stack)), _parsed_4[0])[-1]
        stack.clear()
        i = len(commit_stack) - 1
        while i >= 0:
            stack.push(commit_stack.peek(i))
            i -= 1
        stmts.extend(suffix_stmts)
        last_suffix_op = self._instrs[then_arm_end - 1].op_code
        if last_suffix_op in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST, OpCodes.RAISE_VARARGS):
            next_reachable = -1
            scan = else_arm_end
            while scan < end_idx and scan < len(self._instrs):
                if not self._instrs[scan].is_jump_target:
                    scan += 1
                    continue
                targeted = False
                k = 0
                while k < else_arm_end and k < len(self._instrs):
                    if (
                        _value_5 if (_value_5 := self._instrs[k].target_offset) is not None else -1
                    ) == self._instrs[scan].offset:
                        targeted = True
                        break
                    k += 1
                if targeted:
                    next_reachable = scan
                    break
                scan += 1
            next_idx = next_reachable if next_reachable >= 0 else end_idx
        else:
            next_idx = else_arm_end
        return (True, next_idx)

    def try_decompile_ternary_expr(self, idx, end_idx, stack):
        next_idx = 0
        next_idx = idx
        if not self.is_pre312():
            return (False, next_idx)
        if idx >= end_idx or len(stack) == 0:
            return (False, next_idx)
        ins = self._instrs[idx]
        if ins.op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, next_idx)
        if (
            not (
                type(self._instrs[idx].target_offset) is int
                and (else_target := self._instrs[idx].target_offset) is not None
            )
            or not (else_idx := self._offset_to_index.get(else_target, _MISSING)) is not _MISSING
        ):
            return (False, next_idx)
        jump_idx = -1
        p = idx + 1
        while p < else_idx:
            if self._instrs[p].op_code == OpCodes.JUMP_FORWARD:
                jump_idx = p
                break
            p += 1
        if jump_idx < 0:
            return (False, next_idx)
        if (
            not (
                type(self._instrs[jump_idx].target_offset) is int
                and (join_target := self._instrs[jump_idx].target_offset) is not None
            )
            or not (join_idx := self._offset_to_index.get(join_target, _MISSING)) is not _MISSING
        ):
            return (False, next_idx)
        if join_idx <= else_idx:
            return (False, next_idx)
        then_expr = None
        then_stack = SymbolicStack()
        try:
            self.decompile_range(idx + 1, jump_idx, then_stack)
        except Exception:
            then_stack.clear()
        if len(then_stack) == 1:
            then_expr = then_stack.pop()
        else:
            p = idx + 1
            while p + 3 < jump_idx:
                if (
                    self._instrs[p].op_code != OpCodes.COPY
                    or (_value_1 if (_value_1 := self._instrs[p].op_arg) is not None else 0) != 1
                ):
                    p += 1
                    continue
                if not self._instrs[p + 1].op_code in (
                    OpCodes.POP_JUMP_IF_FALSE,
                    OpCodes.POP_JUMP_IF_TRUE,
                ):
                    p += 1
                    continue
                if self._instrs[p + 2].op_code != OpCodes.POP_TOP:
                    p += 1
                    continue
                left_stack = SymbolicStack()
                right_stack = SymbolicStack()
                try:
                    self.decompile_range(idx + 1, p, left_stack)
                    self.decompile_range(p + 3, jump_idx, right_stack)
                except Exception:
                    break
                if len(left_stack) != 1 or len(right_stack) != 1:
                    break
                op = "and" if self._instrs[p + 1].op_code == OpCodes.POP_JUMP_IF_FALSE else "or"
                then_expr = BinaryExpr(left_stack.pop(), op, right_stack.pop())
                break
                p += 1
            if then_expr is None:
                return (False, next_idx)
        else_stack = SymbolicStack()
        try:
            self.decompile_range(else_idx, join_idx, else_stack)
        except Exception:
            return (False, next_idx)
        if len(else_stack) != 1:
            return (False, next_idx)
        cond = stack.peek()
        stack.pop()
        stack.push(TernaryExpr(cond, then_expr, else_stack.pop()))
        next_idx = join_idx
        return (True, next_idx)

    def try_decompile_conditional(self, idx, end_idx, stack):
        stmt = None
        next_idx = 0
        stmt = None
        next_idx = idx
        if os.getenv("PYCRUNE_TRACE") == "1":
            _trace(
                "[trace] EnterConditional idx={} off={} op={} stack={} end={}".format(
                    idx, self._instrs[idx].offset, self._instrs[idx].op_code, len(stack), end_idx
                )
            )
        is_chained = (
            (_parsed_1 := self.try_match_chained_branch(idx, stack)),
            (cb_cond := _parsed_1[1]),
            (cb_true := _parsed_1[2]),
            (cb_false := _parsed_1[3]),
            (cb_next_idx := _parsed_1[4]),
            (is_assert := _parsed_1[5]),
            _parsed_1[0],
        )[-1]
        if is_chained and is_assert:
            stmt = AssertStmt(cb_cond, None)
            next_idx = cb_next_idx
            return (True, stmt, next_idx)
        condition = None
        jump_if_true = False
        target_offset = 0
        t0 = 0
        f0 = 0
        start_c_idx = 0
        is_ternary_cond = False
        ins = None
        if is_chained:
            condition = cb_cond
            jump_if_true = True
            target_offset = cb_true
            t0 = cb_true
            f0 = cb_false
            start_c_idx = cb_next_idx
        else:
            ins = self._instrs[idx]
            if not (
                (_parsed_2 := self.try_get_branch_info(ins, stack)),
                (condition := _parsed_2[1]),
                (jump_if_true := _parsed_2[2]),
                (target_offset := _parsed_2[3]),
                _parsed_2[0],
            )[-1]:
                if os.getenv("PYCRUNE_TRACE") == "1":
                    _trace(
                        "[trace] Conditional bail: branchinfo idx={} stack={}".format(
                            idx, len(stack)
                        )
                    )
                return (False, stmt, next_idx)
            if (
                not is_chained
                and (not jump_if_true)
                and self.is_active_loop_header(target_offset)
                and (idx + 1 < len(self._instrs))
            ):
                bt_idx = idx + 1
                is_break = False
                if (
                    bt_idx < len(self._instrs)
                    and self._instrs[bt_idx].op_code == OpCodes.POP_TOP
                    and (bt_idx + 1 < len(self._instrs))
                    and (
                        self._instrs[bt_idx + 1].op_code
                        in (OpCodes.JUMP_FORWARD, OpCodes.JUMP_BACKWARD)
                    )
                ):
                    b_target = (
                        _value_4
                        if (_value_4 := self._instrs[bt_idx + 1].target_offset) is not None
                        else -1
                    )
                    if b_target in self._active_loop_exits:
                        is_break = True
                elif bt_idx < len(self._instrs) and self._instrs[bt_idx].op_code in (
                    OpCodes.JUMP_FORWARD,
                    OpCodes.JUMP_BACKWARD,
                ):
                    b_target = (
                        _value_6
                        if (_value_6 := self._instrs[bt_idx].target_offset) is not None
                        else -1
                    )
                    if b_target in self._active_loop_exits:
                        is_break = True
                if is_break:
                    stmt = IfStmt(condition, [BreakStmt()], [])
                    next_idx = end_idx
                    return (True, stmt, next_idx)
                tail_stack = stack.clone()
                tail_body = self.decompile_range(bt_idx, end_idx, tail_stack)
                stmt = IfStmt(condition, tail_body, [])
                next_idx = end_idx
                return (True, stmt, next_idx)
            true_latch_scan = idx + 1
            while true_latch_scan < len(self._instrs) and self._instrs[true_latch_scan].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                true_latch_scan += 1
            if (
                jump_if_true
                and true_latch_scan < len(self._instrs)
                and (
                    self._instrs[true_latch_scan].op_code
                    in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                )
            ):
                backward_target = (
                    _value_9
                    if (_value_9 := self._instrs[true_latch_scan].target_offset) is not None
                    else -1
                )
                if (
                    self.is_active_loop_header(backward_target)
                    and (bt_idx := self._offset_to_index.get(target_offset, _MISSING))
                    is not _MISSING
                ):
                    is_break = False
                    if (
                        bt_idx < len(self._instrs)
                        and self._instrs[bt_idx].op_code == OpCodes.POP_TOP
                    ):
                        if bt_idx + 1 < len(self._instrs) and self._instrs[bt_idx + 1].op_code in (
                            OpCodes.JUMP_FORWARD,
                            OpCodes.JUMP_BACKWARD,
                        ):
                            b_target = (
                                _value_11
                                if (_value_11 := self._instrs[bt_idx + 1].target_offset) is not None
                                else -1
                            )
                            if b_target in self._active_loop_exits:
                                is_break = True
                    elif bt_idx < len(self._instrs) and self._instrs[bt_idx].op_code in (
                        OpCodes.JUMP_FORWARD,
                        OpCodes.JUMP_BACKWARD,
                    ):
                        b_target = (
                            _value_13
                            if (_value_13 := self._instrs[bt_idx].target_offset) is not None
                            else -1
                        )
                        if b_target in self._active_loop_exits:
                            is_break = True
                    if is_break:
                        stmt = IfStmt(condition, [BreakStmt()], [])
                        next_idx = end_idx
                        return (True, stmt, next_idx)
                    br_body_end = min(end_idx, len(self._instrs))
                    if (
                        bt_idx < br_body_end
                        and (
                            (_parsed_14 := self.try_match_break_path(bt_idx, br_body_end)),
                            (break_marker_idx := _parsed_14[1]),
                            _parsed_14[0],
                        )[-1]
                    ):
                        br_stack = stack.clone()
                        br_body = (
                            self.decompile_range(bt_idx, break_marker_idx, br_stack)
                            if break_marker_idx > bt_idx
                            else []
                        )
                        _add(br_body, BreakStmt())
                        stmt = IfStmt(condition, br_body, [])
                        next_idx = end_idx
                        return (True, stmt, next_idx)
                    if not self.is_offset_protected(self._instrs[true_latch_scan].offset):
                        body_end = min(end_idx, len(self._instrs))
                        if bt_idx < body_end:
                            body_stack = stack.clone()
                            if_body = self.decompile_range(bt_idx, body_end, body_stack)
                            stmt = IfStmt(condition, if_body, [])
                            next_idx = body_end
                            return (True, stmt, next_idx)
            if (
                idx + 1 < len(self._instrs)
                and self._instrs[idx + 1].op_code == OpCodes.JUMP_FORWARD
            ):
                jump_branch_offset = target_offset
                fall_branch_offset = self.resolve_jump_target(
                    _value_15
                    if (_value_15 := self._instrs[idx + 1].target_offset) is not None
                    else 0
                )
                if (
                    jump_branch_idx := self._offset_to_index.get(jump_branch_offset, _MISSING)
                ) is not _MISSING and (
                    fall_branch_idx := self._offset_to_index.get(fall_branch_offset, _MISSING)
                ) is not _MISSING:
                    first_idx = min(jump_branch_idx, fall_branch_idx)
                    second_idx = max(jump_branch_idx, fall_branch_idx)
                    second_offset = self._instrs[second_idx].offset
                    has_jump_to_second = False
                    k = first_idx
                    while k < second_idx - 1:
                        if self._instrs[k].target_offset == second_offset:
                            has_jump_to_second = True
                            break
                        k += 1
                    if (
                        not has_jump_to_second
                        and first_idx < second_idx - 1
                        and (second_idx > 0)
                        and (self._instrs[second_idx - 1].op_code == OpCodes.JUMP_FORWARD)
                    ):
                        join_offset = self.resolve_jump_target(
                            _value_16
                            if (_value_16 := self._instrs[second_idx - 1].target_offset) is not None
                            else 0
                        )
                        if (
                            join_idx := self._offset_to_index.get(join_offset, _MISSING)
                        ) is not _MISSING and join_idx >= second_idx:
                            nested_cond_chain = (
                                not jump_if_true
                                and jump_branch_idx < fall_branch_idx
                                and (idx + 4 < len(self._instrs))
                                and (ins is not None)
                                and (ins.target_offset == self._instrs[idx + 2].offset)
                                and (
                                    self._instrs[idx + 1].target_offset
                                    == self._instrs[idx + 3].offset
                                )
                                and (self._instrs[idx + 2].op_code == OpCodes.JUMP_FORWARD)
                                and (self._instrs[idx + 3].op_code == OpCodes.JUMP_FORWARD)
                                and (
                                    self._instrs[idx + 2].target_offset
                                    == self._instrs[idx + 4].offset
                                )
                                and (jump_branch_offset == self._instrs[idx + 4].offset)
                                and (fall_branch_offset == self._instrs[idx + 3].target_offset)
                            )
                            first_stack = stack.clone()
                            first_body = self.decompile_range(
                                first_idx, second_idx - 1, first_stack
                            )
                            second_stack = stack.clone()
                            second_body = self.decompile_range(second_idx, join_idx, second_stack)
                            is_jump_first = jump_branch_idx < fall_branch_idx
                            jump_stack = first_stack if is_jump_first else second_stack
                            fall_stack = second_stack if is_jump_first else first_stack
                            jump_body = first_body if is_jump_first else second_body
                            fall_body = second_body if is_jump_first else first_body
                            true_stack = jump_stack if jump_if_true else fall_stack
                            false_stack = fall_stack if jump_if_true else jump_stack
                            true_body = jump_body if jump_if_true else fall_body
                            false_body = fall_body if jump_if_true else jump_body
                            if nested_cond_chain:
                                condition = TernaryExpr(
                                    TernaryExpr(condition, ConstantExpr(False), ConstantExpr(True)),
                                    ConstantExpr(True),
                                    ConstantExpr(False),
                                )
                                true_stack = jump_stack
                                false_stack = fall_stack
                                true_body = jump_body
                                false_body = fall_body
                            base_cnt = len(stack)
                            if (
                                len(true_body) == 0
                                and len(false_body) == 0
                                and (len(true_stack) > base_cnt)
                                and (len(false_stack) > base_cnt)
                            ):
                                true_val = true_stack.pop()
                                false_val = false_stack.pop()
                                stack.push(TernaryExpr(condition, true_val, false_val))
                                next_idx = join_idx
                                return (True, stmt, next_idx)
                            stmt = IfStmt(condition, true_body, false_body)
                            next_idx = join_idx
                            return (True, stmt, next_idx)
            if target_offset not in self._offset_to_index:
                if os.getenv("PYCRUNE_TRACE") == "1":
                    _trace(
                        "[trace] Conditional bail: target idx={} target={}".format(
                            idx, target_offset
                        )
                    )
                return (False, stmt, next_idx)
            start_c_idx = idx + 1
            t0 = (
                target_offset
                if jump_if_true
                else self._instrs[idx + 1].offset
                if idx + 1 < len(self._instrs)
                else 0
            )
            f0 = (
                (self._instrs[idx + 1].offset if idx + 1 < len(self._instrs) else 0)
                if jump_if_true
                else target_offset
            )
            if (
                not jump_if_true
                and idx + 2 < len(self._instrs)
                and (self._instrs[idx + 1].op_code == OpCodes.JUMP_FORWARD)
                and (self._instrs[idx + 2].op_code == OpCodes.JUMP_FORWARD)
                and (ins is not None)
                and (ins.target_offset == self._instrs[idx + 2].offset)
                and (idx + 3 < len(self._instrs))
                and (self._instrs[idx + 1].target_offset == self._instrs[idx + 3].offset)
            ):
                is_ternary_cond = True
                condition = TernaryExpr(condition, NameExpr("True"), NameExpr("False"))
                t0 = (
                    _value_17
                    if (_value_17 := self._instrs[idx + 1].target_offset) is not None
                    else 0
                )
                f0 = (
                    _value_18
                    if (_value_18 := self._instrs[idx + 2].target_offset) is not None
                    else 0
                )
                target_offset = f0
                start_c_idx = idx + 3
        if not is_chained and t0 == f0:
            stmt = IfStmt(condition, [PassStmt()], [])
            next_idx = (
                no_op_next_idx
                if (no_op_next_idx := self._offset_to_index.get(t0, _MISSING)) is not _MISSING
                else idx + 1
            )
            return (True, stmt, next_idx)
        jumps = []
        jump_start_to_index = {}
        _add(
            jumps,
            ConditionJump(instr_index=idx, condition=condition, true_offset=t0, false_offset=f0),
        )
        jump_start_to_index[self._instrs[idx].offset] = 0
        if (
            is_chained
            and cb_next_idx > 0
            and (cb_next_idx < len(self._instrs))
            and (not self._instrs[cb_next_idx].offset in jump_start_to_index)
        ):
            jump_start_to_index[self._instrs[cb_next_idx].offset] = len(jumps)
        cond_stack = SymbolicStack()
        prev_jump_idx = idx + 2 if is_ternary_cond else idx
        min_true_target = (
            target_offset
            if not is_chained and jump_if_true and (target_offset > self._instrs[idx].offset)
            else 2147483647
        )
        c_idx = start_c_idx
        while c_idx < end_idx:
            if (
                self._instrs[c_idx].op_code
                in (OpCodes.JUMP_FORWARD, OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                and (
                    (_subject_20 := self._instrs[c_idx].target_offset),
                    type(_subject_20) is int and (loop_jump_target := _subject_20) is not None,
                )[-1]
                and (
                    self.is_active_loop_header(loop_jump_target)
                    or loop_jump_target in self._active_loop_exits
                )
            ):
                break
            if c_idx == prev_jump_idx + 1:
                jump_start_to_index[self._instrs[c_idx].offset] = len(jumps)
            if (
                (_parsed_21 := self.try_match_chained_branch(c_idx, cond_stack)),
                (cb_sub_cond := _parsed_21[1]),
                (cb_sub_true := _parsed_21[2]),
                (cb_sub_false := _parsed_21[3]),
                (cb_sub_next := _parsed_21[4]),
                _parsed_21[0],
            )[-1]:
                jump_start_to_index[self._instrs[c_idx].offset] = len(jumps)
                _add(
                    jumps,
                    ConditionJump(
                        instr_index=c_idx,
                        condition=cb_sub_cond,
                        true_offset=cb_sub_true,
                        false_offset=cb_sub_false,
                    ),
                )
                if not cb_sub_true in jump_start_to_index:
                    jump_start_to_index[cb_sub_true] = len(jumps)
                c_idx = cb_sub_next - 1
                c_idx += 1
                continue
            if (
                (_parsed_22 := self.try_decompile_chained_comparison(c_idx, cond_stack)),
                (cc_expr := _parsed_22[1]),
                (cc_next := _parsed_22[2]),
                _parsed_22[0],
            )[-1]:
                cond_stack.push(cc_expr)
                c_idx = cc_next - 1
                c_idx += 1
                continue
            if (
                (_parsed_23 := self.try_decompile_walrus(c_idx, cond_stack)),
                (c_idx := _parsed_23[1]),
                _parsed_23[0],
            )[-1]:
                c_idx -= 1
                c_idx += 1
                continue
            c_ins = self._instrs[c_idx]
            if c_ins.offset >= min_true_target:
                break
            if c_ins.op_code in (
                OpCodes.STORE_NAME,
                OpCodes.STORE_FAST,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
                OpCodes.STORE_SUBSCR,
                OpCodes.STORE_ATTR,
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                OpCodes.POP_TOP,
                OpCodes.RAISE_VARARGS,
                OpCodes.RERAISE,
            ):
                break
            if (
                (_parsed_24 := self.try_get_branch_info(c_ins, cond_stack)),
                (c_cond := _parsed_24[1]),
                (c_jump_if_true := _parsed_24[2]),
                (c_target_offset := _parsed_24[3]),
                _parsed_24[0],
            )[-1]:
                if len(cond_stack) != 0:
                    break
                if (
                    not c_jump_if_true
                    and c_idx + 2 < len(self._instrs)
                    and (self._instrs[c_idx + 1].op_code == OpCodes.JUMP_FORWARD)
                    and (self._instrs[c_idx + 2].op_code == OpCodes.JUMP_FORWARD)
                    and (c_ins.target_offset == self._instrs[c_idx + 2].offset)
                    and (c_idx + 3 < len(self._instrs))
                    and (self._instrs[c_idx + 1].target_offset == self._instrs[c_idx + 3].offset)
                ):
                    c_cond = TernaryExpr(c_cond, NameExpr("True"), NameExpr("False"))
                    tct = (
                        _value_25
                        if (_value_25 := self._instrs[c_idx + 1].target_offset) is not None
                        else 0
                    )
                    tcf = (
                        _value_26
                        if (_value_26 := self._instrs[c_idx + 2].target_offset) is not None
                        else 0
                    )
                    jump_start_to_index[self._instrs[c_idx].offset] = len(jumps)
                    _add(
                        jumps,
                        ConditionJump(
                            instr_index=c_idx, condition=c_cond, true_offset=tct, false_offset=tcf
                        ),
                    )
                    prev_jump_idx = c_idx + 2
                    c_idx += 2
                    c_idx += 1
                    continue
                if (
                    c_target_offset < target_offset
                    and (ct_idx := self._offset_to_index.get(c_target_offset, _MISSING))
                    is not _MISSING
                    and (ct_idx > c_idx)
                    and StructuralDecompiler.has_statement_opcode_between(
                        self._instrs, c_idx + 1, ct_idx
                    )
                ):
                    break
                if (
                    c_jump_if_true
                    and c_target_offset > c_ins.offset
                    and (c_target_offset < min_true_target)
                ):
                    min_true_target = c_target_offset
                c_fall_offset = (
                    self._instrs[c_idx + 1].offset if c_idx + 1 < len(self._instrs) else 0
                )
                ct = c_target_offset if c_jump_if_true else c_fall_offset
                cf = c_fall_offset if c_jump_if_true else c_target_offset
                jump_start_to_index[self._instrs[c_idx].offset] = len(jumps)
                _add(
                    jumps,
                    ConditionJump(
                        instr_index=c_idx, condition=c_cond, true_offset=ct, false_offset=cf
                    ),
                )
                prev_jump_idx = c_idx
                c_idx += 1
                continue
            try:
                self.execute_instruction(c_ins, cond_stack, None)
            except Exception as ex:
                if not not isinstance(ex, FailFastException):
                    raise
                break
            c_idx += 1
        last_jump = jumps[-1]
        then_start_offset = min(last_jump.true_offset, last_jump.false_offset)
        else_start_offset = max(last_jump.true_offset, last_jump.false_offset)
        for jmp in _iter_values(jumps):
            if jmp.true_offset > then_start_offset and jmp.true_offset > else_start_offset:
                else_start_offset = jmp.true_offset
            if jmp.false_offset > then_start_offset and jmp.false_offset > else_start_offset:
                else_start_offset = jmp.false_offset
        condition = self.reconstruct_condition(
            then_start_offset, else_start_offset, jumps, jump_start_to_index
        )
        if os.getenv("PYCRUNE_TRACE") == "1":
            _trace(
                "[trace] TryDecompileConditional idx={} off={} then={} else={} jumps={} cond={}".format(
                    idx,
                    self._instrs[idx].offset,
                    then_start_offset,
                    else_start_offset,
                    len(jumps),
                    PythonEmitter.emit_expr(condition),
                )
            )
            for jj in _iter_values(jumps):
                _trace(
                    "   jump#{} off={} t={} f={} cond={}".format(
                        jj.instr_index,
                        self._instrs[jj.instr_index].offset,
                        jj.true_offset,
                        jj.false_offset,
                        PythonEmitter.emit_expr(jj.condition),
                    )
                )
        if (
            not (then_start_idx := self._offset_to_index.get(then_start_offset, _MISSING))
            is not _MISSING
        ):
            return (False, stmt, next_idx)
        if (
            not (else_start_idx := self._offset_to_index.get(else_start_offset, _MISSING))
            is not _MISSING
        ):
            return (False, stmt, next_idx)
        then_end_idx = else_start_idx
        else_end_idx = else_start_idx
        has_else = False
        if (
            then_end_idx > then_start_idx
            and self._instrs[then_end_idx - 1].op_code == OpCodes.JUMP_FORWARD
        ):
            jump_target = (
                _value_27
                if (_value_27 := self._instrs[then_end_idx - 1].target_offset) is not None
                else 0
            )
            if jump_target in self._active_loop_exits or self.is_active_loop_header(jump_target):
                has_else = False
            elif (
                j_target_idx := self._offset_to_index.get(jump_target, _MISSING)
            ) is not _MISSING and j_target_idx > else_start_idx:
                has_else = True
                then_end_idx -= 1
                else_end_idx = j_target_idx
        elif then_end_idx > then_start_idx and self._instrs[then_end_idx - 1].op_code in (
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
        ):
            jump_target = (
                _value_29
                if (_value_29 := self._instrs[then_end_idx - 1].target_offset) is not None
                else 0
            )
            if self.is_active_loop_header(jump_target) or jump_target in self._active_loop_exits:
                has_else = False
            elif else_start_idx < end_idx:
                has_else = True
                then_end_idx -= 1
                else_end_idx = end_idx
        elif (
            then_end_idx > then_start_idx
            and self._instrs[then_end_idx - 1].op_code == OpCodes.RERAISE
            and (else_start_idx < end_idx)
        ):
            if then_end_idx >= 2 and self._instrs[then_end_idx - 2].op_code == OpCodes.POP_EXCEPT:
                exit_scan = then_end_idx - 1
                while exit_scan > then_start_idx and self._instrs[exit_scan].op_code in (
                    OpCodes.RERAISE,
                    OpCodes.POP_EXCEPT,
                    OpCodes.COPY,
                    OpCodes.EXTENDED_ARG,
                ):
                    exit_scan -= 1
                if (
                    exit_scan > then_start_idx
                    and self._instrs[exit_scan].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (
                            cleanup_else_idx := self._offset_to_index.get(
                                (
                                    _value_31
                                    if (_value_31 := self._instrs[exit_scan].target_offset)
                                    is not None
                                    else 0
                                ),
                                _MISSING,
                            )
                        )
                        is not _MISSING
                    )
                    and (cleanup_else_idx > else_start_idx)
                ):
                    has_else = True
                    then_end_idx = exit_scan
                    else_end_idx = cleanup_else_idx
            else:
                has_else = True
                else_end_idx = end_idx
        elif (
            then_end_idx > then_start_idx
            and self._instrs[then_end_idx - 1].op_code
            in (OpCodes.RAISE_VARARGS, OpCodes.RERAISE, OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST)
            and (else_start_idx < end_idx)
            and (end_idx < len(self._instrs))
            and (self._instrs[end_idx].op_code == OpCodes.NOP)
        ):
            has_else = True
            else_end_idx = end_idx
        target_idx2 = else_end_idx
        while (
            target_idx2 < len(self._instrs)
            and self._instrs[target_idx2].op_code == OpCodes.EXTENDED_ARG
        ):
            target_idx2 += 1
        target_ins = self._instrs[target_idx2] if target_idx2 < len(self._instrs) else None
        is_expression_context = target_ins is not None and target_ins.op_code in (
            OpCodes.STORE_NAME,
            OpCodes.STORE_FAST,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
            OpCodes.STORE_SUBSCR,
            OpCodes.STORE_ATTR,
            OpCodes.RETURN_VALUE,
            OpCodes.RETURN_CONST,
            OpCodes.LIST_APPEND,
            OpCodes.SET_ADD,
            OpCodes.MAP_ADD,
            OpCodes.YIELD_VALUE,
        )
        actual_then_end_idx = then_end_idx if is_expression_context else min(then_end_idx, end_idx)
        actual_else_end_idx = min(else_end_idx, end_idx)
        if (
            not has_else
            and (
                (_parsed_33 := self.try_match_break_path(then_start_idx, actual_then_end_idx)),
                (break_marker_idx2 := _parsed_33[1]),
                _parsed_33[0],
            )[-1]
        ):
            brk_stack = stack.clone()
            brk_body = (
                self.decompile_range(then_start_idx, break_marker_idx2, brk_stack)
                if break_marker_idx2 > then_start_idx
                else []
            )
            _add(brk_body, BreakStmt())
            stmt = IfStmt(condition, brk_body, [])
            next_idx = max(else_start_idx, then_start_idx)
            return (True, stmt, next_idx)
        base_count = len(stack)
        then_stack = stack.clone()
        then_body = self.decompile_range(then_start_idx, actual_then_end_idx, then_stack)
        else_stack = stack.clone()
        else_body = (
            self.decompile_range(else_start_idx, actual_else_end_idx, else_stack)
            if has_else and else_start_idx < actual_else_end_idx
            else []
        )
        if (
            len(then_body) == 0
            and len(then_stack) == base_count + 1
            and (len(else_body) == 0)
            and (len(else_stack) == base_count + 1)
        ):
            t_val = then_stack.pop()
            f_val = else_stack.pop()
            ternary = TernaryExpr(condition, t_val, f_val)
            stack.push(ternary)
            next_idx = else_end_idx
            return (True, stmt, next_idx)
        if (
            base_count > 0
            and len(then_body) == 1
            and (
                (_subject_34 := then_body[0]),
                isinstance(_subject_34, ReturnStmt)
                and _subject_34.from_return_const is False
                and ((t_ret := _subject_34) is not None),
            )[-1]
            and (
                (_subject_35 := _item_field(t_ret, "value", 1)),
                isinstance((t_tup := _subject_35), TupleExpr),
            )[-1]
        ):
            probe_else = else_body
            probe_consumed_to_end = False
            if len(probe_else) == 0 and else_start_idx < end_idx:
                probe_else = self.decompile_range(else_start_idx, end_idx, stack.clone())
                probe_consumed_to_end = True
            if (
                len(probe_else) == 1
                and (
                    (_subject_36 := probe_else[0]),
                    isinstance(_subject_36, ReturnStmt)
                    and _subject_36.from_return_const is False
                    and ((e_ret := _subject_36) is not None),
                )[-1]
                and (
                    (_subject_37 := _item_field(e_ret, "value", 1)),
                    isinstance((e_tup := _subject_37), TupleExpr),
                )[-1]
                and (len(t_tup.elements) == len(e_tup.elements))
                and (len(t_tup.elements) > base_count)
            ):
                prefix_ok = True
                bi = 0
                while bi < base_count:
                    base_val = stack.peek(base_count - 1 - bi)
                    if not StructuralDecompiler.expr_equals(
                        t_tup.elements[bi], base_val
                    ) or not StructuralDecompiler.expr_equals(e_tup.elements[bi], base_val):
                        prefix_ok = False
                        break
                    bi += 1
                if prefix_ok:
                    diff_idx = base_count
                    while diff_idx < len(t_tup.elements) and StructuralDecompiler.expr_equals(
                        t_tup.elements[diff_idx], e_tup.elements[diff_idx]
                    ):
                        diff_idx += 1
                    rest_ok = True
                    j = diff_idx + 1
                    while j < len(t_tup.elements):
                        if not StructuralDecompiler.expr_equals(
                            t_tup.elements[j], e_tup.elements[j]
                        ):
                            rest_ok = False
                            break
                        j += 1
                    if rest_ok and diff_idx < len(t_tup.elements):
                        elems = []
                        j = 0
                        while j < diff_idx:
                            _add(elems, t_tup.elements[j])
                            j += 1
                        _add(
                            elems,
                            TernaryExpr(
                                condition, t_tup.elements[diff_idx], e_tup.elements[diff_idx]
                            ),
                        )
                        j = diff_idx + 1
                        while j < len(t_tup.elements):
                            _add(elems, t_tup.elements[j])
                            j += 1
                        stmt = ReturnStmt(TupleExpr(elems))
                        next_idx = (
                            end_idx
                            if probe_consumed_to_end
                            else (
                                min(else_end_idx, end_idx)
                                if has_else
                                else min(else_start_idx, end_idx)
                            )
                        )
                        return (True, stmt, next_idx)
        if (
            len(then_body) == 1
            and len(else_body) == 0
            and (
                (_subject_38 := then_body[0]),
                isinstance(_subject_38, RaiseStmt)
                and ((raise_exc := _subject_38.exception), True)[-1],
            )[-1]
            and (
                isinstance(raise_exc, AssertionErrorExpr)
                or (
                    isinstance((ce := raise_exc), CallExpr)
                    and isinstance(ce.function, AssertionErrorExpr)
                )
            )
        ):
            assert_msg = None
            if isinstance((call_ex := raise_exc), CallExpr) and len(call_ex.arguments) > 0:
                assert_msg = call_ex.arguments[0]
            stmt = AssertStmt(self.invert_condition(condition), assert_msg)
            next_idx = min(else_end_idx, end_idx) if has_else else min(else_start_idx, end_idx)
            return (True, stmt, next_idx)
        if (
            len(then_body) == 1
            and len(else_body) == 0
            and (
                (_subject_39 := then_body[0]),
                isinstance(_subject_39, AssertStmt)
                and isinstance(_subject_39.test, ConstantExpr)
                and (_item_field(_subject_39.test, "value", 1) is False)
                and ((false_assert := _subject_39) is not None),
            )[-1]
            and (self._instrs[last_jump.instr_index].op_code == OpCodes.POP_JUMP_IF_TRUE)
        ):
            stmt = AssertStmt(self.invert_condition(condition), false_assert.message)
            next_idx = min(else_end_idx, end_idx) if has_else else min(else_start_idx, end_idx)
            return (True, stmt, next_idx)
        if has_else and len(else_body) == 0 and (else_start_idx < actual_else_end_idx):
            only_nops = True
            k = else_start_idx
            while k < actual_else_end_idx and k < len(self._instrs):
                if not self._instrs[k].op_code in (OpCodes.NOP, OpCodes.EXTENDED_ARG):
                    only_nops = False
                    break
                k += 1
            if only_nops:
                else_body = [PassStmt()]
        stmt = IfStmt(condition, then_body, else_body)
        next_idx = min(else_end_idx, end_idx) if has_else else min(else_start_idx, end_idx)
        return (True, stmt, next_idx)

    def invert_condition(self, expr):
        if isinstance((u := expr), UnaryExpr) and _item_field(u, "op", 0) == "not":
            return u.operand
        if (
            isinstance(expr, CompareExpr)
            and len(expr.comparators) == 1
            and ((cmp := expr) is not None)
        ):
            op, right = cmp.comparators[0]
            if op == "is":
                flipped = "is not"
            elif op == "is not":
                flipped = "is"
            elif op == "in":
                flipped = "not in"
            elif op == "not in":
                flipped = "in"
            else:
                flipped = None
            if flipped is not None:
                return CompareExpr(cmp.left, [(flipped, right)])
        if isinstance((b := expr), BoolOpExpr) and len(b.values) > 0:
            inverted_op = "or" if _item_field(b, "op", 0) == "and" else "and"
            flat = []
            for v in _iter_values(b.values):
                iv = self.invert_condition(v)
                if isinstance((vb := iv), BoolOpExpr) and _item_field(vb, "op", 0) == inverted_op:
                    flat.extend(vb.values)
                else:
                    _add(flat, iv)
            return BoolOpExpr(inverted_op, flat)
        return UnaryExpr("not", expr)

    @staticmethod
    def expr_equals(a, b):
        if a is b:
            return True
        if a is None or b is None:
            return False
        _subject_1 = (a, b)
        if (
            _subject_1 is not None
            and isinstance((ca := _subject_1[0]), ConstantExpr)
            and isinstance((cb := _subject_1[1]), ConstantExpr)
        ):
            return _same_value(_item_field(ca, "value", 1), _item_field(cb, "value", 1))
        elif (
            _subject_1 is not None
            and isinstance((na := _subject_1[0]), NameExpr)
            and isinstance((nb := _subject_1[1]), NameExpr)
        ):
            return _item_field(na, "name", 0) == _item_field(nb, "name", 0)
        elif (
            _subject_1 is not None
            and isinstance((ua := _subject_1[0]), UnaryExpr)
            and isinstance((ub := _subject_1[1]), UnaryExpr)
        ):
            return _item_field(ua, "op", 0) == _item_field(
                ub, "op", 0
            ) and StructuralDecompiler.expr_equals(ua.operand, ub.operand)
        elif (
            _subject_1 is not None
            and isinstance((ba := _subject_1[0]), BinaryExpr)
            and isinstance((bb := _subject_1[1]), BinaryExpr)
        ):
            return (
                _item_field(ba, "op", 0) == _item_field(bb, "op", 0)
                and StructuralDecompiler.expr_equals(ba.left, bb.left)
                and StructuralDecompiler.expr_equals(
                    _item_field(ba, "right", 1), _item_field(bb, "right", 1)
                )
            )
        elif (
            _subject_1 is not None
            and isinstance((oa := _subject_1[0]), BoolOpExpr)
            and isinstance((ob := _subject_1[1]), BoolOpExpr)
        ):
            if _item_field(oa, "op", 0) != _item_field(ob, "op", 0) or len(oa.values) != len(
                ob.values
            ):
                return False
            i = 0
            while i < len(oa.values):
                if not StructuralDecompiler.expr_equals(oa.values[i], ob.values[i]):
                    return False
                i += 1
            return True
        elif (
            _subject_1 is not None
            and isinstance((cpa := _subject_1[0]), CompareExpr)
            and isinstance((cpb := _subject_1[1]), CompareExpr)
        ):
            if not StructuralDecompiler.expr_equals(cpa.left, cpb.left) or len(
                cpa.comparators
            ) != len(cpb.comparators):
                return False
            i = 0
            while i < len(cpa.comparators):
                if _item_field(cpa.comparators[i], "op", 0) != _item_field(
                    cpb.comparators[i], "op", 0
                ) or not StructuralDecompiler.expr_equals(
                    _item_field(cpa.comparators[i], "right", 1),
                    _item_field(cpb.comparators[i], "right", 1),
                ):
                    return False
                i += 1
            return True
        elif (
            _subject_1 is not None
            and isinstance((ta := _subject_1[0]), TernaryExpr)
            and isinstance((tb := _subject_1[1]), TernaryExpr)
        ):
            return (
                StructuralDecompiler.expr_equals(ta.condition, tb.condition)
                and StructuralDecompiler.expr_equals(ta.if_true, tb.if_true)
                and StructuralDecompiler.expr_equals(ta.if_false, tb.if_false)
            )
        elif (
            _subject_1 is not None
            and isinstance((fa := _subject_1[0]), CallExpr)
            and isinstance((fb := _subject_1[1]), CallExpr)
        ):
            if (
                not StructuralDecompiler.expr_equals(fa.function, fb.function)
                or len(fa.arguments) != len(fb.arguments)
                or len(fa.keyword_arguments) != len(fb.keyword_arguments)
            ):
                return False
            i = 0
            while i < len(fa.arguments):
                if not StructuralDecompiler.expr_equals(fa.arguments[i], fb.arguments[i]):
                    return False
                i += 1
            i = 0
            while i < len(fa.keyword_arguments):
                if _item_field(fa.keyword_arguments[i], "name", 0) != _item_field(
                    fb.keyword_arguments[i], "name", 0
                ) or not StructuralDecompiler.expr_equals(
                    _item_field(fa.keyword_arguments[i], "value", 1),
                    _item_field(fb.keyword_arguments[i], "value", 1),
                ):
                    return False
                i += 1
            return True
        elif (
            _subject_1 is not None
            and isinstance((sa := _subject_1[0]), SubscriptExpr)
            and isinstance((sb := _subject_1[1]), SubscriptExpr)
        ):
            return StructuralDecompiler.expr_equals(
                sa.target, sb.target
            ) and StructuralDecompiler.expr_equals(
                _item_field(sa, "index", 0), _item_field(sb, "index", 0)
            )
        elif (
            _subject_1 is not None
            and isinstance((aa := _subject_1[0]), AttributeExpr)
            and isinstance((ab := _subject_1[1]), AttributeExpr)
        ):
            return aa.attribute == ab.attribute and StructuralDecompiler.expr_equals(
                aa.target, ab.target
            )
        elif (
            _subject_1 is not None
            and isinstance((la := _subject_1[0]), SliceExpr)
            and isinstance((lb := _subject_1[1]), SliceExpr)
        ):
            return (
                StructuralDecompiler.expr_equals(la.lower, lb.lower)
                and StructuralDecompiler.expr_equals(la.upper, lb.upper)
                and StructuralDecompiler.expr_equals(la.step, lb.step)
            )
        elif (
            _subject_1 is not None
            and isinstance((xa := _subject_1[0]), TupleExpr)
            and isinstance((xb := _subject_1[1]), TupleExpr)
        ):
            return StructuralDecompiler.expr_list_equals(xa.elements, xb.elements)
        elif (
            _subject_1 is not None
            and isinstance((ya := _subject_1[0]), ListExpr)
            and isinstance((yb := _subject_1[1]), ListExpr)
        ):
            return StructuralDecompiler.expr_list_equals(ya.elements, yb.elements)
        elif (
            _subject_1 is not None
            and isinstance((za := _subject_1[0]), SetExpr)
            and isinstance((zb := _subject_1[1]), SetExpr)
        ):
            return StructuralDecompiler.expr_list_equals(za.elements, zb.elements)
        elif (
            _subject_1 is not None
            and isinstance((nwa := _subject_1[0]), NamedExpr)
            and isinstance((nwb := _subject_1[1]), NamedExpr)
        ):
            return StructuralDecompiler.expr_equals(
                nwa.target, nwb.target
            ) and StructuralDecompiler.expr_equals(
                _item_field(nwa, "value", 1), _item_field(nwb, "value", 1)
            )
        elif (
            _subject_1 is not None
            and isinstance((da := _subject_1[0]), DictExpr)
            and isinstance((db := _subject_1[1]), DictExpr)
        ):
            if len(da.entries) != len(db.entries):
                return False
            i = 0
            while i < len(da.entries):
                if not StructuralDecompiler.expr_equals(
                    _item_field(da.entries[i], "key", 0), _item_field(db.entries[i], "key", 0)
                ) or not StructuralDecompiler.expr_equals(
                    _item_field(da.entries[i], "value", 1), _item_field(db.entries[i], "value", 1)
                ):
                    return False
                i += 1
            return True
        elif (
            _subject_1 is not None
            and isinstance((sta := _subject_1[0]), StarredExpr)
            and isinstance((stb := _subject_1[1]), StarredExpr)
        ):
            return StructuralDecompiler.expr_equals(
                _item_field(sta, "value", 1), _item_field(stb, "value", 1)
            )
        else:
            return _same_value(a, b)

    @staticmethod
    def expr_list_equals(a, b):
        if len(a) != len(b):
            return False
        i = 0
        while i < len(a):
            if not StructuralDecompiler.expr_equals(a[i], b[i]):
                return False
            i += 1
        return True

    def combine_and(self, left, right):
        if (
            isinstance(left, ConstantExpr)
            and _item_field(left, "value", 1) is False
            or (isinstance(right, ConstantExpr) and _item_field(right, "value", 1) is False)
        ):
            return ConstantExpr(False)
        if isinstance(left, ConstantExpr) and _item_field(left, "value", 1) is True:
            return right
        if isinstance(right, ConstantExpr) and _item_field(right, "value", 1) is True:
            return left
        if (
            isinstance((rb_or := right), BoolOpExpr)
            and _item_field(rb_or, "op", 0) == "or"
            and (len(rb_or.values) > 0)
            and StructuralDecompiler.expr_equals(self.invert_condition(left), rb_or.values[0])
        ):
            rest = (
                None if len(rb_or.values) == 1 else BoolOpExpr("or", list(_skip(rb_or.values, 1)))
            )
            return left if rest is None else self.combine_and(left, rest)
        if (
            isinstance((lb_or := left), BoolOpExpr)
            and _item_field(lb_or, "op", 0) == "or"
            and (len(lb_or.values) > 0)
            and StructuralDecompiler.expr_equals(self.invert_condition(right), lb_or.values[0])
        ):
            rest = (
                None if len(lb_or.values) == 1 else BoolOpExpr("or", list(_skip(lb_or.values, 1)))
            )
            return right if rest is None else self.combine_and(rest, right)
        values = []
        if isinstance((b_left := left), BoolOpExpr) and _item_field(b_left, "op", 0) == "and":
            values.extend(b_left.values)
        else:
            _add(values, left)
        if isinstance((b_right := right), BoolOpExpr) and _item_field(b_right, "op", 0) == "and":
            values.extend(b_right.values)
        else:
            _add(values, right)
        return BoolOpExpr("and", values)

    def combine_or(self, left, right):
        if (
            isinstance(left, ConstantExpr)
            and _item_field(left, "value", 1) is True
            or (isinstance(right, ConstantExpr) and _item_field(right, "value", 1) is True)
        ):
            return ConstantExpr(True)
        if isinstance(left, ConstantExpr) and _item_field(left, "value", 1) is False:
            return right
        if isinstance(right, ConstantExpr) and _item_field(right, "value", 1) is False:
            return left
        if (
            isinstance((rb_and := right), BoolOpExpr)
            and _item_field(rb_and, "op", 0) == "and"
            and (len(rb_and.values) > 0)
            and StructuralDecompiler.expr_equals(self.invert_condition(left), rb_and.values[0])
        ):
            rest = (
                None
                if len(rb_and.values) == 1
                else BoolOpExpr("and", list(_skip(rb_and.values, 1)))
            )
            return left if rest is None else self.combine_or(left, rest)
        if (
            isinstance((lb_and := left), BoolOpExpr)
            and _item_field(lb_and, "op", 0) == "and"
            and (len(lb_and.values) > 0)
            and StructuralDecompiler.expr_equals(self.invert_condition(right), lb_and.values[0])
        ):
            rest = (
                None
                if len(lb_and.values) == 1
                else BoolOpExpr("and", list(_skip(lb_and.values, 1)))
            )
            return right if rest is None else self.combine_or(rest, right)
        values = []
        if isinstance((b_left := left), BoolOpExpr) and _item_field(b_left, "op", 0) == "or":
            values.extend(b_left.values)
        else:
            _add(values, left)
        if isinstance((b_right := right), BoolOpExpr) and _item_field(b_right, "op", 0) == "or":
            values.extend(b_right.values)
        else:
            _add(values, right)
        return BoolOpExpr("or", values)

    def reconstruct_condition(self, true_offset, false_offset, jumps, jump_index):
        if self._version >= (3, 14):

            def next_instruction(offset):
                index = self._offset_to_index.get(offset)
                if index is None:
                    return offset
                index = self.skip_transparent(index)
                return self._instrs[index].offset if index < len(self._instrs) else offset

            jumps = [
                replace(
                    jump,
                    true_offset=next_instruction(jump.true_offset),
                    false_offset=next_instruction(jump.false_offset),
                )
                for jump in jumps
            ]
            jump_index = {next_instruction(offset): index for offset, index in jump_index.items()}
            true_offset = next_instruction(true_offset)
            false_offset = next_instruction(false_offset)
        return self.build_condition_expr(0, true_offset, false_offset, jumps, jump_index)

    def build_condition_expr(
        self, jump_idx, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
    ):
        if jump_idx >= len(jumps):
            return ConstantExpr(True)
        j = jumps[jump_idx]
        cond = j.condition
        t = j.true_offset
        f = j.false_offset
        if os.getenv("PYCRUNE_TRACE") == "1":
            _trace(
                "[bce] idx={} t={} f={} curT={} curF={} cond={} map=[{}]".format(
                    jump_idx,
                    t,
                    f,
                    cur_true_offset,
                    cur_false_offset,
                    PythonEmitter.emit_expr(cond),
                    _join(
                        ",", _select(jump_start_to_index, lambda kv: "{}:{}".format(kv[0], kv[1]))
                    ),
                )
            )
        if t == cur_true_offset and f == cur_false_offset:
            return cond
        if t == cur_false_offset and f == cur_true_offset:
            return self.invert_condition(cond)
        if f == cur_false_offset:
            if t == cur_true_offset:
                return cond
            if (ni := jump_start_to_index.get(t, _MISSING)) is not _MISSING:
                return self.combine_and(
                    cond,
                    self.build_condition_expr(
                        ni, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
                    ),
                )
            return ConstantExpr(False)
        if t == cur_true_offset:
            if f == cur_false_offset:
                return cond
            if (ni := jump_start_to_index.get(f, _MISSING)) is not _MISSING:
                return self.combine_or(
                    cond,
                    self.build_condition_expr(
                        ni, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
                    ),
                )
            return cond
        if t == cur_false_offset:
            if f == cur_true_offset:
                return self.invert_condition(cond)
            if (ni := jump_start_to_index.get(f, _MISSING)) is not _MISSING:
                return self.combine_and(
                    self.invert_condition(cond),
                    self.build_condition_expr(
                        ni, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
                    ),
                )
            return ConstantExpr(False)
        if f == cur_true_offset:
            if t == cur_false_offset:
                return self.invert_condition(cond)
            if (ni := jump_start_to_index.get(t, _MISSING)) is not _MISSING:
                return self.combine_or(
                    self.invert_condition(cond),
                    self.build_condition_expr(
                        ni, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
                    ),
                )
            return self.invert_condition(cond)
        if f in jump_start_to_index:
            chain = [cond]
            chain_true = t
            chain_false = f
            seen = {jump_idx}
            while (
                (chain_next := jump_start_to_index.get(chain_true, _MISSING)) is not _MISSING
                and chain_next >= 0
                and (chain_next < len(jumps))
                and _add(seen, chain_next)
            ):
                next_jump = jumps[chain_next]
                if next_jump.false_offset == chain_false:
                    _add(chain, next_jump.condition)
                    chain_true = next_jump.true_offset
                elif next_jump.true_offset == chain_false:
                    _add(chain, self.invert_condition(next_jump.condition))
                    chain_true = next_jump.false_offset
                else:
                    break
            if (
                len(chain) > 1
                and chain_true == cur_true_offset
                and (
                    (chain_continuation := jump_start_to_index.get(chain_false, _MISSING))
                    is not _MISSING
                )
            ):
                chain_expr = BoolOpExpr("and", chain)
                return self.combine_or(
                    chain_expr,
                    self.build_condition_expr(
                        chain_continuation,
                        cur_true_offset,
                        cur_false_offset,
                        jumps,
                        jump_start_to_index,
                    ),
                )
            left = self.build_condition_expr(
                jump_idx, cur_true_offset, f, jumps, jump_start_to_index
            )
            reach_f = self.build_condition_expr(
                jump_idx, f, cur_true_offset, jumps, jump_start_to_index
            )
            nxt = jump_start_to_index[f]
            from_f = self.build_condition_expr(
                nxt, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
            )
            return self.combine_or(left, self.combine_and(reach_f, from_f))
        if t in jump_start_to_index:
            left = self.build_condition_expr(
                jump_idx, t, cur_false_offset, jumps, jump_start_to_index
            )
            nxt = jump_start_to_index[t]
            from_t = self.build_condition_expr(
                nxt, cur_true_offset, cur_false_offset, jumps, jump_start_to_index
            )
            return self.combine_and(left, from_t)
        return cond

    def try_decompile_walrus(self, idx, stack):
        ins = self._instrs[idx]
        if ins.op_code == OpCodes.COPY and ins.op_arg == 1 or ins.op_code == OpCodes.LEGACY_DUP_TOP:
            if idx > 0 and self._instrs[idx - 1].op_code in (
                OpCodes.CALL_INTRINSIC_1,
                OpCodes.CALL_INTRINSIC_2,
            ):
                return (False, idx)
            next_ = self._instrs[idx + 1]
            if next_.op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                var_name = (
                    _value_3
                    if (
                        _value_3 := (
                            str(_subject_2)
                            if (
                                _subject_2 := (
                                    next_.arg_repr if next_.arg_repr is not None else next_.arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                if var_name.startswith("NULL + "):
                    var_name = var_name[7:]
                if var_name == "__classcell__":
                    return (False, idx)
                if len(stack) > 0:
                    val = stack.pop()
                    stack.push(NamedExpr(NameExpr(var_name), val))
                    idx += 2
                    return (True, idx)
        return (False, idx)

    def extract_target(self, loop_idx):
        while (
            loop_idx < len(self._instrs) and self._instrs[loop_idx].op_code == OpCodes.EXTENDED_ARG
        ):
            loop_idx += 1
        if loop_idx >= len(self._instrs):
            return (NameExpr("__pycrune_ph_target"), loop_idx)
        cur = self._instrs[loop_idx]
        if cur.op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            name = (
                _value_2
                if (
                    _value_2 := (
                        str(_subject_1)
                        if (
                            _subject_1 := (
                                cur.arg_repr if cur.arg_repr is not None else cur.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else "__pycrune_ph_x"
            )
            if name.startswith("NULL + "):
                name = name[7:]
            loop_idx += 1
            return (NameExpr(name), loop_idx)
        if cur.op_code == OpCodes.UNPACK_SEQUENCE:
            count = cur.op_arg if cur.op_arg is not None else 0
            loop_idx += 1
            targets = []
            i = 0
            while i < count and loop_idx < len(self._instrs):
                _add(
                    targets,
                    (
                        (_parsed_3 := self.extract_target(loop_idx)),
                        (loop_idx := _parsed_3[1]),
                        _parsed_3[0],
                    )[-1],
                )
                i += 1
            return (TupleExpr(targets), loop_idx)
        loop_idx += 1
        return (NameExpr("__pycrune_ph_target"), loop_idx)

    def try_decompile_inlined_comprehension(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if idx >= len(self._instrs):
            return (False, next_idx)
        clear_vars = []
        p = idx
        while p < len(self._instrs) and self._instrs[p].op_code in (
            OpCodes.LOAD_FAST_AND_CLEAR,
            OpCodes.MAKE_CELL,
        ):
            if self._instrs[p].op_code == OpCodes.LOAD_FAST_AND_CLEAR:
                _add(
                    clear_vars,
                    (
                        _value_4
                        if (
                            _value_4 := (
                                str(_subject_3)
                                if (
                                    _subject_3 := (
                                        _value_2
                                        if (_value_2 := self._instrs[p].arg_repr) is not None
                                        else self._instrs[p].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else ""
                    ),
                )
            p += 1
        if len(clear_vars) == 0 or p + 3 >= len(self._instrs):
            return (False, next_idx)
        if self._instrs[p].op_code != OpCodes.SWAP or self._instrs[p].op_arg != len(clear_vars) + 1:
            return (False, next_idx)
        coll_ins = self._instrs[p + 1]
        if (
            not coll_ins.op_code in (OpCodes.BUILD_LIST, OpCodes.BUILD_SET, OpCodes.BUILD_MAP)
            or coll_ins.op_arg != 0
        ):
            return (False, next_idx)
        if self._instrs[p + 2].op_code != OpCodes.SWAP or self._instrs[p + 2].op_arg != 2:
            return (False, next_idx)
        for_iter_pos = p + 3
        if (
            self._instrs[for_iter_pos].op_code == OpCodes.GET_AITER
            and for_iter_pos + 1 < len(self._instrs)
            and (self._instrs[for_iter_pos + 1].op_code == OpCodes.GET_ANEXT)
        ):
            for_iter_pos += 1
        is_async = self._instrs[for_iter_pos].op_code == OpCodes.GET_ANEXT
        if (
            not is_async
            and self._instrs[for_iter_pos].op_code == OpCodes.GET_ITER
            and (for_iter_pos + 1 < len(self._instrs))
            and (self._instrs[for_iter_pos + 1].op_code == OpCodes.FOR_ITER)
        ):
            for_iter_pos += 1
        if not is_async and self._instrs[for_iter_pos].op_code != OpCodes.FOR_ITER:
            return (False, next_idx)
        if len(stack) == 0:
            return (False, next_idx)
        outer_iterator = stack.pop()
        loop_header_offset = self._instrs[for_iter_pos].offset
        end_for_idx = -1
        end_for_offset = 0
        if is_async:
            scan = for_iter_pos + 1
            while scan < len(self._instrs):
                if self._instrs[scan].op_code == OpCodes.END_ASYNC_FOR:
                    end_for_idx = scan
                    end_for_offset = self._instrs[scan].offset
                    break
                scan += 1
            if end_for_idx < 0:
                return (False, next_idx)
        else:
            outer_for_iter = self._instrs[for_iter_pos]
            end_for_offset = (
                outer_for_iter.target_offset if outer_for_iter.target_offset is not None else 0
            )
            if (
                not (end_for_idx := self._offset_to_index.get(end_for_offset, _MISSING))
                is not _MISSING
            ):
                return (False, next_idx)
        loop_idx = for_iter_pos + 1
        if is_async:
            while loop_idx < end_for_idx and self._instrs[loop_idx].op_code != OpCodes.END_SEND:
                loop_idx += 1
            if loop_idx < end_for_idx and self._instrs[loop_idx].op_code == OpCodes.END_SEND:
                loop_idx += 1
        outer_target = (
            (_parsed_5 := self.extract_target(loop_idx)),
            (loop_idx := _parsed_5[1]),
            _parsed_5[0],
        )[-1]
        clauses = []
        cur_target = outer_target
        cur_iterator = outer_iterator
        cur_filters = []
        cur_filter_threaded = []
        inner_stack = SymbolicStack()
        comp_result = None
        while loop_idx < end_for_idx:
            cur = self._instrs[loop_idx]
            if cur.op_code == OpCodes.FOR_ITER:
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, False, cur_filter_threaded
                    ),
                )
                cur_filters = []
                cur_filter_threaded = []
                cur_iterator = inner_stack.pop()
                loop_header_offset = cur.offset
                loop_idx += 1
                cur_target = (
                    (_parsed_6 := self.extract_target(loop_idx)),
                    (loop_idx := _parsed_6[1]),
                    _parsed_6[0],
                )[-1]
                continue
            if cur.op_code == OpCodes.GET_ANEXT:
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, True, cur_filter_threaded
                    ),
                )
                cur_filters = []
                cur_filter_threaded = []
                cur_iterator = inner_stack.pop()
                loop_idx += 1
                while loop_idx < end_for_idx and self._instrs[loop_idx].op_code != OpCodes.END_SEND:
                    loop_idx += 1
                if loop_idx < end_for_idx and self._instrs[loop_idx].op_code == OpCodes.END_SEND:
                    loop_idx += 1
                cur_target = (
                    (_parsed_7 := self.extract_target(loop_idx)),
                    (loop_idx := _parsed_7[1]),
                    _parsed_7[0],
                )[-1]
                continue
            if (
                (_parsed_8 := self.try_decompile_await_or_yield_from(loop_idx, inner_stack)),
                (await_expr := _parsed_8[1]),
                (await_next_idx := _parsed_8[2]),
                _parsed_8[0],
            )[-1]:
                inner_stack.push(await_expr)
                loop_idx = await_next_idx
                continue
            if (
                (
                    _parsed_9 := self.try_decompile_inlined_comprehension(
                        loop_idx, inner_stack, None
                    )
                ),
                (nested_comp_next := _parsed_9[1]),
                _parsed_9[0],
            )[-1]:
                loop_idx = nested_comp_next
                continue
            if (
                (_parsed_10 := self.try_decompile_walrus(loop_idx, inner_stack)),
                (loop_idx := _parsed_10[1]),
                _parsed_10[0],
            )[-1]:
                continue
            if cur.op_code in (
                OpCodes.POP_JUMP_IF_FALSE,
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_NONE,
                OpCodes.POP_JUMP_IF_NOT_NONE,
            ):
                back_scan = loop_idx + 1
                while back_scan < len(self._instrs) and self._instrs[back_scan].op_code in (
                    OpCodes.NOP,
                    OpCodes.NOT_TAKEN,
                ):
                    back_scan += 1
                if (
                    back_scan < len(self._instrs)
                    and self._instrs[back_scan].op_code
                    in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                    and (
                        (
                            _value_13
                            if (_value_13 := self._instrs[back_scan].target_offset) is not None
                            else -1
                        )
                        == loop_header_offset
                    )
                    and (
                        (
                            element_idx := self._offset_to_index.get(
                                cur.target_offset if cur.target_offset is not None else 0, _MISSING
                            )
                        )
                        is not _MISSING
                    )
                    and (element_idx > back_scan)
                ):
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, inner_stack.pop())
                    elif cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                    elif cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                    else:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                    _add(cur_filter_threaded, True)
                    loop_idx = element_idx
                    continue
                if (
                    loop_idx + 1 < len(self._instrs)
                    and self._instrs[loop_idx + 1].op_code == OpCodes.JUMP_BACKWARD
                ):
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, inner_stack.pop())
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                target_off = self.resolve_jump_target(
                    cur.target_offset if cur.target_offset is not None else 0
                )
                is_loop_end_or_continue = (
                    target_off >= end_for_offset or target_off == loop_header_offset
                )
                if (
                    not is_loop_end_or_continue
                    and (t_idx := self._offset_to_index.get(target_off, _MISSING)) is not _MISSING
                ):
                    is_loop_end_or_continue = self._instrs[t_idx].op_code in (
                        OpCodes.JUMP_BACKWARD,
                        OpCodes.END_FOR,
                        OpCodes.FOR_ITER,
                    )
                if is_loop_end_or_continue:
                    if cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, inner_stack.pop())
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
            if (
                (_parsed_15 := self.try_decompile_conditional(loop_idx, end_for_idx, inner_stack)),
                (if_next_idx := _parsed_15[2]),
                _parsed_15[0],
            )[-1]:
                loop_idx = if_next_idx
                continue
            if cur.op_code == OpCodes.LIST_APPEND:
                elt = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, is_async, cur_filter_threaded
                    ),
                )
                comp_result = ListCompExpr(elt, clauses)
                break
            if cur.op_code == OpCodes.SET_ADD:
                elt = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, is_async, cur_filter_threaded
                    ),
                )
                comp_result = SetCompExpr(elt, clauses)
                break
            if cur.op_code == OpCodes.MAP_ADD:
                val = inner_stack.pop()
                key = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, is_async, cur_filter_threaded
                    ),
                )
                comp_result = DictCompExpr(key, val, clauses)
                break
            if cur.op_code == OpCodes.GET_AWAITABLE:
                matched, inner_expr, next_idx = self.try_decompile_await_or_yield_from(
                    loop_idx, inner_stack
                )
                if matched:
                    inner_stack.push(inner_expr)
                    loop_idx = next_idx
                    continue
            self.execute_instruction(cur, inner_stack, None)
            loop_idx += 1
        if comp_result is None:
            return (False, next_idx)
        stack.push(comp_result)
        post_idx = end_for_idx + 1
        if post_idx < len(self._instrs) and self._instrs[post_idx].op_code == OpCodes.JUMP_FORWARD:
            target_off = (
                _value_16 if (_value_16 := self._instrs[post_idx].target_offset) is not None else 0
            )
            if (target_idx := self._offset_to_index.get(target_off, _MISSING)) is not _MISSING:
                post_idx = target_idx
        if (
            post_idx < len(self._instrs)
            and self._instrs[post_idx].op_code
            in (OpCodes.STORE_FAST, OpCodes.STORE_NAME, OpCodes.STORE_GLOBAL)
            and (
                not (
                    _value_20
                    if (
                        _value_20 := (
                            str(_subject_19)
                            if (
                                _subject_19 := (
                                    _value_18
                                    if (_value_18 := self._instrs[post_idx].arg_repr) is not None
                                    else self._instrs[post_idx].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                in clear_vars
            )
        ):
            if stmts is not None:
                self.execute_instruction(self._instrs[post_idx], stack, stmts)
            post_idx += 1
        while post_idx < len(self._instrs):
            ins = self._instrs[post_idx]
            if ins.op_code == OpCodes.SWAP:
                post_idx += 1
            elif (
                ins.op_code in (OpCodes.STORE_FAST, OpCodes.STORE_NAME)
                and (
                    _value_22
                    if (
                        _value_22 := (
                            str(_subject_21)
                            if (
                                _subject_21 := (
                                    ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                in clear_vars
            ):
                post_idx += 1
            else:
                break
        next_idx = post_idx
        return (True, next_idx)

    @staticmethod
    def is_comprehension_code(code):
        return _item_field(code, "name", 0) in (
            "<genexpr>",
            "<listcomp>",
            "<setcomp>",
            "<dictcomp>",
        )

    def decompile_gen_expr(self, outer_iter):
        (
            (_parsed_2 := self.decompile_comprehension_parts(outer_iter)),
            (elt := _parsed_2[1]),
            (clauses := _parsed_2[3]),
            _parsed_2[0],
        )[-1]
        if elt is None:
            raise FailFastException(
                "comprehension '{}' has no element expression".format(
                    _item_field(self._code, "name", 0)
                )
            )
        return GeneratorExpr(elt, clauses)

    def decompile_legacy_comprehension(self, outer_iter):
        (
            (_parsed_2 := self.decompile_comprehension_parts(outer_iter)),
            (elt := _parsed_2[1]),
            (dict_key := _parsed_2[2]),
            (clauses := _parsed_2[3]),
            _parsed_2[0],
        )[-1]
        if _item_field(self._code, "name", 0) == "<dictcomp>":
            if dict_key is None or elt is None:
                raise FailFastException(
                    "dict comprehension '{}' has no key/value expression".format(
                        _item_field(self._code, "name", 0)
                    )
                )
            return DictCompExpr(dict_key, elt, clauses)
        if elt is None:
            raise FailFastException(
                "comprehension '{}' has no element expression".format(
                    _item_field(self._code, "name", 0)
                )
            )
        _subject_3 = _item_field(self._code, "name", 0)
        if _subject_3 == "<listcomp>":
            return ListCompExpr(elt, clauses)
        if _subject_3 == "<setcomp>":
            return SetCompExpr(elt, clauses)
        return GeneratorExpr(elt, clauses)

    def decompile_comprehension_parts(self, outer_iter):
        elt = None
        dict_key = None
        clauses = None
        elt = None
        dict_key = None
        clauses = []
        loop_idx = 0
        while loop_idx < len(self._instrs) and (
            not self._instrs[loop_idx].op_code in (OpCodes.FOR_ITER, OpCodes.GET_ANEXT)
        ):
            loop_idx += 1
        if loop_idx >= len(self._instrs):
            return (None, elt, dict_key, clauses)
        is_async_start = self._instrs[loop_idx].op_code == OpCodes.GET_ANEXT
        loop_header_offset = self._instrs[loop_idx].offset
        end_for_offset = 0
        end_for_idx = 0
        if is_async_start:
            end_send_idx = loop_idx
            while (
                end_send_idx < len(self._instrs)
                and self._instrs[end_send_idx].op_code != OpCodes.END_SEND
            ):
                end_send_idx += 1
            if end_send_idx < len(self._instrs):
                end_for_idx = end_send_idx
                while (
                    end_for_idx < len(self._instrs)
                    and self._instrs[end_for_idx].op_code != OpCodes.END_ASYNC_FOR
                ):
                    end_for_idx += 1
                if end_for_idx >= len(self._instrs):
                    end_for_idx = len(self._instrs) - 1
                end_for_offset = self._instrs[end_for_idx].offset
                loop_idx = end_send_idx + 1
            else:
                legacy_end = loop_idx
                while (
                    legacy_end < len(self._instrs)
                    and self._instrs[legacy_end].op_code != OpCodes.END_ASYNC_FOR
                ):
                    legacy_end += 1
                if legacy_end >= len(self._instrs):
                    return (None, elt, dict_key, clauses)
                end_for_idx = legacy_end
                end_for_offset = self._instrs[legacy_end].offset
                loop_idx += 1
                while loop_idx < legacy_end and self._instrs[loop_idx].op_code in (
                    OpCodes.LOAD_CONST,
                    OpCodes.LEGACY_YIELD_FROM,
                    OpCodes.LEGACY_POP_BLOCK,
                    OpCodes.NOP,
                ):
                    loop_idx += 1
                if (
                    loop_idx < legacy_end
                    and self._instrs[loop_idx].op_code == OpCodes.SEND
                    and (
                        (_subject_2 := self._instrs[loop_idx].target_offset),
                        type(_subject_2) is int and (send_bind_target := _subject_2) is not None,
                    )[-1]
                    and (
                        (send_bind_idx := self._offset_to_index.get(send_bind_target, _MISSING))
                        is not _MISSING
                    )
                ):
                    loop_idx = send_bind_idx
                s = loop_idx
                while s < legacy_end:
                    if (
                        self._instrs[s].op_code == OpCodes.JUMP_BACKWARD
                        and (
                            (_subject_3 := self._instrs[s].target_offset),
                            type(_subject_3) is int and (back_target := _subject_3) is not None,
                        )[-1]
                        and (back_target < self._instrs[s].offset)
                    ):
                        loop_header_offset = back_target
                    s += 1
        else:
            for_iter_ins = self._instrs[loop_idx]
            end_for_offset = (
                for_iter_ins.target_offset if for_iter_ins.target_offset is not None else 0
            )
            (end_for_idx := self._offset_to_index.get(end_for_offset, _MISSING)) is not _MISSING
            loop_idx += 1
        cur_async = is_async_start
        outer_target = (
            (_parsed_4 := self.extract_target(loop_idx)),
            (loop_idx := _parsed_4[1]),
            _parsed_4[0],
        )[-1]
        cur_target = outer_target
        cur_iterator = outer_iter
        cur_filters = []
        inner_stack = SymbolicStack()
        cur_filter_threaded = []
        limit_idx = end_for_idx if end_for_idx > 0 else len(self._instrs)
        while loop_idx < limit_idx:
            if (
                (_parsed_5 := self.try_decompile_walrus(loop_idx, inner_stack)),
                (loop_idx := _parsed_5[1]),
                _parsed_5[0],
            )[-1]:
                continue
            if (
                (
                    _parsed_6 := self.try_decompile_short_circuit_bool_op(
                        loop_idx, limit_idx, inner_stack
                    )
                ),
                (loop_idx := _parsed_6[1]),
                _parsed_6[0],
            )[-1]:
                loop_idx += 1
                continue
            cur = self._instrs[loop_idx]
            if cur.op_code == OpCodes.GET_ANEXT and (not is_async_start):
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, cur_async, cur_filter_threaded
                    ),
                )
                cur_filters = []
                cur_filter_threaded = []
                cur_iterator = inner_stack.pop()
                end_send_idx = loop_idx
                while (
                    end_send_idx < len(self._instrs)
                    and self._instrs[end_send_idx].op_code != OpCodes.END_SEND
                ):
                    end_send_idx += 1
                if end_send_idx >= len(self._instrs):
                    break
                loop_idx = end_send_idx + 1
                cur_target = (
                    (_parsed_7 := self.extract_target(loop_idx)),
                    (loop_idx := _parsed_7[1]),
                    _parsed_7[0],
                )[-1]
                cur_async = True
                continue
            if cur.op_code == OpCodes.FOR_ITER:
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, cur_async, cur_filter_threaded
                    ),
                )
                cur_filters = []
                cur_filter_threaded = []
                cur_iterator = inner_stack.pop()
                loop_header_offset = cur.offset
                loop_idx += 1
                cur_target = (
                    (_parsed_8 := self.extract_target(loop_idx)),
                    (loop_idx := _parsed_8[1]),
                    _parsed_8[0],
                )[-1]
                cur_async = False
                continue
            if cur.op_code in (
                OpCodes.POP_JUMP_IF_FALSE,
                OpCodes.POP_JUMP_IF_TRUE,
                OpCodes.POP_JUMP_IF_NONE,
                OpCodes.POP_JUMP_IF_NOT_NONE,
            ):
                back_scan = loop_idx + 1
                while back_scan < len(self._instrs) and self._instrs[back_scan].op_code in (
                    OpCodes.NOP,
                    OpCodes.NOT_TAKEN,
                ):
                    back_scan += 1
                if (
                    back_scan < len(self._instrs)
                    and self._instrs[back_scan].op_code
                    in (OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
                    and (
                        (
                            _value_11
                            if (_value_11 := self._instrs[back_scan].target_offset) is not None
                            else -1
                        )
                        == loop_header_offset
                    )
                    and (
                        (
                            element_idx := self._offset_to_index.get(
                                cur.target_offset if cur.target_offset is not None else 0, _MISSING
                            )
                        )
                        is not _MISSING
                    )
                    and (element_idx > back_scan)
                ):
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, inner_stack.pop())
                    elif cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                    elif cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                    else:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                    _add(cur_filter_threaded, True)
                    loop_idx = element_idx
                    continue
                if (
                    loop_idx + 1 < len(self._instrs)
                    and self._instrs[loop_idx + 1].op_code == OpCodes.JUMP_BACKWARD
                ):
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, inner_stack.pop())
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                        _add(cur_filter_threaded, True)
                        loop_idx += 2
                        continue
                target_off = self.resolve_jump_target(
                    cur.target_offset if cur.target_offset is not None else 0
                )
                is_loop_end_or_continue = (
                    target_off >= end_for_offset or target_off == loop_header_offset
                )
                if (
                    not is_loop_end_or_continue
                    and (t_idx := self._offset_to_index.get(target_off, _MISSING)) is not _MISSING
                ):
                    is_loop_end_or_continue = self._instrs[t_idx].op_code in (
                        OpCodes.JUMP_BACKWARD,
                        OpCodes.END_FOR,
                        OpCodes.FOR_ITER,
                    )
                if is_loop_end_or_continue:
                    if cur.op_code == OpCodes.POP_JUMP_IF_FALSE:
                        _add(cur_filters, inner_stack.pop())
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_TRUE:
                        _add(cur_filters, UnaryExpr("not", inner_stack.pop()))
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NONE:
                        _add(
                            cur_filters,
                            CompareExpr(inner_stack.pop(), [("is not", NameExpr("None"))]),
                        )
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
                    if cur.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
                        _add(
                            cur_filters, CompareExpr(inner_stack.pop(), [("is", NameExpr("None"))])
                        )
                        _add(cur_filter_threaded, False)
                        loop_idx += 1
                        continue
            if (
                (_parsed_13 := self.try_decompile_conditional(loop_idx, limit_idx, inner_stack)),
                (if_next_idx := _parsed_13[2]),
                _parsed_13[0],
            )[-1]:
                loop_idx = if_next_idx
                continue
            if cur.op_code == OpCodes.YIELD_VALUE:
                elt = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, cur_async, cur_filter_threaded
                    ),
                )
                return (None, elt, dict_key, clauses)
            if cur.op_code in (OpCodes.LIST_APPEND, OpCodes.SET_ADD):
                elt = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, cur_async, cur_filter_threaded
                    ),
                )
                return (None, elt, dict_key, clauses)
            if cur.op_code == OpCodes.MAP_ADD:
                elt = inner_stack.pop()
                dict_key = inner_stack.pop()
                _add(
                    clauses,
                    ComprehensionClause(
                        cur_target, cur_iterator, cur_filters, cur_async, cur_filter_threaded
                    ),
                )
                return (None, elt, dict_key, clauses)
            if cur.op_code == OpCodes.GET_AWAITABLE:
                matched, inner_expr, next_idx = self.try_decompile_await_or_yield_from(
                    loop_idx, inner_stack
                )
                if matched:
                    inner_stack.push(inner_expr)
                    loop_idx = next_idx
                    continue
            self.execute_instruction(cur, inner_stack, None)
            loop_idx += 1
        return (None, elt, dict_key, clauses)

    @staticmethod
    def is_condition_expression_opcode(op):
        return op in (
            OpCodes.LOAD_CONST,
            OpCodes.LOAD_FAST,
            OpCodes.LOAD_FAST_CHECK,
            OpCodes.LOAD_NAME,
            OpCodes.LOAD_GLOBAL,
            OpCodes.LOAD_DEREF,
            OpCodes.LOAD_ATTR,
            OpCodes.COMPARE_OP,
            OpCodes.IS_OP,
            OpCodes.CONTAINS_OP,
            OpCodes.BINARY_OP,
            OpCodes.UNARY_NEGATIVE,
            OpCodes.UNARY_NOT,
            OpCodes.UNARY_INVERT,
            OpCodes.BUILD_TUPLE,
            OpCodes.BUILD_LIST,
            OpCodes.CALL,
            OpCodes.KW_NAMES,
            OpCodes.COPY,
            OpCodes.SWAP,
        )

    @staticmethod
    def same_condition_instruction(a, b):
        return a.op_code == b.op_code and _same_value(a.arg_val, b.arg_val)

    @staticmethod
    def attribute_values(value):
        if isinstance((tuple_ := value), TupleExpr):
            return tuple_.elements
        if (
            isinstance(value, ConstantExpr)
            and isinstance((seq := _item_field(value, "value", 1)), Iterable)
            and (not type(_item_field(value, "value", 1)) is str)
        ):
            return list(_select(_iter_values(seq), lambda x: ConstantExpr(x)))
        return [value]

    @staticmethod
    def apply_function_defaults(parameters, defaults_value):
        defaults = StructuralDecompiler.attribute_values(defaults_value)
        list_ = list(parameters)
        positional = []
        i = 0
        while i < len(list_):
            p = list_[i]
            if p == "/":
                i += 1
                continue
            if p == "*" or p.startswith("*"):
                break
            _add(positional, i)
            i += 1
        apply = min(len(defaults), len(positional))
        k = 0
        while k < apply:
            idx = positional[len(positional) - apply + k]
            if not "=" in list_[idx]:
                list_[idx] = "{}={}".format(list_[idx], PythonEmitter.emit_expr(defaults[k]))
            k += 1
        return list_

    @staticmethod
    def apply_function_kw_defaults(parameters, dict_):
        list_ = list(parameters)
        for key, val in _iter_values(dict_.entries):
            name = (
                s
                if isinstance(key, ConstantExpr)
                and type(_item_field(key, "value", 1)) is str
                and ((s := _item_field(key, "value", 1)) is not None)
                else _item_field(n, "name", 0)
                if isinstance((n := key), NameExpr)
                else None
            )
            if _empty_string(name):
                continue
            i = 0
            while i < len(list_):
                p = list_[i]
                base_name = _trim(_split(p, ":", "=")[0]).lstrip("*")
                if base_name == name and (not "=" in p):
                    list_[i] = "{}={}".format(p, PythonEmitter.emit_expr(val))
                    break
                i += 1
        return list_

    @staticmethod
    def extract_annotate_dict(annotate):
        if not isinstance((fn := annotate), FunctionExpr):
            return None
        for stmt in _iter_values(fn.body):
            if isinstance(stmt, ReturnStmt) and isinstance(
                (dict_ := _item_field(stmt, "value", 1)), DictExpr
            ):
                return dict_
        return None

    @staticmethod
    def build_class_expr_from_build_class_call(pos_args, kw_args):
        class_body_func = pos_args[0]
        class_name_expr = pos_args[1]
        raw_bases = list(_skip(pos_args, 2))
        bases = []
        for b in _iter_values(raw_bases):
            if isinstance((gbe := b), GenericBaseExpr):
                if len(gbe.type_params) > 0:
                    pass
            elif isinstance(b, NameExpr) and _item_field(b, "name", 0) == ".generic_base":
                pass
            else:
                _add(bases, b)
        class_name = (
            name_str
            if isinstance(class_name_expr, ConstantExpr)
            and type(_item_field(class_name_expr, "value", 1)) is str
            and ((name_str := _item_field(class_name_expr, "value", 1)) is not None)
            else "__pycrune_ph_class"
        )
        body_stmts = []
        if isinstance((fn_body := class_body_func), FunctionExpr):
            body_stmts = list(
                _where(
                    fn_body.body,
                    lambda s: (
                        not isinstance((asgn := s), AssignStmt)
                        or len(asgn.targets) == 0
                        or (not isinstance((n := asgn.targets[0]), NameExpr))
                        or (
                            not _item_field(n, "name", 0)
                            in (
                                "__module__",
                                "__qualname__",
                                "__classcell__",
                                "__type_params__",
                                "__firstlineno__",
                                "__static_attributes__",
                            )
                        )
                    ),
                )
            )
            while len(body_stmts) > 0 and (
                isinstance(body_stmts[-1], ReturnStmt)
                or (
                    (
                        (_subject_1 := body_stmts[-1]),
                        isinstance((asgn_end := _subject_1), AssignStmt),
                    )[-1]
                    and len(asgn_end.targets) > 0
                    and (
                        (_subject_2 := asgn_end.targets[0]),
                        isinstance((n_end := _subject_2), NameExpr),
                    )[-1]
                    and (
                        _item_field(n_end, "name", 0)
                        in (
                            "__module__",
                            "__qualname__",
                            "__classcell__",
                            "__type_params__",
                            "__firstlineno__",
                            "__static_attributes__",
                        )
                    )
                )
                or (
                    (
                        (_subject_3 := body_stmts[-1]),
                        isinstance((expr_end := _subject_3), ExprStmt),
                    )[-1]
                    and isinstance((ne_end := expr_end.expression), NamedExpr)
                    and isinstance((net_end := ne_end.target), NameExpr)
                    and (_item_field(net_end, "name", 0) == "__classcell__")
                )
            ):
                body_stmts.pop(len(body_stmts) - 1)
        return ClassExpr(class_name, bases, kw_args, body_stmts, None, None)

    def apply_function_annotations(self, fn, ann_value):
        pairs = []
        if isinstance((tuple_ := ann_value), TupleExpr):
            i = 0
            while i + 1 < len(tuple_.elements):
                if (
                    (_subject_1 := tuple_.elements[i]),
                    isinstance(_subject_1, ConstantExpr)
                    and type(_item_field(_subject_1, "value", 1)) is str
                    and ((s := _item_field(_subject_1, "value", 1)) is not None),
                )[-1]:
                    _add(pairs, (s, tuple_.elements[i + 1]))
                i += 2
        elif isinstance((dict_ := ann_value), DictExpr):
            for key, val in _iter_values(dict_.entries):
                name = (
                    s
                    if isinstance(key, ConstantExpr)
                    and type(_item_field(key, "value", 1)) is str
                    and ((s := _item_field(key, "value", 1)) is not None)
                    else _item_field(n, "name", 0)
                    if isinstance((n := key), NameExpr)
                    else None
                )
                if name is not None:
                    _add(pairs, (name, val))
        elif (
            isinstance(ann_value, ConstantExpr)
            and isinstance((seq := _item_field(ann_value, "value", 1)), Iterable)
            and (not type(_item_field(ann_value, "value", 1)) is str)
        ):
            items = list(_iter_values(seq))
            i = 0
            while i + 1 < len(items):
                name = str(_subject_2) if (_subject_2 := items[i]) is not None else None
                if name is not None:
                    _add(pairs, (name, ConstantExpr(items[i + 1])))
                i += 2
        if len(pairs) == 0:
            return fn
        list_ = list(fn.parameters)
        return_type = fn.return_type
        for name, value in _iter_values(pairs):
            if name == "return":
                return_type = _annotation_text(value, self._code.flags)
                continue
            i = 0
            while i < len(list_):
                p = list_[i]
                base_name = _trim(_split(p, ":", "=")[0]).lstrip("*")
                if base_name == name and (not ":" in p):
                    list_[i] = "{}: {}".format(p, _annotation_text(value, self._code.flags))
                    break
                i += 1
        return replace(fn, parameters=list_, return_type=return_type)

    def try_parse_target(self, p):
        target = None
        target = None
        while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
            p += 1
        if p >= len(self._instrs):
            return (False, p, target)
        ins = self._instrs[p]
        if ins.op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            name = (
                _value_2
                if (
                    _value_2 := (
                        str(_subject_1)
                        if (
                            _subject_1 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else "__pycrune_ph_x"
            )
            target = NameExpr(name)
            p += 1
            return (True, p, target)
        if ins.op_code in (OpCodes.UNPACK_SEQUENCE, OpCodes.UNPACK_EX):
            return (
                (
                    (_parsed_3 := self.try_parse_unpack_group(p)),
                    (p := _parsed_3[1]),
                    (target := _parsed_3[2]),
                    _parsed_3[0],
                )[-1],
                p,
                target,
            )
        start_p = p
        target_stack = SymbolicStack()
        try:
            while p < len(self._instrs):
                cur = self._instrs[p]
                if cur.op_code == OpCodes.STORE_ATTR:
                    if len(target_stack) > 0:
                        obj = target_stack.pop()
                        attr_name = (
                            _value_5
                            if (
                                _value_5 := (
                                    str(_subject_4)
                                    if (
                                        _subject_4 := (
                                            cur.arg_repr
                                            if cur.arg_repr is not None
                                            else cur.arg_val
                                        )
                                    )
                                    is not None
                                    else None
                                )
                            )
                            is not None
                            else "__pycrune_ph_attr"
                        )
                        target = AttributeExpr(obj, attr_name)
                        p += 1
                        return (True, p, target)
                    break
                if cur.op_code == OpCodes.STORE_SUBSCR:
                    if len(target_stack) >= 2:
                        key = target_stack.pop()
                        obj = target_stack.pop()
                        target = SubscriptExpr(obj, key)
                        p += 1
                        return (True, p, target)
                    break
                if cur.op_code in (
                    OpCodes.STORE_FAST,
                    OpCodes.STORE_NAME,
                    OpCodes.STORE_GLOBAL,
                    OpCodes.STORE_DEREF,
                    OpCodes.UNPACK_SEQUENCE,
                    OpCodes.UNPACK_EX,
                ):
                    break
                self.execute_instruction(cur, target_stack, None)
                p += 1
        except Exception as ex:
            if not not isinstance(ex, FailFastException):
                raise
            p = start_p
            return (False, p, target)
        p = start_p
        return (False, p, target)

    def try_decompile_swap_unpack_assign(self, idx, stmts):
        next_idx = 0
        next_idx = idx
        if idx + 4 >= len(self._instrs):
            return (False, next_idx)
        if (
            self._instrs[idx + 2].op_code != OpCodes.SWAP
            or (_value_1 if (_value_1 := self._instrs[idx + 2].op_arg) is not None else 0) != 2
        ):
            return (False, next_idx)
        if not self._instrs[idx + 3].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            return (False, next_idx)
        if not self._instrs[idx + 4].op_code in (OpCodes.UNPACK_SEQUENCE, OpCodes.UNPACK_EX):
            return (False, next_idx)
        value_stack = SymbolicStack()
        try:
            self.execute_instruction(self._instrs[idx], value_stack, None)
            self.execute_instruction(self._instrs[idx + 1], value_stack, None)
        except Exception:
            return (False, next_idx)
        if len(value_stack) != 2:
            return (False, next_idx)
        v2 = value_stack.pop()
        v1 = value_stack.pop()
        p = idx + 3
        try:
            if not (
                (_parsed_2 := self.try_parse_target(p)),
                (p := _parsed_2[1]),
                (target1 := _parsed_2[2]),
                _parsed_2[0],
            )[-1]:
                return (False, next_idx)
            if p != idx + 4:
                return (False, next_idx)
            if not (
                (_parsed_3 := self.try_parse_unpack_group(p)),
                (p := _parsed_3[1]),
                (target2 := _parsed_3[2]),
                _parsed_3[0],
            )[-1]:
                return (False, next_idx)
            if p <= idx + 4:
                return (False, next_idx)
            _add(stmts, AssignStmt([TupleExpr([target1, target2])], TupleExpr([v1, v2])))
            next_idx = p
            return (True, next_idx)
        except Exception:
            return (False, next_idx)

    def try_parse_unpack_group(self, p):
        target = None
        target = None
        while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
            p += 1
        if p >= len(self._instrs):
            return (False, p, target)
        ins = self._instrs[p]
        if ins.op_code == OpCodes.UNPACK_SEQUENCE:
            count = ins.op_arg if ins.op_arg is not None else 0
            p += 1
            list_ = []
            i = 0
            while i < count:
                if not (
                    (_parsed_1 := self.try_parse_target(p)),
                    (p := _parsed_1[1]),
                    (sub := _parsed_1[2]),
                    _parsed_1[0],
                )[-1]:
                    return (False, p, target)
                _add(list_, sub)
                i += 1
            target = TupleExpr(list_)
            return (True, p, target)
        if ins.op_code == OpCodes.UNPACK_EX:
            arg = ins.op_arg if ins.op_arg is not None else 0
            before = arg & 255
            after = arg >> 8 & 255
            p += 1
            list_ = []
            i = 0
            while i < before:
                if not (
                    (_parsed_2 := self.try_parse_target(p)),
                    (p := _parsed_2[1]),
                    (sub := _parsed_2[2]),
                    _parsed_2[0],
                )[-1]:
                    return (False, p, target)
                _add(list_, sub)
                i += 1
            if not (
                (_parsed_3 := self.try_parse_target(p)),
                (p := _parsed_3[1]),
                (star_sub := _parsed_3[2]),
                _parsed_3[0],
            )[-1]:
                return (False, p, target)
            _add(list_, StarredExpr(star_sub))
            i = 0
            while i < after:
                if not (
                    (_parsed_4 := self.try_parse_target(p)),
                    (p := _parsed_4[1]),
                    (sub := _parsed_4[2]),
                    _parsed_4[0],
                )[-1]:
                    return (False, p, target)
                _add(list_, sub)
                i += 1
            target = TupleExpr(list_)
            return (True, p, target)
        return (False, p, target)

    def try_decompile_unpack_assign(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if idx >= len(self._instrs) or len(stack) == 0:
            return (False, next_idx)
        ins = self._instrs[idx]
        if not ins.op_code in (OpCodes.UNPACK_SEQUENCE, OpCodes.UNPACK_EX):
            return (False, next_idx)
        p = idx
        if not (
            (_parsed_1 := self.try_parse_unpack_group(p)),
            (p := _parsed_1[1]),
            (target := _parsed_1[2]),
            _parsed_1[0],
        )[-1]:
            return (False, next_idx)
        rhs = stack.pop()
        _add(stmts, AssignStmt([target], rhs))
        next_idx = p
        return (True, next_idx)

    def try_decompile_parallel_store(self, idx, end_idx, stack):
        widths = {
            OpCodes.LEGACY_ROT_TWO: 2,
            OpCodes.LEGACY_ROT_THREE: 3,
            OpCodes.LEGACY_ROT_FOUR: 4,
        }
        permutation = list(range(len(stack)))
        span = 0
        cursor = idx
        while cursor < end_idx:
            ins = self._instrs[cursor]
            width = ins.op_arg if ins.op_code == OpCodes.SWAP else widths.get(ins.op_code)
            if width is None:
                break
            if not 2 <= width <= len(stack):
                return (None, idx)
            span = max(span, width)
            if ins.op_code == OpCodes.SWAP:
                permutation[-1], permutation[-width] = (permutation[-width], permutation[-1])
            else:
                permutation.insert(len(permutation) - width, permutation.pop())
            cursor += 1
        if not span or cursor + span > end_idx:
            return (None, idx)
        stores = self._instrs[cursor : cursor + span]
        store_opcodes = (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        )
        if any(
            (ins.op_code not in store_opcodes or type(ins.arg_val) is not str for ins in stores)
        ):
            return (None, idx)
        base = len(permutation) - span
        targets = [None] * span
        for store, position in zip(stores, reversed(permutation[-span:])):
            targets[position - base] = NameExpr(store.arg_val)
        return (AssignStmt([TupleExpr(targets)], TupleExpr(stack.pop_count(span))), cursor + span)

    def try_decompile_multi_assign(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if idx >= len(self._instrs) or len(stack) < 2:
            return (False, next_idx)
        k = 0
        while idx + k < len(self._instrs) and self._instrs[idx + k].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            k += 1
        if k < 2 or len(stack) < k:
            return (False, next_idx)
        targets = []
        i = k - 1
        while i >= 0:
            name = (
                _value_4
                if (
                    _value_4 := (
                        str(_subject_3)
                        if (
                            _subject_3 := (
                                _value_2
                                if (_value_2 := self._instrs[idx + i].arg_repr) is not None
                                else self._instrs[idx + i].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else "v{}".format(i)
            )
            _add(targets, NameExpr(name))
            i -= 1
        values = stack.pop_count(k)
        _add(stmts, AssignStmt([TupleExpr(targets)], TupleExpr(values)))
        next_idx = idx + k
        return (True, next_idx)

    def try_decompile_inlined_finally_return(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if (
            _item_field(self._code, "name", 0) == "<module>"
            or idx + 1 >= len(self._instrs)
            or len(stack) == 0
        ):
            return (False, next_idx)
        ins = self._instrs[idx]
        if (
            ins.op_code == OpCodes.SWAP
            and (ins.op_arg if ins.op_arg is not None else 0) >= 2
            and (self._instrs[idx + 1].op_code == OpCodes.POP_TOP)
        ):
            p = idx + 2
            while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                p += 1
            while (
                p < len(self._instrs)
                and self._instrs[p].op_code == OpCodes.SWAP
                and ((_value_1 if (_value_1 := self._instrs[p].op_arg) is not None else 0) >= 2)
                and (p + 1 < len(self._instrs))
                and (self._instrs[p + 1].op_code == OpCodes.POP_TOP)
            ):
                p += 2
                while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                    p += 1
            scan = p
            while scan < len(self._instrs) and scan < p + 30:
                if self._instrs[scan].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                    ret_val = stack.pop()
                    _add(stmts, ReturnStmt(ret_val))
                    next_idx = scan + 1
                    return (True, next_idx)
                scan += 1
        return (False, next_idx)

    def try_decompile_chained_assign(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if idx >= len(self._instrs) or len(stack) == 0:
            return (False, next_idx)
        if self._instrs[idx].op_code == OpCodes.LEGACY_DUP_TOP:
            lp = idx + 1
            legacy_targets = []
            while lp < len(self._instrs) and self._instrs[lp].op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                _add(
                    legacy_targets,
                    NameExpr(
                        _value_4
                        if (
                            _value_4 := (
                                str(_subject_3)
                                if (
                                    _subject_3 := (
                                        _value_2
                                        if (_value_2 := self._instrs[lp].arg_repr) is not None
                                        else self._instrs[lp].arg_val
                                    )
                                )
                                is not None
                                else None
                            )
                        )
                        is not None
                        else ""
                    ),
                )
                lp += 1
            if len(legacy_targets) >= 2:
                value = stack.pop()
                _add(stmts, AssignStmt(legacy_targets, value))
                next_idx = lp
                return (True, next_idx)
            return (False, next_idx)
        if (
            self._instrs[idx].op_code != OpCodes.COPY
            or (_value_5 if (_value_5 := self._instrs[idx].op_arg) is not None else 0) != 1
        ):
            return (False, next_idx)
        p = idx
        targets = []
        while (
            p < len(self._instrs)
            and self._instrs[p].op_code == OpCodes.COPY
            and ((_value_6 if (_value_6 := self._instrs[p].op_arg) is not None else 0) == 1)
        ):
            next_p = p + 1
            if not (
                (_parsed_7 := self.try_parse_target(next_p)),
                (next_p := _parsed_7[1]),
                (t := _parsed_7[2]),
                _parsed_7[0],
            )[-1]:
                break
            _add(targets, t)
            p = next_p
        if len(targets) == 0:
            return (False, next_idx)
        final_p = p
        if not (
            (_parsed_8 := self.try_parse_target(final_p)),
            (final_p := _parsed_8[1]),
            (final_target := _parsed_8[2]),
            _parsed_8[0],
        )[-1]:
            return (False, next_idx)
        _add(targets, final_target)
        rhs = stack.pop()
        _add(stmts, AssignStmt(targets, rhs))
        next_idx = final_p
        return (True, next_idx)

    @staticmethod
    def convert_statements_to_expression(stmts):
        if _any(
            stmts, lambda s: isinstance(s, ExprStmt) and isinstance(s.expression, ConstantExpr)
        ):
            stmts = list(
                _where(
                    stmts,
                    lambda s: (
                        not (isinstance(s, ExprStmt) and isinstance(s.expression, ConstantExpr))
                    ),
                )
            )
        if (
            len(stmts) == 1
            and ((_subject_1 := stmts[0]), isinstance((ret := _subject_1), ReturnStmt))[-1]
            and (_item_field(ret, "value", 1) is not None)
        ):
            return _item_field(ret, "value", 1)
        if (
            len(stmts) == 1
            and ((_subject_2 := stmts[0]), isinstance((if_stmt := _subject_2), IfStmt))[-1]
            and (len(if_stmt.then_body) > 0)
        ):
            if_true = StructuralDecompiler.convert_statements_to_expression(if_stmt.then_body)
            if_false = (
                StructuralDecompiler.convert_statements_to_expression(if_stmt.else_body)
                if len(if_stmt.else_body) > 0
                else ConstantExpr(None)
            )
            return TernaryExpr(if_stmt.condition, if_true, if_false)
        if (
            len(stmts) == 2
            and ((_subject_3 := stmts[0]), isinstance((if_no_else := _subject_3), IfStmt))[-1]
            and (len(if_no_else.else_body) == 0)
            and ((_subject_4 := stmts[1]), isinstance((fallback := _subject_4), ReturnStmt))[-1]
        ):
            if_true = StructuralDecompiler.convert_statements_to_expression(if_no_else.then_body)
            if_false = (
                _value_5
                if (_value_5 := _item_field(fallback, "value", 1)) is not None
                else ConstantExpr(None)
            )
            return TernaryExpr(if_no_else.condition, if_true, if_false)
        if (
            len(stmts) > 0
            and ((_subject_6 := stmts[-1]), isinstance((last_ret := _subject_6), ReturnStmt))[-1]
            and (_item_field(last_ret, "value", 1) is not None)
        ):
            return _item_field(last_ret, "value", 1)
        if len(stmts) == 0:
            return ConstantExpr(None)
        raise FailFastException("lambda body is not a single expression or return")

    def build_parameter_list(self, co, defaults, kwdefaults=None, annotations=None):
        return_annotation = None
        return_annotation = None
        if (
            annotations is not None
            and (ret_ann := annotations.get("return", _MISSING)) is not _MISSING
        ):
            return_annotation = _annotation_text(ret_ann, co.flags)
        param_list = []
        pos_count = _item_field(co, "arg_count", 1)
        pos_only = co.pos_only_arg_count
        kw_only = co.kw_only_arg_count
        default_count = (
            _value_1
            if (_value_1 := (len(defaults) if defaults is not None else None)) is not None
            else 0
        )
        first_default_idx = pos_count - default_count
        i = 0
        while i < pos_count:
            p_name = co.var_names[i] if i < len(co.var_names) else "__pycrune_ph_arg{}".format(i)
            annotated = p_name
            ann_expr = None
            has_ann = (
                annotations is not None
                and (ann_expr := annotations.get(p_name, _MISSING)) is not _MISSING
            )
            if has_ann and ann_expr is not None:
                annotated = "{}: {}".format(p_name, _annotation_text(ann_expr, co.flags))
            if (
                i >= first_default_idx
                and defaults is not None
                and (i - first_default_idx < len(defaults))
            ):
                def_expr = defaults[i - first_default_idx]
                def_str = PythonEmitter.emit_expr(def_expr)
                _add(
                    param_list,
                    (
                        "{} = {}".format(annotated, def_str)
                        if has_ann
                        else "{}={}".format(annotated, def_str)
                    ),
                )
            else:
                _add(param_list, annotated)
            if pos_only > 0 and i == pos_only - 1:
                _add(param_list, "/")
            i += 1
        var_args_idx = pos_count + kw_only
        has_var_args = co.flags & 4 != 0 and var_args_idx < len(co.var_names)
        if has_var_args:
            va_name = co.var_names[var_args_idx]
            if (
                annotations is not None
                and (va_ann := annotations.get(va_name, _MISSING)) is not _MISSING
            ):
                _add(param_list, "*{}: {}".format(va_name, _annotation_text(va_ann, co.flags)))
            else:
                _add(param_list, "*{}".format(va_name))
        elif kw_only > 0:
            _add(param_list, "*")
        k = 0
        while k < kw_only:
            p_idx = pos_count + k
            kw_name = co.var_names[p_idx] if p_idx < len(co.var_names) else "kw{}".format(k)
            annotated_kw = kw_name
            kw_ann = None
            has_kw_ann = (
                annotations is not None
                and (kw_ann := annotations.get(kw_name, _MISSING)) is not _MISSING
            )
            if has_kw_ann and kw_ann is not None:
                annotated_kw = "{}: {}".format(kw_name, _annotation_text(kw_ann, co.flags))
            if (
                kwdefaults is not None
                and (kw_def_val := kwdefaults.get(kw_name, _MISSING)) is not _MISSING
            ):
                kw_def_str = PythonEmitter.emit_expr(kw_def_val)
                _add(
                    param_list,
                    (
                        "{} = {}".format(annotated_kw, kw_def_str)
                        if has_kw_ann
                        else "{}={}".format(annotated_kw, kw_def_str)
                    ),
                )
            else:
                _add(param_list, annotated_kw)
            k += 1
        kw_args_idx = pos_count + kw_only + (1 if has_var_args else 0)
        if co.flags & 8 != 0 and kw_args_idx < len(co.var_names):
            kwa_name = co.var_names[kw_args_idx]
            if (
                annotations is not None
                and (kwa_ann := annotations.get(kwa_name, _MISSING)) is not _MISSING
            ):
                _add(param_list, "**{}: {}".format(kwa_name, _annotation_text(kwa_ann, co.flags)))
            else:
                _add(param_list, "**{}".format(kwa_name))
        return (param_list, return_annotation)

    def execute_instruction(self, ins, stack, stmts):
        if ins.op_code == OpCodes.PUSH_NULL:
            stack.push(NameExpr(""))
        elif (
            ins.op_code == OpCodes.LEGACY_SETUP_FINALLY or ins.op_code == OpCodes.LEGACY_SETUP_WITH
        ):
            raise FailFastException(
                "Unrecognized legacy block '{}' at offset {}".format(ins.op_name, ins.offset)
            )
        elif ins.op_code == OpCodes.LEGACY_DUP_TOP:
            stack.push(stack.peek())
        elif ins.op_code == OpCodes.LEGACY_DUP_TOP_TWO:
            dup_second = stack.peek(1)
            dup_top = stack.peek()
            stack.push(dup_second)
            stack.push(dup_top)
        elif ins.op_code == OpCodes.LEGACY_ROT_TWO:
            rot_top = stack.pop()
            rot_second = stack.pop()
            stack.push(rot_top)
            stack.push(rot_second)
        elif ins.op_code == OpCodes.LEGACY_ROT_THREE:
            rot_top = stack.pop()
            rot_second = stack.pop()
            rot_third = stack.pop()
            stack.push(rot_top)
            stack.push(rot_third)
            stack.push(rot_second)
        elif ins.op_code == OpCodes.LEGACY_ROT_FOUR:
            rot_top = stack.pop()
            rot_second = stack.pop()
            rot_third = stack.pop()
            rot_fourth = stack.pop()
            stack.push(rot_top)
            stack.push(rot_fourth)
            stack.push(rot_third)
            stack.push(rot_second)
        elif ins.op_code == OpCodes.LEGACY_ROT_N:
            rot_n = ins.op_arg if ins.op_arg is not None else 2
            if rot_n <= 1 or len(stack) < rot_n:
                raise DecompilationError("ROT_N {} with stack of {}".format(rot_n, len(stack)))
            rot_items = []
            i = 0
            while i < rot_n:
                _add(rot_items, stack.pop())
                i += 1
            stack.push(rot_items[0])
            i = rot_n - 1
            while i >= 1:
                stack.push(rot_items[i])
                i -= 1
        elif ins.op_code == OpCodes.LEGACY_LIST_TO_TUPLE:
            list_to_tuple = stack.pop()
            stack.push(
                TupleExpr(list_literal.elements)
                if isinstance((list_literal := list_to_tuple), ListExpr)
                else CallExpr(NameExpr("tuple"), [list_to_tuple], [])
            )
        elif ins.op_code == OpCodes.LEGACY_BUILD_LIST_UNPACK:
            list_unpack_count = ins.op_arg if ins.op_arg is not None else 0
            list_unpack_items = stack.pop_count(list_unpack_count)
            list_unpack_elems = []
            for item in _iter_values(list_unpack_items):
                if isinstance((plain_tuple := item), TupleExpr):
                    list_unpack_elems.extend(plain_tuple.elements)
                else:
                    _add(list_unpack_elems, StarredExpr(item))
            stack.push(ListExpr(list_unpack_elems))
        elif (
            ins.op_code == OpCodes.LEGACY_BUILD_TUPLE_UNPACK
            or ins.op_code == OpCodes.LEGACY_BUILD_TUPLE_UNPACK_WITH_CALL
        ):
            tuple_unpack_count = ins.op_arg if ins.op_arg is not None else 0
            tuple_unpack_items = stack.pop_count(tuple_unpack_count)
            tuple_elems = []
            for item in _iter_values(tuple_unpack_items):
                if isinstance((plain_tuple := item), TupleExpr):
                    tuple_elems.extend(plain_tuple.elements)
                else:
                    _add(tuple_elems, StarredExpr(item))
            stack.push(TupleExpr(tuple_elems))
        elif (
            ins.op_code == OpCodes.LEGACY_CALL_FINALLY
            or ins.op_code == OpCodes.LEGACY_COPY_DICT_WITHOUT_KEYS
            or ins.op_code == OpCodes.LEGACY_PREP_RERAISE_STAR
            or (ins.op_code == OpCodes.LEGACY_ASYNC_GEN_WRAP)
        ):
            pass
        elif ins.op_code == OpCodes.LOAD_ASSERTION_ERROR:
            stack.push(AssertionErrorExpr())
        elif ins.op_code == OpCodes.LOAD_LOCALS:
            stack.push(NameExpr("locals()"))
        elif ins.op_code in (OpCodes.LOAD_FROM_DICT_OR_DEREF, OpCodes.LOAD_FROM_DICT_OR_GLOBALS):
            dict_deref_name = (
                _value_2
                if (
                    _value_2 := (
                        str(_subject_1)
                        if (
                            _subject_1 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if len(stack) > 0:
                stack.pop()
            stack.push(NameExpr(dict_deref_name))
        elif ins.op_code == OpCodes.LOAD_CONST:
            if (
                self.is_at_least314()
                and (
                    (_parsed_3 := self.try_reconstruct_folded_constant(ins)),
                    (folded_const := _parsed_3[1]),
                    _parsed_3[0],
                )[-1]
            ):
                stack.push(folded_const)
            elif isinstance((slice_const := ins.arg_val), SliceValue):
                stack.push(
                    SliceExpr(
                        (
                            None
                            if _item_field(slice_const, "start", 0) is None
                            else ConstantExpr(_item_field(slice_const, "start", 0))
                        ),
                        None if slice_const.stop is None else ConstantExpr(slice_const.stop),
                        None if slice_const.step is None else ConstantExpr(slice_const.step),
                    )
                )
            elif (
                isinstance((const_seq := ins.arg_val), Iterable)
                and (not type(ins.arg_val) is str)
                and _any(_iter_values(const_seq), lambda x: isinstance(x, SliceValue))
            ):
                elems = list(
                    _select(
                        _iter_values(const_seq),
                        lambda x: (
                            SliceExpr(
                                (
                                    None
                                    if _item_field(sl, "start", 0) is None
                                    else ConstantExpr(_item_field(sl, "start", 0))
                                ),
                                None if sl.stop is None else ConstantExpr(sl.stop),
                                None if sl.step is None else ConstantExpr(sl.step),
                            )
                            if isinstance((sl := x), SliceValue)
                            else ConstantExpr(x)
                        ),
                    )
                )
                stack.push(TupleExpr(elems))
            else:
                stack.push(ConstantExpr(ins.arg_val))
        elif ins.op_code in (
            OpCodes.LOAD_NAME,
            OpCodes.LOAD_FAST,
            OpCodes.LOAD_FAST_CHECK,
            OpCodes.LOAD_GLOBAL,
            OpCodes.LOAD_DEREF,
        ):
            name = (
                _value_5
                if (
                    _value_5 := (
                        str(_subject_4)
                        if (
                            _subject_4 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if name.startswith("NULL + "):
                name = name[7:]
            elif name.endswith(" + NULL"):
                name = name[:-7]
            stack.push(NameExpr(name))
        elif ins.op_code == OpCodes.LOAD_FAST_AND_CLEAR:
            clear_name = (
                _value_7
                if (
                    _value_7 := (
                        str(_subject_6)
                        if (
                            _subject_6 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if clear_name.startswith("NULL + "):
                clear_name = clear_name[7:]
            elif clear_name.endswith(" + NULL"):
                clear_name = clear_name[:-7]
            if not _empty_string(clear_name):
                _add(self._cleared_fast_vars, clear_name)
        elif ins.op_code == OpCodes.LOAD_ATTR:
            attr_target = stack.pop()
            attr_name = (
                _value_9
                if (
                    _value_9 := (
                        str(_subject_8)
                        if (
                            _subject_8 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if attr_name.startswith("NULL|self + "):
                attr_name = attr_name[12:]
            elif attr_name.startswith("NULL + "):
                attr_name = attr_name[7:]
            elif attr_name.endswith(" + NULL|self"):
                attr_name = attr_name[:-12]
            elif attr_name.endswith(" + NULL"):
                attr_name = attr_name[:-7]
            stack.push(AttributeExpr(attr_target, attr_name))
        elif ins.op_code == OpCodes.LOAD_SUPER_ATTR:
            if len(stack) < 3:
                raise DecompilationError("LOAD_SUPER_ATTR with stack underflow")
            stack.pop_count(3)
            super_attr_name = (
                _value_11
                if (
                    _value_11 := (
                        str(_subject_10)
                        if (
                            _subject_10 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            is_super_method = False
            if super_attr_name.startswith("NULL|self + "):
                super_attr_name = super_attr_name[12:]
                is_super_method = True
            elif super_attr_name.startswith("NULL + "):
                super_attr_name = super_attr_name[7:]
                is_super_method = True
            elif super_attr_name.endswith(" + NULL|self"):
                super_attr_name = super_attr_name[:-12]
                is_super_method = True
            elif super_attr_name.endswith(" + NULL"):
                super_attr_name = super_attr_name[:-7]
                is_super_method = True
            if is_super_method:
                stack.push(NameExpr(""))
            stack.push(AttributeExpr(CallExpr(NameExpr("super"), [], []), super_attr_name))
        elif ins.op_code == OpCodes.LOAD_BUILD_CLASS:
            stack.push(NameExpr("__build_class__"))
        elif ins.op_code in (
            OpCodes.STORE_NAME,
            OpCodes.STORE_FAST,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target_name = (
                _value_13
                if (
                    _value_13 := (
                        str(_subject_12)
                        if (
                            _subject_12 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if (
                len(stack) == 0
                and target_name in self._cleared_fast_vars
                and ((store_idx := self._offset_to_index.get(ins.offset, _MISSING)) is not _MISSING)
                and (store_idx > 0)
                and (self._instrs[store_idx - 1].op_code == OpCodes.SWAP)
            ):
                return
            val = stack.pop()
            if isinstance((ta := val), TypeAliasExpr):
                if stmts is not None:
                    _add(
                        stmts,
                        TypeAliasStmt(
                            _item_field(ta, "name", 0), ta.type_params, _item_field(ta, "value", 1)
                        ),
                    )
            elif isinstance((cls := val), ClassExpr):
                if stmts is not None:
                    _add(
                        stmts,
                        ClassDefStmt(
                            _item_field(cls, "name", 0),
                            cls.bases,
                            cls.keywords,
                            cls.body,
                            cls.decorators,
                            cls.type_params,
                        ),
                    )
            elif isinstance((fn := val), FunctionExpr):
                if stmts is not None:
                    _add(
                        stmts,
                        FunctionDefStmt(
                            target_name,
                            fn.parameters,
                            fn.body,
                            fn.decorators,
                            fn.is_async,
                            fn.type_params,
                            fn.return_type,
                        ),
                    )
            elif isinstance((b_aug := val), BinaryExpr) and _item_field(b_aug, "op", 0).endswith(
                "="
            ):
                if stmts is not None:
                    _add(
                        stmts,
                        AugAssignStmt(
                            NameExpr(target_name),
                            _item_field(b_aug, "op", 0),
                            _item_field(b_aug, "right", 1),
                        ),
                    )
            elif stmts is not None:
                _add(stmts, AssignStmt([NameExpr(target_name)], val))
        elif ins.op_code == OpCodes.STORE_SUBSCR:
            sub_key = stack.pop()
            sub_obj = stack.pop()
            sub_val = stack.pop()
            if (
                stmts is not None
                and isinstance(sub_obj, NameExpr)
                and (sub_obj.name == "__annotations__")
                and isinstance(sub_key, ConstantExpr)
                and (type(sub_key.value) is str)
                and any(
                    (
                        instruction.op_code == OpCodes.SETUP_ANNOTATIONS
                        for instruction in self._instrs
                    )
                )
            ):
                target = NameExpr(sub_key.value)
                value = None
                if stmts and isinstance(stmts[-1], AssignStmt) and (stmts[-1].targets == [target]):
                    value = stmts.pop().value
                if self._code.flags & 16777216 and isinstance(sub_val, ConstantExpr):
                    sub_val = NameExpr(sub_val.value)
                stmts.append(AnnAssignStmt(target, sub_val, value))
                return
            if isinstance((b_sub_aug := sub_val), BinaryExpr) and _item_field(
                b_sub_aug, "op", 0
            ).endswith("="):
                if stmts is not None:
                    _add(
                        stmts,
                        AugAssignStmt(
                            SubscriptExpr(sub_obj, sub_key),
                            _item_field(b_sub_aug, "op", 0),
                            _item_field(b_sub_aug, "right", 1),
                        ),
                    )
            elif stmts is not None:
                _add(stmts, AssignStmt([SubscriptExpr(sub_obj, sub_key)], sub_val))
        elif ins.op_code == OpCodes.STORE_ATTR:
            attr_obj = stack.pop()
            attr_val = stack.pop()
            a_name = (
                _value_15
                if (
                    _value_15 := (
                        str(_subject_14)
                        if (
                            _subject_14 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if isinstance((b_attr_aug := attr_val), BinaryExpr) and _item_field(
                b_attr_aug, "op", 0
            ).endswith("="):
                if stmts is not None:
                    _add(
                        stmts,
                        AugAssignStmt(
                            AttributeExpr(attr_obj, a_name),
                            _item_field(b_attr_aug, "op", 0),
                            _item_field(b_attr_aug, "right", 1),
                        ),
                    )
            elif stmts is not None:
                _add(stmts, AssignStmt([AttributeExpr(attr_obj, a_name)], attr_val))
        elif ins.op_code in (
            OpCodes.DELETE_NAME,
            OpCodes.DELETE_FAST,
            OpCodes.DELETE_GLOBAL,
            OpCodes.DELETE_DEREF,
        ):
            del_name = (
                _value_17
                if (
                    _value_17 := (
                        str(_subject_16)
                        if (
                            _subject_16 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if stmts is not None:
                _add(stmts, DeleteStmt([NameExpr(del_name)]))
        elif ins.op_code == OpCodes.DELETE_SUBSCR:
            del_sub_key = stack.pop()
            del_sub_obj = stack.pop()
            if stmts is not None:
                _add(stmts, DeleteStmt([SubscriptExpr(del_sub_obj, del_sub_key)]))
        elif ins.op_code == OpCodes.DELETE_ATTR:
            del_attr_obj = stack.pop()
            del_attr_name = (
                _value_19
                if (
                    _value_19 := (
                        str(_subject_18)
                        if (
                            _subject_18 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            if stmts is not None:
                _add(stmts, DeleteStmt([AttributeExpr(del_attr_obj, del_attr_name)]))
        elif ins.op_code == OpCodes.POP_TOP:
            if len(stack) > 0:
                popped = stack.pop()
                if not isinstance((n := popped), NameExpr) or not _empty_string(
                    _item_field(n, "name", 0)
                ):
                    if stmts is not None:
                        _add(stmts, ExprStmt(popped))
        elif ins.op_code == OpCodes.UNARY_POSITIVE:
            stack.push(UnaryExpr("+", stack.pop()))
        elif ins.op_code == OpCodes.UNARY_NEGATIVE:
            stack.push(UnaryExpr("-", stack.pop()))
        elif ins.op_code == OpCodes.UNARY_NOT:
            stack.push(UnaryExpr("not", stack.pop()))
        elif ins.op_code == OpCodes.UNARY_INVERT:
            stack.push(UnaryExpr("~", stack.pop()))
        elif ins.op_code == OpCodes.BINARY_OP:
            right = stack.pop()
            left = stack.pop()
            if (ins.op_arg if ins.op_arg is not None else -1) == 26:
                stack.push(SubscriptExpr(left, right))
                return
            op = ins.arg_repr if ins.arg_repr is not None else "+"
            stack.push(BinaryExpr(left, op, right))
        elif ins.op_code == OpCodes.UNPACK_SEQUENCE:
            unpack_count = ins.op_arg if ins.op_arg is not None else 0
            seq_val = stack.pop()
            if isinstance((te_seq := seq_val), TupleExpr) and len(te_seq.elements) == unpack_count:
                i = unpack_count - 1
                while i >= 0:
                    stack.push(te_seq.elements[i])
                    i -= 1
            elif isinstance((le_seq := seq_val), ListExpr) and len(le_seq.elements) == unpack_count:
                i = unpack_count - 1
                while i >= 0:
                    stack.push(le_seq.elements[i])
                    i -= 1
            elif (
                isinstance(seq_val, ConstantExpr)
                and isinstance((list_val := _item_field(seq_val, "value", 1)), list)
                and (len(list_val) == unpack_count)
            ):
                i = unpack_count - 1
                while i >= 0:
                    stack.push(ConstantExpr(list_val[i]))
                    i -= 1
            elif unpack_count == 1:
                stack.push(StarredExpr(seq_val))
            else:
                i = unpack_count - 1
                while i >= 0:
                    stack.push(SubscriptExpr(seq_val, ConstantExpr(i)))
                    i -= 1
        elif ins.op_code == OpCodes.COMPARE_OP:
            comp_right = stack.pop()
            comp_left = stack.pop()
            comp_op = (
                _value_21
                if (
                    _value_21 := (
                        str(_subject_20)
                        if (
                            _subject_20 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else "=="
            )
            stack.push(CompareExpr(comp_left, [(comp_op, comp_right)]))
        elif ins.op_code == OpCodes.IS_OP:
            is_right = stack.pop()
            is_left = stack.pop()
            is_op = "is not" if (ins.op_arg if ins.op_arg is not None else 0) == 1 else "is"
            stack.push(CompareExpr(is_left, [(is_op, is_right)]))
        elif ins.op_code == OpCodes.CONTAINS_OP:
            cont_right = stack.pop()
            cont_left = stack.pop()
            cont_op = "not in" if (ins.op_arg if ins.op_arg is not None else 0) == 1 else "in"
            stack.push(CompareExpr(cont_left, [(cont_op, cont_right)]))
        elif ins.op_code == OpCodes.BUILD_TUPLE:
            tup_count = ins.op_arg if ins.op_arg is not None else 0
            tup_elems = stack.pop_count(tup_count)
            stack.push(TupleExpr(tup_elems))
        elif ins.op_code == OpCodes.BUILD_LIST:
            list_count = ins.op_arg if ins.op_arg is not None else 0
            list_elems = stack.pop_count(list_count)
            stack.push(ListExpr(list_elems))
        elif ins.op_code == OpCodes.BUILD_SET:
            set_count = ins.op_arg if ins.op_arg is not None else 0
            set_elems = stack.pop_count(set_count)
            stack.push(SetExpr(set_elems))
        elif ins.op_code == OpCodes.BUILD_MAP:
            map_count = ins.op_arg if ins.op_arg is not None else 0
            entries = []
            i = 0
            while i < map_count:
                map_val = stack.pop()
                map_key = stack.pop()
                entries.insert(0, (map_key, map_val))
                i += 1
            stack.push(DictExpr(entries))
        elif (
            ins.op_code == OpCodes.LEGACY_BUILD_MAP_UNPACK
            or ins.op_code == OpCodes.LEGACY_BUILD_MAP_UNPACK_WITH_CALL
        ):
            unp_count = ins.op_arg if ins.op_arg is not None else 0
            unp_items = stack.pop_count(unp_count)
            unp_entries = []
            for item in _iter_values(unp_items):
                _add(unp_entries, (None, item))
            stack.push(DictExpr(unp_entries))
        elif ins.op_code == OpCodes.LEGACY_BUILD_SET_UNPACK:
            set_unpack_count = ins.op_arg if ins.op_arg is not None else 0
            set_items = stack.pop_count(set_unpack_count)
            stack.push(
                SetExpr(
                    list(
                        _select(
                            set_items,
                            lambda i: (
                                StarredExpr(se)
                                if isinstance((se := i), SetExpr)
                                else StarredExpr(i)
                            ),
                        )
                    )
                )
            )
        elif ins.op_code == OpCodes.LEGACY_CALL_FUNCTION_KW:
            kw_tuple_val = stack.pop()
            kw_name_list = []
            if isinstance((kw_tuple := kw_tuple_val), TupleExpr):
                for elem in _iter_values(kw_tuple.elements):
                    if (
                        isinstance(elem, ConstantExpr)
                        and type(_item_field(elem, "value", 1)) is str
                        and ((s := _item_field(elem, "value", 1)) is not None)
                    ):
                        _add(kw_name_list, s)
            elif (
                isinstance(kw_tuple_val, ConstantExpr)
                and isinstance((kw_seq := _item_field(kw_tuple_val, "value", 1)), Iterable)
                and (not type(_item_field(kw_tuple_val, "value", 1)) is str)
            ):
                for item in _iter_values(kw_seq):
                    if type(item) is str and (s := item) is not None:
                        _add(kw_name_list, s)
            kw_argc = ins.op_arg if ins.op_arg is not None else 0
            kw_all = stack.pop_count(kw_argc)
            kw_pos_count = max(0, kw_argc - len(kw_name_list))
            kw_pos_args = list(_take(kw_all, kw_pos_count))
            kw_arg_list = []
            i = 0
            while i < len(kw_name_list) and kw_pos_count + i < len(kw_all):
                _add(kw_arg_list, (kw_name_list[i], kw_all[kw_pos_count + i]))
                i += 1
            kw_func = stack.pop()
            if (
                isinstance(kw_func, NameExpr)
                and _item_field(kw_func, "name", 0) == ""
                and (len(stack) > 0)
            ):
                kw_func = stack.pop()
            elif (
                len(stack) > 0
                and (
                    (_subject_22 := stack.peek()),
                    isinstance(_subject_22, NameExpr) and _item_field(_subject_22, "name", 0) == "",
                )[-1]
            ):
                stack.pop()
            if (
                isinstance(kw_func, NameExpr)
                and _item_field(kw_func, "name", 0) == "__build_class__"
                and (len(kw_pos_args) >= 2)
            ):
                stack.push(
                    StructuralDecompiler.build_class_expr_from_build_class_call(
                        kw_pos_args, kw_arg_list
                    )
                )
                return
            stack.push(CallExpr(kw_func, kw_pos_args, kw_arg_list))
        elif ins.op_code == OpCodes.LEGACY_YIELD_FROM:
            stack.pop()
            yield_iter = stack.pop()
            stack.push(YieldFromExpr(yield_iter))
        elif ins.op_code == OpCodes.KW_NAMES:
            kwn_val = ins.arg_val
            kw_list = []
            if isinstance((kwn_en := kwn_val), Iterable) and (not type(kwn_val) is str):
                for item in _iter_values(kwn_en):
                    if item is not None:
                        _add(kw_list, str(item))
            elif (
                type(ins.arg_repr) is str
                and (kwn_repr := ins.arg_repr) is not None
                and kwn_repr.startswith("(")
                and kwn_repr.endswith(")")
            ):
                inner = _trim(kwn_repr[1:-1])
                parts = _split(inner, ",")
                for p in _iter_values(parts):
                    trimmed = _trim(p, " ", "'", '"')
                    if not _empty_string(trimmed):
                        _add(kw_list, trimmed)
            self._active_kw_names = kw_list
        elif ins.op_code == OpCodes.CALL:
            argc = ins.op_arg if ins.op_arg is not None else 0
            kw_names = self._active_kw_names
            self._active_kw_names = None
            kw_count = (
                _value_24
                if (_value_24 := (len(kw_names) if kw_names is not None else None)) is not None
                else 0
            )
            if (
                argc == 0
                and kw_count == 0
                and (len(stack) >= 2)
                and isinstance(stack.peek(1), AssertionErrorExpr)
            ):
                msg_arg = stack.pop()
                assert_func = stack.pop()
                stack.push(CallExpr(assert_func, [msg_arg], []))
                return
            pos_count = argc - kw_count
            all_args = stack.pop_count(argc)
            pos_args = list(_take(all_args, pos_count))
            kw_args = []
            i = 0
            while i < kw_count:
                _add(kw_args, (kw_names[i], all_args[pos_count + i]))
                i += 1
            func = stack.pop()
            if (
                isinstance(func, NameExpr)
                and _item_field(func, "name", 0) == ""
                and (len(stack) > 0)
            ):
                func = stack.pop()
            elif (
                len(stack) > 0
                and (
                    (_subject_25 := stack.peek()),
                    isinstance((null_name := _subject_25), NameExpr),
                )[-1]
                and _empty_string(_item_field(null_name, "name", 0))
            ):
                stack.pop()
            if isinstance((fn_callable := func), FunctionExpr) and argc == 0:
                if _item_field(fn_callable, "name", 0).startswith("<generic parameters of "):
                    inner_val = (
                        _value_27
                        if (
                            _value_27 := (
                                _item_field(_subject_26, "value", 1)
                                if (_subject_26 := _first(_of_type(fn_callable.body, ReturnStmt)))
                                is not None
                                else None
                            )
                        )
                        is not None
                        else fn_callable
                    )
                    stack.push(inner_val)
                    return
                if len(stack) > 0:
                    dec = stack.pop()
                    decs = (
                        list(fn_callable.decorators) if fn_callable.decorators is not None else []
                    )
                    decs.insert(0, dec)
                    stack.push(
                        FunctionExpr(
                            _item_field(fn_callable, "name", 0),
                            fn_callable.parameters,
                            fn_callable.body,
                            decs,
                            fn_callable.is_async,
                            fn_callable.type_params,
                            fn_callable.return_type,
                        )
                    )
                    return
            if (
                len(pos_args) == 1
                and ((_subject_28 := pos_args[0]), isinstance((cls_dec := _subject_28), ClassExpr))[
                    -1
                ]
            ):
                decs = list(cls_dec.decorators) if cls_dec.decorators is not None else []
                decs.insert(0, func)
                stack.push(
                    ClassExpr(
                        _item_field(cls_dec, "name", 0),
                        cls_dec.bases,
                        cls_dec.keywords,
                        cls_dec.body,
                        decs,
                        cls_dec.type_params,
                    )
                )
                return
            if (
                isinstance(func, NameExpr)
                and _item_field(func, "name", 0) == "__build_class__"
                and (len(pos_args) >= 2)
            ):
                class_body_func = pos_args[0]
                class_name_expr = pos_args[1]
                raw_bases = list(_skip(pos_args, 2))
                bases = []
                class_type_params = list(self._type_params) if len(self._type_params) > 0 else None
                for b in _iter_values(raw_bases):
                    if isinstance((gbe := b), GenericBaseExpr):
                        if len(gbe.type_params) > 0:
                            class_type_params = list(gbe.type_params)
                    elif isinstance(b, NameExpr) and _item_field(b, "name", 0) == ".generic_base":
                        pass
                    else:
                        _add(bases, b)
                class_name = (
                    name_str
                    if isinstance(class_name_expr, ConstantExpr)
                    and type(_item_field(class_name_expr, "value", 1)) is str
                    and ((name_str := _item_field(class_name_expr, "value", 1)) is not None)
                    else "__pycrune_ph_class"
                )
                body_stmts = []
                if isinstance((fn_body := class_body_func), FunctionExpr):
                    body_stmts = list(
                        _where(
                            fn_body.body,
                            lambda s: (
                                not isinstance((asgn := s), AssignStmt)
                                or len(asgn.targets) == 0
                                or (not isinstance((n := asgn.targets[0]), NameExpr))
                                or (
                                    not _item_field(n, "name", 0)
                                    in (
                                        "__module__",
                                        "__qualname__",
                                        "__classcell__",
                                        "__type_params__",
                                        "__firstlineno__",
                                        "__static_attributes__",
                                    )
                                )
                            ),
                        )
                    )
                    while len(body_stmts) > 0 and (
                        isinstance(body_stmts[-1], ReturnStmt)
                        or (
                            (
                                (_subject_29 := body_stmts[-1]),
                                isinstance((asgn_end := _subject_29), AssignStmt),
                            )[-1]
                            and len(asgn_end.targets) > 0
                            and (
                                (_subject_30 := asgn_end.targets[0]),
                                isinstance((n_end := _subject_30), NameExpr),
                            )[-1]
                            and (
                                _item_field(n_end, "name", 0)
                                in (
                                    "__module__",
                                    "__qualname__",
                                    "__classcell__",
                                    "__type_params__",
                                    "__firstlineno__",
                                    "__static_attributes__",
                                )
                            )
                        )
                        or (
                            (
                                (_subject_31 := body_stmts[-1]),
                                isinstance((expr_end := _subject_31), ExprStmt),
                            )[-1]
                            and isinstance((ne_end := expr_end.expression), NamedExpr)
                            and isinstance((net_end := ne_end.target), NameExpr)
                            and (_item_field(net_end, "name", 0) == "__classcell__")
                        )
                    ):
                        body_stmts.pop(len(body_stmts) - 1)
                stack.push(
                    ClassExpr(class_name, bases, kw_args, body_stmts, None, class_type_params)
                )
                return
            if isinstance((cls_callable := func), ClassExpr) and argc == 0 and (len(stack) > 0):
                dec = stack.pop()
                decs = list(cls_callable.decorators) if cls_callable.decorators is not None else []
                decs.insert(0, dec)
                stack.push(
                    ClassExpr(
                        _item_field(cls_callable, "name", 0),
                        cls_callable.bases,
                        cls_callable.keywords,
                        cls_callable.body,
                        decs,
                        cls_callable.type_params,
                    )
                )
                return
            if (
                argc == 1
                and len(pos_args) == 1
                and (
                    (_subject_32 := pos_args[0]),
                    isinstance((dec_fn := _subject_32), FunctionExpr),
                )[-1]
                and (len(kw_args) == 0)
            ):
                decs = list(dec_fn.decorators) if dec_fn.decorators is not None else []
                decs.insert(0, func)
                stack.push(
                    FunctionExpr(
                        _item_field(dec_fn, "name", 0),
                        dec_fn.parameters,
                        dec_fn.body,
                        decs,
                        dec_fn.is_async,
                        dec_fn.type_params,
                        dec_fn.return_type,
                    )
                )
                return
            call_instr_idx = (
                cur_call_i
                if (cur_call_i := self._offset_to_index.get(ins.offset, _MISSING)) is not _MISSING
                else -1
            )
            next_instr_idx = call_instr_idx + 1
            while (
                next_instr_idx >= 0
                and next_instr_idx < len(self._instrs)
                and (self._instrs[next_instr_idx].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN))
            ):
                next_instr_idx += 1
            next_op = (
                self._instrs[next_instr_idx].op_code
                if next_instr_idx >= 0 and next_instr_idx < len(self._instrs)
                else -1
            )
            next_arg = (
                (_value_34 if (_value_34 := self._instrs[next_instr_idx].op_arg) is not None else 0)
                if next_instr_idx >= 0 and next_instr_idx < len(self._instrs)
                else -1
            )
            next_consumes_iterable = next_op in (OpCodes.GET_ITER, OpCodes.GET_AITER) or (
                next_op == OpCodes.CALL and next_arg == 0
            )
            if (
                argc == 0
                and (not next_consumes_iterable)
                and (len(stack) > 0)
                and (
                    (_subject_35 := stack.peek()),
                    isinstance(_subject_35, ConstantExpr)
                    and isinstance(
                        (comp_obj := _item_field(_subject_35, "value", 1)), PythonCodeObject
                    ),
                )[-1]
                and StructuralDecompiler.is_comprehension_code(comp_obj)
            ):
                outer_iter = func
                func = stack.pop()
                pos_args.insert(0, outer_iter)
            if (
                isinstance(func, ConstantExpr)
                and isinstance((comp_code := _item_field(func, "value", 1)), PythonCodeObject)
                and StructuralDecompiler.is_comprehension_code(comp_code)
            ):
                if len(pos_args) == 0:
                    raise FailFastException(
                        "comprehension '{}' called without an outer iterator".format(
                            _item_field(comp_code, "name", 0)
                        )
                    )
                comp_decompiler = StructuralDecompiler(
                    comp_code, report_position=self._report_position
                )
                stack.push(comp_decompiler.decompile_legacy_comprehension(pos_args[0]))
                if self._report_position is not None:
                    self._report_position(comp_code, len(comp_code.instructions))
                return
            stack.push(CallExpr(func, pos_args, kw_args))
        elif ins.op_code == OpCodes.BINARY_SLICE:
            slice_stop = stack.pop()
            slice_start = stack.pop()
            slice_container = stack.pop()
            lower = (
                None
                if isinstance(slice_start, ConstantExpr)
                and _item_field(slice_start, "value", 1) is None
                else slice_start
            )
            upper = (
                None
                if isinstance(slice_stop, ConstantExpr)
                and _item_field(slice_stop, "value", 1) is None
                else slice_stop
            )
            stack.push(SubscriptExpr(slice_container, SliceExpr(lower, upper, None)))
        elif ins.op_code == OpCodes.BUILD_SLICE:
            slice_argc = ins.op_arg if ins.op_arg is not None else 2
            step = stack.pop() if slice_argc == 3 else None
            stop_node = stack.pop()
            start_node = stack.pop()
            slice_l = (
                None
                if isinstance(start_node, ConstantExpr)
                and _item_field(start_node, "value", 1) is None
                else start_node
            )
            slice_u = (
                None
                if isinstance(stop_node, ConstantExpr)
                and _item_field(stop_node, "value", 1) is None
                else stop_node
            )
            slice_s = (
                None
                if isinstance(step, ConstantExpr) and _item_field(step, "value", 1) is None
                else step
            )
            stack.push(SliceExpr(slice_l, slice_u, slice_s))
        elif ins.op_code == OpCodes.STORE_SLICE:
            store_slice_stop = stack.pop()
            store_slice_start = stack.pop()
            store_slice_container = stack.pop()
            store_slice_val = stack.pop()
            s_lower = (
                None
                if isinstance(store_slice_start, ConstantExpr)
                and _item_field(store_slice_start, "value", 1) is None
                else store_slice_start
            )
            s_upper = (
                None
                if isinstance(store_slice_stop, ConstantExpr)
                and _item_field(store_slice_stop, "value", 1) is None
                else store_slice_stop
            )
            if stmts is not None:
                _add(
                    stmts,
                    AssignStmt(
                        [SubscriptExpr(store_slice_container, SliceExpr(s_lower, s_upper, None))],
                        store_slice_val,
                    ),
                )
        elif ins.op_code == OpCodes.SET_UPDATE:
            update_arg = stack.pop()
            if (
                len(stack) > 0
                and ((_subject_36 := stack.peek()), isinstance((set_expr := _subject_36), SetExpr))[
                    -1
                ]
            ):
                merged = list(set_expr.elements)
                if isinstance(update_arg, ConstantExpr) and isinstance(
                    (hash_set := _item_field(update_arg, "value", 1)), set
                ):
                    for item in _iter_values(hash_set):
                        _add(merged, ConstantExpr(item))
                elif isinstance(update_arg, ConstantExpr) and isinstance(
                    (frozen_set := _item_field(update_arg, "value", 1)), FrozenSetValue
                ):
                    for item in _iter_values(frozen_set.items):
                        _add(merged, ConstantExpr(item))
                else:
                    _add(merged, StarredExpr(update_arg))
                stack.pop()
                stack.push(SetExpr(merged))
            elif isinstance(update_arg, ConstantExpr) and isinstance(
                (loose_set := _item_field(update_arg, "value", 1)), set
            ):
                stack.push(SetExpr(list(_select(loose_set, lambda item: ConstantExpr(item)))))
            elif isinstance(update_arg, ConstantExpr) and isinstance(
                (loose_frozen := _item_field(update_arg, "value", 1)), FrozenSetValue
            ):
                stack.push(
                    SetExpr(list(_select(loose_frozen.items, lambda item: ConstantExpr(item))))
                )
            else:
                stack.push(SetExpr([StarredExpr(update_arg)]))
        elif ins.op_code == OpCodes.LIST_APPEND:
            if len(stack) == 0:
                return
            appended_val = stack.pop()
            list_depth = (ins.op_arg if ins.op_arg is not None else 1) - 1
            if (
                list_depth >= 0
                and list_depth < len(stack)
                and (
                    (_subject_37 := stack.peek(list_depth)),
                    isinstance((target_list := _subject_37), ListExpr),
                )[-1]
            ):
                if isinstance((m_list := target_list.elements), list):
                    _add(m_list, appended_val)
                else:
                    merged = [*target_list.elements, appended_val]
                    stack.pop()
                    stack.push(ListExpr(merged))
        elif ins.op_code == OpCodes.SET_ADD:
            if len(stack) == 0:
                return
            set_val = stack.pop()
            set_depth = (ins.op_arg if ins.op_arg is not None else 1) - 1
            if (
                set_depth >= 0
                and set_depth < len(stack)
                and (
                    (_subject_38 := stack.peek(set_depth)),
                    isinstance((target_set := _subject_38), SetExpr),
                )[-1]
            ):
                if isinstance((m_set := target_set.elements), list):
                    _add(m_set, set_val)
                else:
                    merged = [*target_set.elements, set_val]
                    stack.pop()
                    stack.push(SetExpr(merged))
        elif ins.op_code == OpCodes.MAP_ADD:
            if len(stack) < 2:
                return
            map_v = stack.pop()
            map_k = stack.pop()
            map_depth = (ins.op_arg if ins.op_arg is not None else 1) - 1
            if (
                map_depth >= 0
                and map_depth < len(stack)
                and (
                    (_subject_39 := stack.peek(map_depth)),
                    isinstance((target_dict := _subject_39), DictExpr),
                )[-1]
            ):
                if isinstance((m_dict := target_dict.entries), list):
                    _add(m_dict, (map_k, map_v))
                else:
                    merged = [*target_dict.entries, (map_k, map_v)]
                    stack.pop()
                    stack.push(DictExpr(merged))
        elif ins.op_code == OpCodes.MATCH_SEQUENCE:
            stack.push(NameExpr("__pycrune_ph_match_subject"))
        elif ins.op_code == OpCodes.GET_LEN:
            if len(stack) == 0:
                raise DecompilationError("GET_LEN with empty stack")
            len_target = stack.peek()
            stack.push(CallExpr(NameExpr("len"), [len_target], []))
        elif ins.op_code == OpCodes.MATCH_CLASS:
            if len(stack) < 2:
                raise DecompilationError("MATCH_CLASS with stack underflow")
            match_count = ins.op_arg if ins.op_arg is not None else 0
            stack.pop()
            stack.pop()
            match_args = list(
                _select(
                    _range(0, match_count), lambda i: NameExpr("__pycrune_ph_match_{}".format(i))
                )
            )
            stack.push(TupleExpr(match_args))
        elif ins.op_code == OpCodes.MATCH_MAPPING:
            stack.push(NameExpr("__pycrune_ph_match_subject"))
        elif ins.op_code == OpCodes.MATCH_KEYS:
            if len(stack) == 0:
                raise DecompilationError("MATCH_KEYS with empty stack")
            key_tuple = stack.pop()
            k_count = 1
            if isinstance((te_keys := key_tuple), TupleExpr):
                k_count = len(te_keys.elements)
            elif isinstance(key_tuple, ConstantExpr) and isinstance(
                (col := _item_field(key_tuple, "value", 1)), Collection
            ):
                k_count = len(col)
            key_vals = list(
                _select(_range(0, k_count), lambda i: NameExpr("__pycrune_ph_val_{}".format(i)))
            )
            stack.push(TupleExpr(key_vals))
        elif ins.op_code == OpCodes.UNPACK_EX:
            if len(stack) == 0:
                raise DecompilationError("UNPACK_EX with empty stack")
            ex_arg = ins.op_arg if ins.op_arg is not None else 0
            ex_before = ex_arg & 255
            ex_after = ex_arg >> 8 & 255
            ex_seq = stack.pop()
            total = ex_before + 1 + ex_after
            i = total - 1
            while i >= 0:
                stack.push(SubscriptExpr(ex_seq, ConstantExpr(i)))
                i -= 1
        elif ins.op_code == OpCodes.CALL_INTRINSIC_1:
            intrin1 = ins.op_arg if ins.op_arg is not None else 0
            if intrin1 == 5 and len(stack) > 0:
                stack.push(UnaryExpr("+", stack.pop()))
            elif intrin1 == 6 and len(stack) > 0:
                list_top = stack.pop()
                if isinstance((list_expr_top := list_top), ListExpr):
                    stack.push(TupleExpr(list_expr_top.elements))
                else:
                    stack.push(list_top)
            elif intrin1 in (7, 8, 9) and len(stack) > 0:
                t_name = stack.pop()
                tp_str = (
                    s_val
                    if isinstance(t_name, ConstantExpr)
                    and type(_item_field(t_name, "value", 1)) is str
                    and ((s_val := _item_field(t_name, "value", 1)) is not None)
                    else (
                        _item_field(ne, "name", 0)
                        if isinstance((ne := t_name), NameExpr)
                        else str(t_name)
                    )
                )
                if intrin1 == 8:
                    tp_str = "**{}".format(tp_str)
                elif intrin1 == 9:
                    tp_str = "*{}".format(tp_str)
                if not _empty_string(tp_str) and (not tp_str in self._type_params):
                    _add(self._type_params, tp_str)
                stack.push(t_name)
            elif intrin1 == 10 and len(stack) > 0:
                tp_arg = stack.pop()
                tp_list = list(self._type_params)
                if isinstance((te_gen := tp_arg), TupleExpr):
                    extracted = list(
                        _where(
                            _select(
                                te_gen.elements,
                                lambda e: (
                                    _item_field(n, "name", 0)
                                    if isinstance((n := e), NameExpr)
                                    else (
                                        (
                                            _value_41
                                            if (
                                                _value_41 := (
                                                    str(_subject_40)
                                                    if (_subject_40 := _item_field(c, "value", 1))
                                                    is not None
                                                    else None
                                                )
                                            )
                                            is not None
                                            else ""
                                        )
                                        if isinstance((c := e), ConstantExpr)
                                        else ""
                                    )
                                ),
                            ),
                            lambda s: not _empty_string(s),
                        )
                    )
                    if len(extracted) > 0:
                        tp_list = extracted
                stack.push(GenericBaseExpr(tp_list))
            elif intrin1 == 11 and len(stack) > 0:
                alias_tuple = stack.pop()
                alias_name = None
                type_params = []
                alias_value = None
                if isinstance((te_alias := alias_tuple), TupleExpr) and len(te_alias.elements) == 3:
                    if (
                        (_subject_42 := te_alias.elements[0]),
                        isinstance(_subject_42, ConstantExpr)
                        and type(_item_field(_subject_42, "value", 1)) is str
                        and ((name_str := _item_field(_subject_42, "value", 1)) is not None),
                    )[-1]:
                        alias_name = name_str
                    if (
                        (_subject_43 := te_alias.elements[1]),
                        isinstance((tp_tuple := _subject_43), TupleExpr),
                    )[-1]:
                        type_params = list(
                            _where(
                                _select(
                                    tp_tuple.elements,
                                    lambda e: (
                                        (
                                            s := (
                                                _item_field(n, "name", 0)
                                                if isinstance((n := e), NameExpr)
                                                else (
                                                    (
                                                        _value_45
                                                        if (
                                                            _value_45 := (
                                                                str(_subject_44)
                                                                if (
                                                                    _subject_44 := _item_field(
                                                                        c, "value", 1
                                                                    )
                                                                )
                                                                is not None
                                                                else None
                                                            )
                                                        )
                                                        is not None
                                                        else ""
                                                    )
                                                    if isinstance((c := e), ConstantExpr)
                                                    else ""
                                                )
                                            )
                                        ),
                                        (
                                            matched := _first(
                                                self._type_params,
                                                lambda tp: (
                                                    tp == s
                                                    or tp == "**{}".format(s)
                                                    or tp == "*{}".format(s)
                                                    or tp.startswith("{}:".format(s))
                                                    or tp.startswith("{} ".format(s))
                                                ),
                                            )
                                        ),
                                        matched if matched is not None else s,
                                    )[-1],
                                ),
                                lambda s: not _empty_string(s),
                            )
                        )
                    if (
                        (_subject_46 := te_alias.elements[2]),
                        isinstance((fn_val := _subject_46), FunctionExpr),
                    )[-1]:
                        alias_value = (
                            _item_field(_subject_47, "value", 1)
                            if (_subject_47 := _first(_of_type(fn_val.body, ReturnStmt)))
                            is not None
                            else None
                        )
                if alias_name is None or alias_value is None:
                    raise FailFastException("malformed PEP 695 type alias intrinsic")
                stack.push(TypeAliasExpr(alias_name, type_params, alias_value))
        elif ins.op_code == OpCodes.CALL_INTRINSIC_2:
            intrin2 = ins.op_arg if ins.op_arg is not None else 0
            if (intrin2 == 2 or intrin2 == 3) and len(stack) >= 2:
                bound_top = stack.pop()
                name_top = stack.pop()
                tp_name = (
                    s_val
                    if isinstance(name_top, ConstantExpr)
                    and type(_item_field(name_top, "value", 1)) is str
                    and ((s_val := _item_field(name_top, "value", 1)) is not None)
                    else (
                        _item_field(ne, "name", 0)
                        if isinstance((ne := name_top), NameExpr)
                        else str(name_top)
                    )
                )
                bound_expr = None
                if isinstance((fe_bound := bound_top), FunctionExpr):
                    bound_expr = (
                        _item_field(_subject_48, "value", 1)
                        if (_subject_48 := _first(_of_type(fe_bound.body, ReturnStmt))) is not None
                        else None
                    )
                else:
                    bound_expr = bound_top
                tp_str = (
                    "{}: {}".format(tp_name, PythonEmitter.emit_expr(bound_expr))
                    if bound_expr is not None
                    else tp_name
                )
                if not _empty_string(tp_str) and (not tp_str in self._type_params):
                    _add(self._type_params, tp_str)
                stack.push(name_top)
            elif intrin2 == 4 and len(stack) >= 2:
                first_top = stack.pop()
                second_top = stack.pop()
                fe = (
                    _value_51
                    if (
                        _value_51 := (
                            _value_49
                            if isinstance((_value_49 := first_top), FunctionExpr)
                            else None
                        )
                    )
                    is not None
                    else _value_50
                    if isinstance((_value_50 := second_top), FunctionExpr)
                    else None
                )
                te_tp = (
                    _value_54
                    if (
                        _value_54 := (
                            _value_52 if isinstance((_value_52 := first_top), TupleExpr) else None
                        )
                    )
                    is not None
                    else _value_53
                    if isinstance((_value_53 := second_top), TupleExpr)
                    else None
                )
                tp_list = []
                if te_tp is not None:
                    tp_list = list(
                        _where(
                            _select(
                                te_tp.elements,
                                lambda e: (
                                    (
                                        s := (
                                            _item_field(n, "name", 0)
                                            if isinstance((n := e), NameExpr)
                                            else (
                                                (
                                                    _value_56
                                                    if (
                                                        _value_56 := (
                                                            str(_subject_55)
                                                            if (
                                                                _subject_55 := _item_field(
                                                                    c, "value", 1
                                                                )
                                                            )
                                                            is not None
                                                            else None
                                                        )
                                                    )
                                                    is not None
                                                    else ""
                                                )
                                                if isinstance((c := e), ConstantExpr)
                                                else str(e)
                                            )
                                        )
                                    ),
                                    (
                                        matched := _first(
                                            self._type_params,
                                            lambda tp: (
                                                tp == s
                                                or tp == "**{}".format(s)
                                                or tp == "*{}".format(s)
                                                or tp.startswith("{}:".format(s))
                                                or tp.startswith("{} ".format(s))
                                            ),
                                        )
                                    ),
                                    matched if matched is not None else s,
                                )[-1],
                            ),
                            lambda s: not _empty_string(s),
                        )
                    )
                if fe is not None:
                    stack.push(
                        FunctionExpr(
                            _item_field(fe, "name", 0),
                            fe.parameters,
                            fe.body,
                            fe.decorators,
                            fe.is_async,
                            tp_list,
                            fe.return_type,
                        )
                    )
                else:
                    stack.push(first_top)
        elif ins.op_code in (OpCodes.RETURN_GENERATOR, OpCodes.RESUME, OpCodes.END_FOR):
            pass
        elif ins.op_code == OpCodes.SWAP:
            swap_n = (ins.op_arg if ins.op_arg is not None else 2) - 1
            if len(stack) > swap_n:
                stack.swap(swap_n)
        elif ins.op_code == OpCodes.COPY:
            copy_n = ins.op_arg if ins.op_arg is not None else 1
            if len(stack) >= copy_n:
                stack.copy(copy_n)
        elif ins.op_code == OpCodes.POP_EXCEPT:
            pass
        elif ins.op_code in (
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
        ):
            if len(stack) > 0:
                stack.pop()
        elif ins.op_code in (
            OpCodes.JUMP_FORWARD,
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
        ):
            target_offset = ins.target_offset if ins.target_offset is not None else -1
            if target_offset in self._active_loop_exits:
                if stmts is not None:
                    _add(stmts, BreakStmt())
            elif self.is_active_loop_header(target_offset):
                if stmts is not None:
                    _add(stmts, ContinueStmt())
        elif ins.op_code == OpCodes.RETURN_CONST:
            const_ret_val = ConstantExpr(ins.arg_val)
            if _item_field(self._code, "name", 0) != "<module>" and (
                _item_field(const_ret_val, "value", 1) is not None
                or (
                    (r_idx := self._offset_to_index.get(ins.offset, _MISSING)) is not _MISSING
                    and r_idx < len(self._instrs) - 1
                )
            ):
                if stmts is not None:
                    _add(stmts, ReturnStmt(const_ret_val, from_return_const=True))
        elif ins.op_code == OpCodes.RETURN_VALUE:
            if len(stack) > 0:
                ret_val = stack.pop()
                is_last = (
                    r_idx2 := self._offset_to_index.get(ins.offset, _MISSING)
                ) is not _MISSING and r_idx2 == len(self._instrs) - 1
                operand_is_jump_target = r_idx2 > 0 and self._instrs[r_idx2 - 1].is_jump_target
                is_trailing = is_last or self.is_followed_only_by_handler_code(ins)
                must_keep = (
                    not isinstance((c := ret_val), ConstantExpr)
                    or _item_field(c, "value", 1) is not None
                    or (not is_trailing)
                    or (
                        operand_is_jump_target
                        if is_last
                        else self.is_targeted_by_conditional_jump(ins.offset)
                    )
                )
                if _item_field(self._code, "name", 0) != "<module>" and must_keep:
                    if stmts is not None:
                        _add(stmts, ReturnStmt(ret_val))
        elif ins.op_code == OpCodes.RAISE_VARARGS:
            raise_argc = ins.op_arg if ins.op_arg is not None else 0
            exc = None
            cause = None
            if raise_argc == 2:
                cause = stack.pop()
                exc = stack.pop()
            elif raise_argc == 1:
                exc = stack.pop()
            if StructuralDecompiler.try_build_assert_from_raise(exc, stmts):
                return
            if stmts is not None:
                _add(stmts, RaiseStmt(exc, cause))
        elif ins.op_code == OpCodes.RERAISE:
            pass
        elif ins.op_code == OpCodes.GET_ITER:
            pass
        elif ins.op_code == OpCodes.GET_AITER:
            pass
        elif ins.op_code == OpCodes.GET_ANEXT:
            stack.push(NameExpr("anext"))
        elif ins.op_code == OpCodes.END_ASYNC_FOR:
            pass
        elif ins.op_code == OpCodes.CLEANUP_THROW:
            pass
        elif ins.op_code == OpCodes.BUILD_CONST_KEY_MAP:
            bck_count = ins.op_arg if ins.op_arg is not None else 0
            keys_tuple = stack.pop()
            bck_vals = stack.pop_count(bck_count)
            bck_entries = []
            if isinstance((te := keys_tuple), TupleExpr) and len(te.elements) == bck_count:
                i = 0
                while i < bck_count:
                    _add(bck_entries, (te.elements[i], bck_vals[i]))
                    i += 1
            elif isinstance(keys_tuple, ConstantExpr) and isinstance(
                (list_ := _item_field(keys_tuple, "value", 1)), Iterable
            ):
                i = 0
                for k in _iter_values(list_):
                    if i < len(bck_vals):
                        _add(bck_entries, (ConstantExpr(k), bck_vals[(i := (i + 1)) - 1]))
            stack.push(DictExpr(bck_entries))
        elif ins.op_code == OpCodes.CONVERT_VALUE:
            conv_val = stack.pop()
            _subject_57 = ins.op_arg if ins.op_arg is not None else 0
            if _subject_57 == 1:
                conv = "s"
            elif _subject_57 == 2:
                conv = "r"
            elif _subject_57 == 3:
                conv = "a"
            else:
                conv = None
            stack.push(FormattedValueExpr(conv_val, conv, None))
        elif ins.op_code == OpCodes.FORMAT_SIMPLE:
            simple_val = stack.pop()
            stack.push(
                FormattedValueExpr(JoinedStrExpr([nested_fv]))
                if isinstance(simple_val, FormattedValueExpr)
                and simple_val.conversion is None
                and ((nested_fv := simple_val) is not None)
                else (
                    sfv
                    if isinstance((sfv := simple_val), FormattedValueExpr)
                    else FormattedValueExpr(simple_val)
                )
            )
        elif ins.op_code == OpCodes.FORMAT_WITH_SPEC:
            spec_val = stack.pop()
            fmt_val = stack.pop()
            stack.push(
                replace(wfv, format_spec=spec_val)
                if isinstance((wfv := fmt_val), FormattedValueExpr)
                else FormattedValueExpr(fmt_val, None, spec_val)
            )
        elif ins.op_code == OpCodes.SET_FUNCTION_ATTRIBUTE:
            attr_fn = stack.pop()
            attr_value = stack.pop()
            if isinstance((fn_expr := attr_fn), FunctionExpr):
                kind = ins.op_arg if ins.op_arg is not None else 0
                if kind == 1:
                    fn_expr = replace(
                        fn_expr,
                        parameters=StructuralDecompiler.apply_function_defaults(
                            fn_expr.parameters, attr_value
                        ),
                    )
                elif kind == 2 and isinstance((kw_dict := attr_value), DictExpr):
                    fn_expr = replace(
                        fn_expr,
                        parameters=StructuralDecompiler.apply_function_kw_defaults(
                            fn_expr.parameters, kw_dict
                        ),
                    )
                elif kind == 4:
                    fn_expr = self.apply_function_annotations(fn_expr, attr_value)
                elif kind == 16:
                    ann_dict = StructuralDecompiler.extract_annotate_dict(attr_value)
                    if ann_dict is not None:
                        fn_expr = self.apply_function_annotations(fn_expr, ann_dict)
                stack.push(fn_expr)
            elif isinstance((lam := attr_fn), LambdaExpr):
                kind = ins.op_arg if ins.op_arg is not None else 0
                if kind == 1:
                    lam = replace(
                        lam,
                        parameters=StructuralDecompiler.apply_function_defaults(
                            lam.parameters, attr_value
                        ),
                    )
                elif kind == 2 and isinstance((lam_kw := attr_value), DictExpr):
                    lam = replace(
                        lam,
                        parameters=StructuralDecompiler.apply_function_kw_defaults(
                            lam.parameters, lam_kw
                        ),
                    )
                stack.push(lam)
            else:
                stack.push(attr_fn)
        elif ins.op_code == OpCodes.FORMAT_VALUE:
            fv_flags = ins.op_arg if ins.op_arg is not None else 0
            fv_fmt_spec = None
            if fv_flags & 4 != 0:
                if len(stack) == 0:
                    raise DecompilationError("FORMAT_VALUE format spec missing")
                fv_fmt_spec = stack.pop()
            if len(stack) == 0:
                raise DecompilationError("FORMAT_VALUE with empty stack")
            fv_val = stack.pop()
            _subject_58 = fv_flags & 3
            if _subject_58 == 1:
                fv_conv = "s"
            elif _subject_58 == 2:
                fv_conv = "r"
            elif _subject_58 == 3:
                fv_conv = "a"
            else:
                fv_conv = None
            stack.push(FormattedValueExpr(fv_val, fv_conv, fv_fmt_spec))
        elif ins.op_code == OpCodes.BUILD_STRING:
            str_count = ins.op_arg if ins.op_arg is not None else 0
            str_parts = stack.pop_count(str_count)
            if _all(str_parts, lambda p: isinstance(p, ConstantExpr)):
                combined = _concat(
                    _select(
                        str_parts,
                        lambda p: (
                            _value_60
                            if (
                                _value_60 := (
                                    str(_subject_59)
                                    if (_subject_59 := _item_field(p, "value", 1)) is not None
                                    else None
                                )
                            )
                            is not None
                            else ""
                        ),
                    )
                )
                stack.push(ConstantExpr(combined))
            else:
                stack.push(JoinedStrExpr(str_parts))
        elif ins.op_code == OpCodes.MAKE_FUNCTION:
            code_val = stack.pop()
            if (
                isinstance(code_val, ConstantExpr)
                and type(_item_field(code_val, "value", 1)) is str
                and ((qname := _item_field(code_val, "value", 1)) is not None)
                and (not qname.startswith("<code:"))
                and (len(stack) > 0)
                and (
                    (_subject_61 := stack.peek()),
                    isinstance(_subject_61, ConstantExpr)
                    and type(_item_field(_subject_61, "value", 1)) is str
                    and ((candidate_code_str := _item_field(_subject_61, "value", 1)) is not None),
                )[-1]
                and candidate_code_str.startswith("<code:")
            ):
                code_val = stack.pop()
            flags = ins.op_arg if ins.op_arg is not None else 0
            if flags & 8 != 0 and len(stack) > 0:
                stack.pop()
            annotations = None
            if flags & 4 != 0 and len(stack) > 0:
                ann_val = stack.pop()
                if isinstance((ann_tuple := ann_val), TupleExpr):
                    annotations = {}
                    ai = 0
                    while ai + 1 < len(ann_tuple.elements):
                        key_expr = ann_tuple.elements[ai]
                        val_expr = ann_tuple.elements[ai + 1]
                        key = (
                            k_str
                            if isinstance(key_expr, ConstantExpr)
                            and type(_item_field(key_expr, "value", 1)) is str
                            and ((k_str := _item_field(key_expr, "value", 1)) is not None)
                            else (
                                _item_field(kn, "name", 0)
                                if isinstance((kn := key_expr), NameExpr)
                                else str(key_expr)
                            )
                        )
                        if not _empty_string(key):
                            annotations[key] = val_expr
                        ai += 2
                elif isinstance((ann_dict := ann_val), DictExpr):
                    annotations = {}
                    for k, v in _iter_values(ann_dict.entries):
                        key = (
                            k_str
                            if isinstance(k, ConstantExpr)
                            and type(_item_field(k, "value", 1)) is str
                            and ((k_str := _item_field(k, "value", 1)) is not None)
                            else (
                                _item_field(kn, "name", 0)
                                if isinstance((kn := k), NameExpr)
                                else str(k)
                                if k is not None
                                else None
                            )
                        )
                        if not _empty_string(key):
                            annotations[key] = v
                elif (
                    isinstance(ann_val, ConstantExpr)
                    and isinstance((ann_seq := _item_field(ann_val, "value", 1)), Iterable)
                    and (not type(_item_field(ann_val, "value", 1)) is str)
                ):
                    annotations = {}
                    ann_items = list(_iter_values(ann_seq))
                    ai = 0
                    while ai + 1 < len(ann_items):
                        key = (
                            str(_subject_62) if (_subject_62 := ann_items[ai]) is not None else None
                        )
                        if not _empty_string(key):
                            annotations[key] = ConstantExpr(ann_items[ai + 1])
                        ai += 2
            kwdefaults = None
            if flags & 2 != 0 and len(stack) > 0:
                kwd_val = stack.pop()
                if isinstance((de := kwd_val), DictExpr):
                    kwdefaults = {}
                    for k, v in _iter_values(de.entries):
                        if (
                            isinstance(k, ConstantExpr)
                            and type(_item_field(k, "value", 1)) is str
                            and ((k_str := _item_field(k, "value", 1)) is not None)
                        ):
                            kwdefaults[k_str] = v
                        elif isinstance((n := k), NameExpr):
                            kwdefaults[_item_field(n, "name", 0)] = v
            defaults = None
            if flags & 1 != 0 and len(stack) > 0:
                d_val = stack.pop()
                if isinstance((te_def := d_val), TupleExpr):
                    defaults = list(te_def.elements)
                elif isinstance(d_val, ConstantExpr) and isinstance(
                    (arr := _item_field(d_val, "value", 1)), list
                ):
                    defaults = list(_select(arr, lambda x: ConstantExpr(x)))
                elif (
                    isinstance(d_val, ConstantExpr)
                    and isinstance((list_ := _item_field(d_val, "value", 1)), Iterable)
                    and (not type(_item_field(d_val, "value", 1)) is str)
                ):
                    defaults = list(_select(_iter_values(list_), lambda x: ConstantExpr(x)))
            func_name = "__pycrune_ph_func"
            nested_code = None
            if (
                isinstance(code_val, ConstantExpr)
                and type(_item_field(code_val, "value", 1)) is str
                and ((code_str := _item_field(code_val, "value", 1)) is not None)
                and code_str.startswith("<code:")
            ):
                inner = code_str[6:-1]
                parts = _split(inner, ":")
                func_name = parts[0]
                if (
                    len(parts) > 1
                    and (nested_idx := _parse_int(parts[1])) is not None
                    and (nested_idx < len(self._code.nested_codes))
                ):
                    nested_code = self._code.nested_codes[nested_idx]
            if (
                nested_code is None
                and isinstance(code_val, ConstantExpr)
                and (type(_item_field(code_val, "value", 1)) is str)
                and ((code_name := _item_field(code_val, "value", 1)) is not None)
                and code_name.startswith("<code:")
            ):
                unique = None
                for candidate in _iter_values(self._code.nested_codes):
                    if _item_field(candidate, "name", 0) != func_name:
                        continue
                    if unique is not None:
                        raise FailFastException(
                            "ambiguous nested code object '{}' in '{}'".format(
                                func_name, _item_field(self._code, "name", 0)
                            )
                        )
                    unique = candidate
                nested_code = (
                    unique
                    if unique is not None
                    else _raise(
                        FailFastException(
                            "nested code object '{}' not found in '{}'".format(
                                func_name, _item_field(self._code, "name", 0)
                            )
                        )
                    )
                )
            if nested_code is not None:
                if func_name in ("<genexpr>", "<listcomp>", "<setcomp>", "<dictcomp>"):
                    stack.push(ConstantExpr(nested_code))
                    return
                decompiler = StructuralDecompiler(
                    nested_code, report_position=self._report_position
                )
                func_stmts = decompiler.decompile()
                parameters = (
                    (
                        _parsed_63 := self.build_parameter_list(
                            nested_code, defaults, kwdefaults, annotations
                        )
                    ),
                    (return_annotation := _parsed_63[1]),
                    _parsed_63[0],
                )[-1]
                is_async = nested_code.flags & 128 != 0 or nested_code.flags & 512 != 0
                if func_name == "<lambda>":
                    lambda_body = StructuralDecompiler.convert_statements_to_expression(func_stmts)
                    stack.push(LambdaExpr(parameters, lambda_body))
                else:
                    is_generator = nested_code.flags & 32 != 0 or nested_code.flags & 512 != 0
                    if (
                        len(func_stmts) > 0
                        and (
                            (_subject_64 := func_stmts[-1]),
                            isinstance(_subject_64, ReturnStmt)
                            and (
                                _item_field(_subject_64, "value", 1) is None
                                or (
                                    isinstance(_item_field(_subject_64, "value", 1), ConstantExpr)
                                    and _item_field(
                                        _item_field(_subject_64, "value", 1), "value", 1
                                    )
                                    is None
                                )
                            )
                            and ((tail_ret := _subject_64) is not None),
                        )[-1]
                        and (tail_ret.from_return_const or is_generator)
                    ):
                        func_stmts.pop(len(func_stmts) - 1)
                    stack.push(
                        FunctionExpr(
                            _item_field(nested_code, "name", 0),
                            parameters,
                            func_stmts,
                            None,
                            is_async,
                            None,
                            return_annotation,
                        )
                    )
            else:
                stack.push(code_val)
        elif ins.op_code == OpCodes.DICT_MERGE:
            if len(stack) >= 2:
                merge_source = stack.pop()
                merge_target = stack.pop()
                if isinstance((dt := merge_target), DictExpr):
                    new_entries = list(dt.entries)
                    _add(new_entries, (None, merge_source))
                    stack.push(DictExpr(new_entries))
                else:
                    stack.push(DictExpr([(None, merge_target), (None, merge_source)]))
        elif ins.op_code == OpCodes.CALL_FUNCTION_EX:
            has_kw = (ins.op_arg if ins.op_arg is not None else 0) & 1 != 0
            kw_val = None
            if self.is_at_least314():
                if len(stack) > 0:
                    kw_slot = stack.pop()
                    if not (
                        isinstance(kw_slot, NameExpr) and _item_field(kw_slot, "name", 0) == ""
                    ):
                        kw_val = kw_slot
            elif has_kw:
                if len(stack) == 0:
                    raise DecompilationError("CALL_FUNCTION_EX keyword mapping missing")
                kw_val = stack.pop()
            if len(stack) == 0:
                raise DecompilationError("CALL_FUNCTION_EX positional tuple missing")
            varargs_val = stack.pop()
            if len(stack) == 0:
                raise DecompilationError("CALL_FUNCTION_EX callable missing")
            callable_ = stack.pop()
            if (
                isinstance(callable_, NameExpr)
                and _item_field(callable_, "name", 0) == ""
                and (len(stack) > 0)
            ):
                callable_ = stack.pop()
            elif (
                len(stack) > 0
                and (
                    (_subject_65 := stack.peek()),
                    isinstance((null_name_ex := _subject_65), NameExpr),
                )[-1]
                and _empty_string(_item_field(null_name_ex, "name", 0))
            ):
                stack.pop()
            call_args = []
            call_kwargs = []
            kw_sequence = None
            if isinstance((te_args := varargs_val), TupleExpr):
                call_args.extend(te_args.elements)
            elif isinstance((le_args := varargs_val), ListExpr):
                _add(call_args, StarredExpr(le_args))
            elif (
                isinstance(varargs_val, ConstantExpr)
                and isinstance(_item_field(varargs_val, "value", 1), list)
                and (len(_item_field(varargs_val, "value", 1)) == 0)
                and ((empty_args := varargs_val) is not None)
            ):
                empty_tuple_is_last_const = (
                    len(self._code.constants) > 0
                    and (
                        (_subject_66 := self._code.constants[-1]),
                        isinstance(_subject_66, list) and len(_subject_66) == 0,
                    )[-1]
                )
                if not empty_tuple_is_last_const:
                    _add(call_args, StarredExpr(empty_args))
            elif varargs_val is not None:
                _add(call_args, StarredExpr(varargs_val))
            if isinstance((de_kw := kw_val), DictExpr) and (
                not _any(de_kw.entries, lambda e: _item_field(e, "key", 0) is None)
            ):
                _add(call_args, DoubleStarredExpr(de_kw))
            elif isinstance((de_kw_merged := kw_val), DictExpr):
                kw_sequence = []
                for k, v in _iter_values(de_kw_merged.entries):
                    if (
                        isinstance(k, ConstantExpr)
                        and type(_item_field(k, "value", 1)) is str
                        and ((k_str := _item_field(k, "value", 1)) is not None)
                    ):
                        _add(call_kwargs, (k_str, v))
                        _add(kw_sequence, (k_str, v))
                    elif isinstance((kn := k), NameExpr):
                        _add(call_kwargs, (_item_field(kn, "name", 0), v))
                        _add(kw_sequence, (_item_field(kn, "name", 0), v))
                    elif k is None:
                        _add(kw_sequence, (None, v))
                    else:
                        _add(call_kwargs, (str(k), v))
                        _add(kw_sequence, (str(k), v))
            elif kw_val is not None:
                _add(call_args, DoubleStarredExpr(kw_val))
            stack.push(CallExpr(callable_, call_args, call_kwargs, kw_sequence))
        elif ins.op_code == OpCodes.BINARY_SUBSCR:
            sub_index = stack.pop()
            sub_target = stack.pop()
            stack.push(SubscriptExpr(sub_target, sub_index))
        elif ins.op_code == OpCodes.LIST_EXTEND:
            extend_items = stack.pop()
            base_list = stack.pop()
            if isinstance((le := base_list), ListExpr):
                elements = list(le.elements)
                if (
                    isinstance(extend_items, ConstantExpr)
                    and isinstance((col := _item_field(extend_items, "value", 1)), Iterable)
                    and (not type(_item_field(extend_items, "value", 1)) is str)
                    and (len(elements) == 0)
                ):
                    for item in _iter_values(col):
                        _add(elements, ConstantExpr(item))
                else:
                    _add(elements, StarredExpr(extend_items))
                stack.push(ListExpr(elements))
            else:
                stack.push(base_list)
        elif ins.op_code in (OpCodes.DICT_UPDATE, OpCodes.DICT_MERGE):
            update_items = stack.pop()
            base_dict = stack.pop()
            if isinstance((target_dict_expr := base_dict), DictExpr):
                du_entries = list(target_dict_expr.entries)
                _add(du_entries, (None, update_items))
                stack.push(DictExpr(du_entries))
            else:
                stack.push(base_dict)
        elif ins.op_code == OpCodes.LOAD_CLOSURE:
            closure_name = (
                _value_68
                if (
                    _value_68 := (
                        str(_subject_67)
                        if (
                            _subject_67 := (
                                ins.arg_repr if ins.arg_repr is not None else ins.arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            stack.push(NameExpr(closure_name))
        elif ins.op_code == OpCodes.YIELD_VALUE:
            yield_val = stack.pop()
            stack.push(YieldExpr(yield_val))
        elif ins.op_code in (OpCodes.MAKE_CELL, OpCodes.COPY_FREE_VARS, OpCodes.RETURN_GENERATOR):
            pass
        elif ins.op_code in (OpCodes.RESUME, OpCodes.NOP, OpCodes.CACHE, OpCodes.EXTENDED_ARG):
            pass

    @staticmethod
    def get_compare_op_string(ins):
        if ins.op_code == OpCodes.IS_OP:
            return "is not" if (ins.op_arg if ins.op_arg is not None else 0) == 1 else "is"
        if ins.op_code == OpCodes.CONTAINS_OP:
            return "not in" if (ins.op_arg if ins.op_arg is not None else 0) == 1 else "in"
        return (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else "=="
        )

    def try_decompile_except_return(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        if _item_field(self._code, "name", 0) == "<module>":
            return (False, next_idx)
        ins = self._instrs[idx]
        if ins.op_code == OpCodes.SWAP and (ins.op_arg if ins.op_arg is not None else 2) == 2:
            p = idx + 1
            while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                p += 1
            if p < len(self._instrs) and self._instrs[p].op_code == OpCodes.POP_EXCEPT:
                p += 1
                while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                    p += 1
                (
                    (_parsed_2 := self.skip_synthetic_var_cleanup(p)),
                    (p := _parsed_2[1]),
                    _parsed_2[0],
                )[-1]
                while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                    p += 1
                if p < len(self._instrs) and self._instrs[p].op_code in (
                    OpCodes.RETURN_VALUE,
                    OpCodes.RETURN_CONST,
                ):
                    ret_val = (
                        ConstantExpr(self._instrs[p].arg_val)
                        if self._instrs[p].op_code == OpCodes.RETURN_CONST
                        else stack.pop()
                        if len(stack) > 0
                        else None
                    )
                    _add(
                        stmts,
                        ReturnStmt(
                            ret_val,
                            from_return_const=self._instrs[p].op_code == OpCodes.RETURN_CONST,
                        ),
                    )
                    next_idx = p + 1
                    return (True, next_idx)
        elif ins.op_code == OpCodes.POP_EXCEPT:
            p = idx + 1
            while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                p += 1
            ((_parsed_5 := self.skip_synthetic_var_cleanup(p)), (p := _parsed_5[1]), _parsed_5[0])[
                -1
            ]
            while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.EXTENDED_ARG:
                p += 1
            if p < len(self._instrs) and self._instrs[p].op_code in (
                OpCodes.RETURN_VALUE,
                OpCodes.RETURN_CONST,
            ):
                ret_val = (
                    (
                        ConstantExpr(self._instrs[p].arg_val)
                        if self._instrs[p].arg_val is not None
                        else None
                    )
                    if self._instrs[p].op_code == OpCodes.RETURN_CONST
                    else stack.pop()
                    if len(stack) > 0
                    else None
                )
                _add(stmts, ReturnStmt(ret_val))
                next_idx = p + 1
                return (True, next_idx)
        return (False, next_idx)

    def try_match_chained_branch(self, idx, stack):
        condition = None
        true_offset = 0
        false_offset = 0
        next_idx = 0
        is_assert = False
        condition = None
        true_offset = -1
        false_offset = -1
        next_idx = idx
        is_assert = False
        if len(stack) < 2:
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        if idx + 3 >= len(self._instrs):
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        if (
            self._instrs[idx].op_code != OpCodes.SWAP
            or (_value_1 if (_value_1 := self._instrs[idx].op_arg) is not None else 0) != 2
        ):
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        if (
            self._instrs[idx + 1].op_code != OpCodes.COPY
            or (_value_2 if (_value_2 := self._instrs[idx + 1].op_arg) is not None else 0) != 2
        ):
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        if not self._instrs[idx + 2].op_code in (
            OpCodes.COMPARE_OP,
            OpCodes.IS_OP,
            OpCodes.CONTAINS_OP,
        ):
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        if self._instrs[idx + 3].op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, condition, true_offset, false_offset, next_idx, is_assert)
        op1 = StructuralDecompiler.get_compare_op_string(self._instrs[idx + 2])
        first_false_target = (
            _value_3 if (_value_3 := self._instrs[idx + 3].target_offset) is not None else 0
        )
        second = stack.pop()
        first = stack.pop()
        comparators = [(op1, second)]
        cur_idx = idx + 4
        eval_stack = SymbolicStack()
        final_false_target = first_false_target
        final_jump_is_true = False
        matched = False
        while cur_idx < len(self._instrs):
            if (
                self._instrs[cur_idx].op_code == OpCodes.SWAP
                and (_value_4 if (_value_4 := self._instrs[cur_idx].op_arg) is not None else 0) == 2
                and (cur_idx + 3 < len(self._instrs))
                and (self._instrs[cur_idx + 1].op_code == OpCodes.COPY)
                and (
                    (_value_5 if (_value_5 := self._instrs[cur_idx + 1].op_arg) is not None else 0)
                    == 2
                )
                and (
                    self._instrs[cur_idx + 2].op_code
                    in (OpCodes.COMPARE_OP, OpCodes.IS_OP, OpCodes.CONTAINS_OP)
                )
                and (self._instrs[cur_idx + 3].op_code == OpCodes.POP_JUMP_IF_FALSE)
            ):
                if len(eval_stack) == 0:
                    raise DecompilationError("chained comparison operand missing")
                next_right = eval_stack.pop()
                next_op = StructuralDecompiler.get_compare_op_string(self._instrs[cur_idx + 2])
                _add(comparators, (next_op, next_right))
                cur_idx += 4
                eval_stack.clear()
                continue
            if (
                self._instrs[cur_idx].op_code
                in (OpCodes.COMPARE_OP, OpCodes.IS_OP, OpCodes.CONTAINS_OP)
                and cur_idx + 1 < len(self._instrs)
                and (
                    self._instrs[cur_idx + 1].op_code
                    in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                )
            ):
                if len(eval_stack) == 0:
                    raise DecompilationError("chained comparison final operand missing")
                last_right = eval_stack.pop()
                last_op = StructuralDecompiler.get_compare_op_string(self._instrs[cur_idx])
                _add(comparators, (last_op, last_right))
                final_false_target = (
                    _value_9
                    if (_value_9 := self._instrs[cur_idx + 1].target_offset) is not None
                    else 0
                )
                final_jump_is_true = self._instrs[cur_idx + 1].op_code == OpCodes.POP_JUMP_IF_TRUE
                cur_idx += 2
                matched = True
                break
            if (
                (_parsed_10 := self.try_decompile_walrus(cur_idx, eval_stack)),
                (cur_idx := _parsed_10[1]),
                _parsed_10[0],
            )[-1]:
                continue
            self.execute_instruction(self._instrs[cur_idx], eval_stack, None)
            cur_idx += 1
        if (
            matched
            and cur_idx < len(self._instrs)
            and (self._instrs[cur_idx].op_code == OpCodes.LOAD_ASSERTION_ERROR)
            and (cur_idx + 1 < len(self._instrs))
            and (self._instrs[cur_idx + 1].op_code == OpCodes.RAISE_VARARGS)
        ):
            is_assert = True
            is_true_jump = self._instrs[cur_idx - 1].op_code == OpCodes.POP_JUMP_IF_TRUE
            condition = (
                CompareExpr(first, comparators)
                if is_true_jump
                else self.invert_condition(CompareExpr(first, comparators))
            )
            true_offset = final_false_target
            false_offset = self._instrs[cur_idx].offset
            next_idx = (
                t_idx
                if (t_idx := self._offset_to_index.get(true_offset, _MISSING)) is not _MISSING
                else cur_idx + 2
            )
            return (True, condition, true_offset, false_offset, next_idx, is_assert)
        if (
            matched
            and cur_idx < len(self._instrs)
            and (self._instrs[cur_idx].op_code == OpCodes.JUMP_FORWARD)
        ):
            if final_jump_is_true:
                true_offset = final_false_target
                false_offset = (
                    _value_11
                    if (_value_11 := self._instrs[cur_idx].target_offset) is not None
                    else 0
                )
            else:
                true_offset = (
                    _value_12
                    if (_value_12 := self._instrs[cur_idx].target_offset) is not None
                    else 0
                )
                false_offset = final_false_target
            condition = CompareExpr(first, comparators)
            continuation_offset = false_offset if final_jump_is_true else true_offset
            next_idx = (
                t_idx
                if (t_idx := self._offset_to_index.get(continuation_offset, _MISSING))
                is not _MISSING
                else cur_idx + 1
            )
            return (True, condition, true_offset, false_offset, next_idx, is_assert)
        stack.push(first)
        stack.push(second)
        return (False, condition, true_offset, false_offset, next_idx, is_assert)

    def try_match_chain_link(self, idx):
        op = None
        link_len = 0
        jump_target = 0
        has_copy = False
        op = ""
        link_len = 0
        jump_target = -1
        has_copy = False
        if idx + 5 >= len(self._instrs):
            return (False, op, link_len, jump_target, has_copy)
        if (
            self._instrs[idx].op_code != OpCodes.SWAP
            or (_value_1 if (_value_1 := self._instrs[idx].op_arg) is not None else 0) != 2
        ):
            return (False, op, link_len, jump_target, has_copy)
        if idx + 3 >= len(self._instrs):
            return (False, op, link_len, jump_target, has_copy)
        if (
            self._instrs[idx + 1].op_code != OpCodes.COPY
            or (_value_2 if (_value_2 := self._instrs[idx + 1].op_arg) is not None else 0) != 2
        ):
            return (False, op, link_len, jump_target, has_copy)
        if not self._instrs[idx + 2].op_code in (
            OpCodes.COMPARE_OP,
            OpCodes.IS_OP,
            OpCodes.CONTAINS_OP,
        ):
            return (False, op, link_len, jump_target, has_copy)
        if (
            self._instrs[idx + 3].op_code != OpCodes.COPY
            or (_value_3 if (_value_3 := self._instrs[idx + 3].op_arg) is not None else 0) != 1
        ):
            return (False, op, link_len, jump_target, has_copy)
        p = idx + 4
        while p < len(self._instrs) and self._instrs[p].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            p += 1
        if p >= len(self._instrs) or self._instrs[p].op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, op, link_len, jump_target, has_copy)
        jump_idx = (p := (p + 1)) - 1
        while p < len(self._instrs) and self._instrs[p].op_code == OpCodes.NOT_TAKEN:
            p += 1
        if p >= len(self._instrs) or self._instrs[p].op_code != OpCodes.POP_TOP:
            return (False, op, link_len, jump_target, has_copy)
        op = StructuralDecompiler.get_compare_op_string(self._instrs[idx + 2])
        has_copy = True
        jump_target = (
            _value_5 if (_value_5 := self._instrs[jump_idx].target_offset) is not None else 0
        )
        link_len = p + 1 - idx
        return (True, op, link_len, jump_target, has_copy)

    def try_match_legacy_chain_link(self, idx):
        op = None
        link_len = 0
        jump_target = 0
        op = ""
        link_len = 0
        jump_target = -1
        if idx + 5 >= len(self._instrs):
            return (False, op, link_len, jump_target)
        if self._instrs[idx].op_code != OpCodes.LEGACY_DUP_TOP:
            return (False, op, link_len, jump_target)
        if self._instrs[idx + 1].op_code != OpCodes.LEGACY_ROT_THREE:
            return (False, op, link_len, jump_target)
        if not self._instrs[idx + 2].op_code in (
            OpCodes.COMPARE_OP,
            OpCodes.IS_OP,
            OpCodes.CONTAINS_OP,
        ):
            return (False, op, link_len, jump_target)
        if (
            self._instrs[idx + 3].op_code != OpCodes.COPY
            or (_value_1 if (_value_1 := self._instrs[idx + 3].op_arg) is not None else 0) != 1
        ):
            return (False, op, link_len, jump_target)
        if self._instrs[idx + 4].op_code != OpCodes.POP_JUMP_IF_FALSE:
            return (False, op, link_len, jump_target)
        if self._instrs[idx + 5].op_code != OpCodes.POP_TOP:
            return (False, op, link_len, jump_target)
        op = StructuralDecompiler.get_compare_op_string(self._instrs[idx + 2])
        jump_target = (
            _value_2 if (_value_2 := self._instrs[idx + 4].target_offset) is not None else 0
        )
        link_len = 6
        return (True, op, link_len, jump_target)

    def try_match_any_chain_link(self, idx):
        op = None
        link_len = 0
        exit_offset = 0
        has_copy = False
        if (
            (_parsed_1 := self.try_match_chain_link(idx)),
            (op := _parsed_1[1]),
            (link_len := _parsed_1[2]),
            (exit_offset := _parsed_1[3]),
            (has_copy := _parsed_1[4]),
            _parsed_1[0],
        )[-1]:
            return (True, op, link_len, exit_offset, has_copy)
        if (
            (_parsed_2 := self.try_match_legacy_chain_link(idx)),
            (op := _parsed_2[1]),
            (link_len := _parsed_2[2]),
            (exit_offset := _parsed_2[3]),
            _parsed_2[0],
        )[-1]:
            has_copy = True
            return (True, op, link_len, exit_offset, has_copy)
        return (False, op, link_len, exit_offset, has_copy)

    def try_decompile_chained_comparison(self, idx, stack):
        chained_expr = None
        next_idx = 0
        chained_expr = None
        next_idx = idx
        is_legacy_chain = (
            (_parsed_1 := self.try_match_legacy_chain_link(idx)),
            (op1 := _parsed_1[1]),
            (link_len := _parsed_1[2]),
            (exit_offset := _parsed_1[3]),
            _parsed_1[0],
        )[-1]
        if not is_legacy_chain and (
            not (
                (_parsed_2 := self.try_match_chain_link(idx)),
                (op1 := _parsed_2[1]),
                (link_len := _parsed_2[2]),
                (exit_offset := _parsed_2[3]),
                _parsed_2[0],
            )[-1]
        ):
            return (False, chained_expr, next_idx)
        has_copy = True
        if len(stack) < 2:
            return (False, chained_expr, next_idx)
        second = stack.pop()
        first = stack.pop()
        exit_idx = self._offset_to_index.get(exit_offset)
        if exit_idx is None:
            return (False, chained_expr, next_idx)
        comparators = [(op1, second)]
        cur_idx = idx + link_len
        sub_stack = SymbolicStack()
        while cur_idx < len(self._instrs):
            if (
                (_parsed_3 := self.try_match_any_chain_link(cur_idx)),
                (next_op := _parsed_3[1]),
                (next_len := _parsed_3[2]),
                _parsed_3[0],
            )[-1]:
                next_right = sub_stack.pop()
                _add(comparators, (next_op, next_right))
                cur_idx += next_len
                continue
            if self._instrs[cur_idx].op_code in (
                OpCodes.COMPARE_OP,
                OpCodes.IS_OP,
                OpCodes.CONTAINS_OP,
            ):
                last_op = StructuralDecompiler.get_compare_op_string(self._instrs[cur_idx])
                if len(sub_stack) == 0:
                    raise DecompilationError("chained comparison operand missing")
                last_right = sub_stack.pop()
                _add(comparators, (last_op, last_right))
                cur_idx += 1
                if (
                    is_legacy_chain
                    and cur_idx < len(self._instrs)
                    and (self._instrs[cur_idx].op_code == OpCodes.JUMP_FORWARD)
                    and (
                        (_subject_5 := self._instrs[cur_idx].target_offset),
                        type(_subject_5) is int and (chain_end_target := _subject_5) is not None,
                    )[-1]
                    and (
                        (chain_end_idx := self._offset_to_index.get(chain_end_target, _MISSING))
                        is not _MISSING
                    )
                ):
                    cur_idx = chain_end_idx
                    has_copy = False
                break
            if (
                (_parsed_6 := self.try_decompile_walrus(cur_idx, sub_stack)),
                (cur_idx := _parsed_6[1]),
                _parsed_6[0],
            )[-1]:
                continue
            self.execute_instruction(self._instrs[cur_idx], sub_stack, None)
            cur_idx += 1
        chained_expr = CompareExpr(first, comparators)
        if has_copy:
            after_exit_idx = exit_idx
            while after_exit_idx < len(self._instrs) and self._instrs[after_exit_idx].op_code in (
                OpCodes.SWAP,
                OpCodes.POP_TOP,
            ):
                after_exit_idx += 1
            next_idx = max(cur_idx, after_exit_idx)
        else:
            next_idx = cur_idx
        return (True, chained_expr, next_idx)

    @staticmethod
    def is_assertion_error_load(ins):
        if ins.op_code == OpCodes.LOAD_ASSERTION_ERROR:
            return True
        if not ins.op_code in (OpCodes.LOAD_GLOBAL, OpCodes.LOAD_NAME):
            return False
        name = (
            _value_2
            if (
                _value_2 := (
                    str(_subject_1)
                    if (_subject_1 := (ins.arg_repr if ins.arg_repr is not None else ins.arg_val))
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        return name == "AssertionError"

    def try_decompile_assert(self, idx, stack, stmts):
        next_idx = 0
        next_idx = idx
        ins = self._instrs[idx]
        if os.getenv("PYCRUNE_TRACE") == "1":
            _trace(
                "[trace] TryDecompileAssert idx={} off={} op={} stack={}".format(
                    idx, ins.offset, ins.op_code, len(stack)
                )
            )
        if not ins.op_code in (
            OpCodes.POP_JUMP_IF_TRUE,
            OpCodes.POP_JUMP_IF_FALSE,
            OpCodes.POP_JUMP_IF_NONE,
            OpCodes.POP_JUMP_IF_NOT_NONE,
        ):
            return (False, next_idx)
        target_offset = ins.target_offset if ins.target_offset is not None else 0
        if not (target_idx := self._offset_to_index.get(target_offset, _MISSING)) is not _MISSING:
            return (False, next_idx)
        if idx + 1 < len(self._instrs) and StructuralDecompiler.is_assertion_error_load(
            self._instrs[idx + 1]
        ):
            if len(stack) == 0:
                return (False, next_idx)
            cond = stack.pop()
            if ins.op_code == OpCodes.POP_JUMP_IF_FALSE:
                cond = UnaryExpr("not", cond)
            elif ins.op_code == OpCodes.POP_JUMP_IF_NONE:
                cond = CompareExpr(cond, [("is", ConstantExpr(None))])
            elif ins.op_code == OpCodes.POP_JUMP_IF_NOT_NONE:
                cond = CompareExpr(cond, [("is not", ConstantExpr(None))])
            msg = None
            raise_idx = idx + 2
            while (
                raise_idx < target_idx and self._instrs[raise_idx].op_code != OpCodes.RAISE_VARARGS
            ):
                raise_idx += 1
            if raise_idx > idx + 2 and raise_idx < target_idx:
                msg_start = idx + 2
                if (
                    raise_idx - 1 > idx
                    and self._instrs[raise_idx - 1].op_code == OpCodes.CALL
                    and (
                        (
                            _value_1
                            if (_value_1 := self._instrs[raise_idx - 1].op_arg) is not None
                            else 0
                        )
                        > 0
                    )
                ):
                    msg_start = idx + 1
                msg_stack = SymbolicStack()
                self.decompile_range(msg_start, raise_idx, msg_stack)
                if len(msg_stack) > 0:
                    msg = msg_stack.pop()
            if isinstance((ce := msg), CallExpr):
                is_assert_ctor = isinstance(ce.function, AssertionErrorExpr) or (
                    isinstance(ce.function, NameExpr)
                    and _item_field(ce.function, "name", 0) == "AssertionError"
                )
                if is_assert_ctor and len(ce.arguments) == 1:
                    msg = ce.arguments[0]
                elif len(ce.arguments) == 0:
                    msg = ce.function
            _add(stmts, AssertStmt(cond, msg))
            next_idx = target_idx
            return (True, next_idx)
        return (False, next_idx)

    @staticmethod
    def try_build_assert_from_raise(exc, stmts):
        if stmts is None:
            return False
        if isinstance(exc, AssertionErrorExpr):
            _add(stmts, AssertStmt(ConstantExpr(False), None))
            return True
        elif (
            isinstance(exc, CallExpr)
            and isinstance(exc.function, AssertionErrorExpr)
            and ((ce := exc) is not None)
        ):
            _add(
                stmts,
                AssertStmt(ConstantExpr(False), ce.arguments[0] if len(ce.arguments) > 0 else None),
            )
            return True
        else:
            return False

    def try_decompile_import(self, idx, stmts):
        instructions = self._instrs
        limit = len(instructions)
        store_ops = {
            OpCodes.STORE_NAME,
            OpCodes.STORE_FAST,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        }

        def skip_extended(index):
            while index < limit and instructions[index].op_code == OpCodes.EXTENDED_ARG:
                index += 1
            return index

        def at(index, opcode):
            return index < limit and instructions[index].op_code == opcode

        def operand(index):
            return instructions[index].arg_val

        level_idx = skip_extended(idx)
        if not at(level_idx, OpCodes.LOAD_CONST):
            return False, idx
        from_idx = skip_extended(level_idx + 1)
        if not at(from_idx, OpCodes.LOAD_CONST):
            return False, idx
        module_idx = skip_extended(from_idx + 1)
        if not at(module_idx, OpCodes.IMPORT_NAME):
            return False, idx
        level = operand(level_idx)
        from_list = operand(from_idx)
        module = operand(module_idx)
        if type(level) is not int or not isinstance(module, str):
            return False, idx
        if module.endswith(" + eager"):
            module = module[:-8]
        cur = skip_extended(module_idx + 1)
        if from_list is None:
            parts = module.split(".")
            # A direct store binds the first component of a dotted import.
            if cur < limit and instructions[cur].op_code in store_ops:
                target = operand(cur)
                stmts.append(
                    ImportStmt([ImportAlias(module, None if target == parts[0] else target)])
                )
                return True, cur + 1
            # An alias for a dotted import walks intermediate components on the stack.
            for position, part in enumerate(parts[1:], start=1):
                if not at(cur, OpCodes.IMPORT_FROM) or operand(cur) != part:
                    return False, idx
                cur = skip_extended(cur + 1)
                if position == len(parts) - 1:
                    if (
                        cur < limit
                        and instructions[cur].op_code in store_ops
                        and at(cur + 1, OpCodes.POP_TOP)
                    ):
                        stmts.append(ImportStmt([ImportAlias(module, operand(cur))]))
                        return True, cur + 2
                    return False, idx
                swap = at(cur, OpCodes.LEGACY_ROT_TWO) or (
                    at(cur, OpCodes.SWAP) and instructions[cur].op_arg == 2
                )
                if not swap or not at(cur + 1, OpCodes.POP_TOP):
                    return False, idx
                cur = skip_extended(cur + 2)
            return False, idx
        if at(cur, OpCodes.IMPORT_STAR):
            stmts.append(ImportFromStmt(module, [ImportAlias("*", None)], level))
            return True, cur + 1
        if (
            at(cur, OpCodes.CALL_INTRINSIC_1)
            and instructions[cur].op_arg == 2
            and at(cur + 1, OpCodes.POP_TOP)
        ):
            stmts.append(ImportFromStmt(module, [ImportAlias("*", None)], level))
            return True, cur + 2

        aliases = []
        while at(cur, OpCodes.IMPORT_FROM):
            imported = operand(cur)
            cur = skip_extended(cur + 1)
            if cur >= limit or instructions[cur].op_code not in store_ops:
                return False, idx
            stored = operand(cur)
            aliases.append(ImportAlias(imported, None if imported == stored else stored))
            cur = skip_extended(cur + 1)
        if aliases and at(cur, OpCodes.POP_TOP):
            stmts.append(ImportFromStmt(module, aliases, level))
            return True, cur + 1
        return False, idx

    def try_decompile_with314(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if idx + 5 >= end_idx or len(stack) == 0:
            return (False, with_stmt, next_idx)
        if (
            self._instrs[idx].op_code != OpCodes.COPY
            or (_value_1 if (_value_1 := self._instrs[idx].op_arg) is not None else 0) != 1
        ):
            return (False, with_stmt, next_idx)
        if self._instrs[idx + 1].op_code != OpCodes.LOAD_SPECIAL:
            return (False, with_stmt, next_idx)
        exit_name = (
            _value_4
            if (
                _value_4 := (
                    str(_subject_3)
                    if (
                        _subject_3 := (
                            _value_2
                            if (_value_2 := self._instrs[idx + 1].arg_repr) is not None
                            else self._instrs[idx + 1].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        is_async = exit_name == "__aexit__"
        if not is_async and exit_name != "__exit__":
            return (False, with_stmt, next_idx)
        enter_name = "__aenter__" if is_async else "__enter__"
        if (
            self._instrs[idx + 2].op_code != OpCodes.SWAP
            or (_value_5 if (_value_5 := self._instrs[idx + 2].op_arg) is not None else 0) != 2
        ):
            return (False, with_stmt, next_idx)
        if (
            self._instrs[idx + 3].op_code != OpCodes.SWAP
            or (_value_6 if (_value_6 := self._instrs[idx + 3].op_arg) is not None else 0) != 3
        ):
            return (False, with_stmt, next_idx)
        if self._instrs[idx + 4].op_code != OpCodes.LOAD_SPECIAL:
            return (False, with_stmt, next_idx)
        actual_enter = (
            _value_9
            if (
                _value_9 := (
                    str(_subject_8)
                    if (
                        _subject_8 := (
                            _value_7
                            if (_value_7 := self._instrs[idx + 4].arg_repr) is not None
                            else self._instrs[idx + 4].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if actual_enter != enter_name:
            return (False, with_stmt, next_idx)
        if (
            self._instrs[idx + 5].op_code != OpCodes.CALL
            or (_value_10 if (_value_10 := self._instrs[idx + 5].op_arg) is not None else 0) != 0
        ):
            return (False, with_stmt, next_idx)
        items = []
        ctx_expr = stack.peek()
        cur = idx + 6
        guard = 0
        while (guard := (guard + 1)) - 1 < 32:
            if is_async:
                cur = self.skip_with_await314(cur, end_idx)
                if cur < 0:
                    return (False, with_stmt, next_idx)
            target = None
            if cur < end_idx and self._instrs[cur].op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                target = NameExpr(
                    _value_14
                    if (
                        _value_14 := (
                            str(_subject_13)
                            if (
                                _subject_13 := (
                                    _value_12
                                    if (_value_12 := self._instrs[cur].arg_repr) is not None
                                    else self._instrs[cur].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                cur += 1
            elif cur < end_idx and self._instrs[cur].op_code == OpCodes.POP_TOP:
                cur += 1
            else:
                return (False, with_stmt, next_idx)
            _add(items, WithItem(ctx_expr, target))
            if cur >= end_idx:
                break
            next_copy = -1
            p = cur
            while p + 1 < end_idx and p < cur + 64:
                if self._instrs[p].op_code in (
                    OpCodes.FOR_ITER,
                    OpCodes.GET_AITER,
                    OpCodes.GET_ANEXT,
                    OpCodes.SEND,
                    OpCodes.RETURN_VALUE,
                    OpCodes.RETURN_CONST,
                    OpCodes.RAISE_VARARGS,
                    OpCodes.JUMP_BACKWARD,
                    OpCodes.JUMP_FORWARD,
                    OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                ):
                    break
                if (
                    self._instrs[p].op_code == OpCodes.COPY
                    and (_value_16 if (_value_16 := self._instrs[p].op_arg) is not None else 0) == 1
                    and (self._instrs[p + 1].op_code == OpCodes.LOAD_SPECIAL)
                ):
                    next_copy = p
                    break
                p += 1
            if next_copy < 0:
                break
            if not self.is_with_header314(next_copy, end_idx, is_async):
                break
            next_stack = SymbolicStack()
            try:
                between = self.decompile_range(cur, next_copy, next_stack)
            except (DecompilationError, IndexError, ValueError):
                break
            if between or len(next_stack) != 1:
                break
            ctx_expr = next_stack.pop()
            cur = next_copy + 6
        if not self.is311() and (not self.is_at_least314()) and (len(items) > 1):
            return (False, with_stmt, next_idx)
        body_start = cur
        if body_start >= end_idx:
            return (False, with_stmt, next_idx)
        body_probe = body_start
        while body_probe < end_idx and self._instrs[body_probe].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            body_probe += 1
        if body_probe >= end_idx:
            return (False, with_stmt, next_idx)
        body_offset = self._instrs[body_probe].offset
        handler_idx = -1
        handler_offset = -1
        body_start_offset = self._instrs[body_start].offset
        best_entry_start = -1
        for entry in _iter_values(self._code.exception_table):
            if (
                _item_field(entry, "start", 0) > body_offset
                or _item_field(entry, "end", 1) < body_start_offset
            ):
                continue
            if not (
                h_idx := self._offset_to_index.get(entry.target, _MISSING)
            ) is not _MISSING or h_idx + 1 >= len(self._instrs):
                continue
            if (
                self._instrs[h_idx].op_code != OpCodes.PUSH_EXC_INFO
                or self._instrs[h_idx + 1].op_code != OpCodes.WITH_EXCEPT_START
            ):
                continue
            if _item_field(entry, "start", 0) > best_entry_start:
                best_entry_start = _item_field(entry, "start", 0)
                handler_idx = h_idx
                handler_offset = entry.target
        cleanup_limit = min(handler_idx, end_idx) if handler_idx >= 0 else end_idx
        all_cleanup_starts = self.find_with_cleanup_starts314(body_start, cleanup_limit, 2147483647)
        cleanup_starts = []
        if handler_offset >= 0:
            if len(items) > 1:
                first = max(0, len(all_cleanup_starts) - len(items))
                c = first
                while c < len(all_cleanup_starts):
                    _add(cleanup_starts, all_cleanup_starts[c])
                    c += 1
            else:
                anchor = -1
                c = len(all_cleanup_starts) - 1
                while c >= 0:
                    if self.is_nested_with_cleanup314(all_cleanup_starts[c], handler_offset):
                        c -= 1
                        continue
                    if self.is_normal_with_exit314(
                        all_cleanup_starts[c], handler_idx, is_async, end_idx
                    ):
                        anchor = c
                        break
                    c -= 1
                if anchor < 0:
                    c = len(all_cleanup_starts) - 1
                    while c >= 0:
                        if not self.is_nested_with_cleanup314(
                            all_cleanup_starts[c], handler_offset
                        ):
                            anchor = c
                            break
                        c -= 1
                if anchor >= 0:
                    _add(cleanup_starts, all_cleanup_starts[anchor])
        elif len(all_cleanup_starts) > len(items):
            cleanup_starts = list(_skip(all_cleanup_starts, len(all_cleanup_starts) - len(items)))
        else:
            cleanup_starts = all_cleanup_starts
        if handler_idx < 0:
            if len(cleanup_starts) > 0:
                last_cleanup = cleanup_starts[-1]
                i = last_cleanup
                while i < len(self._instrs):
                    if self._instrs[i].op_code != OpCodes.PUSH_EXC_INFO:
                        i += 1
                        continue
                    if _any(
                        self._code.exception_table, lambda e: e.target == self._instrs[i].offset
                    ):
                        handler_idx = i
                        break
                    i += 1
            if handler_idx < 0:
                handler_target = -1
                for entry in _iter_values(self._code.exception_table):
                    if (
                        not entry.encloses(body_offset)
                        and _item_field(entry, "start", 0) != body_offset
                        or entry.target <= body_offset
                    ):
                        continue
                    if not (
                        h_idx := self._offset_to_index.get(entry.target, _MISSING)
                    ) is not _MISSING or h_idx >= len(self._instrs):
                        continue
                    if self._instrs[h_idx].op_code != OpCodes.PUSH_EXC_INFO:
                        continue
                    if entry.target > handler_target:
                        handler_target = entry.target
                        handler_idx = h_idx
            if handler_idx < 0:
                return (False, with_stmt, next_idx)
            cleanup_starts = self.find_with_cleanup_starts314(body_start, handler_idx, len(items))
        if len(cleanup_starts) != len(items) or cleanup_starts[0] < body_start:
            if len(cleanup_starts) == 0 and len(items) == 1 and (handler_idx > body_start):
                abrupt_body = self.decompile_range(body_start, handler_idx, stack.clone())
                after_handler = self.find_handler_normal_exit(handler_idx, end_idx)
                if after_handler < 0:
                    return (False, with_stmt, next_idx)
                while after_handler < end_idx and self._instrs[after_handler].op_code in (
                    OpCodes.POP_TOP,
                    OpCodes.NOP,
                    OpCodes.NOT_TAKEN,
                ):
                    after_handler += 1
                if (
                    after_handler < end_idx
                    and self._instrs[after_handler].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (_subject_19 := self._instrs[after_handler].target_offset),
                        type(_subject_19) is int and (jt := _subject_19) is not None,
                    )[-1]
                    and ((j_idx := self._offset_to_index.get(jt, _MISSING)) is not _MISSING)
                ):
                    after_handler = j_idx
                stack.pop()
                with_stmt = WithStmt(items, abrupt_body, is_async)
                next_idx = after_handler
                return (True, with_stmt, next_idx)
            return (False, with_stmt, next_idx)
        body_stack = SymbolicStack()
        body = self.decompile_range(body_start, cleanup_starts[0], body_stack)
        body_return_idx = -1
        q = cleanup_starts[0]
        while q < handler_idx:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                body_return_idx = q
                break
            q += 1
        body_has_return = body_return_idx >= 0 and len(body_stack) > 0
        if body_has_return:
            _add(body, ReturnStmt(body_stack.pop()))
        if body_has_return:
            _add(self._handler_stop_indices, handler_idx)
            stack.pop()
            with_stmt = WithStmt(items, body, is_async)
            next_idx = body_return_idx + 1
            return (True, with_stmt, next_idx)
        after = cleanup_starts[0]
        for _ in _iter_values(cleanup_starts):
            while after < end_idx and (
                not (
                    self._instrs[after].op_code == OpCodes.CALL
                    and (_value_21 if (_value_21 := self._instrs[after].op_arg) is not None else 0)
                    == 3
                )
            ):
                after += 1
            if after >= end_idx:
                return (False, with_stmt, next_idx)
            after += 1
            if is_async:
                after = self.skip_with_await314(after, end_idx)
                if after < 0:
                    return (False, with_stmt, next_idx)
            if after < end_idx and self._instrs[after].op_code == OpCodes.POP_TOP:
                after += 1
        if (
            len(body) == 0
            and after < len(self._instrs)
            and (self._instrs[after].offset in self._active_loop_exits)
        ):
            _add(body, BreakStmt())
        stack.pop()
        with_stmt = WithStmt(items, body, is_async)
        next_idx = after
        return (True, with_stmt, next_idx)

    def skip_with_await314(self, cur, end_idx):
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.GET_AWAITABLE:
            return -1
        cur += 1
        while cur < end_idx and self._instrs[cur].op_code in (
            OpCodes.PUSH_NULL,
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LOAD_CONST:
            return -1
        cur += 1
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.SEND:
            return -1
        if (
            not (
                type(self._instrs[cur].target_offset) is int
                and (target := self._instrs[cur].target_offset) is not None
            )
            or not (end_send_idx := self._offset_to_index.get(target, _MISSING)) is not _MISSING
        ):
            return -1
        if end_send_idx >= end_idx or self._instrs[end_send_idx].op_code != OpCodes.END_SEND:
            return -1
        return end_send_idx + 1

    def is_with_header314(self, at, end_idx, is_async):
        if at + 5 >= end_idx:
            return False
        if (
            self._instrs[at].op_code != OpCodes.COPY
            or (_value_1 if (_value_1 := self._instrs[at].op_arg) is not None else 0) != 1
        ):
            return False
        if self._instrs[at + 1].op_code != OpCodes.LOAD_SPECIAL:
            return False
        exit_name = (
            _value_4
            if (
                _value_4 := (
                    str(_subject_3)
                    if (
                        _subject_3 := (
                            _value_2
                            if (_value_2 := self._instrs[at + 1].arg_repr) is not None
                            else self._instrs[at + 1].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if exit_name != ("__aexit__" if is_async else "__exit__"):
            return False
        if (
            self._instrs[at + 2].op_code != OpCodes.SWAP
            or (_value_5 if (_value_5 := self._instrs[at + 2].op_arg) is not None else 0) != 2
        ):
            return False
        if (
            self._instrs[at + 3].op_code != OpCodes.SWAP
            or (_value_6 if (_value_6 := self._instrs[at + 3].op_arg) is not None else 0) != 3
        ):
            return False
        if self._instrs[at + 4].op_code != OpCodes.LOAD_SPECIAL:
            return False
        enter_name = (
            _value_9
            if (
                _value_9 := (
                    str(_subject_8)
                    if (
                        _subject_8 := (
                            _value_7
                            if (_value_7 := self._instrs[at + 4].arg_repr) is not None
                            else self._instrs[at + 4].arg_val
                        )
                    )
                    is not None
                    else None
                )
            )
            is not None
            else ""
        )
        if enter_name != ("__aenter__" if is_async else "__enter__"):
            return False
        return (
            self._instrs[at + 5].op_code == OpCodes.CALL
            and (_value_10 if (_value_10 := self._instrs[at + 5].op_arg) is not None else 0) == 0
        )

    def is_nested_with_cleanup314(self, cleanup_idx, handler_offset):
        start_offset = self._instrs[cleanup_idx].offset
        return _any(
            self._code.exception_table,
            lambda e: (
                e.target <= handler_offset
                and _item_field(e, "start", 0) <= start_offset
                and (start_offset < _item_field(e, "end", 1))
            ),
        )

    def is_normal_with_exit314(self, cleanup_idx, handler_idx, is_async, end_idx):
        exit_idx = self.find_handler_normal_exit(handler_idx, len(self._instrs))
        if exit_idx < 0:
            return False
        continuation = (
            (_value_1 if (_value_1 := self._instrs[exit_idx].target_offset) is not None else -1)
            if (
                (_subject_2 := self._instrs[exit_idx].op_code),
                _subject_2 == OpCodes.JUMP_FORWARD
                or _subject_2 == OpCodes.JUMP_BACKWARD
                or _subject_2 == OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            )[-1]
            else self._instrs[exit_idx].offset
        )
        if continuation < 0:
            return False
        limit = len(self._instrs)
        after = cleanup_idx
        while after < limit and (
            not (
                self._instrs[after].op_code == OpCodes.CALL
                and (_value_3 if (_value_3 := self._instrs[after].op_arg) is not None else 0) == 3
            )
        ):
            after += 1
        if after >= limit:
            return False
        after += 1
        if is_async:
            after = self.skip_with_await314(after, limit)
            if after < 0:
                return False
        if after < limit and self._instrs[after].op_code == OpCodes.POP_TOP:
            after += 1
        if after >= limit:
            return False
        if self._instrs[after].offset == continuation:
            return True
        return (
            self._instrs[after].op_code
            in (OpCodes.JUMP_FORWARD, OpCodes.JUMP_BACKWARD, OpCodes.JUMP_BACKWARD_NO_INTERRUPT)
            and (_value_5 if (_value_5 := self._instrs[after].target_offset) is not None else -1)
            == continuation
        )

    def find_with_cleanup_starts314(self, body_start, handler_idx, count):
        starts = []
        i = body_start
        while i < handler_idx:
            if (
                self._instrs[i].op_code != OpCodes.CALL
                or (_value_1 if (_value_1 := self._instrs[i].op_arg) is not None else 0) != 3
            ):
                i += 1
                continue
            q = i - 1
            consts = 0
            first_const = -1
            while q >= body_start and self._instrs[q].op_code in (
                OpCodes.LOAD_CONST,
                OpCodes.PUSH_NULL,
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                if self._instrs[q].op_code == OpCodes.LOAD_CONST:
                    consts += 1
                    first_const = q
                q -= 1
            if consts < 3:
                i += 1
                continue
            start = first_const
            before = first_const - 1
            while before >= body_start and self._instrs[before].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                before -= 1
            if (
                before >= body_start
                and self._instrs[before].op_code == OpCodes.SWAP
                and (
                    (_value_4 if (_value_4 := self._instrs[before].op_arg) is not None else 0) == 2
                )
            ):
                before -= 1
                while before >= body_start and self._instrs[before].op_code in (
                    OpCodes.NOP,
                    OpCodes.NOT_TAKEN,
                ):
                    before -= 1
                if (
                    before >= body_start
                    and self._instrs[before].op_code == OpCodes.SWAP
                    and (
                        (_value_6 if (_value_6 := self._instrs[before].op_arg) is not None else 0)
                        == 3
                    )
                ):
                    start = before
            _add(starts, start)
            i += 1
        if len(starts) > count:
            _remove_range(starts, 0, len(starts) - count)
        return starts

    def try_decompile_with311(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if not self.is_at_least311():
            return (False, with_stmt, next_idx)
        if idx >= len(self._instrs) or self._instrs[idx].op_code != OpCodes.BEFORE_WITH:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        items = []
        ctx_expr = stack.peek()
        cur = idx + 1
        guard = 0
        while (guard := (guard + 1)) - 1 < 32:
            target = None
            if cur < end_idx and self._instrs[cur].op_code in (
                OpCodes.STORE_FAST,
                OpCodes.STORE_NAME,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                target = NameExpr(
                    _value_4
                    if (
                        _value_4 := (
                            str(_subject_3)
                            if (
                                _subject_3 := (
                                    _value_2
                                    if (_value_2 := self._instrs[cur].arg_repr) is not None
                                    else self._instrs[cur].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                cur += 1
            elif cur < end_idx and self._instrs[cur].op_code == OpCodes.POP_TOP:
                cur += 1
            _add(items, WithItem(ctx_expr, target))
            next_before = -1
            p = cur
            while p < end_idx and p < cur + 64:
                op = self._instrs[p].op_code
                if op == OpCodes.BEFORE_WITH:
                    next_before = p
                    break
                if OpCodes.is_jump(op) or OpCodes.is_terminator(op):
                    break
                p += 1
            if next_before < 0:
                break
            next_stack = SymbolicStack()
            try:
                between = self.decompile_range(cur, next_before, next_stack)
            except (DecompilationError, IndexError, ValueError):
                break
            if between or len(next_stack) != 1:
                break
            ctx_expr = next_stack.pop()
            cur = next_before + 1
        body_start = cur
        if body_start >= end_idx:
            return (False, with_stmt, next_idx)
        body_probe = body_start
        while body_probe < end_idx and self._instrs[body_probe].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
        ):
            body_probe += 1
        if body_probe >= end_idx:
            return (False, with_stmt, next_idx)
        body_offset = self._instrs[body_probe].offset
        handler_idx = -1
        handler_target = -1
        for entry in _iter_values(self._code.exception_table):
            if (
                not entry.encloses(body_offset)
                and _item_field(entry, "start", 0) != body_offset
                or entry.target <= body_offset
            ):
                continue
            if not (
                h_idx := self._offset_to_index.get(entry.target, _MISSING)
            ) is not _MISSING or h_idx >= len(self._instrs):
                continue
            if self._instrs[h_idx].op_code != OpCodes.PUSH_EXC_INFO:
                continue
            if entry.target > handler_target:
                handler_target = entry.target
                handler_idx = h_idx
        if handler_idx < 0:
            return (False, with_stmt, next_idx)
        cleanup_starts = []
        cleanup_limit = handler_idx if len(items) == 1 else end_idx
        i = body_start
        while i < cleanup_limit:
            if (
                self._instrs[i].op_code != OpCodes.CALL
                or (_value_6 if (_value_6 := self._instrs[i].op_arg) is not None else 0) != 2
            ):
                i += 1
                continue
            q = i - 1
            consts = 0
            while q >= body_start and self._instrs[q].op_code in (
                OpCodes.LOAD_CONST,
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                if self._instrs[q].op_code == OpCodes.LOAD_CONST:
                    if not self.is_none_constant(self._instrs[q]):
                        break
                    consts += 1
                q -= 1
            if consts != 3:
                i += 1
                continue
            after = self.skip_transparent(i + 1)
            if after >= cleanup_limit or self._instrs[after].op_code != OpCodes.POP_TOP:
                i += 1
                continue
            start = q + 1
            if q >= body_start and self._instrs[q].op_code == OpCodes.SWAP:
                if self._instrs[q].op_arg == 2:
                    start = q
            if len(items) == 1:
                cleanup_starts[:] = [start]
            elif len(cleanup_starts) < len(items):
                cleanup_starts.append(start)
            i += 1
        if len(cleanup_starts) != len(items):
            if len(cleanup_starts) == 0 and len(items) == 1 and (handler_idx > body_start):
                abrupt_body = self.decompile_range(body_start, handler_idx, stack.clone())
                after_handler = self.find_handler_normal_exit(handler_idx, end_idx)
                if after_handler < 0:
                    return (False, with_stmt, next_idx)
                while after_handler < end_idx and self._instrs[after_handler].op_code in (
                    OpCodes.POP_TOP,
                    OpCodes.NOP,
                    OpCodes.NOT_TAKEN,
                ):
                    after_handler += 1
                if (
                    after_handler < end_idx
                    and self._instrs[after_handler].op_code == OpCodes.JUMP_FORWARD
                    and (
                        (_subject_9 := self._instrs[after_handler].target_offset),
                        type(_subject_9) is int and (jt := _subject_9) is not None,
                    )[-1]
                    and ((j_idx := self._offset_to_index.get(jt, _MISSING)) is not _MISSING)
                ):
                    after_handler = j_idx
                stack.pop()
                with_stmt = WithStmt(items, abrupt_body, is_async=False)
                next_idx = after_handler
                return (True, with_stmt, next_idx)
            return (False, with_stmt, next_idx)
        if not self.is311():
            b = body_start
            while b < cleanup_starts[0]:
                if self._instrs[b].op_code in (OpCodes.BEFORE_WITH, OpCodes.BEFORE_ASYNC_WITH):
                    return (False, with_stmt, next_idx)
                b += 1
        body_return_scan_start = cleanup_starts[0]
        if len(items) == 1 and len(cleanup_starts) > 1:
            cleanup_starts = [cleanup_starts[-1]]
        body_stack = SymbolicStack()
        body = self.decompile_range(body_start, cleanup_starts[0], body_stack)
        body_return_idx = -1
        q = body_return_scan_start
        while q < handler_idx:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                body_return_idx = q
                break
            q += 1
        body_has_return = body_return_idx >= 0 and len(body_stack) > 0
        if body_has_return:
            _add(body, ReturnStmt(body_stack.pop()))
        if body_has_return:
            _add(self._handler_stop_indices, handler_idx)
            stack.pop()
            with_stmt = WithStmt(items, body, is_async=False)
            next_idx = body_return_idx + 1
            return (True, with_stmt, next_idx)
        after = cleanup_starts[0]
        for _ in _iter_values(cleanup_starts):
            while after < end_idx and (
                not (
                    self._instrs[after].op_code == OpCodes.CALL
                    and (_value_12 if (_value_12 := self._instrs[after].op_arg) is not None else 0)
                    == 2
                )
            ):
                after += 1
            if after >= end_idx:
                return (False, with_stmt, next_idx)
            after += 1
            while after < end_idx and self._instrs[after].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                after += 1
            if after < end_idx and self._instrs[after].op_code == OpCodes.POP_TOP:
                after += 1
            while after < end_idx and self._instrs[after].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                after += 1
            if after < end_idx and self._instrs[after].op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                jt = (
                    _value_16
                    if (_value_16 := self._instrs[after].target_offset) is not None
                    else -1
                )
                if self.is_active_loop_header(jt):
                    _add(self._handler_stop_indices, handler_idx)
                    after = handler_idx
                elif jt in self._active_loop_exits:
                    pass
                elif (j_idx := self._offset_to_index.get(jt, _MISSING)) is not _MISSING:
                    after = j_idx
        stack.pop()
        with_stmt = WithStmt(items, body, is_async=False)
        next_idx = after
        return (True, with_stmt, next_idx)

    def try_decompile_async_with311(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if idx >= len(self._instrs) or self._instrs[idx].op_code != OpCodes.BEFORE_ASYNC_WITH:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        cur = idx + 1
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.GET_AWAITABLE:
            return (False, with_stmt, next_idx)
        cur += 1
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LOAD_CONST:
            return (False, with_stmt, next_idx)
        cur += 1
        while cur < end_idx and self._instrs[cur].op_code in (OpCodes.NOP, OpCodes.NOT_TAKEN):
            cur += 1
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.SEND:
            return (False, with_stmt, next_idx)
        if (
            not (
                type(self._instrs[cur].target_offset) is int
                and (send_target := self._instrs[cur].target_offset) is not None
            )
            or not (bind_idx := self._offset_to_index.get(send_target, _MISSING)) is not _MISSING
        ):
            return (False, with_stmt, next_idx)
        body_start = bind_idx
        if body_start < end_idx and self._instrs[body_start].op_code == OpCodes.END_SEND:
            body_start += 1
        target = None
        if body_start < end_idx and self._instrs[body_start].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target = NameExpr(
                _value_7
                if (
                    _value_7 := (
                        str(_subject_6)
                        if (
                            _subject_6 := (
                                _value_5
                                if (_value_5 := self._instrs[body_start].arg_repr) is not None
                                else self._instrs[body_start].arg_val
                            )
                        )
                        is not None
                        else None
                    )
                )
                is not None
                else ""
            )
            body_start += 1
        elif body_start < end_idx and self._instrs[body_start].op_code == OpCodes.POP_TOP:
            body_start += 1
        if body_start >= end_idx:
            return (False, with_stmt, next_idx)
        cleanup_start = -1
        call_idx = -1
        handler_idx = -1
        if self.is311():
            i = body_start
            while i < end_idx:
                if self._instrs[i].op_code == OpCodes.CALL and (
                    _value_8 if (_value_8 := self._instrs[i].op_arg) is not None else 0
                ) in (2, 3):
                    q = i - 1
                    consts = 0
                    while q >= body_start and self._instrs[q].op_code in (
                        OpCodes.LOAD_CONST,
                        OpCodes.PUSH_NULL,
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        if self._instrs[q].op_code == OpCodes.LOAD_CONST:
                            consts += 1
                        q -= 1
                    if consts >= 3:
                        cleanup_start = q + 1
                        call_idx = i
                i += 1
            if cleanup_start < 0 or cleanup_start < body_start:
                return (False, with_stmt, next_idx)
            cleanup_cursor = call_idx + 1
            while cleanup_cursor < end_idx and self._instrs[cleanup_cursor].op_code in (
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                cleanup_cursor += 1
            if (
                cleanup_cursor < end_idx
                and self._instrs[cleanup_cursor].op_code == OpCodes.GET_AWAITABLE
            ):
                cleanup_cursor += 1
            if (
                cleanup_cursor < end_idx
                and self._instrs[cleanup_cursor].op_code == OpCodes.LOAD_CONST
            ):
                cleanup_cursor += 1
            if (
                cleanup_cursor < end_idx
                and self._instrs[cleanup_cursor].op_code == OpCodes.SEND
                and (
                    (_subject_12 := self._instrs[cleanup_cursor].target_offset),
                    type(_subject_12) is int and (cleanup_send_target := _subject_12) is not None,
                )[-1]
                and (
                    (cleanup_send_idx := self._offset_to_index.get(cleanup_send_target, _MISSING))
                    is not _MISSING
                )
            ):
                cleanup_cursor = cleanup_send_idx
                if (
                    cleanup_cursor < end_idx
                    and self._instrs[cleanup_cursor].op_code == OpCodes.END_SEND
                ):
                    cleanup_cursor += 1
            if cleanup_cursor < end_idx and self._instrs[cleanup_cursor].op_code == OpCodes.POP_TOP:
                cleanup_cursor += 1
            cleanup_end_offset = (
                self._instrs[cleanup_cursor].offset if cleanup_cursor < end_idx else 2147483647
            )
            i = call_idx + 1
            while i < end_idx:
                if self._instrs[i].op_code != OpCodes.PUSH_EXC_INFO:
                    i += 1
                    continue
                handler_offset = self._instrs[i].offset
                if not _any(
                    self._code.exception_table,
                    lambda entry: (
                        entry.target == handler_offset
                        and _item_field(entry, "start", 0) < cleanup_end_offset
                    ),
                ):
                    i += 1
                    continue
                if i > handler_idx:
                    handler_idx = i
                i += 1
        else:
            body_offset = self._instrs[body_start].offset
            for entry in _iter_values(self._code.exception_table):
                if (
                    not entry.encloses(body_offset)
                    and (
                        not (
                            _item_field(entry, "start", 0) < body_offset
                            and _item_field(entry, "end", 1) >= body_offset
                        )
                    )
                    or entry.target <= body_offset
                ):
                    continue
                if (
                    not (h_idx := self._offset_to_index.get(entry.target, _MISSING)) is not _MISSING
                    or h_idx >= end_idx
                ):
                    continue
                if self._instrs[h_idx].op_code == OpCodes.PUSH_EXC_INFO:
                    handler_idx = h_idx
                    break
            if handler_idx < 0:
                return (False, with_stmt, next_idx)
            has_nested_with = False
            b = body_start
            while b < handler_idx:
                if self._instrs[b].op_code in (OpCodes.BEFORE_ASYNC_WITH, OpCodes.BEFORE_WITH):
                    has_nested_with = True
                    break
                b += 1
            i = body_start
            while i < handler_idx:
                if self._instrs[i].op_code == OpCodes.CALL and (
                    _value_14 if (_value_14 := self._instrs[i].op_arg) is not None else 0
                ) in (2, 3):
                    q = i - 1
                    consts = 0
                    while q >= body_start and self._instrs[q].op_code in (
                        OpCodes.LOAD_CONST,
                        OpCodes.PUSH_NULL,
                        OpCodes.NOP,
                        OpCodes.NOT_TAKEN,
                    ):
                        if self._instrs[q].op_code == OpCodes.LOAD_CONST:
                            consts += 1
                        q -= 1
                    if consts >= 3:
                        if not has_nested_with and cleanup_start >= 0:
                            i += 1
                            continue
                        cleanup_start = q + 1
                        call_idx = i
                i += 1
        if cleanup_start < 0 or cleanup_start < body_start or handler_idx < 0:
            return (False, with_stmt, next_idx)
        context = stack.peek()
        stack.pop()
        body_stack = SymbolicStack()
        body = self.decompile_range(body_start, cleanup_start, body_stack)
        body_return_idx = -1
        q = cleanup_start
        while q < handler_idx:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                body_return_idx = q
                break
            q += 1
        body_has_return = body_return_idx >= 0 and len(body_stack) > 0
        if body_has_return:
            _add(body, ReturnStmt(body_stack.pop()))
        if body_has_return:
            _add(self._handler_stop_indices, handler_idx)
            with_stmt = WithStmt([WithItem(context, target)], body, is_async=True)
            next_idx = body_return_idx + 1
            return (True, with_stmt, next_idx)
        after = self.find_legacy_async_exit_resume(call_idx, handler_idx, end_idx)
        with_stmt = WithStmt([WithItem(context, target)], body, is_async=True)
        next_idx = after
        return (True, with_stmt, next_idx)

    def find_legacy_async_exit_resume(self, call_idx, handler_idx, end_idx):
        p = self.skip_transparent(call_idx + 1)
        if p < end_idx and self._instrs[p].op_code == OpCodes.GET_AWAITABLE:
            p = self.skip_transparent(p + 1)
            if p < end_idx and self._instrs[p].op_code == OpCodes.LOAD_CONST:
                p = self.skip_transparent(p + 1)
            if p < end_idx and self._instrs[p].op_code in (OpCodes.LEGACY_YIELD_FROM, OpCodes.SEND):
                if (
                    self._instrs[p].op_code == OpCodes.SEND
                    and (
                        (_subject_2 := self._instrs[p].target_offset),
                        type(_subject_2) is int and (send_target := _subject_2) is not None,
                    )[-1]
                    and (
                        (send_end := self._offset_to_index.get(send_target, _MISSING))
                        is not _MISSING
                    )
                ):
                    p = send_end
                    if p < end_idx and self._instrs[p].op_code == OpCodes.END_SEND:
                        p += 1
                else:
                    p = self.skip_transparent(p + 1)
            p = self.skip_transparent(p)
            if p < end_idx and self._instrs[p].op_code == OpCodes.POP_TOP:
                p += 1
            p = self.skip_transparent(p)
        if p < end_idx and self._instrs[p].op_code in (
            OpCodes.JUMP_FORWARD,
            OpCodes.JUMP_BACKWARD,
            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
        ):
            target = _value_4 if (_value_4 := self._instrs[p].target_offset) is not None else -1
            if self.is_active_loop_header(target):
                h = handler_idx
                while h < end_idx:
                    if (
                        self._instrs[h].op_code
                        in (
                            OpCodes.JUMP_BACKWARD,
                            OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
                            OpCodes.JUMP_FORWARD,
                        )
                        and self._instrs[h].target_offset == target
                    ):
                        return h + 1
                    if self._instrs[h].op_code == OpCodes.END_ASYNC_FOR:
                        break
                    h += 1
                return end_idx
            if (target_idx := self._offset_to_index.get(target, _MISSING)) is not _MISSING:
                return target_idx
        _add(self._handler_stop_indices, handler_idx)
        return p

    def try_decompile_legacy_async_with(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if idx >= len(self._instrs) or self._instrs[idx].op_code != OpCodes.BEFORE_ASYNC_WITH:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        cur = self.skip_transparent(idx + 1)
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.GET_AWAITABLE:
            return (False, with_stmt, next_idx)
        cur = self.skip_transparent(cur + 1)
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LOAD_CONST:
            return (False, with_stmt, next_idx)
        if not StructuralDecompiler.is_none_constant(self._instrs[cur]):
            return (False, with_stmt, next_idx)
        cur = self.skip_transparent(cur + 1)
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LEGACY_YIELD_FROM:
            return (False, with_stmt, next_idx)
        cur = self.skip_transparent(cur + 1)
        if cur >= end_idx or self._instrs[cur].op_code != OpCodes.LEGACY_SETUP_WITH:
            return (False, with_stmt, next_idx)
        handler_offset = (
            _value_1 if (_value_1 := self._instrs[cur].target_offset) is not None else -1
        )
        if not (handler_idx := self._offset_to_index.get(handler_offset, _MISSING)) is not _MISSING:
            return (False, with_stmt, next_idx)
        body_start = cur + 1
        target = None
        body_start = self.skip_transparent(body_start)
        if body_start < handler_idx and self._instrs[body_start].op_code in (
            OpCodes.STORE_FAST,
            OpCodes.STORE_NAME,
            OpCodes.STORE_GLOBAL,
            OpCodes.STORE_DEREF,
        ):
            target_name = (
                str(_subject_4)
                if (
                    _subject_4 := (
                        _value_3
                        if (_value_3 := self._instrs[body_start].arg_repr) is not None
                        else self._instrs[body_start].arg_val
                    )
                )
                is not None
                else None
            )
            if _empty_string(target_name):
                raise FailFastException("async with target name missing")
            target = NameExpr(target_name)
            body_start += 1
        elif body_start < handler_idx and self._instrs[body_start].op_code == OpCodes.POP_TOP:
            body_start += 1
        if not (
            (
                _parsed_5 := self.try_find_legacy_block_pop(
                    body_start, handler_idx, prefer_last=True
                )
            ),
            (body_end := _parsed_5[1]),
            _parsed_5[0],
        )[-1]:
            p = handler_idx - 1
            while p > body_start:
                if self._instrs[p].op_code == OpCodes.LEGACY_POP_BLOCK:
                    body_end = p
                    break
                p -= 1
            if body_end < 0:
                return (False, with_stmt, next_idx)
        body_stack = SymbolicStack()
        body = self.decompile_range(body_start, body_end, body_stack)
        body_return_idx = -1
        q = body_end + 1
        while q < handler_idx:
            if self._instrs[q].op_code in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                body_return_idx = q
                break
            q += 1
        body_has_return = body_return_idx >= 0 and len(body_stack) > 0
        if body_has_return:
            _add(body, ReturnStmt(body_stack.pop()))
        after = body_end + 1
        if body_has_return:
            _add(self._handler_stop_indices, handler_idx)
            after = body_return_idx + 1
        else:
            end_finally_idx = after
            while (
                end_finally_idx < end_idx
                and self._instrs[end_finally_idx].op_code != OpCodes.LEGACY_END_FINALLY
            ):
                end_finally_idx += 1
            if (
                end_finally_idx < end_idx
                and self._instrs[end_finally_idx].op_code == OpCodes.LEGACY_END_FINALLY
            ):
                after = end_finally_idx + 1
            else:
                call_idx = body_end + 1
                while call_idx < handler_idx and (
                    not (
                        self._instrs[call_idx].op_code == OpCodes.CALL
                        and (
                            _value_7
                            if (_value_7 := self._instrs[call_idx].op_arg) is not None
                            else 0
                        )
                        in (2, 3)
                    )
                ):
                    call_idx += 1
                if call_idx >= handler_idx:
                    return (False, with_stmt, next_idx)
                after = self.find_legacy_async_exit_resume(call_idx, handler_idx, end_idx)
        context = stack.peek()
        stack.pop()
        items = list([WithItem(context, target)])
        while (
            len(body) == 1
            and ((_subject_9 := body[0]), isinstance((inner_with := _subject_9), WithStmt))[-1]
            and inner_with.is_async
        ):
            items.extend(inner_with.items)
            body = list(inner_with.body)
        with_stmt = WithStmt(items, body, is_async=True)
        next_idx = after
        return (True, with_stmt, next_idx)

    def try_decompile_with_stmt(self, idx, end_idx, stack):
        with_stmt = None
        next_idx = 0
        with_stmt = None
        next_idx = idx
        if idx >= len(self._instrs):
            return (False, with_stmt, next_idx)
        op = self._instrs[idx].op_code
        if op != OpCodes.BEFORE_WITH and op != OpCodes.BEFORE_ASYNC_WITH:
            return (False, with_stmt, next_idx)
        if len(stack) == 0:
            return (False, with_stmt, next_idx)
        is_async = op == OpCodes.BEFORE_ASYNC_WITH
        ctx_expr = stack.pop()
        cur = idx + 1
        if is_async:
            while cur < len(self._instrs) and self._instrs[cur].op_code != OpCodes.END_SEND:
                cur += 1
            if cur < len(self._instrs) and self._instrs[cur].op_code == OpCodes.END_SEND:
                cur += 1
        target_var = None
        while cur < len(self._instrs) and self._instrs[cur].op_code == OpCodes.EXTENDED_ARG:
            cur += 1
        if cur < len(self._instrs):
            if self._instrs[cur].op_code in (
                OpCodes.STORE_NAME,
                OpCodes.STORE_FAST,
                OpCodes.STORE_GLOBAL,
                OpCodes.STORE_DEREF,
            ):
                var_name = (
                    _value_4
                    if (
                        _value_4 := (
                            str(_subject_3)
                            if (
                                _subject_3 := (
                                    _value_2
                                    if (_value_2 := self._instrs[cur].arg_repr) is not None
                                    else self._instrs[cur].arg_val
                                )
                            )
                            is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                )
                target_var = NameExpr(var_name)
                cur += 1
            elif self._instrs[cur].op_code == OpCodes.POP_TOP:
                cur += 1
        scan = cur
        with_nesting = 0
        found_cleanup = False
        while scan < len(self._instrs) and scan < end_idx:
            if self._instrs[scan].op_code in (OpCodes.BEFORE_WITH, OpCodes.BEFORE_ASYNC_WITH):
                with_nesting += 1
            elif (
                not is_async
                and self._instrs[scan].op_code == OpCodes.PUSH_EXC_INFO
                and (with_nesting == 0)
                and (scan + 1 < len(self._instrs))
                and (self._instrs[scan + 1].op_code == OpCodes.WITH_EXCEPT_START)
            ):
                break
            elif (
                self._instrs[scan].op_code == OpCodes.LOAD_CONST
                and self._instrs[scan].arg_val is None
                and (scan + 3 < len(self._instrs))
                and (self._instrs[scan + 3].op_code == OpCodes.CALL)
                and (
                    (_value_6 if (_value_6 := self._instrs[scan + 3].op_arg) is not None else 0)
                    == 2
                )
            ):
                if with_nesting == 0:
                    found_cleanup = True
                    break
                with_nesting -= 1
            scan += 1
        body_end_idx = scan
        if not found_cleanup:
            s = cur
            while s + 2 < len(self._instrs) and s < end_idx:
                if (
                    self._instrs[s].op_code
                    in (
                        OpCodes.RAISE_VARARGS,
                        OpCodes.RETURN_VALUE,
                        OpCodes.RETURN_CONST,
                        OpCodes.RERAISE,
                    )
                    and self._instrs[s + 1].op_code == OpCodes.PUSH_EXC_INFO
                    and (self._instrs[s + 2].op_code == OpCodes.WITH_EXCEPT_START)
                ):
                    body_end_idx = s + 1
                    break
                s += 1
        with_items = list([WithItem(ctx_expr, target_var)])
        body_stack = SymbolicStack()
        with_body = self.decompile_range(cur, body_end_idx, body_stack)
        while (
            len(with_body) == 1
            and ((_subject_8 := with_body[0]), isinstance((inner_with := _subject_8), WithStmt))[-1]
            and (inner_with.is_async == is_async)
        ):
            with_items.extend(inner_with.items)
            with_body = list(inner_with.body)
        if (
            body_end_idx < len(self._instrs)
            and self._instrs[body_end_idx].op_code == OpCodes.PUSH_EXC_INFO
        ):
            handler_exit_idx = self.find_with_handler_exit(body_end_idx)
            if handler_exit_idx > body_end_idx:
                with_stmt = WithStmt(with_items, with_body, is_async)
                next_idx = handler_exit_idx
                return (True, with_stmt, next_idx)
        cleanup_idx = body_end_idx
        k = 0
        while k < len(with_items):
            if cleanup_idx >= len(self._instrs):
                break
            if (
                self._instrs[cleanup_idx].op_code != OpCodes.LOAD_CONST
                or self._instrs[cleanup_idx].arg_val is not None
            ):
                break
            while cleanup_idx < len(self._instrs):
                if self._instrs[cleanup_idx].op_code == OpCodes.POP_TOP:
                    cleanup_idx += 1
                    break
                if self._instrs[cleanup_idx].op_code in (
                    OpCodes.PUSH_EXC_INFO,
                    OpCodes.RETURN_VALUE,
                    OpCodes.RETURN_CONST,
                ):
                    break
                cleanup_idx += 1
            k += 1
        if cleanup_idx < len(self._instrs) and self._instrs[cleanup_idx].op_code in (
            OpCodes.RETURN_CONST,
            OpCodes.RETURN_VALUE,
        ):
            ret_ins = self._instrs[cleanup_idx]
            if ret_ins.op_code == OpCodes.RETURN_CONST and ret_ins.arg_val is not None:
                _add(with_body, ReturnStmt(ConstantExpr(ret_ins.arg_val)))
                cleanup_idx += 1
        with_stmt = WithStmt(with_items, with_body, is_async)
        next_idx = cleanup_idx
        return (True, with_stmt, next_idx)

    def find_handler_normal_exit(self, h_idx, end_idx):
        q = h_idx
        while q < end_idx:
            if self._instrs[q].op_code != OpCodes.POP_EXCEPT:
                q += 1
                continue
            r = q + 1
            while r < end_idx and self._instrs[r].op_code in (
                OpCodes.POP_TOP,
                OpCodes.NOP,
                OpCodes.NOT_TAKEN,
            ):
                r += 1
            if r < end_idx and (not self._instrs[r].op_code in (OpCodes.RERAISE, OpCodes.COPY)):
                return r
            q += 1
        return -1

    def find_with_handler_exit(self, h_idx):
        h_offset = self._instrs[h_idx].offset
        nesting = 0
        k = h_idx + 1
        while k < len(self._instrs):
            op = self._instrs[k].op_code
            if op == OpCodes.PUSH_EXC_INFO:
                nesting += 1
                k += 1
                continue
            if op == OpCodes.POP_EXCEPT:
                if nesting > 0:
                    nesting -= 1
                    k += 1
                    continue
                q = k + 1
                while q < len(self._instrs):
                    qop = self._instrs[q].op_code
                    if (
                        qop == OpCodes.JUMP_FORWARD
                        and (
                            _value_1
                            if (_value_1 := self._instrs[q].target_offset) is not None
                            else 0
                        )
                        > h_offset
                    ):
                        return (
                            ti
                            if (
                                ti := self._offset_to_index.get(
                                    self._instrs[q].target_offset, _MISSING
                                )
                            )
                            is not _MISSING
                            else -1
                        )
                    if qop in (OpCodes.RETURN_VALUE, OpCodes.RETURN_CONST):
                        return q + 1
                    if qop in (OpCodes.PUSH_EXC_INFO, OpCodes.POP_EXCEPT, OpCodes.RERAISE):
                        break
                    q += 1
                return -1
            if (
                op == OpCodes.RERAISE
                and nesting == 0
                and ((_value_2 if (_value_2 := self._instrs[k].op_arg) is not None else 0) <= 1)
            ):
                q = k + 1
                if (
                    q + 2 < len(self._instrs)
                    and self._instrs[q].op_code == OpCodes.COPY
                    and (self._instrs[q + 1].op_code == OpCodes.POP_EXCEPT)
                    and (self._instrs[q + 2].op_code == OpCodes.RERAISE)
                ):
                    return q + 3
                return k + 1
            k += 1
        return -1

    @staticmethod
    def is_async_comprehension_expr(expr):
        if isinstance((lc := expr), ListCompExpr):
            return _any(lc.clauses, lambda c: c.is_async)
        if isinstance((sc := expr), SetCompExpr):
            return _any(sc.clauses, lambda c: c.is_async)
        if isinstance((dc := expr), DictCompExpr):
            return _any(dc.clauses, lambda c: c.is_async)
        if isinstance((ge := expr), GeneratorExpr):
            return _any(ge.clauses, lambda c: c.is_async)
        return False

    def try_decompile_await_or_yield_from(self, idx, stack):
        expr = None
        next_idx = 0
        expr = None
        next_idx = idx
        if idx >= len(self._instrs):
            return (False, expr, next_idx)
        op = self._instrs[idx].op_code
        if (
            op != OpCodes.GET_AWAITABLE
            and op != OpCodes.GET_YIELD_FROM_ITER
            and (op != OpCodes.GET_ITER)
        ):
            return (False, expr, next_idx)
        is_await = op == OpCodes.GET_AWAITABLE
        if len(stack) == 0:
            return (False, expr, next_idx)
        if (
            (op == OpCodes.GET_YIELD_FROM_ITER or op == OpCodes.GET_AWAITABLE)
            and idx + 2 < len(self._instrs)
            and (self._instrs[idx + 1].op_code == OpCodes.LOAD_CONST)
            and (self._instrs[idx + 2].op_code == OpCodes.LEGACY_YIELD_FROM)
        ):
            legacy_target = stack.pop()
            expr = (
                (
                    legacy_target
                    if StructuralDecompiler.is_async_comprehension_expr(legacy_target)
                    else AwaitExpr(legacy_target)
                )
                if is_await
                else YieldFromExpr(legacy_target)
            )
            next_idx = idx + 3
            return (True, expr, next_idx)
        cur = idx + 1
        while cur < len(self._instrs) and self._instrs[cur].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
            OpCodes.PUSH_NULL,
        ):
            cur += 1
        if cur >= len(self._instrs) or self._instrs[cur].op_code != OpCodes.LOAD_CONST:
            return (False, expr, next_idx)
        cur += 1
        while cur < len(self._instrs) and self._instrs[cur].op_code in (
            OpCodes.NOP,
            OpCodes.NOT_TAKEN,
            OpCodes.PUSH_NULL,
        ):
            cur += 1
        if cur >= len(self._instrs) or self._instrs[cur].op_code != OpCodes.SEND:
            return (False, expr, next_idx)
        send_target_offset = (
            _value_3 if (_value_3 := self._instrs[cur].target_offset) is not None else -1
        )
        if not (
            send_end_idx2 := self._offset_to_index.get(send_target_offset, _MISSING)
        ) is not _MISSING or send_end_idx2 >= len(self._instrs):
            return (False, expr, next_idx)
        send_next_idx = (
            send_end_idx2 + 1
            if self._instrs[send_end_idx2].op_code == OpCodes.END_SEND
            else send_end_idx2
        )
        target_expr = stack.pop()
        expr = (
            (
                target_expr
                if StructuralDecompiler.is_async_comprehension_expr(target_expr)
                else AwaitExpr(target_expr)
            )
            if is_await
            else YieldFromExpr(target_expr)
        )
        next_idx = send_next_idx
        return (True, expr, next_idx)
