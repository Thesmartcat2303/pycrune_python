"""Render syntax nodes as Python source."""

from __future__ import annotations

import math

from ._support import (
    FailFastException,
    _cast_int,
    _empty_string,
    _index_of,
    _is_negative,
    _item_field,
    _iter_values,
    _join,
    _negative_infinity,
    _positive_infinity,
    _SourceBuffer,
)
from .model import (
    FrozenSetValue,
    SliceValue,
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
    CompareExpr,
    ConstantExpr,
    ContinueStmt,
    DeleteStmt,
    DictCompExpr,
    DictExpr,
    DoubleStarredExpr,
    ExprStmt,
    FormattedValueExpr,
    ForStmt,
    FunctionDefStmt,
    GeneratorExpr,
    GlobalStmt,
    IfStmt,
    ImportFromStmt,
    ImportStmt,
    JoinedStrExpr,
    LambdaExpr,
    ListCompExpr,
    ListExpr,
    MatchAsPattern,
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
    TypeAliasStmt,
    UnaryExpr,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)


class PythonEmitter:
    def __init__(self):
        self._sb = _SourceBuffer()
        self._indent = 0

    @staticmethod
    def emit(statements):
        emitter = PythonEmitter()
        emitter.emit_statements(statements)
        return str(emitter._sb)

    @staticmethod
    def emit_expr(expr):
        emitter = PythonEmitter()
        emitter.write_expr(expr, 0)
        return str(emitter._sb)

    def emit_statements(self, statements):
        if len(statements) == 0:
            self.write_indent()
            self._sb.append_line("pass")
            return
        for stmt in _iter_values(statements):
            self.emit_statement(stmt)

    def emit_statement(self, stmt):
        if isinstance((s := stmt), ExprStmt):
            self.write_indent()
            self.write_expr(s.expression, 1)
            self._sb.append_line()
        elif isinstance((s := stmt), AssignStmt):
            self.write_indent()
            i = 0
            while i < len(s.targets):
                self.write_expr(s.targets[i], 0)
                self._sb.append(" = ")
                i += 1
            self.write_expr(_item_field(s, "value", 1), 1)
            self._sb.append_line()
        elif isinstance(stmt, AnnAssignStmt):
            self.write_indent()
            self.write_expr(stmt.target, 0)
            self._sb.append(": ")
            self.write_expr(stmt.annotation, 1)
            if stmt.value is not None:
                self._sb.append(" = ")
                self.write_expr(stmt.value, 1)
            self._sb.append_line()
        elif isinstance((s := stmt), AugAssignStmt):
            self.write_indent()
            self.write_expr(s.target, 0)
            self._sb.append(" {} ".format(_item_field(s, "op", 0)))
            self.write_expr(_item_field(s, "value", 1), 1)
            self._sb.append_line()
        elif isinstance((s := stmt), IfStmt):
            self.write_indent()
            self._sb.append("if ")
            self.write_expr(s.condition, 0)
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.then_body)
            self._indent -= 1
            if len(s.else_body) > 0:
                if (
                    len(s.else_body) == 1
                    and ((_subject_1 := s.else_body[0]), isinstance((elif_ := _subject_1), IfStmt))[
                        -1
                    ]
                ):
                    self.emit_elif(elif_)
                else:
                    self.write_indent()
                    self._sb.append_line("else:")
                    self._indent += 1
                    self.emit_statements(s.else_body)
                    self._indent -= 1
        elif isinstance((s := stmt), WhileStmt):
            self.write_indent()
            self._sb.append("while ")
            self.write_expr(s.condition, 0)
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
            if len(s.else_body) > 0:
                self.write_indent()
                self._sb.append_line("else:")
                self._indent += 1
                self.emit_statements(s.else_body)
                self._indent -= 1
        elif isinstance((s := stmt), ForStmt):
            self.write_indent()
            if s.is_async:
                self._sb.append("async ")
            self._sb.append("for ")
            self.write_expr(s.target, 0)
            self._sb.append(" in ")
            self.write_expr(s.iterator, 1)
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
            if len(s.else_body) > 0:
                self.write_indent()
                self._sb.append_line("else:")
                self._indent += 1
                self.emit_statements(s.else_body)
                self._indent -= 1
        elif isinstance((s := stmt), TryStmt):
            self.write_indent()
            self._sb.append_line("try:")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
            for h in _iter_values(s.handlers):
                self.write_indent()
                self._sb.append("except*" if h.is_star else "except")
                if h.exception_type is not None:
                    self._sb.append(" ")
                    self.write_expr(h.exception_type, 0)
                    if not _empty_string(h.variable_name):
                        PythonEmitter.guard_identifier(h.variable_name)
                        self._sb.append(" as {}".format(h.variable_name))
                self._sb.append_line(":")
                self._indent += 1
                self.emit_statements(h.body)
                self._indent -= 1
            if len(s.else_body) > 0:
                self.write_indent()
                self._sb.append_line("else:")
                self._indent += 1
                self.emit_statements(s.else_body)
                self._indent -= 1
            if s.finally_body is not None:
                self.write_indent()
                self._sb.append_line("finally:")
                self._indent += 1
                self.emit_statements(s.finally_body)
                self._indent -= 1
            elif len(s.handlers) == 0:
                self.write_indent()
                self._sb.append_line("except:")
                self._indent += 1
                self.write_indent()
                self._sb.append_line("pass")
                self._indent -= 1
        elif isinstance((s := stmt), RaiseStmt):
            self.write_indent()
            self._sb.append("raise")
            if s.exception is not None:
                self._sb.append(" ")
                self.write_expr(s.exception, 1)
                if s.from_ is not None:
                    self._sb.append(" from ")
                    self.write_expr(s.from_, 1)
            self._sb.append_line()
        elif isinstance((s := stmt), ReturnStmt):
            self.write_indent()
            self._sb.append("return")
            if _item_field(s, "value", 1) is not None:
                self._sb.append(" ")
                self.write_expr(_item_field(s, "value", 1), 1)
            self._sb.append_line()
        elif isinstance(stmt, BreakStmt):
            self.write_indent()
            self._sb.append_line("break")
        elif isinstance(stmt, ContinueStmt):
            self.write_indent()
            self._sb.append_line("continue")
        elif isinstance(stmt, PassStmt):
            self.write_indent()
            self._sb.append_line("pass")
        elif isinstance((s := stmt), AssertStmt):
            self.write_indent()
            self._sb.append("assert ")
            self.write_expr(s.test, 1)
            if s.message is not None:
                self._sb.append(", ")
                self.write_expr(s.message, 1)
            self._sb.append_line()
        elif isinstance((s := stmt), NonlocalStmt):
            self.write_indent()
            self._sb.append("nonlocal ")
            self._sb.append_line(_join(", ", s.names))
        elif isinstance((s := stmt), GlobalStmt):
            self.write_indent()
            self._sb.append("global ")
            self._sb.append_line(_join(", ", s.names))
        elif isinstance((s := stmt), FunctionDefStmt):
            if s.decorators is not None:
                for dec in _iter_values(s.decorators):
                    self.write_indent()
                    self._sb.append("@")
                    self.write_expr(dec, 0)
                    self._sb.append_line()
            PythonEmitter.guard_identifier(_item_field(s, "name", 0))
            if s.type_params is not None:
                for tp in _iter_values(s.type_params):
                    PythonEmitter.guard_identifier(tp)
            for param in _iter_values(s.parameters):
                if "__pycrune_ph_" in param:
                    raise FailFastException(
                        "internal placeholder leaked into parameter: {}".format(param)
                    )
            self.write_indent()
            if s.is_async:
                self._sb.append("async ")
            self._sb.append("def {}".format(_item_field(s, "name", 0)))
            if s.type_params is not None and len(s.type_params) > 0:
                self._sb.append("[{}]".format(_join(", ", s.type_params)))
            self._sb.append("({})".format(_join(", ", s.parameters)))
            if not _empty_string(s.return_type):
                self._sb.append(" -> {}".format(s.return_type))
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
        elif isinstance((s := stmt), ClassDefStmt):
            if s.decorators is not None:
                for dec in _iter_values(s.decorators):
                    self.write_indent()
                    self._sb.append("@")
                    self.write_expr(dec, 0)
                    self._sb.append_line()
            PythonEmitter.guard_identifier(_item_field(s, "name", 0))
            if s.type_params is not None:
                for tp in _iter_values(s.type_params):
                    PythonEmitter.guard_identifier(tp)
            self.write_indent()
            self._sb.append("class {}".format(_item_field(s, "name", 0)))
            if s.type_params is not None and len(s.type_params) > 0:
                self._sb.append("[{}]".format(_join(", ", s.type_params)))
            if len(s.bases) > 0 or len(s.keywords) > 0:
                self._sb.append("(")
                first = True
                for b in _iter_values(s.bases):
                    if not first:
                        self._sb.append(", ")
                    first = False
                    self.write_expr(b, 1)
                for kw, val in _iter_values(s.keywords):
                    if not first:
                        self._sb.append(", ")
                    first = False
                    self._sb.append("{}=".format(kw))
                    self.write_expr(val, 1)
                self._sb.append(")")
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
        elif isinstance((s := stmt), TypeAliasStmt):
            PythonEmitter.guard_identifier(_item_field(s, "name", 0))
            for tp in _iter_values(s.type_params):
                PythonEmitter.guard_identifier(tp)
            self.write_indent()
            self._sb.append("type {}".format(_item_field(s, "name", 0)))
            if len(s.type_params) > 0:
                self._sb.append("[{}]".format(_join(", ", s.type_params)))
            self._sb.append(" = ")
            self.write_expr(_item_field(s, "value", 1), 1)
            self._sb.append_line()
        elif isinstance((s := stmt), DeleteStmt):
            self.write_indent()
            self._sb.append("del ")
            i = 0
            while i < len(s.targets):
                if i > 0:
                    self._sb.append(", ")
                self.write_expr(s.targets[i], 0)
                i += 1
            self._sb.append_line()
        elif isinstance((s := stmt), ImportStmt):
            self.write_indent()
            self._sb.append("import ")
            i = 0
            while i < len(s.names):
                if i > 0:
                    self._sb.append(", ")
                self._sb.append(_item_field(s.names[i], "name", 0))
                if not _empty_string(s.names[i].as_name):
                    self._sb.append(" as {}".format(s.names[i].as_name))
                i += 1
            self._sb.append_line()
        elif isinstance((s := stmt), ImportFromStmt):
            self.write_indent()
            self._sb.append("from ")
            if s.level > 0:
                self._sb.append("." * s.level)
            if not _empty_string(s.module):
                self._sb.append(s.module)
            self._sb.append(" import ")
            i = 0
            while i < len(s.names):
                if i > 0:
                    self._sb.append(", ")
                self._sb.append(_item_field(s.names[i], "name", 0))
                if not _empty_string(s.names[i].as_name):
                    self._sb.append(" as {}".format(s.names[i].as_name))
                i += 1
            self._sb.append_line()
        elif isinstance((s := stmt), WithStmt):
            self.write_indent()
            if s.is_async:
                self._sb.append("async ")
            self._sb.append("with ")
            i = 0
            while i < len(s.items):
                if i > 0:
                    self._sb.append(", ")
                self.write_expr(s.items[i].context_expr, 1)
                if s.items[i].optional_vars is not None:
                    self._sb.append(" as ")
                    self.write_expr(s.items[i].optional_vars, 0)
                i += 1
            self._sb.append_line(":")
            self._indent += 1
            self.emit_statements(s.body)
            self._indent -= 1
        elif isinstance((s := stmt), MatchStmt):
            self.write_indent()
            self._sb.append("match ")
            self.write_expr(s.subject, 0)
            self._sb.append_line(":")
            self._indent += 1
            for c in _iter_values(s.cases):
                self.write_indent()
                self._sb.append("case ")
                self.emit_pattern(_item_field(c, "pattern", 1))
                if c.guard is not None:
                    self._sb.append(" if ")
                    self.write_expr(c.guard, 0)
                self._sb.append_line(":")
                self._indent += 1
                self.emit_statements(c.body)
                self._indent -= 1
            self._indent -= 1

    def emit_pattern(self, pattern):
        if isinstance(pattern, MatchWildcardPattern):
            self._sb.append("_")
        elif isinstance((v := pattern), MatchValuePattern):
            self.write_expr(_item_field(v, "value", 1), 0)
        elif isinstance((st := pattern), MatchStarPattern):
            self._sb.append("*")
            self._sb.append(_item_field(st, "name", 0))
        elif isinstance((a := pattern), MatchAsPattern):
            if _item_field(a, "pattern", 1) is not None:
                self.emit_pattern(_item_field(a, "pattern", 1))
                self._sb.append(" as ")
            self._sb.append(_item_field(a, "name", 0))
        elif isinstance((o := pattern), MatchOrPattern):
            i = 0
            while i < len(o.patterns):
                if i > 0:
                    self._sb.append(" | ")
                self.emit_pattern(o.patterns[i])
                i += 1
        elif isinstance((seq := pattern), MatchSequencePattern):
            self._sb.append("[" if seq.is_list else "(")
            i = 0
            while i < len(seq.patterns):
                if i > 0:
                    self._sb.append(", ")
                self.emit_pattern(seq.patterns[i])
                i += 1
            if not seq.is_list and len(seq.patterns) == 1:
                self._sb.append(",")
            self._sb.append("]" if seq.is_list else ")")
        elif isinstance((cls := pattern), MatchClassPattern):
            self.write_expr(cls.class_name, 0)
            self._sb.append("(")
            i = 0
            while i < len(cls.patterns):
                if i > 0:
                    self._sb.append(", ")
                self.emit_pattern(cls.patterns[i])
                i += 1
            self._sb.append(")")
        elif isinstance((kw := pattern), MatchKeywordPattern):
            self._sb.append("{}=".format(kw.attribute))
            self.emit_pattern(_item_field(kw, "pattern", 1))
        elif isinstance((m := pattern), MatchMappingPattern):
            self._sb.append("{")
            first = True
            for k, p in _iter_values(m.entries):
                if not first:
                    self._sb.append(", ")
                first = False
                self.write_expr(k, 0)
                self._sb.append(": ")
                self.emit_pattern(p)
            if m.rest is not None:
                if not first:
                    self._sb.append(", ")
                if isinstance((st_rest := m.rest), MatchStarPattern):
                    self._sb.append("**")
                    self._sb.append(_item_field(st_rest, "name", 0))
                else:
                    self.emit_pattern(m.rest)
            self._sb.append("}")

    def emit_elif(self, elif_):
        self.write_indent()
        self._sb.append("elif ")
        self.write_expr(elif_.condition, 0)
        self._sb.append_line(":")
        self._indent += 1
        self.emit_statements(elif_.then_body)
        self._indent -= 1
        if len(elif_.else_body) > 0:
            if (
                len(elif_.else_body) == 1
                and (
                    (_subject_1 := elif_.else_body[0]),
                    isinstance((nested_elif := _subject_1), IfStmt),
                )[-1]
            ):
                self.emit_elif(nested_elif)
            else:
                self.write_indent()
                self._sb.append_line("else:")
                self._indent += 1
                self.emit_statements(elif_.else_body)
                self._indent -= 1

    @staticmethod
    def guard_identifier(name):
        if name.startswith("__pycrune_ph_"):
            raise FailFastException("internal placeholder leaked into output: {}".format(name))

    def write_expr(self, expr, parent_prec):
        my_prec = PythonEmitter.get_precedence(expr)
        need_parens = my_prec < parent_prec
        if need_parens:
            self._sb.append("(")
        if isinstance((c := expr), ConstantExpr):
            self.write_constant(_item_field(c, "value", 1))
        elif isinstance((n := expr), NameExpr):
            PythonEmitter.guard_identifier(_item_field(n, "name", 0))
            self._sb.append(_item_field(n, "name", 0))
        elif isinstance(expr, AssertionErrorExpr):
            self._sb.append("AssertionError")
        elif isinstance((u := expr), UnaryExpr):
            self._sb.append(_item_field(u, "op", 0))
            if _item_field(u, "op", 0) == "not":
                self._sb.append(" ")
            self.write_expr(u.operand, my_prec)
        elif isinstance((b := expr), BinaryExpr):
            self.write_expr(b.left, my_prec)
            self._sb.append(" {} ".format(_item_field(b, "op", 0)))
            self.write_expr(_item_field(b, "right", 1), my_prec + 1)
        elif isinstance((comp := expr), CompareExpr):
            self.write_expr(comp.left, 99 if isinstance(comp.left, CompareExpr) else my_prec)
            for op, right in _iter_values(comp.comparators):
                self._sb.append(" {} ".format(op))
                self.write_expr(right, 99 if isinstance(right, CompareExpr) else my_prec + 1)
        elif isinstance((bool_op := expr), BoolOpExpr):
            i = 0
            while i < len(bool_op.values):
                if i > 0:
                    self._sb.append(" {} ".format(_item_field(bool_op, "op", 0)))
                self.write_expr(bool_op.values[i], my_prec)
                i += 1
        elif isinstance((t := expr), TernaryExpr):
            self.write_expr(t.if_true, my_prec + 1)
            self._sb.append(" if ")
            self.write_expr(t.condition, my_prec + 1)
            self._sb.append(" else ")
            self.write_expr(t.if_false, my_prec)
        elif isinstance((call := expr), CallExpr):
            self.write_expr(call.function, 16)
            self._sb.append("(")
            first = True
            for arg in _iter_values(call.arguments):
                if not first:
                    self._sb.append(", ")
                first = False
                self.write_expr(arg, 0)
            if call.keyword_sequence is not None:
                for kw, val in _iter_values(call.keyword_sequence):
                    if not first:
                        self._sb.append(", ")
                    first = False
                    if kw is None:
                        self._sb.append("**")
                        self.write_expr(val, 1)
                    else:
                        self._sb.append("{}=".format(kw))
                        self.write_expr(val, 1)
            else:
                for kw, val in _iter_values(call.keyword_arguments):
                    if not first:
                        self._sb.append(", ")
                    first = False
                    self._sb.append("{}=".format(kw))
                    self.write_expr(val, 1)
            self._sb.append(")")
        elif isinstance((sub := expr), SubscriptExpr):
            self.write_expr(sub.target, 16)
            self._sb.append("[")
            if (
                (_subject_1 := _item_field(sub, "index", 0)),
                isinstance((sub_tuple := _subject_1), TupleExpr),
            )[-1]:
                i = 0
                while i < len(sub_tuple.elements):
                    if i > 0:
                        self._sb.append(", ")
                    self.write_expr(sub_tuple.elements[i], 1)
                    i += 1
            else:
                self.write_expr(_item_field(sub, "index", 0), 1)
            self._sb.append("]")
        elif isinstance((att := expr), AttributeExpr):
            PythonEmitter.guard_identifier(att.attribute)
            self.write_expr(att.target, 16)
            self._sb.append(".{}".format(att.attribute))
        elif isinstance((sl := expr), SliceExpr):
            if sl.lower is not None:
                self.write_expr(sl.lower, 2)
            self._sb.append(":")
            if sl.upper is not None:
                self.write_expr(sl.upper, 2)
            if sl.step is not None:
                self._sb.append(":")
                self.write_expr(sl.step, 2)
        elif isinstance((l := expr), ListExpr):
            self._sb.append("[")
            i = 0
            while i < len(l.elements):
                if i > 0:
                    self._sb.append(", ")
                self.write_expr(l.elements[i], 1)
                i += 1
            self._sb.append("]")
        elif isinstance((tup := expr), TupleExpr):
            self._sb.append("(")
            i = 0
            while i < len(tup.elements):
                if i > 0:
                    self._sb.append(", ")
                self.write_expr(tup.elements[i], 1)
                i += 1
            if len(tup.elements) == 1:
                self._sb.append(",")
            self._sb.append(")")
        elif isinstance((set_ := expr), SetExpr):
            if len(set_.elements) == 0:
                self._sb.append("set()")
            else:
                self._sb.append("{")
                i = 0
                while i < len(set_.elements):
                    if i > 0:
                        self._sb.append(", ")
                    self.write_expr(set_.elements[i], 1)
                    i += 1
                self._sb.append("}")
        elif isinstance((d := expr), DictExpr):
            self._sb.append("{")
            i = 0
            while i < len(d.entries):
                if i > 0:
                    self._sb.append(", ")
                if _item_field(d.entries[i], "key", 0) is None:
                    self._sb.append("**")
                    self.write_expr(_item_field(d.entries[i], "value", 1), 2)
                else:
                    self.write_expr(_item_field(d.entries[i], "key", 0), 2)
                    self._sb.append(": ")
                    self.write_expr(_item_field(d.entries[i], "value", 1), 2)
                i += 1
            self._sb.append("}")
        elif isinstance((lc := expr), ListCompExpr):
            self._sb.append("[")
            self.write_expr(lc.element, 1)
            self.write_clauses(lc.clauses)
            self._sb.append("]")
        elif isinstance((sc := expr), SetCompExpr):
            self._sb.append("{")
            self.write_expr(sc.element, 1)
            self.write_clauses(sc.clauses)
            self._sb.append("}")
        elif isinstance((dc := expr), DictCompExpr):
            self._sb.append("{")
            self.write_expr(_item_field(dc, "key", 0), 2)
            self._sb.append(": ")
            self.write_expr(_item_field(dc, "value", 1), 2)
            self.write_clauses(dc.clauses)
            self._sb.append("}")
        elif isinstance((ge := expr), GeneratorExpr):
            self._sb.append("(")
            self.write_expr(ge.element, 1)
            self.write_clauses(ge.clauses)
            self._sb.append(")")
        elif isinstance((ne := expr), NamedExpr):
            self.write_expr(ne.target, 2)
            self._sb.append(" := ")
            self.write_expr(_item_field(ne, "value", 1), 1)
        elif isinstance((lam := expr), LambdaExpr):
            if len(lam.parameters) > 0:
                self._sb.append("lambda {}: ".format(_join(", ", lam.parameters)))
            else:
                self._sb.append("lambda: ")
            self.write_expr(lam.body, 2)
        elif isinstance((y := expr), YieldExpr):
            self._sb.append("yield")
            if _item_field(y, "value", 1) is not None:
                self._sb.append(" ")
                self.write_expr(_item_field(y, "value", 1), 1)
        elif isinstance((yf := expr), YieldFromExpr):
            self._sb.append("yield from ")
            self.write_expr(_item_field(yf, "value", 1), my_prec)
        elif isinstance((a := expr), AwaitExpr):
            self._sb.append("await ")
            self.write_expr(_item_field(a, "value", 1), my_prec)
        elif isinstance((fv := expr), FormattedValueExpr):
            self._sb.append('f"')
            self.write_formatted_value(fv)
            self._sb.append('"')
        elif isinstance((st := expr), StarredExpr):
            self._sb.append("*")
            self.write_expr(_item_field(st, "value", 1), 1)
        elif isinstance((dst := expr), DoubleStarredExpr):
            self._sb.append("**")
            self.write_expr(_item_field(dst, "value", 1), 1)
        elif isinstance((js := expr), JoinedStrExpr):
            self._sb.append('f"')
            for part in _iter_values(js.parts):
                if (
                    isinstance(part, ConstantExpr)
                    and type(_item_field(part, "value", 1)) is str
                    and ((s := _item_field(part, "value", 1)) is not None)
                ):
                    literal = PythonEmitter.format_string_literal(s)[1:-1]
                    self._sb.append(literal.replace("{", "{{").replace("}", "}}"))
                elif isinstance((fv_part := part), FormattedValueExpr):
                    self.write_formatted_value(fv_part)
                else:
                    self._sb.append("{")
                    self.write_expr(part, 2)
                    self._sb.append("}")
            self._sb.append('"')
        if need_parens:
            self._sb.append(")")

    def write_constant(self, val):
        if val is None:
            self._sb.append("None")
        elif val is Ellipsis:
            self._sb.append("...")
        elif type(val) is bool and (b := val) is not None:
            self._sb.append("True" if b else "False")
        elif type(val) is str and (s := val) is not None:
            self._sb.append(PythonEmitter.format_string_literal(s))
        elif type(val) is bytes and (bytes_ := val) is not None:
            self._sb.append(PythonEmitter.format_bytes(bytes_))
        elif type(val) is float and (d := val) is not None:
            self._sb.append(PythonEmitter.format_double(d))
        elif type(val) is float and (f := val) is not None:
            self._sb.append(PythonEmitter.format_double(f))
        elif (
            type(val) is int
            or type(val) is int
            or type(val) is int
            or (type(val) is int)
            or (type(val) is int)
        ):
            self._sb.append(str(val))
        elif type(val) is complex and (cx := val) is not None:
            self._sb.append(PythonEmitter.format_complex(cx))
        elif isinstance((sl := val), SliceValue):
            self._sb.append("slice(")
            self.write_constant(_item_field(sl, "start", 0))
            self._sb.append(", ")
            self.write_constant(sl.stop)
            self._sb.append(", ")
            self.write_constant(sl.step)
            self._sb.append(")")
        elif isinstance((list_ := val), list):
            self._sb.append("(")
            i = 0
            while i < len(list_):
                if i > 0:
                    self._sb.append(", ")
                self.write_constant(list_[i])
                i += 1
            if len(list_) == 1:
                self._sb.append(",")
            self._sb.append(")")
        elif isinstance((set_ := val), set):
            self._sb.append("{")
            first_set_item = True
            for item in _iter_values(set_):
                if not first_set_item:
                    self._sb.append(", ")
                first_set_item = False
                self.write_constant(item)
            self._sb.append("}")
        elif isinstance((frozen_set := val), FrozenSetValue):
            self._sb.append("{")
            first_frozen_item = True
            for item in _iter_values(frozen_set.items):
                if not first_frozen_item:
                    self._sb.append(", ")
                first_frozen_item = False
                self.write_constant(item)
            self._sb.append("}")
        else:
            self._sb.append(str(val))

    @staticmethod
    def format_complex(c):
        if c.imag == 0.0:
            return "({}+0j)".format(PythonEmitter.format_double(c.real))
        real = PythonEmitter.format_double(c.real)
        imag = PythonEmitter.format_double(c.imag)
        if c.real == 0.0 and (not _is_negative(c.real)):
            return "{}j".format(imag)
        sign = "+" if c.imag >= 0 or _is_negative(c.imag) == False else ""
        return "({}{}{}j)".format(real, sign, imag)

    @staticmethod
    def format_double(d):
        if math.isnan(d):
            return 'float("nan")'
        if _positive_infinity(d):
            return "1e999"
        if _negative_infinity(d):
            return "-1e999"
        if d == 0.0:
            return "-0.0" if _is_negative(d) else "0.0"
        s = str(d)
        if _index_of(s, ".") < 0 and _index_of(s, "E") < 0 and (_index_of(s, "e") < 0):
            s += ".0"
        return s

    @staticmethod
    def format_bytes(bytes_):
        sb = _SourceBuffer("b'")
        for b in _iter_values(bytes_):
            if b == _cast_int("\\"):
                sb.append("\\\\")
            elif b == _cast_int("'"):
                sb.append("\\'")
            elif b == _cast_int("\n"):
                sb.append("\\n")
            elif b == _cast_int("\r"):
                sb.append("\\r")
            elif b == _cast_int("\t"):
                sb.append("\\t")
            elif b >= 32 and b < 127:
                sb.append(chr(b))
            else:
                sb.append("\\x{:02x}".format(b))
        sb.append("'")
        return str(sb)

    @staticmethod
    def format_string_literal(s):
        sb = _SourceBuffer()
        sb.append('"')
        for ch in _iter_values(s):
            if 55296 <= ord(ch) <= 57343:
                sb.append("\\u{:04x}".format(ord(ch)))
            elif ch == "\\":
                sb.append("\\\\")
            elif ch == '"':
                sb.append('\\"')
            elif ch == "\n":
                sb.append("\\n")
            elif ch == "\r":
                sb.append("\\r")
            elif ch == "\t":
                sb.append("\\t")
            elif ord(ch) < 32 or ch == 127:
                sb.append("\\x{:02x}".format(_cast_int(ch)))
            else:
                sb.append(ch)
        sb.append('"')
        return str(sb)

    @staticmethod
    def get_precedence(expr):
        if isinstance(expr, LambdaExpr):
            return 1
        if (
            isinstance(expr, YieldExpr)
            or isinstance(expr, YieldFromExpr)
            or isinstance(expr, AwaitExpr)
        ):
            return 1
        if isinstance(expr, TernaryExpr):
            return 2
        if isinstance((b := expr), BoolOpExpr):
            return 3 if _item_field(b, "op", 0) == "or" else 4
        if isinstance(expr, NamedExpr):
            return 0
        if isinstance((u := expr), UnaryExpr) and _item_field(u, "op", 0) == "not":
            return 5
        if isinstance(expr, CompareExpr):
            return 6
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) == "|":
            return 7
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) == "^":
            return 8
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) == "&":
            return 9
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) in ("<<", ">>"):
            return 10
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) in ("+", "-"):
            return 11
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) in (
            "*",
            "/",
            "//",
            "%",
            "@",
        ):
            return 12
        if isinstance(expr, UnaryExpr):
            return 13
        if isinstance((b := expr), BinaryExpr) and _item_field(b, "op", 0) == "**":
            return 14
        if (
            isinstance(expr, CallExpr)
            or isinstance(expr, SubscriptExpr)
            or isinstance(expr, AttributeExpr)
        ):
            return 16
        if (
            isinstance(expr, ConstantExpr)
            or isinstance(expr, NameExpr)
            or isinstance(expr, AssertionErrorExpr)
            or isinstance(expr, ListExpr)
            or isinstance(expr, TupleExpr)
            or isinstance(expr, DictExpr)
            or isinstance(expr, SetExpr)
            or isinstance(expr, ListCompExpr)
            or isinstance(expr, SetCompExpr)
            or isinstance(expr, DictCompExpr)
            or isinstance(expr, GeneratorExpr)
        ):
            return 17
        if (
            isinstance(expr, SliceExpr)
            or isinstance(expr, StarredExpr)
            or isinstance(expr, DoubleStarredExpr)
            or isinstance(expr, FormattedValueExpr)
            or isinstance(expr, JoinedStrExpr)
        ):
            return 17
        return 0

    def write_clauses(self, clauses):
        for clause in _iter_values(clauses):
            self._sb.append(" async for " if clause.is_async else " for ")
            self.write_expr(clause.target, 0)
            self._sb.append(" in ")
            self.write_expr(clause.iterator, 1)
            i = 0
            while i < len(clause.filters):
                threaded = (
                    clause.filter_threaded is None
                    or i >= len(clause.filter_threaded)
                    or clause.filter_threaded[i]
                )
                if not threaded:
                    self._sb.append("\n")
                    self._sb.append(" " * ((self._indent + 1) * 4))
                self._sb.append(" if ")
                self.write_expr(clause.filters[i], 1)
                i += 1

    def write_formatted_value(self, fv):
        self._sb.append("{")
        self.write_expr(_item_field(fv, "value", 1), 2)
        if not _empty_string(fv.conversion):
            self._sb.append("!")
            self._sb.append(fv.conversion)
        if fv.format_spec is not None:
            self._sb.append(":")
            self.write_format_spec(fv.format_spec)
        self._sb.append("}")

    def write_format_spec(self, spec):
        if (
            isinstance(spec, ConstantExpr)
            and type(_item_field(spec, "value", 1)) is str
            and ((s := _item_field(spec, "value", 1)) is not None)
        ):
            self._sb.append(s)
        elif isinstance((js := spec), JoinedStrExpr):
            for part in _iter_values(js.parts):
                if (
                    isinstance(part, ConstantExpr)
                    and type(_item_field(part, "value", 1)) is str
                    and ((ps := _item_field(part, "value", 1)) is not None)
                ):
                    self._sb.append(ps)
                elif isinstance((fv_part := part), FormattedValueExpr):
                    self.write_formatted_value(fv_part)
                else:
                    self._sb.append("{")
                    self.write_expr(part, 0)
                    self._sb.append("}")
        else:
            self.write_expr(spec, 2)

    def write_indent(self):
        self._sb.append(" " * (self._indent * 4))
