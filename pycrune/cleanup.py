"""Remove compiler-generated exception and control-flow scaffolding."""

from __future__ import annotations

from ._support import (
    _add,
    _all,
    _any,
    _empty_string,
    _item_field,
    _iter_values,
)
from .nodes import (
    AssignStmt,
    AttributeExpr,
    CallExpr,
    ClassDefStmt,
    ConstantExpr,
    DeleteStmt,
    ExceptClause,
    ExprStmt,
    ForStmt,
    FunctionDefStmt,
    IfStmt,
    NameExpr,
    TryStmt,
    WhileStmt,
    WithStmt,
)


class SyntheticCleanupPass:
    @staticmethod
    def clean_statements(statements):
        cleaned = SyntheticCleanupPass.clean_exception_variable(statements)
        result = []
        for stmt in _iter_values(cleaned):
            _add(result, SyntheticCleanupPass.clean_statement(stmt))
        return result

    @staticmethod
    def clean_statement(statement):
        if isinstance((try_stmt := statement), TryStmt):
            return SyntheticCleanupPass.clean_try(try_stmt)
        if isinstance((with_stmt := statement), WithStmt):
            return SyntheticCleanupPass.clean_with(with_stmt)
        if isinstance((if_stmt := statement), IfStmt):
            return IfStmt(
                if_stmt.condition,
                SyntheticCleanupPass.clean_statements(if_stmt.then_body),
                SyntheticCleanupPass.clean_statements(if_stmt.else_body),
            )
        if isinstance((while_stmt := statement), WhileStmt):
            return WhileStmt(
                while_stmt.condition,
                SyntheticCleanupPass.clean_statements(while_stmt.body),
                SyntheticCleanupPass.clean_statements(while_stmt.else_body),
            )
        if isinstance((for_stmt := statement), ForStmt):
            return ForStmt(
                for_stmt.target,
                for_stmt.iterator,
                SyntheticCleanupPass.clean_statements(for_stmt.body),
                SyntheticCleanupPass.clean_statements(for_stmt.else_body),
                for_stmt.is_async,
            )
        if isinstance((fn := statement), FunctionDefStmt):
            return FunctionDefStmt(
                _item_field(fn, "name", 0),
                fn.parameters,
                SyntheticCleanupPass.clean_statements(fn.body),
                fn.decorators,
                fn.is_async,
                fn.type_params,
                fn.return_type,
            )
        if isinstance((cls := statement), ClassDefStmt):
            return ClassDefStmt(
                _item_field(cls, "name", 0),
                cls.bases,
                cls.keywords,
                SyntheticCleanupPass.clean_class_body(cls.body),
                cls.decorators,
                cls.type_params,
            )
        return statement

    @staticmethod
    def clean_exception_variable(statements, variable_name=None):
        result = []
        count = len(statements)
        i = 0
        while i < count:
            stmt = statements[i]
            if not _empty_string(variable_name):
                if SyntheticCleanupPass.is_exception_variable_cleanup(stmt, variable_name):
                    i += 1
                    continue
            elif (
                i + 1 < count
                and (
                    (_parsed_1 := SyntheticCleanupPass.is_exception_variable_assign(stmt)),
                    (assign_name := _parsed_1[1]),
                    _parsed_1[0],
                )[-1]
                and SyntheticCleanupPass.is_exception_variable_delete(
                    statements[i + 1], assign_name
                )
            ):
                i += 1
                i += 1
                continue
            _add(result, stmt)
            i += 1
        return result

    @staticmethod
    def clean_context_manager_exit(statements):
        list_ = list(statements)
        while len(list_) > 0 and SyntheticCleanupPass.is_synthetic_context_manager_exit(list_[-1]):
            list_.pop(len(list_) - 1)
        return list_

    @staticmethod
    def is_synthetic_exception_cleanup(statement, variable_name=None):
        if isinstance((assign := statement), AssignStmt):
            if (
                (_subject_1 := _item_field(assign, "value", 1)),
                isinstance(_subject_1, ConstantExpr)
                and _item_field(_subject_1, "value", 1) is None,
            )[-1] and len(assign.targets) > 0:
                if _empty_string(variable_name):
                    return _all(assign.targets, lambda t: isinstance(t, NameExpr))
                return _any(
                    assign.targets,
                    lambda t: (
                        isinstance((n := t), NameExpr)
                        and _item_field(n, "name", 0) == variable_name
                    ),
                )
        elif isinstance((del_ := statement), DeleteStmt):
            if len(del_.targets) > 0:
                if _empty_string(variable_name):
                    return _all(del_.targets, lambda t: isinstance(t, NameExpr))
                return _any(
                    del_.targets,
                    lambda t: (
                        isinstance((n := t), NameExpr)
                        and _item_field(n, "name", 0) == variable_name
                    ),
                )
        return False

    @staticmethod
    def is_synthetic_context_manager_exit(statement):
        if isinstance((expr_stmt := statement), ExprStmt) and isinstance(
            (call := expr_stmt.expression), CallExpr
        ):
            if isinstance((attr := call.function), AttributeExpr) and attr.attribute in (
                "__exit__",
                "__aexit__",
            ):
                return len(call.arguments) == 0 or _all(
                    call.arguments,
                    lambda a: isinstance(a, ConstantExpr) and _item_field(a, "value", 1) is None,
                )
            if isinstance((name := call.function), NameExpr) and _item_field(name, "name", 0) in (
                "__exit__",
                "__aexit__",
            ):
                return len(call.arguments) == 0 or _all(
                    call.arguments,
                    lambda a: isinstance(a, ConstantExpr) and _item_field(a, "value", 1) is None,
                )
        return False

    @staticmethod
    def clean_try(try_stmt):
        clean_body = SyntheticCleanupPass.clean_statements(try_stmt.body)
        clean_handlers = []
        for handler in _iter_values(try_stmt.handlers):
            cleaned_handler_body = SyntheticCleanupPass.clean_statements(handler.body)
            if not _empty_string(handler.variable_name):
                cleaned_handler_body = SyntheticCleanupPass.clean_exception_variable(
                    cleaned_handler_body, handler.variable_name
                )
            _add(
                clean_handlers,
                ExceptClause(
                    handler.exception_type,
                    handler.variable_name,
                    cleaned_handler_body,
                    handler.is_star,
                ),
            )
        clean_else = SyntheticCleanupPass.clean_statements(try_stmt.else_body)
        clean_finally = (
            SyntheticCleanupPass.clean_statements(try_stmt.finally_body)
            if try_stmt.finally_body is not None
            else None
        )
        return TryStmt(clean_body, clean_handlers, clean_else, clean_finally)

    @staticmethod
    def clean_with(with_stmt):
        clean_body = SyntheticCleanupPass.clean_statements(with_stmt.body)
        clean_body = SyntheticCleanupPass.clean_context_manager_exit(clean_body)
        return WithStmt(with_stmt.items, clean_body, with_stmt.is_async)

    @staticmethod
    def clean_class_body(statements):
        cleaned = SyntheticCleanupPass.clean_statements(statements)
        result = []
        for stmt in _iter_values(cleaned):
            if SyntheticCleanupPass.is_synthetic_class_statement(stmt):
                continue
            _add(result, stmt)
        return result

    @staticmethod
    def is_synthetic_class_statement(stmt):
        if isinstance((assign := stmt), AssignStmt):
            if (
                len(assign.targets) == 1
                and (
                    (_subject_1 := assign.targets[0]),
                    isinstance((target_name := _subject_1), NameExpr),
                )[-1]
            ):
                if _item_field(target_name, "name", 0) in (
                    "__classdict__",
                    "__classdictcell__",
                    "__classcell__",
                    "__type_params__",
                ):
                    return True
        return False

    @staticmethod
    def is_exception_variable_cleanup(stmt, variable_name):
        if isinstance((assign := stmt), AssignStmt):
            return (
                (_subject_1 := _item_field(assign, "value", 1)),
                isinstance(_subject_1, ConstantExpr)
                and _item_field(_subject_1, "value", 1) is None,
            )[-1] and _any(
                assign.targets,
                lambda t: (
                    isinstance((n := t), NameExpr) and _item_field(n, "name", 0) == variable_name
                ),
            )
        if isinstance((del_ := stmt), DeleteStmt):
            return _any(
                del_.targets,
                lambda t: (
                    isinstance((n := t), NameExpr) and _item_field(n, "name", 0) == variable_name
                ),
            )
        return False

    @staticmethod
    def is_exception_variable_assign(stmt):
        var_name = None
        if (
            isinstance((assign := stmt), AssignStmt)
            and (
                (_subject_1 := _item_field(assign, "value", 1)),
                isinstance(_subject_1, ConstantExpr)
                and _item_field(_subject_1, "value", 1) is None,
            )[-1]
            and (len(assign.targets) == 1)
            and ((_subject_2 := assign.targets[0]), isinstance((n := _subject_2), NameExpr))[-1]
        ):
            var_name = _item_field(n, "name", 0)
            return (True, var_name)
        var_name = None
        return (False, var_name)

    @staticmethod
    def is_exception_variable_delete(stmt, var_name):
        if (
            isinstance((del_ := stmt), DeleteStmt)
            and len(del_.targets) == 1
            and ((_subject_1 := del_.targets[0]), isinstance((n := _subject_1), NameExpr))[-1]
            and (_item_field(n, "name", 0) == var_name)
        ):
            return True
        return False
