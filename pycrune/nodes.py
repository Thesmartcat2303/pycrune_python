"""Syntax nodes reconstructed by the decompiler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AstNode:
    pass


@dataclass
class ExprNode(AstNode):
    pass


@dataclass
class ConstantExpr(ExprNode):
    value: object | None


@dataclass
class NameExpr(ExprNode):
    name: str


@dataclass
class AssertionErrorExpr(ExprNode):
    pass


@dataclass
class UnaryExpr(ExprNode):
    op: str
    operand: ExprNode


@dataclass
class BinaryExpr(ExprNode):
    left: ExprNode
    op: str
    right: ExprNode


@dataclass
class CompareExpr(ExprNode):
    left: ExprNode
    comparators: list[tuple[str, ExprNode]]


@dataclass
class BoolOpExpr(ExprNode):
    op: str
    values: list[ExprNode]


@dataclass
class TernaryExpr(ExprNode):
    condition: ExprNode
    if_true: ExprNode
    if_false: ExprNode


@dataclass
class CallExpr(ExprNode):
    function: ExprNode
    arguments: list[ExprNode]
    keyword_arguments: list[tuple[str, ExprNode]]
    keyword_sequence: list[tuple[str | None, ExprNode]] | None = None


@dataclass
class SubscriptExpr(ExprNode):
    target: ExprNode
    index: ExprNode


@dataclass
class AttributeExpr(ExprNode):
    target: ExprNode
    attribute: str


@dataclass
class SliceExpr(ExprNode):
    lower: ExprNode | None
    upper: ExprNode | None
    step: ExprNode | None


@dataclass
class ListExpr(ExprNode):
    elements: list[ExprNode]


@dataclass
class TupleExpr(ExprNode):
    elements: list[ExprNode]


@dataclass
class SetExpr(ExprNode):
    elements: list[ExprNode]


@dataclass
class DictExpr(ExprNode):
    entries: list[tuple[ExprNode | None, ExprNode]]


@dataclass
class ComprehensionClause:
    target: ExprNode
    iterator: ExprNode
    filters: list[ExprNode]
    is_async: bool = False
    filter_threaded: list[bool] | None = None


@dataclass
class ListCompExpr(ExprNode):
    element: ExprNode
    clauses: list[ComprehensionClause]


@dataclass
class SetCompExpr(ExprNode):
    element: ExprNode
    clauses: list[ComprehensionClause]


@dataclass
class DictCompExpr(ExprNode):
    key: ExprNode
    value: ExprNode
    clauses: list[ComprehensionClause]


@dataclass
class GeneratorExpr(ExprNode):
    element: ExprNode
    clauses: list[ComprehensionClause]


@dataclass
class NamedExpr(ExprNode):
    target: ExprNode
    value: ExprNode


@dataclass
class LambdaExpr(ExprNode):
    parameters: list[str]
    body: ExprNode


@dataclass
class YieldExpr(ExprNode):
    value: ExprNode | None


@dataclass
class YieldFromExpr(ExprNode):
    value: ExprNode


@dataclass
class AwaitExpr(ExprNode):
    value: ExprNode


@dataclass
class FormattedValueExpr(ExprNode):
    value: ExprNode
    conversion: str | None = None
    format_spec: ExprNode | None = None


@dataclass
class JoinedStrExpr(ExprNode):
    parts: list[ExprNode]


@dataclass
class StarredExpr(ExprNode):
    value: ExprNode


@dataclass
class DoubleStarredExpr(ExprNode):
    value: ExprNode


@dataclass
class FunctionExpr(ExprNode):
    name: str
    parameters: list[str]
    body: list[StmtNode]
    decorators: list[ExprNode] | None = None
    is_async: bool = False
    type_params: list[str] | None = None
    return_type: str | None = None


@dataclass
class ClassExpr(ExprNode):
    name: str
    bases: list[ExprNode]
    keywords: list[tuple[str, ExprNode]]
    body: list[StmtNode]
    decorators: list[ExprNode] | None = None
    type_params: list[str] | None = None


@dataclass
class TypeAliasExpr(ExprNode):
    name: str
    type_params: list[str]
    value: ExprNode


@dataclass
class GenericBaseExpr(ExprNode):
    type_params: list[str]


@dataclass
class StmtNode(AstNode):
    pass


@dataclass
class ExprStmt(StmtNode):
    expression: ExprNode


@dataclass
class AssignStmt(StmtNode):
    targets: list[ExprNode]
    value: ExprNode


@dataclass
class AnnAssignStmt(StmtNode):
    target: NameExpr
    annotation: ExprNode
    value: ExprNode | None = None


@dataclass
class AugAssignStmt(StmtNode):
    target: ExprNode
    op: str
    value: ExprNode


@dataclass
class DeleteStmt(StmtNode):
    targets: list[ExprNode]


@dataclass
class IfStmt(StmtNode):
    condition: ExprNode
    then_body: list[StmtNode]
    else_body: list[StmtNode]


@dataclass
class WhileStmt(StmtNode):
    condition: ExprNode
    body: list[StmtNode]
    else_body: list[StmtNode]


@dataclass
class ForStmt(StmtNode):
    target: ExprNode
    iterator: ExprNode
    body: list[StmtNode]
    else_body: list[StmtNode]
    is_async: bool = False


@dataclass
class ExceptClause:
    exception_type: ExprNode | None
    variable_name: str | None
    body: list[StmtNode]
    is_star: bool = False


@dataclass
class TryStmt(StmtNode):
    body: list[StmtNode]
    handlers: list[ExceptClause]
    else_body: list[StmtNode]
    finally_body: list[StmtNode] | None = None


@dataclass
class RaiseStmt(StmtNode):
    exception: ExprNode | None
    from_: ExprNode | None


@dataclass
class ReturnStmt(StmtNode):
    value: ExprNode | None
    from_return_const: bool = False


@dataclass
class BreakStmt(StmtNode):
    pass


@dataclass
class ContinueStmt(StmtNode):
    pass


@dataclass
class PassStmt(StmtNode):
    pass


@dataclass
class AssertStmt(StmtNode):
    test: ExprNode
    message: ExprNode | None


@dataclass
class NonlocalStmt(StmtNode):
    names: list[str]


@dataclass
class GlobalStmt(StmtNode):
    names: list[str]


@dataclass
class FunctionDefStmt(StmtNode):
    name: str
    parameters: list[str]
    body: list[StmtNode]
    decorators: list[ExprNode] | None = None
    is_async: bool = False
    type_params: list[str] | None = None
    return_type: str | None = None


@dataclass
class ClassDefStmt(StmtNode):
    name: str
    bases: list[ExprNode]
    keywords: list[tuple[str, ExprNode]]
    body: list[StmtNode]
    decorators: list[ExprNode] | None = None
    type_params: list[str] | None = None


@dataclass
class TypeAliasStmt(StmtNode):
    name: str
    type_params: list[str]
    value: ExprNode


@dataclass
class ImportAlias:
    name: str
    as_name: str | None


@dataclass
class ImportStmt(StmtNode):
    names: list[ImportAlias]


@dataclass
class ImportFromStmt(StmtNode):
    module: str | None
    names: list[ImportAlias]
    level: int


@dataclass
class WithItem:
    context_expr: ExprNode
    optional_vars: ExprNode | None


@dataclass
class WithStmt(StmtNode):
    items: list[WithItem]
    body: list[StmtNode]
    is_async: bool = False


@dataclass
class PatternNode(AstNode):
    pass


@dataclass
class MatchValuePattern(PatternNode):
    value: ExprNode


@dataclass
class MatchOrPattern(PatternNode):
    patterns: list[PatternNode]


@dataclass
class MatchSequencePattern(PatternNode):
    patterns: list[PatternNode]
    is_list: bool


@dataclass
class MatchClassPattern(PatternNode):
    class_name: ExprNode
    patterns: list[PatternNode]


@dataclass
class MatchKeywordPattern(PatternNode):
    attribute: str
    pattern: PatternNode


@dataclass
class MatchMappingPattern(PatternNode):
    entries: list[tuple[ExprNode, PatternNode]]
    rest: PatternNode | None


@dataclass
class MatchStarPattern(PatternNode):
    name: str


@dataclass
class MatchAsPattern(PatternNode):
    pattern: PatternNode | None
    name: str


@dataclass
class MatchWildcardPattern(PatternNode):
    pass


@dataclass
class MatchCase:
    pattern: PatternNode
    guard: ExprNode | None
    body: list[StmtNode]


@dataclass
class MatchStmt(StmtNode):
    subject: ExprNode
    cases: list[MatchCase]
