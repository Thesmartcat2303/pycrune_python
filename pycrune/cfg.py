"""Control-flow graphs, dominance, exception scopes and natural loops."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ._support import (
    _MISSING,
    _add,
    _any,
    _distinct,
    _enumerator,
    _first,
    _group_by,
    _item_field,
    _iter_values,
    _last,
    _order_by,
    _remove,
    _select,
    _select_many,
    _sort_ascending_descending,
    _where,
)
from .model import (
    ExceptionTableEntry,
    OpCodes,
    PythonCodeObject,
    PythonInstruction,
)


@dataclass(eq=False)
class BasicBlock:
    id_: int = 0
    start_offset: int = 0
    instructions: list[PythonInstruction] = field(default_factory=list)
    predecessors: list[BasicBlock] = field(default_factory=list)
    successors: list[BasicBlock] = field(default_factory=list)
    is_exception_target: bool = False
    exception_depth: int = 0

    @property
    def end_offset(self):
        return self.instructions[-1].offset if len(self.instructions) > 0 else self.start_offset

    def __str__(self):
        return "Block #{} [{}..{}] (Preds: {}, Succs: {})".format(
            self.id_,
            self.start_offset,
            self.end_offset,
            len(self.predecessors),
            len(self.successors),
        )


@dataclass(eq=False)
class ControlFlowGraph:
    code: PythonCodeObject = None
    blocks: list[BasicBlock] = None
    offset_to_block: dict[int, BasicBlock] = None

    @staticmethod
    def build(code):
        instrs = code.instructions
        if len(instrs) == 0:
            return ControlFlowGraph(code=code, blocks=[], offset_to_block=[])
        leaders = {0}
        for entry in _iter_values(code.exception_table):
            _add(leaders, _item_field(entry, "start", 0))
            _add(leaders, _item_field(entry, "end", 1))
            _add(leaders, entry.target)
        i = 0
        while i < len(instrs):
            ins = instrs[i]
            if ins.target_offset is not None:
                _add(leaders, ins.target_offset)
            if OpCodes.is_jump(ins.op_code) or OpCodes.is_terminator(ins.op_code):
                if i + 1 < len(instrs):
                    _add(leaders, instrs[i + 1].offset)
            i += 1
        blocks = []
        offset_to_block = {}
        current_block = None
        block_id = 0
        for ins in _iter_values(instrs):
            if ins.offset in leaders:
                current_block = BasicBlock(
                    id_=(block_id := (block_id + 1)) - 1, start_offset=ins.offset
                )
                _add(blocks, current_block)
            if current_block is not None:
                _add(current_block.instructions, ins)
                offset_to_block[ins.offset] = current_block
        exc_targets = set(_select(code.exception_table, lambda e: e.target))
        for b in _iter_values(blocks):
            if b.start_offset in exc_targets:
                b.is_exception_target = True
        i = 0
        while i < len(blocks):
            b = blocks[i]
            if len(b.instructions) == 0:
                i += 1
                continue
            last = b.instructions[-1]
            next_block = blocks[i + 1] if i + 1 < len(blocks) else None
            if last.op_code in (
                OpCodes.JUMP_FORWARD,
                OpCodes.JUMP_BACKWARD,
                OpCodes.JUMP_BACKWARD_NO_INTERRUPT,
            ):
                if (
                    last.target_offset is not None
                    and (target := offset_to_block.get(last.target_offset, _MISSING))
                    is not _MISSING
                ):
                    ControlFlowGraph.add_edge(b, target)
            elif OpCodes.is_relative_forward_jump(last.op_code) or last.op_code == OpCodes.FOR_ITER:
                if (
                    last.target_offset is not None
                    and (target := offset_to_block.get(last.target_offset, _MISSING))
                    is not _MISSING
                ):
                    ControlFlowGraph.add_edge(b, target)
                if next_block is not None:
                    ControlFlowGraph.add_edge(b, next_block)
            elif not OpCodes.is_terminator(last.op_code):
                if next_block is not None:
                    ControlFlowGraph.add_edge(b, next_block)
            i += 1
        return ControlFlowGraph(code=code, blocks=blocks, offset_to_block=offset_to_block)

    @staticmethod
    def add_edge(from_, to):
        if not to in from_.successors:
            _add(from_.successors, to)
        if not from_ in to.predecessors:
            _add(to.predecessors, from_)


class DominanceEngine:
    def __init__(self, root, all_nodes, get_predecessors, get_successors):
        self._idom = {}
        self._tin = {}
        self._tout = {}
        self._reachable = set()

        def intersect(b1, b2):
            finger1 = b1
            finger2 = b2
            while not finger1 is finger2:
                while post_order_index[finger1] < post_order_index[finger2]:
                    finger1 = doms[finger1]
                while post_order_index[finger2] < post_order_index[finger1]:
                    finger2 = doms[finger2]
            return finger1

        if root is None:
            return
        post_order = []
        post_order_index = {}
        visited = set()
        stack = []
        _add(visited, root)
        stack.append((root, _enumerator(get_successors(root))))
        while len(stack) > 0:
            top = stack[-1]
            if _item_field(top, "enumerator", 1).move_next():
                succ = _item_field(top, "enumerator", 1).current
                if _add(visited, succ):
                    stack.append((succ, _enumerator(get_successors(succ))))
            else:
                stack.pop()
                post_order_index[_item_field(top, "node", 0)] = len(post_order)
                _add(post_order, _item_field(top, "node", 0))
        for node in _iter_values(visited):
            _add(self._reachable, node)
        rpo = list(post_order)
        rpo.reverse()
        doms = {}
        doms[root] = root
        changed = True
        while changed:
            changed = False
            i = 1
            while i < len(rpo):
                b = rpo[i]
                new_idom = None
                for p in _iter_values(get_predecessors(b)):
                    if p in doms:
                        new_idom = p
                        break
                if new_idom is None:
                    i += 1
                    continue
                for p in _iter_values(get_predecessors(b)):
                    if not p is new_idom and p in doms:
                        new_idom = intersect(p, new_idom)
                if (
                    not (current_idom := doms.get(b, _MISSING)) is not _MISSING
                    or not current_idom is new_idom
                ):
                    doms[b] = new_idom
                    changed = True
                i += 1
        self._idom[root] = None
        i = 1
        while i < len(rpo):
            b = rpo[i]
            if (parent := doms.get(b, _MISSING)) is not _MISSING:
                self._idom[b] = parent
            i += 1
        dom_children = {}
        for b in _iter_values(visited):
            dom_children[b] = []
        i = 1
        while i < len(rpo):
            b = rpo[i]
            if (parent := doms.get(b, _MISSING)) is not _MISSING:
                _add(dom_children[parent], b)
            i += 1
        timer = 0
        tree_stack = []
        self._tin[root] = (timer := (timer + 1))
        tree_stack.append((root, _enumerator(dom_children[root])))
        while len(tree_stack) > 0:
            top = tree_stack[-1]
            if _item_field(top, "enumerator", 1).move_next():
                child = _item_field(top, "enumerator", 1).current
                self._tin[child] = (timer := (timer + 1))
                tree_stack.append((child, _enumerator(dom_children[child])))
            else:
                tree_stack.pop()
                self._tout[_item_field(top, "node", 0)] = (timer := (timer + 1))

    def is_reachable(self, block):
        return block in self._reachable

    def get_immediate_dominator(self, block):
        return self._idom.get(block)

    def dominates(self, a, b):
        if a is b:
            return a in self._reachable
        if (
            not (in_a := self._tin.get(a, _MISSING)) is not _MISSING
            or not (in_b := self._tin.get(b, _MISSING)) is not _MISSING
        ):
            return False
        return in_a <= in_b and self._tout[a] >= self._tout[b]


class DominatorTree:
    def __init__(self, blocks, entry=None):
        if isinstance(blocks, ControlFlowGraph):
            blocks = blocks.blocks
        self._forward_engine = None
        self._reverse_engine = None
        self._artificial_exit = None
        self._frontiers = {}
        self.entry = None
        self.blocks = None
        self.blocks = blocks
        self.entry = entry if entry is not None else blocks[0] if len(blocks) > 0 else None
        self._forward_engine = DominanceEngine(
            self.entry, blocks, lambda b: b.predecessors, lambda b: b.successors
        )
        exit_blocks = list(
            _where(
                blocks, lambda b: self._forward_engine.is_reachable(b) and len(b.successors) == 0
            )
        )
        if len(exit_blocks) == 0 and len(blocks) > 0:
            fallback = (
                _value_1
                if (_value_1 := _last(blocks, self._forward_engine.is_reachable)) is not None
                else blocks[-1]
            )
            _add(exit_blocks, fallback)
        if len(exit_blocks) > 1:

            def get_reverse_predecessors(b):
                if b is self._artificial_exit:
                    return []
                if b in exit_set:
                    return [*b.successors, self._artificial_exit]
                return b.successors

            def get_reverse_successors(b):
                if b is self._artificial_exit:
                    return exit_blocks
                return b.predecessors

            self._artificial_exit = BasicBlock(id_=-1, start_offset=-1)
            exit_set = set(exit_blocks)
            self._reverse_engine = DominanceEngine(
                self._artificial_exit,
                [*blocks, self._artificial_exit],
                get_reverse_predecessors,
                get_reverse_successors,
            )
        else:
            self._artificial_exit = None
            exit_root = exit_blocks[0] if len(exit_blocks) == 1 else None
            self._reverse_engine = DominanceEngine(
                exit_root, blocks, lambda b: b.successors, lambda b: b.predecessors
            )
        for block in _iter_values(blocks):
            self._frontiers[block] = set()
        for b in _iter_values(blocks):
            if self._forward_engine.is_reachable(b) and len(b.predecessors) >= 2:
                b_idom = self._forward_engine.get_immediate_dominator(b)
                for p in _iter_values(b.predecessors):
                    if not self._forward_engine.is_reachable(p):
                        continue
                    runner = p
                    while runner is not None and (not runner is b_idom):
                        if (frontier := self._frontiers.get(runner, _MISSING)) is not _MISSING:
                            _add(frontier, b)
                        runner = self._forward_engine.get_immediate_dominator(runner)

    def dominates(self, a, b):
        return self._forward_engine.dominates(a, b)

    def post_dominates(self, a, b):
        if a is b:
            return self._reverse_engine.is_reachable(a)
        return self._reverse_engine.dominates(a, b)

    def get_immediate_dominator(self, b):
        return self._forward_engine.get_immediate_dominator(b)

    def get_immediate_post_dominator(self, b):
        ipdom = self._reverse_engine.get_immediate_dominator(b)
        return (
            None if self._artificial_exit is not None and ipdom is self._artificial_exit else ipdom
        )

    def get_dominance_frontier(self, b):
        return [*set_] if (set_ := self._frontiers.get(b, _MISSING)) is not _MISSING else []


class ExceptionHandlerKind(Enum):
    except_ = 0
    except_star = 1
    finally_ = 2


@dataclass(eq=False)
class ExceptionHandler:
    kind: ExceptionHandlerKind = None
    entry_block: BasicBlock = None
    target_offset: int = 0
    blocks: set[BasicBlock] = None
    table_entry: ExceptionTableEntry | None = None

    @property
    def is_finally(self):
        return self.kind == ExceptionHandlerKind.finally_

    @property
    def is_star(self):
        return self.kind == ExceptionHandlerKind.except_star

    @property
    def start_offset(self):
        return self.entry_block.start_offset

    @property
    def end_offset(self):
        return (
            max(_select(self.blocks, lambda b: b.end_offset))
            if len(self.blocks) > 0
            else self.entry_block.end_offset
        )

    def get_instructions(self):
        return list(
            _select_many(_order_by(self.blocks, lambda b: b.start_offset), lambda b: b.instructions)
        )


@dataclass(eq=False)
class ExceptionScope:
    id_: int = 0
    start_offset: int = 0
    end_offset: int = 0
    try_blocks: set[BasicBlock] = None
    handlers: list[ExceptionHandler] = None
    finally_handler: ExceptionHandler | None = None
    else_blocks: set[BasicBlock] = None
    exit_block: BasicBlock | None = None
    parent: ExceptionScope | None = None
    children: list[ExceptionScope] = field(default_factory=list)

    @property
    def has_handlers(self):
        return len(self.handlers) > 0

    @property
    def has_finally(self):
        return self.finally_handler is not None

    @property
    def has_else(self):
        return len(self.else_blocks) > 0

    def contains_offset(self, offset):
        return offset >= self.start_offset and offset < self.end_offset

    def contains_block(self, block):
        return (
            block in self.try_blocks
            or _any(self.handlers, lambda h: block in h.blocks)
            or (self.finally_handler is not None and block in self.finally_handler.blocks)
            or (block in self.else_blocks)
        )

    def get_try_instructions(self):
        return list(
            _select_many(
                _order_by(self.try_blocks, lambda b: b.start_offset), lambda b: b.instructions
            )
        )

    def get_else_instructions(self):
        return list(
            _select_many(
                _order_by(self.else_blocks, lambda b: b.start_offset), lambda b: b.instructions
            )
        )

    def get_finally_instructions(self):
        return self.finally_handler.get_instructions() if self.finally_handler is not None else []

    @staticmethod
    def analyze(cfg, dom_tree):
        entries = cfg.code.exception_table
        if len(entries) == 0 or len(cfg.blocks) == 0:
            return []
        offset_to_block = cfg.offset_to_block
        blocks = cfg.blocks
        user_entries = list(
            _sort_ascending_descending(
                _where(entries, lambda e: e.target in offset_to_block),
                lambda e: _item_field(e, "start", 0),
                lambda e: _item_field(e, "end", 1),
            )
        )
        if len(user_entries) == 0:
            return []
        scopes = []
        grouped_by_range = _sort_ascending_descending(
            _group_by(
                user_entries, lambda e: (_item_field(e, "start", 0), _item_field(e, "end", 1))
            ),
            lambda g: _item_field(_item_field(g, "key", 0), "start", 0),
            lambda g: _item_field(_item_field(g, "key", 0), "end", 1),
        )
        scope_id = 0
        for group in _iter_values(grouped_by_range):
            start, end = _item_field(group, "key", 0)
            try_blocks = set(
                _where(blocks, lambda b: b.start_offset >= start and b.start_offset < end)
            )
            if len(try_blocks) == 0:
                continue
            primary_entries = list(group)
            except_entries = list(
                _where(
                    primary_entries,
                    lambda e: not ExceptionScope.is_finally_entry(e, offset_to_block[e.target]),
                )
            )
            finally_entries = list(
                _where(
                    primary_entries,
                    lambda e: ExceptionScope.is_finally_entry(e, offset_to_block[e.target]),
                )
            )
            candidate_chain_heads = list(
                _distinct(_select(except_entries, lambda e: offset_to_block[e.target]))
            )
            all_handler_entries = set()
            raw_handlers = []
            for head in _iter_values(candidate_chain_heads):
                cur = head
                while cur is not None:
                    if not _add(all_handler_entries, cur):
                        break
                    check_match = _first(
                        cur.instructions,
                        lambda i: i.op_code in (OpCodes.CHECK_EXC_MATCH, OpCodes.CHECK_EG_MATCH),
                    )
                    kind = (
                        ExceptionHandlerKind.except_star
                        if check_match is not None and check_match.op_code == OpCodes.CHECK_EG_MATCH
                        else ExceptionHandlerKind.except_
                    )
                    _add(raw_handlers, (cur, kind))
                    if check_match is None:
                        break
                    jump_ins = _first(
                        cur.instructions,
                        lambda i: (
                            i.op_code in (OpCodes.POP_JUMP_IF_FALSE, OpCodes.POP_JUMP_IF_TRUE)
                        ),
                    )
                    if (jump_ins.target_offset if jump_ins is not None else None) is not None and (
                        next_block := offset_to_block.get(jump_ins.target_offset, _MISSING)
                    ) is not _MISSING:
                        if _any(
                            next_block.instructions,
                            lambda i: (
                                i.op_code == OpCodes.RERAISE
                                and (i.op_arg if i.op_arg is not None else 0) == 0
                            ),
                        ):
                            break
                        cur = next_block
                    else:
                        break
            finally_entry_block = None
            finally_table_entry = None
            if len(finally_entries) > 0:
                finally_table_entry = finally_entries[0]
                finally_entry_block = offset_to_block[finally_table_entry.target]
                _add(all_handler_entries, finally_entry_block)
            all_clause_exits = set()
            for tb in _iter_values(try_blocks):
                for succ in _iter_values(tb.successors):
                    if not succ in try_blocks and (not succ in all_handler_entries):
                        _add(all_clause_exits, succ)
            for h_entry, _ in _iter_values(raw_handlers):
                for succ in _iter_values(h_entry.successors):
                    if not succ in try_blocks and (not succ in all_handler_entries):
                        _add(all_clause_exits, succ)
            if finally_entry_block is not None:
                for succ in _iter_values(finally_entry_block.successors):
                    if not succ in try_blocks and (not succ in all_handler_entries):
                        _add(all_clause_exits, succ)
            exit_block = None
            if len(all_clause_exits) > 0:
                exit_block = ExceptionScope.find_convergence_block(all_clause_exits, dom_tree)
            elif finally_entry_block is not None:
                exit_block = finally_entry_block
            handlers = []
            for h_entry, kind in _iter_values(raw_handlers):
                h_blocks = ExceptionScope.collect_handler_blocks(
                    h_entry, exit_block, try_blocks, all_handler_entries
                )
                matching_entry = _first(except_entries, lambda e: e.target == h_entry.start_offset)
                _add(
                    handlers,
                    ExceptionHandler(
                        kind=kind,
                        entry_block=h_entry,
                        target_offset=h_entry.start_offset,
                        blocks=h_blocks,
                        table_entry=matching_entry,
                    ),
                )
            finally_handler = None
            if finally_entry_block is not None:
                f_blocks = ExceptionScope.collect_handler_blocks(
                    finally_entry_block, exit_block, try_blocks, all_handler_entries
                )
                finally_handler = ExceptionHandler(
                    kind=ExceptionHandlerKind.finally_,
                    entry_block=finally_entry_block,
                    target_offset=finally_entry_block.start_offset,
                    blocks=f_blocks,
                    table_entry=finally_table_entry,
                )
            else_blocks = set()
            if exit_block is not None:
                for tb in _iter_values(try_blocks):
                    for succ in _iter_values(tb.successors):
                        if (
                            not succ in try_blocks
                            and succ != exit_block
                            and (not succ in all_handler_entries)
                        ):
                            collected = ExceptionScope.collect_else_blocks(
                                succ, exit_block, try_blocks, all_handler_entries
                            )
                            for b in _iter_values(collected):
                                _add(else_blocks, b)
            scope = ExceptionScope(
                id_=(scope_id := (scope_id + 1)) - 1,
                start_offset=start,
                end_offset=end,
                try_blocks=try_blocks,
                handlers=handlers,
                finally_handler=finally_handler,
                else_blocks=else_blocks,
                exit_block=exit_block,
            )
            _add(scopes, scope)
        ExceptionScope.build_scope_hierarchy(scopes)
        return scopes

    @staticmethod
    def is_finally_entry(entry, target_block):
        if entry.last_i:
            return True
        instrs = target_block.instructions
        if len(instrs) == 0:
            return False
        has_check_match = _any(
            instrs, lambda i: i.op_code in (OpCodes.CHECK_EXC_MATCH, OpCodes.CHECK_EG_MATCH)
        )
        if has_check_match:
            return False
        return _any(
            instrs,
            lambda i: (
                i.op_code == OpCodes.RERAISE and (i.op_arg if i.op_arg is not None else 0) == 0
            ),
        )

    @staticmethod
    def find_convergence_block(candidate_exits, dom_tree):
        candidates = list(candidate_exits)
        if len(candidates) == 0:
            return None
        lca = candidates[0]
        i = 1
        while i < len(candidates):
            target = candidates[i]
            while lca is not None and (not dom_tree.post_dominates(lca, target)):
                lca = dom_tree.get_immediate_post_dominator(lca)
            if lca is None:
                break
            i += 1
        return lca

    @staticmethod
    def collect_handler_blocks(entry_block, exit_block, try_blocks, all_handler_entries):
        result = set()
        visited = set()
        queue = deque()
        queue.append(entry_block)
        _add(visited, entry_block)
        while len(queue) > 0:
            cur = queue.popleft()
            _add(result, cur)
            for succ in _iter_values(cur.successors):
                if succ == exit_block or succ in try_blocks:
                    continue
                if succ in all_handler_entries and succ != entry_block:
                    continue
                if _add(visited, succ):
                    queue.append(succ)
        return result

    @staticmethod
    def collect_else_blocks(start_block, exit_block, try_blocks, all_handler_entries):
        result = set()
        visited = set()
        queue = deque()
        queue.append(start_block)
        _add(visited, start_block)
        while len(queue) > 0:
            cur = queue.popleft()
            _add(result, cur)
            for succ in _iter_values(cur.successors):
                if succ == exit_block or succ in try_blocks or succ in all_handler_entries:
                    continue
                if _add(visited, succ):
                    queue.append(succ)
        return result

    @staticmethod
    def build_scope_hierarchy(scopes):
        i = 0
        while i < len(scopes):
            j = 0
            while j < len(scopes):
                if i == j:
                    j += 1
                    continue
                a = scopes[i]
                b = scopes[j]
                if a.start_offset <= b.start_offset and a.end_offset >= b.end_offset:
                    if b.parent is None or (
                        b.parent.start_offset <= a.start_offset
                        and b.parent.end_offset >= a.end_offset
                    ):
                        _remove(b.parent.children if b.parent is not None else None, b)
                        b.parent = a
                        _add(a.children, b)
                j += 1
            i += 1


@dataclass(eq=False)
class NaturalLoop:
    header: BasicBlock = None
    body: set[BasicBlock] = None
    exits: set[BasicBlock] = None
    latches: set[BasicBlock] = None
    exiting_blocks: set[BasicBlock] = None

    def contains(self, block):
        return block in self.body

    def __str__(self):
        return "Loop (Header: Block #{}, Body: {} blocks, Latches: {}, Exits: {})".format(
            self.header.id_, len(self.body), len(self.latches), len(self.exits)
        )


class LoopAnalysis:
    def __init__(self, blocks, dom_tree=None):
        if isinstance(blocks, ControlFlowGraph):
            graph = blocks
            blocks = graph.blocks
            if dom_tree is None:
                dom_tree = DominatorTree(graph)
        self.loops = None
        self.dominator_tree = None
        self.dominator_tree = dom_tree
        self.loops = LoopAnalysis.detect_loops(blocks, dom_tree)

    def is_loop_header(self, block):
        return _any(self.loops, lambda l: l.header is block)

    def get_loop_by_header(self, header):
        return _first(self.loops, lambda l: l.header is header)

    def get_loops_for_block(self, block):
        return list(_where(self.loops, lambda l: block in l))

    def get_innermost_loop(self, block):
        return _first(_order_by(_where(self.loops, lambda l: block in l), lambda l: len(l.body)))

    def is_in_loop(self, block):
        return _any(self.loops, lambda l: block in l)

    @staticmethod
    def detect_loops(blocks, dom_tree):
        back_edges = {}
        for tail in _iter_values(blocks):
            for head in _iter_values(tail.successors):
                if dom_tree.dominates(head, tail):
                    if not (latches := back_edges.get(head, _MISSING)) is not _MISSING:
                        latches = []
                        back_edges[head] = latches
                    _add(latches, tail)
        result = []
        for header, latches in _iter_values(back_edges):
            body = {header}
            work_list = []
            for latch in _iter_values(latches):
                if _add(body, latch):
                    work_list.append(latch)
            while len(work_list) > 0:
                current = work_list.pop()
                for pred in _iter_values(current.predecessors):
                    if _add(body, pred):
                        work_list.append(pred)
            exits = set()
            exiting_blocks = set()
            for b in _iter_values(body):
                for succ in _iter_values(b.successors):
                    if not succ in body:
                        _add(exits, succ)
                        _add(exiting_blocks, b)
            _add(
                result,
                NaturalLoop(
                    header=header,
                    body=body,
                    exits=exits,
                    latches=latches,
                    exiting_blocks=exiting_blocks,
                ),
            )
        return list(_order_by(result, lambda l: l.header.start_offset))
