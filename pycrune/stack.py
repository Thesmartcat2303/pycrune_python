"""The symbolic operand stack."""

from __future__ import annotations

from ._support import (
    DecompilationError,
    _add,
    _get_range,
    _remove_range,
)


class SymbolicStack:
    def __init__(self):
        self._items = []

    def __len__(self):
        return len(self._items)

    def clear(self):
        self._items.clear()

    def push(self, expr):
        _add(self._items, expr)

    def pop(self):
        if len(self._items) == 0:
            raise DecompilationError("Stack underflow")
        last = self._items[-1]
        self._items.pop(len(self._items) - 1)
        return last

    def peek(self, depth=0):
        if depth >= len(self._items):
            raise DecompilationError(
                "Stack depth out of range: {} >= {}".format(depth, len(self._items))
            )
        return self._items[len(self._items) - 1 - depth]

    def swap(self, n):
        if n >= len(self._items):
            raise DecompilationError(
                "Swap index out of range: {} >= {}".format(n, len(self._items))
            )
        top_idx = len(self._items) - 1
        target_idx = top_idx - n
        top = self._items[top_idx]
        self._items[top_idx] = self._items[target_idx]
        self._items[target_idx] = top

    def copy(self, n):
        if n > len(self._items):
            raise DecompilationError("Copy index out of range: {} > {}".format(n, len(self._items)))
        _add(self._items, self._items[len(self._items) - n])

    def pop_count(self, count):
        if count > len(self._items):
            raise DecompilationError(
                "Cannot pop {} items from stack of {}".format(count, len(self._items))
            )
        result = _get_range(self._items, len(self._items) - count, count)
        _remove_range(self._items, len(self._items) - count, count)
        return result

    def clone(self):
        copy = SymbolicStack()
        copy._items.extend(self._items)
        return copy

    def restore_from(self, other):
        self._items.clear()
        self._items.extend(other._items)
