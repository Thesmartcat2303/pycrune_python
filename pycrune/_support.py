"""Shared helpers for symbolic decompilation."""

from __future__ import annotations

import math
import re
import sys
from itertools import chain, islice


class DecompilationError(RuntimeError):
    def __init__(self, message, cause=None):
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


class FailFastException(DecompilationError):
    pass


class PatParseException(DecompilationError):
    def __init__(self, message="unsupported match pattern"):
        super().__init__(message)


class _SourceBuffer:
    def __init__(self, prefix=""):
        self._parts = [prefix] if isinstance(prefix, str) and prefix else []

    def append(self, value):
        if value is not None:
            self._parts.append(str(value))
        return self

    def append_line(self, line=""):
        self.append(line)
        self._parts.append("\n")
        return self

    def __str__(self):
        return "".join(self._parts)


class _Enumeration:
    def __init__(self, values):
        self._iterator = iter(values)
        self.current = None

    def move_next(self):
        try:
            self.current = next(self._iterator)
        except StopIteration:
            return False
        return True


class _Groups:
    def __init__(self, key, values):
        self.key = key
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


_MISSING = object()


def _item_field(value, name, tuple_index):
    if isinstance(value, tuple):
        return value[tuple_index]
    return getattr(value, name)


def _iter_values(values):
    return values.items() if isinstance(values, dict) else values


def _enumerator(values):
    return _Enumeration(_iter_values(values))


def _where(values, predicate):
    return filter(predicate, _iter_values(values))


def _select(values, transform):
    return map(transform, _iter_values(values))


def _select_indexed(values, transform):
    return (transform(value, i) for i, value in enumerate(_iter_values(values)))


def _select_many(values, transform):
    return chain.from_iterable(map(transform, _iter_values(values)))


def _first(values, predicate=None, *, default=None):
    return next(
        iter(_iter_values(values) if predicate is None else _where(values, predicate)), default
    )


def _first_required(values, predicate=None):
    result = _first(values, predicate, default=_MISSING)
    if result is _MISSING:
        raise DecompilationError("empty sequence")
    return result


def _last(values, predicate=None):
    result = None
    for result in _iter_values(values) if predicate is None else _where(values, predicate):
        pass
    return result


def _any(values, predicate=None):
    return _first(values, predicate, default=_MISSING) is not _MISSING


def _all(values, predicate):
    return all(map(predicate, _iter_values(values)))


def _count(values, predicate=None):
    if predicate is None and hasattr(values, "__len__"):
        return len(values)
    return sum(
        (1 for _ in (_iter_values(values) if predicate is None else _where(values, predicate)))
    )


def _of_type(values, kind):
    return (value for value in _iter_values(values) if isinstance(value, kind))


def _skip(values, count):
    return islice(_iter_values(values), max(count, 0), None)


def _take(values, count):
    return islice(_iter_values(values), max(count, 0))


def _distinct(values):
    return tuple(dict.fromkeys(_iter_values(values)))


def _order_by(values, key):
    return sorted(_iter_values(values), key=key)


def _order_by_descending(values, key):
    return sorted(_iter_values(values), key=key, reverse=True)


def _sort_ascending_descending(values, primary_key, secondary_key):
    return sorted(sorted(_iter_values(values), key=secondary_key, reverse=True), key=primary_key)


def _group_by(values, key):
    groups = {}
    for value in _iter_values(values):
        groups.setdefault(key(value), []).append(value)
    return [_Groups(key, values) for key, values in groups.items()]


def _default_if_empty(values, default=None):
    iterator = iter(_iter_values(values))
    first = next(iterator, _MISSING)
    if first is _MISSING:
        yield default
    else:
        yield first
        yield from iterator


def _add(collection, value):
    if isinstance(collection, list):
        collection.append(value)
        return True
    original_size = len(collection)
    collection.add(value)
    return len(collection) != original_size


def _remove(collection, value):
    if isinstance(collection, dict):
        return collection.pop(value, _MISSING) is not _MISSING
    try:
        collection.remove(value)
    except (KeyError, ValueError):
        return False
    return True


def _remove_all(values, predicate):
    values[:] = [value for value in values if not predicate(value)]


def _remove_range(values, start, count):
    del values[start : start + count]


def _get_range(values, start, count):
    return values[start : start + count]


def _find_last_index(values, predicate):
    return next((i for i in range(len(values) - 1, -1, -1) if predicate(values[i])), -1)


def _index_of(values, value):
    try:
        return values.index(value)
    except ValueError:
        return -1


def _zip(left, right):
    return zip(left, right)


def _try_add(mapping, key, value):
    if key in mapping:
        return False
    mapping[key] = value
    return True


def _split(value, *separators):
    if len(separators) == 1:
        return value.split(separators[0])
    return re.split("[" + re.escape("".join(separators)) + "]", value)


def _trim(value, *chars):
    return value.strip("".join(chars) if chars else None)


def _parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _cast_int(value):
    return ord(value) if type(value) is str and len(value) == 1 else int(value)


def _compare_to(a, b):
    return (a > b) - (a < b)


def _empty_string(val):
    return val is None or val == ""


def _join(separator, values):
    return separator.join((str(value) if value is not None else "" for value in values))


def _concat(values):
    return _join("", values)


def _empty():
    return []


def _range(start, count):
    return range(start, start + count)


def _positive_infinity(value):
    return value == math.inf


def _negative_infinity(value):
    return value == -math.inf


def _is_negative(value):
    return math.copysign(1.0, value) < 0


def _raise(exc):
    raise exc


def _trace(message):
    print(message, file=sys.stderr)
