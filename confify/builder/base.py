from typing import Any, Self, NamedTuple, Optional, Union, TypeVar, Callable, Iterator
import itertools
from pathlib import Path
from enum import Enum
import json
import shlex

from ..base import ConfifyOptions


class _ArgEntry(NamedTuple):
    suffix: Optional[str]
    kv: list[tuple[str, Any]]


_T = TypeVar("_T")
_FileLikeT = TypeVar("_FileLikeT", str, Path)


class _CLIBuilderResult(NamedTuple):
    name: str
    args: list[str]
    arg_list: list[str]

    def __repr__(self):
        return f"{self.name}: {' '.join(self.args)}"


def _stringify_impl(v: Any, is_root: bool = True) -> str:
    if isinstance(v, (str, Path)):
        s = json.dumps(str(v))
        if is_root:
            return s[1:-1]
        else:
            return s
    elif isinstance(v, (bool, int, float)) or v is None:
        return str(v)
    elif isinstance(v, Enum):
        return v.name
    elif isinstance(v, list):
        return "[" + ", ".join([_stringify_impl(e, False) for e in v]) + "]"
    elif isinstance(v, tuple):
        return "(" + ", ".join([_stringify_impl(e, False) for e in v]) + ")"
    else:
        raise ValueError(f"Unsupported type: {type(v)}")


def _stringify(v: Any) -> str:
    return shlex.quote(_stringify_impl(v))


class CLIBuilder:
    def __init__(self, base_name: str = ""):
        self.entries: list[list[_ArgEntry]] = []
        self.base_name = base_name

    def add(self, key: str, value: Any) -> Self:
        self.entries.append([_ArgEntry(None, [(key, _stringify(value))])])
        return self

    def add_sweep(
        self,
        key: str,
        values: list[_T],
        suffix: Union[str, list[str], None] = None,
        suffix_fn: Optional[Callable[[_T, str], str]] = None,
    ) -> Self:
        suffix_: list[str] = []
        if suffix_fn is not None:
            if suffix is not None:
                raise ValueError("Cannot specify both suffix and suffix_fn")
            suffix_ = [suffix_fn(v, key) for v in values]
        elif suffix is not None:
            if isinstance(suffix, str):
                suffix_ = [suffix] * len(values)
            else:
                if len(suffix) != len(values):
                    raise ValueError("suffix must have the same length as values")
                suffix_ = list(suffix)
        else:
            raise ValueError("Must specify either suffix or suffix_fn")
        if len(set(suffix_)) != len(suffix_):
            raise ValueError("suffix must be unique")
        self.entries.append([_ArgEntry(s, [(key, _stringify(v))]) for s, v in zip(suffix_, values)])
        return self

    def add_sweep_set(
        self,
        sets: dict[str, dict[str, _T]],
    ) -> Self:
        self.entries.append([_ArgEntry(s, [(k, _stringify(v)) for k, v in set_.items()]) for s, set_ in sets.items()])
        return self

    def _build(self) -> Iterator[_CLIBuilderResult]:
        for all_args in itertools.product(*self.entries):
            yield _CLIBuilderResult(
                name=self.base_name + "".join([s for s, _ in all_args if s is not None]),
                args=[f"{k} {v}" for _, kvs in all_args for k, v in kvs],
                arg_list=[x for _, kvs in all_args for kv in kvs for x in kv],
            )

    def build(self) -> list[_CLIBuilderResult]:
        return list(self._build())

    def build_one(self) -> _CLIBuilderResult:
        return next(self._build())
