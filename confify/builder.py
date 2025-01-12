from typing import Any, Self, NamedTuple, Optional, Union, TypeVar, Callable, Iterator
import itertools
from pathlib import Path
from enum import Enum
import json
import shlex

from .base import ConfifyCLIConfig


class _ArgEntry(NamedTuple):
    key: str
    values: list[str]
    suffix: Optional[list[str]]


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

def default_suffix_fn(v: Any, key: str) -> str:
    k = key.rsplit(".", 1)[-1].replace("_", "-")
    return f"_{k}={v}"

class CLIBuilder:
    def __init__(self, base_name: str = "", config: ConfifyCLIConfig = ConfifyCLIConfig()):
        self.entries: list[_ArgEntry] = []
        self.base_name = base_name
        self.config = config

    def _add(self, key: str, value: Any, suffix: Optional[str] = None) -> Self:
        suffix_ = [suffix] if suffix is not None else None
        self.entries.append(_ArgEntry(key, [_stringify(value)], suffix_))
        return self

    def add(self, key: str, value: Any, suffix: Optional[str] = None) -> Self:
        return self._add(self.config.prefix + key, value, suffix)

    def add_yaml(self, key: str, value: Union[str, Path], suffix: Optional[str] = None) -> Self:
        return self._add(self.config.yaml_prefix + key, value, suffix)

    def _add_sweep(
        self,
        prefix: str,
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
        self.entries.append(_ArgEntry(prefix + key, [_stringify(v) for v in values], suffix_))
        return self

    def add_sweep(
        self,
        key: str,
        values: list[_T],
        suffix: Union[str, list[str], None] = None,
        suffix_fn: Optional[Callable[[_T, str], str]] = None,
    ) -> Self:
        return self._add_sweep(self.config.prefix, key, values, suffix, suffix_fn)

    def add_yaml_sweep(
        self,
        key: str,
        values: list[_FileLikeT],
        suffix: Union[str, list[str], None] = None,
        suffix_fn: Optional[Callable[[_FileLikeT, str], str]] = None,
    ) -> Self:
        return self._add_sweep(self.config.yaml_prefix, key, values, suffix, suffix_fn)

    def _build(self) -> Iterator[_CLIBuilderResult]:
        entries: list[list[tuple[str, Any, Optional[str]]]] = []
        for e in self.entries:
            key_list = [e.key] * len(e.values)
            suffix_list = [None] * len(e.values) if e.suffix is None else e.suffix
            entries.append(list(zip(key_list, e.values, suffix_list)))
        for all_args in itertools.product(*entries):
            arg_list: list[str] = []
            for k, v, _ in all_args:
                arg_list.append(k)
                arg_list.append(v)
            yield _CLIBuilderResult(
                name=self.base_name + "".join([s for _, _, s in all_args if s is not None]),
                args=[f"{k} {v}" for k, v, _ in all_args],
                arg_list=arg_list,
            )

    def build(self) -> list[_CLIBuilderResult]:
        return list(self._build())

    def build_one(self) -> _CLIBuilderResult:
        return next(self._build())
