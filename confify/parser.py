import pickle
from pathlib import Path
from typing import (
    Any,
    Type,
    Union,
    TypeVar,
    cast,
    Optional,
    List,
    Tuple,
    Dict,
    get_origin,
    Literal,
    get_args,
    is_typeddict,
    Required,
    NotRequired,
    Annotated,
    Generic,
)
from enum import Enum
from types import GenericAlias
from inspect import isclass
import sys
from dataclasses import fields, is_dataclass, MISSING, dataclass, field
import yaml
from importlib import import_module
from functools import partialmethod
import warnings
import ast

from .base import ConfifyParseError, _warning, ConfifyOptions
from .yaml import ConfifyDumper, ConfifyLoader
from .utils import classname, import_string
from .schema import UnresolvedString, _ParseWarningEntry, _ParseResult, Schema

_T = TypeVar("_T")


# Wait for PEP 747
_TypeFormT = Any


def parse(d: Any, cls: type[_T], options: Optional[ConfifyOptions] = None, schema: Optional[Schema] = None) -> _T:
    options = ConfifyOptions.get_default() if options is None else options

    if schema is None:
        schema = Schema.from_typeform(cls)
    res = schema._parse(d, "<root>", options)

    if len(res.warnings) > 0:
        warns: list[str] = []
        for w in res.warnings:
            warns.append(f"-> {w.message}")
        _warning("Some warnings were encountered during parsing:\n" + "\n".join(warns))
    return res.value


# alias to avoid type errors in tests
def _parse(d: Any, cls: _TypeFormT, options: Optional[ConfifyOptions] = None) -> Any:
    return parse(d, cls, options)


def read_yaml(file: Union[str, Path]) -> Any:
    return yaml.load(Path(file).open("r", encoding="utf-8"), Loader=ConfifyLoader)


def parse_yaml(
    file: Union[str, Path], cls: type[_T], /, keys: str = "", options: Optional[ConfifyOptions] = None
) -> _T:
    """
    Parse a YAML file into a specified type.

    Args:
        `file`: The path to the YAML file.
        `cls`: The type to parse the YAML file into.
        `keys`: A dotlist notation of keys to parse. If empty, parse the entire file.
    """
    value = read_yaml(file)
    keys_ = keys.split(".") if keys else []
    for k in keys_:
        if k not in value:
            raise ConfifyParseError(f"Key {k} not found in YAML file.")
        value = value[k]
    return parse(value, cls, options=options)


################################################################################
# Dumping
################################################################################


def _config_dump_impl(config: Any, ignore: list[str], options: ConfifyOptions) -> Any:
    field_ignore: dict[str, list[str]] = {}
    ignored_fields: set[str] = set()
    for f in ignore:
        _fields = f.split(".", 1)
        if len(_fields) == 1:
            ignored_fields.add(f)
        else:
            field, subfield = _fields
            if field not in field_ignore:
                field_ignore[field] = []
            field_ignore[field].append(subfield)
    if is_dataclass(config):
        res = {
            f.name: _config_dump_impl(getattr(config, f.name), ignore=field_ignore.get(f.name, []), options=options)
            for f in fields(config)
            if f.name not in ignored_fields
        }
        res[options.type_key] = classname(config)
        return res
    elif isinstance(config, dict):
        return {k: _config_dump_impl(v, ignore=field_ignore.get(k, []), options=options) for k, v in config.items()}
    elif isinstance(config, list):
        return [_config_dump_impl(v, ignore=[], options=options) for v in config]
    elif isinstance(config, tuple):
        return tuple(_config_dump_impl(v, ignore=[], options=options) for v in config)
    else:
        return config


def config_dump_yaml(
    config: Any, file: Union[str, Path], /, ignore: list[str] = [], options: Optional[ConfifyOptions] = None
) -> Any:
    """
    Dump a config to a YAML file.

    Args:
        `config`: The config to dump.
        `file`: The path to the YAML file.
        `ignore`: A list of fields in dotlist notations to ignore.
    """
    options = ConfifyOptions.get_default() if options is None else options
    data = _config_dump_impl(config, ignore=ignore, options=options)
    yaml.dump(data, Path(file).open("w", encoding="utf-8"), Dumper=ConfifyDumper)
    return data


def config_dump(config: Any, /, ignore: list[str] = [], options: Optional[ConfifyOptions] = None) -> Any:
    """
    Dump a config to a Python dict

    Args:
        `config`: The config to dump.
        `ignore`: A list of fields in dotlist notations to ignore.
    """
    options = ConfifyOptions.get_default() if options is None else options
    return _config_dump_impl(config, ignore=ignore, options=options)
