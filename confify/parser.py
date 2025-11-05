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
from .schema import _UnresolvedString, _ParseWarningEntry, _ParseResult, Schema

_T = TypeVar("_T")


# Wait for PEP 747
_TypeFormT = Any


def parse(d: Any, cls: type[_T], options: Optional[ConfifyOptions] = None) -> _T:
    options = ConfifyOptions.get_default() if options is None else options

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
# CLI
################################################################################


def _insert_dict(d: dict[str, Any], keys: list[str], value: Any, prefix: str = "") -> None:
    if len(keys) == 0:
        if not isinstance(value, dict):
            raise ConfifyParseError(f"[{prefix}] Value must be a dict. Got [{value}]")
        d.update(value)
    elif len(keys) == 1:
        key = keys[0]
        if key in d:
            dd = d[key]
            if isinstance(dd, dict) and isinstance(value, dict):
                for k, v in value.items():
                    _insert_dict(dd, [k], v)
            else:
                d[key] = value
        else:
            d[key] = value
    else:
        if keys[0] not in d:
            d[keys[0]] = {}
        dd = d[keys[0]]
        if not isinstance(dd, dict):
            _warning(f"[{prefix}] Overriding non-dict value [{dd}] with [{value}]")
            d[keys[0]] = {}
        _insert_dict(dd, keys[1:], value)


def read_config_from_argv(Config: Type[_T], argv: list[str], options: Optional[ConfifyOptions] = None) -> _T:
    options = ConfifyOptions.get_default() if options is None else options
    args: dict = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        value = argv[i + 1]
        if key.startswith(options.yaml_prefix):
            key = key[len(options.yaml_prefix) :]
            value = read_yaml(value)
        elif key.startswith(options.prefix):
            key = key[len(options.prefix) :]
            value = _UnresolvedString(value)
        else:
            raise ValueError(f"Invalid argument: {key}. Must start with {options.prefix} or {options.yaml_prefix}")
        _insert_dict(args, key.split(".") if key else [], value)
        i += 2
    return parse(args, Config)


def read_config_from_cli(Config: Type[_T], options: Optional[ConfifyOptions] = None) -> _T:
    return read_config_from_argv(Config, sys.argv[1:], options=options)


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
