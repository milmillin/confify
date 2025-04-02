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

_T = TypeVar("_T")


def __get_subclasses(cls: Type[_T]) -> list[Type[_T]]:
    res: list[Type[_T]] = [cls]
    for subclass in cls.__subclasses__():
        res.extend(__get_subclasses(subclass))
    return res


def _cached_import(module_path, class_name):
    # Check whether module is loaded and fully initialized.
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def _import_string(dotted_path: str):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    try:
        return _cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)) from err


class _UnresolvedString:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def resolve_as_sequence(self) -> list["_UnresolvedString"]:
        s = self.value
        s = s.strip()
        if not ((s[0] == "[" and s[-1] == "]") or (s[0] == "(" and s[-1] == ")")):
            raise ValueError(f"Invalid sequence: {s}")
        s = s[1:-1]
        stk: list[str] = []
        res: list[str] = []
        last: str = ""
        quote: Optional[str] = None
        i = 0
        while i < len(s):
            c = s[i]
            if quote is not None:
                if c == "\\":
                    if s[i + 1] in ["\\", "'", '"']:
                        last += c + s[i + 1]
                        i += 1
                    else:
                        last += c
                elif c == quote:
                    last += c
                    quote = None
                else:
                    last += c
            else:
                if c == "," and len(stk) == 0:
                    res.append(last.strip())
                    last = ""
                elif c == '"' or c == "'":
                    quote = c
                    last += c
                else:
                    if c == "[" or c == "(":
                        stk.append(c)
                    elif c == "]" or c == ")":
                        if len(stk) == 0:
                            raise ValueError(f"Invalid sequence: {s}")
                        if (c == "]" and stk[-1] != "[") or (c == ")" and stk[-1] != "("):
                            raise ValueError(f"Invalid sequence: {s}")
                        stk.pop()
                    last += c
            i += 1
        if len(stk) > 0 or quote is not None:
            raise ValueError(f"Invalid sequence: {s}")
        res.append(last.strip())
        if len(res) == 1 and res[0] == "":
            return []
        return [_UnresolvedString(r) for r in res]


_TYPE_RANK = {
    None: 1,
    type(None): 2,
    bool: 3,
    int: 4,
    float: 5,
    str: 1000,
}

# Wait for PEP 747
_TypeFormT = Any


@dataclass
class _ParseWarningEntry:
    loc: str
    type: str
    value: Any
    message: str


@dataclass
class _ParseResult:
    value: Any
    warnings: list[_ParseWarningEntry] = field(default_factory=list)


def _sanitize(d):
    """Recursively resolve all _UnresolvedString in d."""
    if isinstance(d, _UnresolvedString):
        return d.value
    elif isinstance(d, dict):
        return {k: _sanitize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_sanitize(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(_sanitize(v) for v in d)
    else:
        return d


def _parse_impl(d: Any, cls: _TypeFormT, prefix: str, options: ConfifyOptions) -> _ParseResult:
    """
    Returns:
    - parsed object
    - cost
    """
    warns: list[_ParseWarningEntry] = []
    try:
        if get_origin(cls) is Annotated:
            cls = get_args(cls)[0]
        if cls == Any:
            return _ParseResult(_sanitize(d))
        elif cls == int:
            if isinstance(d, int) and not isinstance(d, bool):
                return _ParseResult(d)
            elif isinstance(d, _UnresolvedString):
                return _ParseResult(int(d.value))
        elif cls == float:
            if isinstance(d, float):
                return _ParseResult(d)
            elif isinstance(d, int):
                return _ParseResult(float(d))
            elif isinstance(d, _UnresolvedString):
                return _ParseResult(float(d.value))
        elif cls == bool:
            if isinstance(d, bool):
                return _ParseResult(d)
            elif isinstance(d, _UnresolvedString):
                d = d.value.strip().lower()
                if d in ["true", "on", "yes"]:
                    return _ParseResult(True)
                elif d in ["false", "off", "no"]:
                    return _ParseResult(False)
        elif cls == str:
            if isinstance(d, _UnresolvedString):
                d = d.value
                if (d.startswith('"') and d.endswith('"')) or (d.startswith("'") and d.endswith("'")):
                    return _ParseResult(ast.literal_eval(d))
                return _ParseResult(d)
            elif isinstance(d, str):
                return _ParseResult(d)
        elif cls == None or cls == type(None):
            if d is None:
                return _ParseResult(None)
            elif isinstance(d, _UnresolvedString):
                d = d.value.strip().lower()
                if d in ["null", "~", "none"]:
                    return _ParseResult(None)
        elif cls == Path:
            if isinstance(d, Path):
                return _ParseResult(d)
            elif isinstance(d, str):
                return _ParseResult(Path(d))
            elif isinstance(d, _UnresolvedString):
                return _ParseResult(Path(d.value))
        elif cls == list or cls == tuple or get_origin(cls) == list or get_origin(cls) == tuple:
            args = get_args(cls)
            elems = None
            if isinstance(d, _UnresolvedString):
                elems = d.resolve_as_sequence()
            elif isinstance(d, (list, tuple)):
                elems = d
            if elems is not None:
                if cls == list or get_origin(cls) == list:
                    if len(args) == 0:
                        args = (Any,)
                    if len(args) == 1:
                        results = [_parse_impl(d, args[0], f"{prefix}[{i}]", options) for i, d in enumerate(elems)]
                        return _ParseResult(
                            [r.value for r in results],
                            warnings=sum([r.warnings for r in results], []),
                        )
                else:
                    # Compatibility hack to distinguish between unparametrized and empty tuple
                    # (tuple[()]), necessary due to https://github.com/python/cpython/issues/91137
                    if len(args) == 0 and (cls is tuple or cls is Tuple):
                        args = (Any, ...)
                    if len(args) == 2 and args[1] == Ellipsis:
                        results = [_parse_impl(d, args[0], f"{prefix}[{i}]", options) for i, d in enumerate(elems)]
                        return _ParseResult(
                            tuple([r.value for r in results]),
                            warnings=sum([r.warnings for r in results], []),
                        )
                    elif len(args) == len(elems):
                        results = [_parse_impl(d, args[i], f"{prefix}[{i}]", options) for i, d in enumerate(elems)]
                        return _ParseResult(
                            tuple([r.value for r in results]),
                            warnings=sum([r.warnings for r in results], []),
                        )
        elif cls == dict or get_origin(cls) == dict:
            args = get_args(cls)
            if len(args) == 0:
                args = (str, Any)
            if len(args) == 2:
                if isinstance(d, dict):
                    entries = [
                        (
                            _parse_impl(k, args[0], f"{prefix}({k})", options),
                            _parse_impl(v, args[1], f"{prefix}[{k}].", options),
                        )
                        for k, v in d.items()
                    ]
                    return _ParseResult(
                        {k.value: v.value for k, v in entries},
                        warnings=sum([r.warnings for kv in entries for r in kv], []),
                    )
        elif get_origin(cls) is Literal:
            args = get_args(cls)
            candidates: list[_ParseResult] = []
            for arg in args:
                try:
                    val = _parse_impl(d, type(arg), f"{prefix}", options)
                    if val.value == arg:
                        candidates.append(val)
                except ConfifyParseError:
                    pass
            candidates.sort(key=lambda x: _TYPE_RANK.get(type(x.value), 0))
            non_str_candidates = [c for c in candidates if not isinstance(c.value, str)]
            if len(non_str_candidates) > 1:
                options_ = ", ".join([f"{repr(c.value)}: {type(c.value).__qualname__}" for c in candidates])
                candidates[0].warnings.insert(
                    0,
                    _ParseWarningEntry(
                        loc=prefix,
                        type=cls,
                        value=d,
                        message=f"Ambiguous input for type [{cls}] at [{prefix}]. Using [{repr(candidates[0].value)}: {type(candidates[0].value).__qualname__}]. Options are [{options_}].",
                    ),
                )
            if len(candidates) >= 1:
                return candidates[0]
        elif get_origin(cls) is Union:
            args = get_args(cls)
            candidates: list[_ParseResult] = []
            for arg in args:
                try:
                    candidates.append(_parse_impl(d, arg, f"{prefix}", options))
                except ConfifyParseError:
                    pass
            candidates.sort(key=lambda x: _TYPE_RANK.get(type(x.value), 0))
            non_str_candidates = [c for c in candidates if not isinstance(c.value, str)]
            if len(non_str_candidates) > 1:
                options_ = ", ".join([f"{repr(c.value)}: {type(c.value).__qualname__}" for c in candidates])
                candidates[0].warnings.insert(
                    0,
                    _ParseWarningEntry(
                        loc=prefix,
                        type=cls,
                        value=d,
                        message=f"Ambiguous input for type [{cls}] at [{prefix}]. Using [{repr(candidates[0].value)}: {type(candidates[0].value).__qualname__}]. Options are [{options_}].",
                    ),
                )
            if len(candidates) >= 1:
                return candidates[0]
        elif isclass(cls) and issubclass(cls, Enum):
            if isinstance(d, str):
                # string loaded from YAML
                d = d.strip()
                return _ParseResult(cls[d])
            elif isinstance(d, _UnresolvedString):
                d = d.value.strip()
                return _ParseResult(cls[d])
            elif isinstance(d, cls):
                return _ParseResult(d)
        elif isclass(cls) and is_dataclass(cls):
            if isinstance(d, dict):
                args = {}
                warns: list[_ParseWarningEntry] = []
                if options.type_key in d:
                    new_cls = _import_string(str(d[options.type_key]))
                    d = dict(d)
                    del d[options.type_key]
                    if not issubclass(new_cls, cls):  # type: ignore
                        warns.append(
                            _ParseWarningEntry(
                                loc=f"{prefix}",
                                type=cls.__name__,
                                value=d,
                                message=f"Type [{new_cls}] is not a subtype of [{cls}] at [{prefix}]",
                            )
                        )
                    else:
                        cls = new_cls
                for f in fields(cls):
                    if f.name not in d:
                        if f.default == MISSING and f.default_factory == MISSING:
                            raise ValueError(f"Missing field: {f.name}")
                    else:
                        results = _parse_impl(d[f.name], f.type, f"{prefix}.{f.name}", options)
                        warns.extend(results.warnings)
                        args[f.name] = results.value
                for k in d.keys():
                    if k not in [f.name for f in fields(cls)]:
                        raise ValueError(f"Got extra field: {k}")
                return _ParseResult(cls(**args), warnings=warns)
            elif isinstance(d, cls):
                return _ParseResult(d)
        elif is_typeddict(cls):
            if isinstance(d, dict):
                args = {}
                warns: list[_ParseWarningEntry] = []
                required_fields: set[str] = set(cls.__required_keys__)  # type: ignore
                all_fields: set[str] = set(cls.__annotations__.keys())
                for field, typ in cls.__annotations__.items():
                    if field not in d:
                        if field in required_fields:
                            raise ValueError(f"Missing field: {field}")
                    else:
                        if get_origin(typ) is NotRequired or get_origin(typ) is Required:
                            typ = get_args(typ)[0]
                        results = _parse_impl(d[field], typ, f"{prefix}.{field}", options)
                        warns.extend(results.warnings)
                        args[field] = results.value
                for k in d.keys():
                    if k not in all_fields:
                        raise ValueError(f"Got extra field: {k}")
                return _ParseResult(cls(**args), warnings=warns)
        else:
            raise ConfifyParseError(f"Unsupported type [{cls}] at [{prefix}].")
    except (ValueError, KeyError) as e:
        warn_msg = ""
        if warns:
            warn_msg = "\nWarnings:\n" + "\n".join([f"\t{w.message}" for w in warns])
        raise ConfifyParseError(
            f"Invalid data for type [{cls}] at [{prefix}]. Got [{repr(d)}].\n\t{type(e).__name__}: {e}" + warn_msg
        )
    raise ConfifyParseError(f"Invalid data for type [{cls}] at [{prefix}]. Got [{repr(d)}].")


def parse(d: Any, cls: type[_T], options: Optional[ConfifyOptions] = None) -> _T:
    options = ConfifyOptions.get_default() if options is None else options
    res = _parse_impl(d, cls, "<root>", options)
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


def _classname(obj: Any) -> str:
    """
    Get the fully qualified name of a class of an object.
    """
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name


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
        res[options.type_key] = _classname(config)
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
