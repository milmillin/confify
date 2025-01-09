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
    Annotated
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

_T = TypeVar("_T")


class ConfifyError(Exception):
    pass


class ConfifyWarning(UserWarning):
    pass


def _warning(msg: str) -> None:
    warnings.warn(msg, ConfifyWarning)


def _insert_dict(d: dict[str, Any], keys: list[str], value: Any, prefix: str = "") -> None:
    if len(keys) == 0:
        if not isinstance(value, dict):
            raise ConfifyError(f"[{prefix}] Value must be a dict. Got [{value}]")
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


def __get_subclasses(cls: Type[_T]) -> list[Type[_T]]:
    res: list[Type[_T]] = [cls]
    for subclass in cls.__subclasses__():
        res.extend(__get_subclasses(subclass))
    return res


def cached_import(module_path, class_name):
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
        return cached_import(module_path, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)) from err


# TODO: test
def _str_to_sequence(s: str) -> list[str]:
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
    return res


def _unescape_quotes(s: str) -> str:
    return s.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")


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


def _parse_impl(d: Any, cls: _TypeFormT, prefix: str = "<root>") -> _ParseResult:
    """
    Returns:
    - parsed object
    - cost
    """
    try:
        if get_origin(cls) is Annotated:
            cls = get_args(cls)[0]
        if cls == int:
            if isinstance(d, int):
                return _ParseResult(d)
            elif isinstance(d, str):
                return _ParseResult(int(d))
        elif cls == float:
            if isinstance(d, float):
                return _ParseResult(d)
            elif isinstance(d, (int, str)):
                return _ParseResult(float(d))
        elif cls == bool:
            if isinstance(d, bool):
                return _ParseResult(d)
            elif isinstance(d, str):
                d = d.strip().lower()
                if d in ["true", "1"]:
                    return _ParseResult(True)
                elif d in ["false", "0"]:
                    return _ParseResult(False)
            elif isinstance(d, int):
                if d == 1:
                    return _ParseResult(True)
                elif d == 0:
                    return _ParseResult(False)
        elif cls == str:
            if isinstance(d, str):
                if (d.startswith('"') and d.endswith('"')) or (d.startswith("'") and d.endswith("'")):
                    return _ParseResult(ast.literal_eval(d))
                return _ParseResult(d)
            else:
                return _ParseResult(str(d))
        elif cls == None or cls == type(None):
            if d is None:
                return _ParseResult(None)
            elif isinstance(d, str):
                d = d.strip().lower()
                if d in ["null"]:
                    return _ParseResult(None)
        elif cls == Path:
            if isinstance(d, Path):
                return _ParseResult(d)
            elif isinstance(d, str):
                return _ParseResult(Path(d))
        elif cls == list or cls == tuple or get_origin(cls) == list or get_origin(cls) == tuple:
            args = get_args(cls)
            elems = None
            if isinstance(d, str):
                elems = _str_to_sequence(d)
            elif isinstance(d, (list, tuple)):
                elems = d
            if elems is not None:
                if cls == list or get_origin(cls) == list:
                    if len(args) == 0:
                        return _ParseResult(list(elems))
                    elif len(args) == 1:
                        results = [_parse_impl(d, args[0], f"{prefix}[{i}]") for i, d in enumerate(elems)]
                        return _ParseResult(
                            [r.value for r in results],
                            warnings=sum([r.warnings for r in results], []),
                        )
                else:
                    # Compatibility hack to distinguish between unparametrized and empty tuple
                    # (tuple[()]), necessary due to https://github.com/python/cpython/issues/91137
                    if len(args) == 0 and (cls is tuple or cls is Tuple):
                        return _ParseResult(tuple(elems))
                    elif len(args) == 2 and args[1] == Ellipsis:
                        results = [_parse_impl(d, args[0], f"{prefix}[{i}]") for i, d in enumerate(elems)]
                        return _ParseResult(
                            tuple([r.value for r in results]),
                            warnings=sum([r.warnings for r in results], []),
                        )
                    elif len(args) == len(elems):
                        results = [_parse_impl(d, args[i], f"{prefix}[{i}]") for i, d in enumerate(elems)]
                        return _ParseResult(
                            tuple([r.value for r in results]),
                            warnings=sum([r.warnings for r in results], []),
                        )
        elif cls == dict or get_origin(cls) == dict:
            args = get_args(cls)
            if len(args) == 0:
                if isinstance(d, dict):
                    return _ParseResult(d)
            elif len(args) == 2:
                if isinstance(d, dict):
                    entries = [
                        (
                            _parse_impl(k, args[0], f"{prefix}({k})"),
                            _parse_impl(v, args[1], f"{prefix}[{k}]."),
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
                    val = _parse_impl(d, type(arg), f"{prefix}")
                    if val.value == arg:
                        candidates.append(val)
                except ConfifyError:
                    pass
            candidates.sort(key=lambda x: _TYPE_RANK.get(type(x.value), 0))
            non_str_candidates = [c for c in candidates if not isinstance(c.value, str)]
            if len(non_str_candidates) > 1:
                options = ", ".join([f"{c.value}: {type(c.value).__qualname__}" for c in candidates])
                candidates[0].warnings.insert(
                    0,
                    _ParseWarningEntry(
                        loc=prefix,
                        type=cls,
                        value=d,
                        message=f"Ambiguous input for type [{cls}] at [{prefix}]. Using [{candidates[0].value}: {type(candidates[0].value).__qualname__}]. Options are [{options}].",
                    ),
                )
            if len(candidates) >= 1:
                return candidates[0]
        elif get_origin(cls) is Union:
            args = get_args(cls)
            candidates: list[_ParseResult] = []
            for arg in args:
                try:
                    candidates.append(_parse_impl(d, arg, f"{prefix}"))
                except ConfifyError:
                    pass
            candidates.sort(key=lambda x: _TYPE_RANK.get(type(x.value), 0))
            non_str_candidates = [c for c in candidates if not isinstance(c.value, str)]
            if len(non_str_candidates) > 1:
                options = ", ".join([f"{c.value}: {type(c.value).__qualname__}" for c in candidates])
                candidates[0].warnings.insert(
                    0,
                    _ParseWarningEntry(
                        loc=prefix,
                        type=cls,
                        value=d,
                        message=f"Ambiguous input for type [{cls}] at [{prefix}]. Using [{candidates[0].value}: {type(candidates[0].value).__qualname__}]. Options are [{options}].",
                    ),
                )
            if len(candidates) >= 1:
                return candidates[0]
        elif isclass(cls) and issubclass(cls, Enum):
            if isinstance(d, str):
                d = d.strip()
                return _ParseResult(cls[d])
            elif isinstance(d, cls):
                return _ParseResult(d)
        elif isclass(cls) and is_dataclass(cls):
            if isinstance(d, dict):
                if "$type" in d:
                    new_cls = _import_string(d["$type"])
                    d = dict(d)
                    del d["$type"]
                    if not issubclass(new_cls, cls):  # type: ignore
                        raise ValueError(f"Type {new_cls} is not a is not assignable to {cls}")
                    cls = new_cls
                args = {}
                warns: list[_ParseWarningEntry] = []
                for f in fields(cls):
                    if f.name not in d:
                        if f.default == MISSING and f.default_factory == MISSING:
                            raise ValueError(f"Missing field: {f.name}")
                    else:
                        results = _parse_impl(d[f.name], f.type, f"{prefix}.{f.name}")
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
                        results = _parse_impl(d[field], typ, f"{prefix}.{field}")
                        warns.extend(results.warnings)
                        args[field] = results.value
                for k in d.keys():
                    if k not in all_fields:
                        raise ValueError(f"Got extra field: {k}")
                return _ParseResult(cls(**args), warnings=warns)
        else:
            raise ConfifyError(f"Unsupported type [{cls}] at [{prefix}].")
    except (ValueError, KeyError) as e:
        raise ConfifyError(f"Invalid data for type [{cls}] at [{prefix}]. Got [{d}].\n\t{type(e).__name__}: {e}")
    raise ConfifyError(f"Invalid data for type [{cls}] at [{prefix}]. Got [{d}].")


def parse(d: Any, cls: type[_T]) -> _T:
    res = _parse_impl(d, cls, "<root>")
    if len(res.warnings) > 0:
        warns: list[str] = []
        for w in res.warnings:
            warns.append(f"-> {w.message}")
        _warning("Some warnings were encountered during parsing:\n" + "\n".join(warns))
    return res.value


# alias to avoid type errors in tests
def _parse(d: Any, cls: _TypeFormT) -> Any:
    return parse(d, cls)


def parse_yaml(file: Union[str, Path], cls: type[_T]) -> _T:
    value = yaml.full_load(Path(file).open("r", encoding="utf-8"))
    return parse(value, cls)


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


def config_dump_python(config: Any, ignore: list[str] = []) -> Any:
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
            f.name: config_dump_python(getattr(config, f.name), ignore=field_ignore.get(f.name, []))
            for f in fields(config)
            if f.name not in ignored_fields
        }
        res["$type"] = _classname(config)
        return res
    elif isinstance(config, dict):
        return {k: config_dump_python(v, ignore=field_ignore.get(k, [])) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_dump_python(v) for v in config]
    elif isinstance(config, tuple):
        return tuple(config_dump_python(v) for v in config)
    else:
        return config


def config_dump_yaml(file: Union[str, Path], config: Any, ignore: list[str] = []) -> Any:
    data = config_dump_python(config, ignore=ignore)
    yaml.dump(data, Path(file).open("w", encoding="utf-8"))
    return data


def read_config_from_argv(Config: Type[_T], argv: list[str]) -> _T:
    args: dict = {}
    i = 0
    while i < len(argv):
        key = argv[i]
        value = argv[i + 1]
        if key.startswith("---"):
            key = key[3:]
            value = yaml.full_load(Path(value).open("r", encoding="utf-8"))
        elif key.startswith("--"):
            key = key[2:]
        else:
            raise ValueError(f"Invalid argument: {key}. Must start with -- or ---")
        _insert_dict(args, key.split(".") if key else [], value)
        i += 2
    return parse(args, Config)


def read_config_from_cli(Config: Type[_T]) -> _T:
    return read_config_from_argv(Config, sys.argv[1:])
