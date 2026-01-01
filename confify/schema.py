from enum import Enum
from typing import (
    Any,
    Union,
    Annotated,
    get_origin,
    get_args,
    Iterable,
    Sequence,
    Literal,
    Type,
    is_typeddict,
    Required,
    NotRequired,
    Optional,
    Never,
    Tuple,
    TypeVar,
)
from collections.abc import Iterable as CollectionsIterable, Sequence as CollectionsSequence
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import fields, is_dataclass, MISSING, dataclass, field
from inspect import isclass
import ast

from .base import ConfifyTypeError, ConfifyOptions, _warning, ConfifyParseError, ConfifyBuilderError
from .utils import classname_of_cls, import_string, repr_of_typeform

# Wait for PEP 747
_TypeFormT = Any


class UnresolvedString:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def resolve_as_any(self, options: ConfifyOptions = ConfifyOptions.get_default()) -> Any:
        s = self.value
        s = s.strip()
        if s == "":
            return ""
        if s[0] == "[" and s[-1] == "]":
            return [x.resolve_as_any() for x in self.resolve_as_sequence()]
        elif s[0] == "(" and s[-1] == ")":
            return tuple(x.resolve_as_any() for x in self.resolve_as_sequence())
        elif (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'"):
            return ast.literal_eval(s)
        else:
            for schema in [_none_schema, _bool_schema, _int_schema, _float_schema]:
                try:
                    return schema.parse(self)
                except ConfifyParseError:
                    pass
            return s

    def resolve_as_sequence(self) -> list["UnresolvedString"]:
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
        return [UnresolvedString(r) for r in res]

    @classmethod
    def sanitize(cls, d: Any) -> Any:
        """Recursively resolve all _UnresolvedString in d."""
        if isinstance(d, UnresolvedString):
            return d.resolve_as_any()
        elif isinstance(d, dict):
            return {k: cls.sanitize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [cls.sanitize(v) for v in d]
        elif isinstance(d, tuple):
            return tuple(cls.sanitize(v) for v in d)
        else:
            return d


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


def _substitute_typevars(T: _TypeFormT, type_dict: dict[TypeVar, _TypeFormT]) -> _TypeFormT:
    if isinstance(T, TypeVar):
        return type_dict[T]
    Origin = get_origin(T)
    if Origin is None:
        return T
    args = get_args(T)
    if len(args) == 0:
        return Origin
    return Origin[tuple(_substitute_typevars(a, type_dict) for a in args)]


def _parse_dataclass_inner(T: type, type_args: tuple[_TypeFormT, ...], prefix: str) -> dict[int, _TypeFormT]:
    type_params: tuple[_TypeFormT, ...] = getattr(T, "__parameters__", ())

    # If type_args is empty, we fill it with Any
    if len(type_args) == 0:
        type_args = (Any,) * len(type_params)

    if len(type_args) != len(type_params):
        raise ConfifyTypeError(
            f"Invalid type arguments for `{T}{list(type_params)}` at `{prefix}`; actual {len(type_args)}, expected {len(type_params)}"
        )

    type_dict = dict(zip(type_params, type_args))

    field_map: dict[int, _TypeFormT] = {}
    bases = getattr(T, "__orig_bases__", T.__bases__)
    for BaseT in bases[::-1]:
        BaseT: type
        Orig: Optional[type] = get_origin(BaseT)
        args_ = get_args(BaseT)
        if is_dataclass(BaseT):
            field_map_ = _parse_dataclass_inner(
                BaseT,
                (),
                prefix,
            )
        elif Orig is not None and is_dataclass(Orig):
            field_map_ = _parse_dataclass_inner(
                Orig,
                tuple(type_dict.get(p, p) for p in args_),
                prefix,
            )
        else:
            continue
        field_map.update(field_map_)

    cur_field_map: dict[int, _TypeFormT] = {}
    for f in fields(T):
        id_ = id(f)
        if id_ in field_map:
            cur_field_map[id_] = field_map[id_]
        else:
            cur_field_map[id_] = _substitute_typevars(f.type, type_dict)
    return cur_field_map


def _parse_dataclass(
    T: type, type_args: tuple[_TypeFormT, ...], prefix: str
) -> tuple[dict[str, _TypeFormT], dict[str, _TypeFormT]]:
    field_map = _parse_dataclass_inner(T, type_args, prefix)
    required_fields: dict[str, _TypeFormT] = {}
    optional_fields: dict[str, _TypeFormT] = {}
    for f in fields(T):
        if f.default == MISSING and f.default_factory == MISSING:
            required_fields[f.name] = field_map[id(f)]
        else:
            optional_fields[f.name] = field_map[id(f)]
    return required_fields, optional_fields


class Schema(ABC):
    annotation: _TypeFormT

    def __init__(self, annotation: _TypeFormT):
        self.annotation = annotation

    @classmethod
    def _from_typeform(cls, T: _TypeFormT, prefix: str) -> "Schema":
        OgT = T
        if get_origin(T) is Annotated:
            T = get_args(T)[0]

        Origin = get_origin(T)
        args = get_args(T)

        if T == Any:
            return AnySchema(OgT)
        elif T == int:
            return IntSchema(OgT)
        elif T == float:
            return FloatSchema(OgT)
        elif T == bool:
            return BoolSchema(OgT)
        elif T == str:
            return StrSchema(OgT)
        elif T == None or T == type(None):
            return NoneSchema(OgT)
        elif T == Path:
            return PathSchema(OgT)
        elif T == list or Origin == list or Origin == CollectionsIterable or Origin == CollectionsSequence:
            if len(args) == 0:
                return ListSchema(OgT, AnySchema(Any))
            elif len(args) == 1:
                return ListSchema(OgT, Schema._from_typeform(args[0], f"{prefix}[0]"))
            else:
                raise ConfifyTypeError(f"Invalid list type: `{T}` at `{prefix}`")
        elif T == tuple or Origin == tuple:
            # Compatibility hack to distinguish between unparametrized and empty tuple
            # (tuple[()]), necessary due to https://github.com/python/cpython/issues/91137
            if len(args) == 0 and (T is tuple or T is Tuple):
                # This will match unparametrized tuple
                return TupleSchema(OgT, AnySchema(Any))
            elif len(args) == 2 and args[1] == Ellipsis:
                return TupleSchema(OgT, Schema._from_typeform(args[0], f"{prefix}.$items"))
            else:
                return TupleSchema(OgT, [Schema._from_typeform(a, f"{prefix}.{i}") for i, a in enumerate(args)])
        elif T == dict or get_origin(T) == dict:
            if len(args) == 0:
                return MappingSchema(OgT, StrSchema(str), AnySchema(Any))
            elif len(args) == 2:
                return MappingSchema(
                    OgT,
                    Schema._from_typeform(args[0], f"{prefix}.$key"),
                    Schema._from_typeform(args[1], f"{prefix}.$value"),
                )
            else:
                raise ConfifyTypeError(f"Invalid dict type: `{T}` at `{prefix}`")
        elif Origin is Literal:
            return LiteralSchema(OgT, args)
        elif Origin is Union:
            return UnionSchema(OgT, [Schema._from_typeform(a, f"{prefix}.{i}") for i, a in enumerate(args)])
        elif isclass(T) and issubclass(T, Enum):
            return EnumSchema(OgT, T)
        elif is_typeddict(T):
            optional_fields: dict[str, Schema] = {}
            required_fields: dict[str, Schema] = {}
            required_keys: set[str] = set(T.__required_keys__)  # type: ignore
            for field, typ in T.__annotations__.items():
                if get_origin(typ) is NotRequired or get_origin(typ) is Required:
                    typ = get_args(typ)[0]
                if field in required_keys:
                    required_fields[field] = Schema._from_typeform(typ, f"{prefix}.{field}")
                else:
                    optional_fields[field] = Schema._from_typeform(typ, f"{prefix}.{field}")
            return DictSchema(OgT, required_fields, optional_fields, T, ())
        elif isclass(T) and is_dataclass(T):
            required_fields_, optional_fields_ = _parse_dataclass(T, (), prefix)
            return DictSchema(
                OgT,
                {k: Schema._from_typeform(v, f"{prefix}.{k}") for k, v in required_fields_.items()},
                {k: Schema._from_typeform(v, f"{prefix}.{k}") for k, v in optional_fields_.items()},
                T,
                (),
            )
        elif isclass(Origin) and is_dataclass(Origin):
            required_fields_, optional_fields_ = _parse_dataclass(Origin, args, prefix)
            return DictSchema(
                OgT,
                {k: Schema._from_typeform(v, f"{prefix}.{k}") for k, v in required_fields_.items()},
                {k: Schema._from_typeform(v, f"{prefix}.{k}") for k, v in optional_fields_.items()},
                T,
                args,
            )
        else:
            raise ConfifyTypeError(f"Unsupported type `{T}` at `{prefix}`")

    @classmethod
    def from_typeform(cls, T: _TypeFormT) -> "Schema":
        return cls._from_typeform(T, "<root>")

    @abstractmethod
    def _repr(self, indent: int = 0) -> str: ...

    def __repr__(self) -> str:
        return self._repr()

    @abstractmethod
    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult: ...

    @abstractmethod
    def equals(self, other: "Schema") -> bool: ...

    def parse(self, d: Any, options: Optional[ConfifyOptions] = None) -> Any:
        options = ConfifyOptions.get_default() if options is None else options
        res = self._parse(d, "<root>", options)
        if len(res.warnings) > 0:
            warns: list[str] = []
            for w in res.warnings:
                warns.append(f"-> {w.message}")
            _warning("Some warnings were encountered during parsing:\n" + "\n".join(warns))
        return res.value

    def raise_parse_error(self, d: Any, prefix: str, message: str = "") -> Never:
        raise ConfifyParseError(
            f"Invalid data for type `{self.annotation}` at `{prefix}`. Got `{repr(d)}`.\n-> {message}"
        )

    @abstractmethod
    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]: ...


class StrSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Str"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, UnresolvedString):
            d = d.value
            if (d.startswith('"') and d.endswith('"')) or (d.startswith("'") and d.endswith("'")):
                return _ParseResult(ast.literal_eval(d))
            return _ParseResult(d)
        elif isinstance(d, str):
            return _ParseResult(d)
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, StrSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, str), [self])


class IntSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Int"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, int) and not isinstance(d, bool):
            return _ParseResult(d)
        elif isinstance(d, UnresolvedString):
            try:
                return _ParseResult(int(d.value))
            except ValueError:
                return self.raise_parse_error(d, prefix, f"Invalid integer: {d.value}")
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, IntSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, int), [self])


class FloatSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Float"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, float):
            return _ParseResult(d)
        elif isinstance(d, int):
            return _ParseResult(float(d))
        elif isinstance(d, UnresolvedString):
            try:
                return _ParseResult(float(d.value))
            except ValueError:
                return self.raise_parse_error(d, prefix, f"Invalid float: {d.value}")
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, FloatSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, (float, int)), [self])


class BoolSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Bool"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, bool):
            return _ParseResult(d)
        elif isinstance(d, UnresolvedString):
            d = d.value.strip().lower()
            if d in ["true", "on", "yes"]:
                return _ParseResult(True)
            elif d in ["false", "off", "no"]:
                return _ParseResult(False)
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, BoolSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, bool), [self])


class NoneSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "None"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if d is None:
            return _ParseResult(None)
        elif isinstance(d, UnresolvedString):
            d = d.value.strip().lower()
            if d in ["null", "~", "none"]:
                return _ParseResult(None)
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, NoneSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (T is type(None) or T is None, [self])


# Value schemas
_none_schema = NoneSchema(None)
_bool_schema = BoolSchema(bool)
_int_schema = IntSchema(int)
_float_schema = FloatSchema(float)


class PathSchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Path"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, Path):
            return _ParseResult(d)
        elif isinstance(d, str):
            return _ParseResult(Path(d))
        elif isinstance(d, UnresolvedString):
            return _ParseResult(Path(d.value))
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, PathSchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, (Path, str)), [self])


class AnySchema(Schema):
    def _repr(self, indent: int = 0) -> str:
        return "Any"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        return _ParseResult(UnresolvedString.sanitize(d))

    def equals(self, other: Schema) -> bool:
        return isinstance(other, AnySchema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (True, [self])


class EnumSchema(Schema):
    def __init__(self, annotation: _TypeFormT, EnumType: Type[Enum]):
        super().__init__(annotation)
        self.EnumType = EnumType

    def _repr(self, indent: int = 0) -> str:
        return f"Enum[{classname_of_cls(self.EnumType)}]"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, str):
            # string loaded from YAML
            d = d.strip()
            try:
                return _ParseResult(self.EnumType[d])
            except KeyError:
                self.raise_parse_error(d, prefix, f"Invalid enum value: {d}")
        elif isinstance(d, UnresolvedString):
            d = d.value.strip()
            try:
                return _ParseResult(self.EnumType[d])
            except KeyError:
                self.raise_parse_error(d, prefix, f"Invalid enum value: {d}")
        elif isinstance(d, self.EnumType):
            return _ParseResult(d)
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, EnumSchema) and self.EnumType == other.EnumType

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, self.EnumType), [self])


_TYPE_RANK = {
    None: 1,
    type(None): 2,
    bool: 3,
    int: 4,
    float: 5,
    str: 1000,
}


class LiteralSchema(Schema):
    def __init__(self, annotation: _TypeFormT, values: tuple[Any, ...]):
        super().__init__(annotation)
        self.values = tuple(values)

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        str_ = f"Literal\n"
        for v in self.values:
            str_ += f"{indent_str}- {v}\n"
        return str_[:-1]

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        candidates: list[_ParseResult] = []

        for i, arg in enumerate(self.values):
            try:
                val = Schema._from_typeform(type(arg), f"{prefix}.{i}")._parse(d, f"{prefix}.{i}", options)
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
                    type=self.annotation,
                    value=d,
                    message=f"Ambiguous input for type `{self.annotation}` at `{prefix}`. Using `{repr(candidates[0].value)}: {type(candidates[0].value).__qualname__}`. Options are `{options_}`.",
                ),
            )
        if len(candidates) >= 1:
            return candidates[0]
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return isinstance(other, LiteralSchema) and self.values == other.values

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (False, [self])


class ListSchema(Schema):
    def __init__(self, annotation: _TypeFormT, val_schema: Schema):
        super().__init__(annotation)
        self.val_schema = val_schema

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"List\n{indent_str}- $items: {self.val_schema._repr(indent + 2)}"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, UnresolvedString):
            try:
                elems = d.resolve_as_sequence()
            except ValueError as e:
                self.raise_parse_error(d, prefix, str(e))
        elif isinstance(d, (list, tuple)):
            elems = d
        else:
            self.raise_parse_error(d, prefix)

        results = [self.val_schema._parse(e, f"{prefix}[{i}]", options) for i, e in enumerate(elems)]
        return _ParseResult([r.value for r in results], warnings=sum([r.warnings for r in results], []))

    def equals(self, other: Schema) -> bool:
        return isinstance(other, ListSchema) and self.val_schema.equals(other.val_schema)

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, list), [self])


class TupleSchema(Schema):
    def __init__(self, annotation: _TypeFormT, val_schemas: Union[Schema, list[Schema]]):
        super().__init__(annotation)
        if isinstance(val_schemas, list):
            self.val_schemas = tuple(val_schemas)
        else:
            self.val_schemas = val_schemas

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        str_ = f"Tuple\n"
        if isinstance(self.val_schemas, tuple):
            for i, v in enumerate(self.val_schemas):
                str_ += f"{indent_str}- {i}: {v._repr(indent + 2)}\n"
        else:
            str_ += f"{indent_str}- $items: {self.val_schemas._repr(indent + 2)}\n"
        return str_[:-1]

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, UnresolvedString):
            try:
                elems = d.resolve_as_sequence()
            except ValueError as e:
                self.raise_parse_error(d, prefix, str(e))
        elif isinstance(d, (list, tuple)):
            elems = d
        else:
            self.raise_parse_error(d, prefix)

        if isinstance(self.val_schemas, tuple):
            if len(elems) != len(self.val_schemas):
                self.raise_parse_error(
                    d, prefix, f"Invalid tuple length. Expected {len(self.val_schemas)}, got {len(elems)}"
                )
            results = [self.val_schemas[i]._parse(e, f"{prefix}[{i}]", options) for i, e in enumerate(elems)]
        else:
            results = [self.val_schemas._parse(e, f"{prefix}[{i}]", options) for i, e in enumerate(elems)]
        return _ParseResult(tuple(r.value for r in results), warnings=sum([r.warnings for r in results], []))

    def equals(self, other: Schema) -> bool:
        if not isinstance(other, TupleSchema):
            return False
        # Both must be the same type (either both Schema or both tuple of Schemas)
        if isinstance(self.val_schemas, tuple) != isinstance(other.val_schemas, tuple):
            return False
        if isinstance(self.val_schemas, tuple) and isinstance(other.val_schemas, tuple):
            # Fixed-length tuple case
            if len(self.val_schemas) != len(other.val_schemas):
                return False
            return all(s.equals(o) for s, o in zip(self.val_schemas, other.val_schemas))
        else:
            # Variable-length tuple case (tuple[int, ...])
            return self.val_schemas.equals(other.val_schemas)  # type: ignore

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, tuple), [self])


class MappingSchema(Schema):
    def __init__(self, annotation: _TypeFormT, key_schema: Schema, val_schema: Schema):
        super().__init__(annotation)
        self.key_schema = key_schema
        self.val_schema = val_schema

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        return f"Mapping\n{indent_str}- $key: {self.key_schema._repr(indent + 2)}\n{indent_str}- $value: {self.val_schema._repr(indent + 2)}"

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, dict):
            entries = [
                (
                    self.key_schema._parse(k, f"{prefix}({k})", options),
                    self.val_schema._parse(v, f"{prefix}[{v}]", options),
                )
                for k, v in d.items()
            ]
            return _ParseResult(
                {k.value: v.value for k, v in entries},
                warnings=sum([r.warnings for kv in entries for r in kv], []),
            )
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        return (
            isinstance(other, MappingSchema)
            and self.key_schema.equals(other.key_schema)
            and self.val_schema.equals(other.val_schema)
        )

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, dict), [self])


class DictSchema(Schema):
    def __init__(
        self,
        annotation: _TypeFormT,
        required_fields: dict[str, Schema],
        optional_fields: dict[str, Schema],
        BaseClass: Type,
        type_args: tuple[_TypeFormT, ...],
    ):
        super().__init__(annotation)
        self.required_fields = required_fields
        self.optional_fields = optional_fields
        self.BaseClass = BaseClass
        params = getattr(BaseClass, "__parameters__", ())
        if len(type_args) == 0:
            type_args = (Any,) * len(params)
        self.type_args = type_args

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        str_ = f"TypedDict[{repr_of_typeform(self.annotation)}]\n"
        for k, v in self.required_fields.items():
            str_ += f"{indent_str}- {k}: {v._repr(indent + 2)}\n"
        for k, v in self.optional_fields.items():
            str_ += f"{indent_str}- {k}: {v._repr(indent + 2)}\n"
        return str_[:-1]

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        if isinstance(d, dict):
            warns: list[_ParseWarningEntry] = []
            if options.type_key in d:
                new_cls = import_string(str(d[options.type_key]))
                if new_cls != self.BaseClass:
                    if not issubclass(new_cls, self.BaseClass):
                        if options.strict_subclass_check:
                            self.raise_parse_error(
                                d,
                                prefix,
                                f"Type `{classname_of_cls(new_cls)}` is not a subtype of `{classname_of_cls(self.BaseClass)}`",
                            )
                        else:
                            warns.append(
                                _ParseWarningEntry(
                                    loc=f"{prefix}",
                                    type=self.annotation,
                                    value=d,
                                    message=f"Type `{classname_of_cls(new_cls)}` is not a subtype of `{classname_of_cls(self.BaseClass)}`. Using `{classname_of_cls(new_cls)}`.",
                                )
                            )
                    else:
                        return Schema._from_typeform(new_cls, prefix)._parse(d, prefix, options)
                d = dict(d)
                del d[options.type_key]

            args = {}
            missing_fields: list[str] = []
            extra_fields: list[str] = []
            for f, schema in self.required_fields.items():
                if f not in d:
                    missing_fields.append(f)
                else:
                    results = schema._parse(d[f], f"{prefix}.{f}", options)
                    warns.extend(results.warnings)
                    args[f] = results.value
            for f, schema in self.optional_fields.items():
                if f in d:
                    results = schema._parse(d[f], f"{prefix}.{f}", options)
                    warns.extend(results.warnings)
                    args[f] = results.value
            for k in d.keys():
                if k not in self.required_fields and k not in self.optional_fields:
                    extra_fields.append(k)

            if len(missing_fields) > 0:
                msg = f"Missing fields: {', '.join(missing_fields)}"
                if len(extra_fields) > 0:
                    msg += f"\nExtra fields: {', '.join(extra_fields)}"
                self.raise_parse_error(d, prefix, msg)
            if len(extra_fields) > 0:
                if options.ignore_extra_fields:
                    warns.append(
                        _ParseWarningEntry(
                            loc=f"{prefix}",
                            type=classname_of_cls(self.BaseClass),
                            value=d,
                            message=f"Ignoring extra fields: {', '.join(extra_fields)}",
                        )
                    )
                else:
                    self.raise_parse_error(d, prefix, f"Got extra fields: {', '.join(extra_fields)}")
            return _ParseResult(self.BaseClass(**args), warnings=warns)
        elif isinstance(d, self.BaseClass):
            return _ParseResult(d)
        self.raise_parse_error(d, prefix)

    def equals(self, other: Schema) -> bool:
        if not isinstance(other, DictSchema):
            return False
        # Compare BaseClass and type_args
        if self.BaseClass != other.BaseClass or self.type_args != other.type_args:
            return False
        # Compare required fields
        if set(self.required_fields.keys()) != set(other.required_fields.keys()):
            return False
        for key in self.required_fields:
            if not self.required_fields[key].equals(other.required_fields[key]):
                return False
        # Compare optional fields
        if set(self.optional_fields.keys()) != set(other.optional_fields.keys()):
            return False
        for key in self.optional_fields:
            if not self.optional_fields[key].equals(other.optional_fields[key]):
                return False
        return True

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        return (issubclass(T, self.BaseClass), [self])


class UnionSchema(Schema):
    def __init__(self, annotation: _TypeFormT, schemas: list[Schema]):
        super().__init__(annotation)
        self.schemas = tuple(schemas)

    def _repr(self, indent: int = 0) -> str:
        indent_str = " " * indent
        str_ = f"Union\n"
        for i, s in enumerate(self.schemas):
            str_ += f"{indent_str}- {i}: {s._repr(indent + 2)}\n"
        return str_[:-1]

    def _parse(self, d: Any, prefix: str, options: ConfifyOptions) -> _ParseResult:
        candidates: list[_ParseResult] = []
        errors = ""
        for i, s in enumerate(self.schemas):
            try:
                val = s._parse(d, f"{prefix}.{i}", options)
                candidates.append(val)
            except ConfifyParseError as e:
                errors += f"\n{e}"

        candidates.sort(key=lambda x: _TYPE_RANK.get(type(x.value), 0))
        non_str_candidates = [c for c in candidates if not isinstance(c.value, str)]
        if len(non_str_candidates) > 1:
            options_ = ", ".join([f"{repr(c.value)}: {type(c.value).__qualname__}" for c in candidates])
            candidates[0].warnings.insert(
                0,
                _ParseWarningEntry(
                    loc=prefix,
                    type=f"{self.annotation}",
                    value=d,
                    message=f"Ambiguous input for type `{self.annotation}` at `{prefix}`. Using `{repr(candidates[0].value)}: {type(candidates[0].value).__qualname__}`. Options are `{options_}`.",
                ),
            )
        if len(candidates) >= 1:
            return candidates[0]
        self.raise_parse_error(d, prefix, "Cannot parse to any of the union types (see below for details):\n" + errors)

    def equals(self, other: Schema) -> bool:
        if not isinstance(other, UnionSchema):
            return False
        if len(self.schemas) != len(other.schemas):
            return False
        return all(s.equals(o) for s, o in zip(self.schemas, other.schemas))

    def assignable_from(self, T: Type) -> tuple[bool, list["Schema"]]:
        work = False
        considered: list[Schema] = []
        for s in self.schemas:
            assignable, candidates = s.assignable_from(T)
            if assignable:
                work = True
                break
            considered.extend(candidates)
        return (work, considered)
