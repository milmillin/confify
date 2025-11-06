from dataclasses import dataclass
from typing import (
    TypeVar,
    Type,
    Generic,
    Callable,
    Self,
    Union,
    Any,
    Optional,
    Sequence,
    Unpack,
    cast,
    NamedTuple,
    Iterable,
    ClassVar,
)
from pathlib import Path
import itertools
import sys

from .base import ConfifyOptions, ConfifyBuilderError, ConfifyError, _warning, ConfifyParseError
from .schema import Schema, DictSchema, MappingSchema
from .parser import read_yaml, UnresolvedString, parse

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T1 = TypeVar("T1")

# region ConfigDuckTyped


class ConfigDuckTyped:
    def __init__(self, schema: Schema, prefixes: list[str]):
        self.prefixes = list(prefixes)
        self.schema = schema

    def get_dotnotation(self) -> str:
        return ".".join(self.prefixes)

    def __getattr__(self, name: str) -> "ConfigDuckTyped":
        if isinstance(self.schema, DictSchema):
            if name in self.schema.required_fields:
                return ConfigDuckTyped(self.schema.required_fields[name], self.prefixes + [name])
            elif name in self.schema.optional_fields:
                return ConfigDuckTyped(self.schema.optional_fields[name], self.prefixes + [name])
        raise ConfifyBuilderError(f"Cannot access field `{name}` in `{self.schema.annotation}`")


# endregion

# region DSL


class SetRecord(Generic[T]):
    def __init__(self, duck_typed: ConfigDuckTyped, value: Any, *, from_yaml: bool = False):
        self.duck_typed = duck_typed
        self.value = value
        self.from_yaml = from_yaml

    def __repr__(self):
        return f"{self.duck_typed.get_dotnotation()} = {self.value}"


class Set(Generic[T]):
    def __init__(self, field: T):
        if not isinstance(field, ConfigDuckTyped):
            raise ConfifyBuilderError(f"Invalid syntax. Expected `ConfigDuckTyped`, got `{type(field).__qualname__}`")
        self.duck_typed = field

    def to(self, value: T) -> SetRecord[T]:
        return SetRecord(self.duck_typed, value, from_yaml=False)

    def from_yaml(self, path: Path) -> SetRecord[T]:
        return SetRecord(self.duck_typed, path, from_yaml=True)


class SetTypeRecordWithStatements(Generic[T]):
    def __init__(self, duck_typed: ConfigDuckTyped, to_type: Type[T], stmts: "ConfigStatements"):
        self.duck_typed = duck_typed
        self.to_type = to_type
        self.stmts = stmts


class SetTypeRecord(Generic[T]):
    def __init__(self, duck_typed: ConfigDuckTyped, to_type: Type[T]):
        self.duck_typed = duck_typed
        self.to_type = to_type

    def __repr__(self):
        return f"{self.duck_typed.get_dotnotation()} = {self.to_type.__name__}"


class AsWithStatements(Generic[T_co]):
    def __init__(self, to_type: Type[T_co], stmts_fn: Callable[[T_co], "ConfigStatements"]):
        self.to_type = to_type
        self.stmts_fn = stmts_fn


class As(Generic[T_co]):
    def __init__(self, to_type: Type[T_co]):
        self.to_type = to_type

    def then(self, stmts_fn: Callable[[T_co], "ConfigStatements"]) -> AsWithStatements[T_co]:
        return AsWithStatements(self.to_type, stmts_fn)


class SetType(Generic[T]):
    def __init__(self, field: T):
        if not isinstance(field, ConfigDuckTyped):
            raise ConfifyBuilderError(f"Invalid syntax. Expected `ConfigDuckTyped`, got `{type(field).__qualname__}`")
        self.duck_typed = field

    def __call__(
        self, as_: Union[As[T], AsWithStatements[T]]
    ) -> Union[SetTypeRecord[T], SetTypeRecordWithStatements[T]]:
        if not isinstance(self.duck_typed.schema, (DictSchema)):
            raise ConfifyBuilderError(f"Only dict type can be set to a type")
        to_type = as_.to_type
        if not issubclass(to_type, self.duck_typed.schema.BaseClass):
            raise ConfifyBuilderError(f"`{to_type}` is not a subtype of `{self.duck_typed.schema.BaseClass}`")
        if isinstance(as_, As):
            return SetTypeRecord(self.duck_typed, to_type)
        elif isinstance(as_, AsWithStatements):
            to_type_duck_typed = ConfigDuckTyped(Schema.from_typeform(to_type), self.duck_typed.prefixes)
            to_type_duck_typed = cast(T, to_type_duck_typed)
            stmts = as_.stmts_fn(to_type_duck_typed)
            return SetTypeRecordWithStatements(self.duck_typed, to_type, stmts)
        else:
            raise ConfifyBuilderError(
                f"Invalid syntax. Expected `As` or `AsWithStatements`, got `{type(as_).__qualname__}`"
            )


class Sweep:
    def __init__(self, _: Optional["ConfigStatements"] = None, /, **sweeps: "ConfigStatements"):
        if _ is not None:
            self.sweeps = {"": _, **sweeps}
        else:
            self.sweeps = dict(sweeps)


PureConfigStatements = Sequence[Union[SetRecord[Any], SetTypeRecord[Any]]]
ConfigStatements = Sequence[Union[SetRecord[Any], Sweep, SetTypeRecord[Any], SetTypeRecordWithStatements[Any]]]


@dataclass
class NamedPureConfigStatements:
    name: str
    stmts: PureConfigStatements
    empty: ClassVar["NamedPureConfigStatements"]

    def __add__(self, other: "NamedPureConfigStatements") -> "NamedPureConfigStatements":
        return NamedPureConfigStatements(self.name + other.name, [*self.stmts, *other.stmts])


NamedPureConfigStatements.empty = NamedPureConfigStatements("", [])


def execute_sweep(sweep: Sweep) -> list[NamedPureConfigStatements]:
    res: list[NamedPureConfigStatements] = []
    for key, stmts in sweep.sweeps.items():
        res.extend(execute(stmts, base_name=key))
    return res


def execute(stmts: ConfigStatements, base_name: str) -> Iterable[NamedPureConfigStatements]:
    operands: list[list[NamedPureConfigStatements]] = [[NamedPureConfigStatements(base_name, [])]]
    for stmt in stmts:
        if isinstance(stmt, Sweep):
            operands.append(execute_sweep(stmt))
        elif isinstance(stmt, SetTypeRecordWithStatements):
            operands.append([NamedPureConfigStatements("", [SetTypeRecord(stmt.duck_typed, stmt.to_type)])])
            operands.append(list(execute(stmt.stmts, "")))
        else:
            operands.append([NamedPureConfigStatements("", [stmt])])
    for x in itertools.product(*operands):
        yield sum(x, NamedPureConfigStatements.empty)


# endregion

# region CLI


class ConfifyCLIError(ConfifyError):
    pass


def _insert_dict(d: dict[str, Any], keys: list[str], value: Any, prefix: str = "") -> None:
    if len(keys) == 0:
        if not isinstance(value, dict):
            raise ConfifyCLIError(f"`{prefix}` Value must be a dict. Got `{value}`")
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
            _warning(f"`{prefix}` Overriding non-dict value `{dd}` with `{value}`")
            d[keys[0]] = {}
        _insert_dict(dd, keys[1:], value)


def read_config_from_argv(
    Config: Type[T], argv: list[str], options: Optional[ConfifyOptions] = None, schema: Optional[Schema] = None
) -> T:
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
            value = UnresolvedString(value)
        else:
            raise ValueError(f"Invalid argument: {key}. Must start with {options.prefix} or {options.yaml_prefix}")
        _insert_dict(args, key.split(".") if key else [], value)
        i += 2
    return parse(args, Config, schema=schema)


def read_config_from_cli(Config: Type[T], options: Optional[ConfifyOptions] = None) -> T:
    return read_config_from_argv(Config, sys.argv[1:], options=options)


class Confify(Generic[T]):
    main_fn: Optional[Callable[[T], Any]] = None
    gen_fns: dict[str, Callable[[T], ConfigStatements]] = {}

    def __init__(self, Config: Type[T], options: Optional[ConfifyOptions] = None):
        self.Config = Config
        self.options = options or ConfifyOptions.get_default()
        self.schema = Schema.from_typeform(Config)

    def main(self):
        """
        Decorator to set the main function.
        """

        def decorator(f: Callable[[T], Any]) -> Callable[[], Any]:
            self.main_fn = f
            return self.main_wrapper

        return decorator

    def main_wrapper(self):
        if self.main_fn is None:
            raise ConfifyCLIError("Main function is not set.")
        argv = sys.argv[1:]
        if len(argv) == 0 or argv[0].startswith(self.options.prefix) or argv[0].startswith(self.options.yaml_prefix):
            # Run main function
            try:
                config = read_config_from_argv(self.Config, argv, self.options, schema=self.schema)
                return self.main_fn(config)
            except ConfifyParseError as e:
                print(e)
        elif argv[0] in ["g", "gen", "generate"]:
            # Run generator
            not_found: list[str] = []
            for name in argv[1:]:
                if name not in self.gen_fns:
                    not_found.append(name)
            if len(not_found) > 0:
                raise ConfifyCLIError(f"Generator not found: {', '.join(not_found)}")
            duck_typed = cast(T, ConfigDuckTyped(self.schema, []))
            for name in argv[1:]:
                prog = self.gen_fns[name](duck_typed)
                configs = execute(prog, base_name=name)
                for config in configs:
                    print(config.name)
                    print(config.stmts)
                    print("--------")

        else:
            # TODO: print help
            print("Help")
            pass

    def gen(self, name: Optional[str] = None):
        """
        Decorator to set a generator function.
        """

        def decorator(f: Callable[[T], ConfigStatements]) -> Callable[[T], ConfigStatements]:
            nonlocal name
            if name is None:
                name = f.__name__
            self.gen_fns[name] = f
            return f

        return decorator


# endregion
