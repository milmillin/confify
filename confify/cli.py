from dataclasses import dataclass, fields, is_dataclass
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
    overload,
    TypedDict,
)
from pathlib import Path
import itertools
import sys
from enum import Enum
import json
import shlex
from abc import abstractmethod, ABC
import shutil
from datetime import datetime
import os

from .base import ConfifyOptions, ConfifyBuilderError, ConfifyError, _warning, ConfifyParseError
from .schema import Schema, DictSchema, MappingSchema
from .parser import read_yaml, UnresolvedString, parse
from .utils import classname_of_cls, classname

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T1 = TypeVar("T1")

# region ConfigDuckTyped


class ConfigDuckTyped:
    def __init__(self, schema: Schema, prefixes: list[str]):
        self.prefixes = list(prefixes)
        self._schema_ = schema

    def get_dotnotation(self) -> str:
        return ".".join(self.prefixes)

    def __getattr__(self, name: str) -> "ConfigDuckTyped":
        if isinstance(self._schema_, DictSchema):
            if name in self._schema_.required_fields:
                return ConfigDuckTyped(self._schema_.required_fields[name], self.prefixes + [name])
            elif name in self._schema_.optional_fields:
                return ConfigDuckTyped(self._schema_.optional_fields[name], self.prefixes + [name])
        raise ConfifyBuilderError(f"Cannot access field `{name}` in `{self._schema_.annotation}`")


# endregion

# region DSL


# Template String
class L(str): ...


class SetRecord(Generic[T]):
    def __init__(self, duck_typed: ConfigDuckTyped, value: Any, *, from_yaml: bool = False):
        self.duck_typed = duck_typed
        self.value = value
        self.from_yaml = from_yaml
        if not isinstance(value, L):
            # TODO: pass options
            self.duck_typed._schema_.parse(value)

    def __repr__(self):
        return f"{self.duck_typed.get_dotnotation()} = {self.value}"


class Set(Generic[T]):
    def __init__(self, field: T):
        if not isinstance(field, ConfigDuckTyped):
            raise ConfifyBuilderError(f"Invalid syntax. Expected `ConfigDuckTyped`, got `{type(field).__qualname__}`")
        self.duck_typed = field

    @overload
    def to(self: "Set[Path]", value: Union[Path, L, str]) -> SetRecord[Path]: ...
    @overload
    def to(self: "Set[Optional[Path]]", value: Union[Path, L, str, None]) -> SetRecord[Optional[Path]]: ...
    @overload
    def to(self, value: T) -> SetRecord[T]: ...
    def to(self, value):
        return SetRecord(self.duck_typed, value, from_yaml=False)

    def from_yaml(self, path: Union[Path, L, str]) -> SetRecord[T]:
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
        if not isinstance(self.duck_typed._schema_, (DictSchema)):
            raise ConfifyBuilderError(f"Only dict type can be set to a type")
        to_type = as_.to_type
        if not issubclass(to_type, self.duck_typed._schema_.BaseClass):
            raise ConfifyBuilderError(f"`{to_type}` is not a subtype of `{self.duck_typed._schema_.BaseClass}`")
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


PureConfigStatements = list[Union[SetRecord[Any], SetTypeRecord[Any]]]
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


# region LStr Default


def create_lstr_kwargs(script_name: str = "", name: str = "", generator_name: str = "") -> dict[str, str]:
    """
    Create LStrKwargs with current datetime information prepopulated.
    """
    now = datetime.now()
    return dict(
        script_name=script_name,
        name=name,
        generator_name=generator_name,
        Y=now.strftime("%Y"),
        m=now.strftime("%m"),
        d=now.strftime("%d"),
        H=now.strftime("%H"),
        M=now.strftime("%M"),
        s=now.strftime("%S"),
    )


# endregion


# region Compilation


def compile_to_config(
    stmts: PureConfigStatements,
    schema: Schema,
    options: ConfifyOptions,
    lstr_kwargs: dict[str, str],
) -> Any:
    res: dict[str, Any] = {}
    for stmt in stmts:
        if isinstance(stmt, SetRecord):
            if stmt.from_yaml:
                value = read_yaml(stmt.value)
            else:
                if isinstance(stmt.value, L):
                    value = stmt.value.format(**lstr_kwargs)
                else:
                    value = stmt.value
            _insert_dict(res, stmt.duck_typed.prefixes, value)
        elif isinstance(stmt, SetTypeRecord):
            _insert_dict(res, stmt.duck_typed.prefixes + [options.type_key], classname_of_cls(stmt.to_type))
        else:
            raise ConfifyCLIError(f"Invalid statement: {stmt}")
    return schema.parse(res, options)


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
        return "[" + ", ".join([_stringify_impl(e) for e in v]) + "]"
    elif isinstance(v, tuple):
        return "(" + ", ".join([_stringify_impl(e) for e in v]) + ")"
    else:
        raise ConfifyCLIError(f"Unsupported type for stringification: {type(v)}")


def _any_to_args(v: Any, options: ConfifyOptions, prefix: str = "") -> list[str]:
    if isinstance(v, (str, Path, bool, int, float, Enum, list, tuple)):
        return [f"{options.prefix}{prefix}", _stringify_impl(v)]
    elif isinstance(v, dict):
        res: list[str] = []
        for k, v in v.items():
            res.extend(_any_to_args(v, options, f"{prefix}.{k}"))
        return res
    elif is_dataclass(v):
        res: list[str] = []
        res.extend([f"{options.prefix}{prefix}.{options.type_key}", classname(v)])
        for f in fields(v):
            res.extend(_any_to_args(getattr(v, f.name), options=options, prefix=f"{prefix}.{f.name}"))
        return res
    else:
        raise ConfifyCLIError(f"Invalid type: {type(v)}")


def compile_to_args(
    stmts: PureConfigStatements,
    options: ConfifyOptions,
    lstr_kwargs: dict[str, str],
    shell_escape: bool = True,
) -> list[str]:
    args: list[str] = []
    for stmt in stmts:
        if isinstance(stmt, SetRecord):
            dotnotation = stmt.duck_typed.get_dotnotation()
            value = stmt.value
            if isinstance(value, L):
                value = value.format(**lstr_kwargs)
            if stmt.from_yaml:
                args.append(f"{options.yaml_prefix}{dotnotation}")
                args.append(value)
            else:
                args.extend(_any_to_args(value, options=options, prefix=dotnotation))
        elif isinstance(stmt, SetTypeRecord):
            args.extend(
                [
                    f"{options.prefix}{stmt.duck_typed.get_dotnotation()}.{options.type_key}",
                    classname_of_cls(stmt.to_type),
                ]
            )
        else:
            raise ConfifyCLIError(f"Invalid statement: {stmt}")
    if shell_escape:
        args = [shlex.quote(arg) for arg in args]
    return args


# endregion

# region Exporter


@dataclass
class ConfifyExporterConfig:
    shell_escape: bool = True


class ConfifyExporter:
    config: ClassVar[ConfifyExporterConfig] = ConfifyExporterConfig()

    def pre_run(self, lstr_kwargs: dict[str, str]) -> dict[str, str]:
        return {}

    @abstractmethod
    def run(self, args: list[str], lstr_kwargs: dict[str, str]): ...

    def post_run(self, lstr_kwargs: dict[str, str]):
        pass


class ShellExporter(ConfifyExporter):
    def __init__(
        self, output_dir_fmt: str = "_generated/{script_name}_{generator_name}", output_fmt: str = "{name}.sh"
    ):
        self.output_dir_fmt = output_dir_fmt
        self.output_fmt = output_fmt

    def pre_run(self, lstr_kwargs: dict[str, str]) -> dict[str, str]:
        output_dir = Path(self.output_dir_fmt.format(**lstr_kwargs))

        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        return {}

    def run(self, args: list[str], lstr_kwargs: dict[str, str]):
        assert len(args) % 2 == 0
        args = [f"{k} {v}" for k, v in zip(args[::2], args[1::2])]
        args_str = " \\\n    ".join(args)

        sh = f"""#!/bin/bash

python {sys.argv[0]} \\
    {args_str}
"""

        output_dir = Path(self.output_dir_fmt.format(**lstr_kwargs))
        out_fn = output_dir / self.output_fmt.format(**lstr_kwargs)
        out_fn.write_text(sh)
        out_fn.chmod(0o755)

        print(out_fn)


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
            raise ConfifyCLIError(f"Invalid argument: {key}. Must start with {options.prefix} or {options.yaml_prefix}")
        _insert_dict(args, key.split(".") if key else [], value)
        i += 2
    return parse(args, Config, schema=schema)


def read_config_from_cli(Config: Type[T], options: Optional[ConfifyOptions] = None) -> T:
    return read_config_from_argv(Config, sys.argv[1:], options=options)


class Confify(Generic[T]):
    main_fn: Optional[Callable[[T], Any]] = None
    gen_fns: dict[str, Callable[[T], ConfigStatements]] = {}
    # kwargs, fn
    exporter_fns: dict[str, ConfifyExporter] = {}

    def __init__(self, Config: Type[T], options: Optional[ConfifyOptions] = None, cwd: Union[str, Path, None] = None):
        self.Config = Config
        self.options = options or ConfifyOptions.get_default()
        self.schema = Schema.from_typeform(Config)
        self.duck_typed = cast(T, ConfigDuckTyped(self.schema, []))

        # Add default exporter
        self.exporter_fns["shell"] = ShellExporter()

        if cwd is not None:
            os.chdir(cwd)

    def main(self):
        """
        Decorator to set the main function.
        """

        def decorator(f: Callable[[T], Any]) -> Callable[[], Any]:
            self.main_fn = f
            return self.main_wrapper

        return decorator

    def _handle_main_execution(self, argv: list[str]):
        """Handle main execution mode: parse config from CLI args and run main function."""
        if self.main_fn is None:
            raise ConfifyCLIError("Main function is not set.")
        config = read_config_from_argv(self.Config, argv, self.options, schema=self.schema)
        return self.main_fn(config)

    def _handle_list_command(self, argv: list[str]):
        """Handle list command: list configurations from specified generators."""
        not_found: list[str] = []
        for generator_name in argv[1:]:
            if generator_name not in self.gen_fns:
                not_found.append(generator_name)
        if len(not_found) > 0:
            raise ConfifyCLIError(f"Generator not found: {', '.join(not_found)}")
        for generator_name in argv[1:]:
            prog = self.gen_fns[generator_name](self.duck_typed)
            stmtss = execute(prog, base_name=generator_name)
            for stmts in stmtss:
                print(stmts.name)

    def _handle_generate_command(self, argv: list[str]):
        """Handle generate command: generate configuration files using an exporter."""
        if len(argv) != 3:
            raise ConfifyCLIError("Invalid arguments. Expected `gen <exporter-name> <generator-name>`")

        generator_name = argv[2]
        if generator_name not in self.gen_fns:
            raise ConfifyCLIError(f"Generator not found: {generator_name}")
        generator = self.gen_fns[generator_name]

        exporter_name = argv[1]
        if exporter_name not in self.exporter_fns:
            raise ConfifyCLIError(f"Exporter not found: {exporter_name}")
        exporter = self.exporter_fns[exporter_name]

        shell_escape = exporter.config.shell_escape

        lstr_kwargs = create_lstr_kwargs(
            script_name=Path(sys.argv[0]).stem,
            name="",
            generator_name=generator_name,
        )
        cnt = 0
        extra_kwargs = exporter.pre_run(lstr_kwargs)
        lstr_kwargs = {**lstr_kwargs, **extra_kwargs}
        prog = generator(self.duck_typed)
        stmtss = execute(prog, base_name=generator_name)
        for stmts in stmtss:
            lstr_kwargs["name"] = stmts.name
            args = compile_to_args(stmts.stmts, self.options, shell_escape=shell_escape, lstr_kwargs=lstr_kwargs)
            exporter.run(args, lstr_kwargs)
            cnt += 1
        exporter.post_run(lstr_kwargs)

        print(f"\nExported {cnt} configs.")

    def _handle_run_command(self, argv: list[str]):
        """Handle run command: run a named configuration directly."""
        if self.main_fn is None:
            raise ConfifyCLIError("Main function is not set.")
        if not len(argv) == 2:
            raise ConfifyCLIError("Invalid arguments. Expected `run <config-name>`")

        run_name = argv[1]
        config = None
        for gen_name, gen_fn in self.gen_fns.items():
            if run_name.startswith(gen_name):
                prog = gen_fn(self.duck_typed)
                stmtss = execute(prog, base_name=gen_name)
                for stmts in stmtss:
                    if stmts.name == run_name:
                        lstr_kwargs = create_lstr_kwargs(
                            script_name=Path(sys.argv[0]).stem,
                            name=run_name,
                            generator_name=gen_name,
                        )
                        config = compile_to_config(stmts.stmts, self.schema, self.options, lstr_kwargs)
                        break
        if config is None:
            raise ConfifyCLIError(f"Config not found: {run_name}")
        return self.main_fn(config)

    def _handle_help_command(self):
        """Handle help command: display usage information."""
        print("Usage:")
        print(f"  python {sys.argv[0]} [--<key> <value>]...           Run main with config from CLI args")
        print(f"  python {sys.argv[0]} list [<generator>]...         List configurations")
        print(f"  python {sys.argv[0]} gen <exporter> <generator>    Generate configuration files")
        print(f"  python {sys.argv[0]} run <config-name>             Run a named configuration")

    def main_wrapper(self):
        """Main entry point with centralized error handling and command dispatching."""
        try:
            if self.main_fn is None:
                raise ConfifyCLIError("Main function is not set.")

            argv = sys.argv[1:]

            # Dispatch to appropriate command handler
            if (
                len(argv) == 0
                or argv[0].startswith(self.options.prefix)
                or argv[0].startswith(self.options.yaml_prefix)
            ):
                return self._handle_main_execution(argv)
            elif argv[0] in ["l", "ls", "list"]:
                return self._handle_list_command(argv)
            elif argv[0] in ["g", "gen", "generate"]:
                return self._handle_generate_command(argv)
            elif argv[0] in ["r", "run"]:
                return self._handle_run_command(argv)
            else:
                return self._handle_help_command()
        except ConfifyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    def generator(self, name: Optional[str] = None):
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

    def register_exporter(self, name: str, exporter: ConfifyExporter):
        """
        Register an exporter function.
        """
        self.exporter_fns[name] = exporter


# endregion
