from typing import Any, Type, TypeVar, get_args, get_origin, Annotated, Union, Literal
import sys
from importlib import import_module
from inspect import isclass


def repr_of_typeform(cls: Union[type, None]) -> str:
    if cls is None:
        # Python is cursed
        return "NoneType"
    Origin = get_origin(cls)
    args = get_args(cls)
    if Origin is Annotated:
        x = args[0]
        Origin = get_origin(x)
        args = get_args(x)

    if Origin is Literal:
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    elif Origin is not None:
        if len(args) == 0:
            return classname_of_cls(Origin)
        args_str = ", ".join("..." if a is Ellipsis else repr_of_typeform(a) for a in args)
        return f"{classname_of_cls(Origin)}[{args_str}]"
    return classname_of_cls(cls)


# alias to format typeforms
FT = repr_of_typeform


def classname_of_cls(cls: type) -> str:
    """
    Get the fully qualified name of a class.
    """
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__" and module != "builtins":
        name = module + "." + name
    return name


def classname(obj: Any) -> str:
    """
    Get the fully qualified name of a class of an object.
    """
    return classname_of_cls(type(obj))


def _cached_import(module_path, class_name):
    # Check whether module is loaded and fully initialized.
    if not (
        (module := sys.modules.get(module_path))
        and (spec := getattr(module, "__spec__", None))
        and getattr(spec, "_initializing", False) is False
    ):
        module = import_module(module_path)
    return getattr(module, class_name)


def import_string(dotted_path: str):
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


_T = TypeVar("_T")


def get_subclasses(cls: Type[_T]) -> list[Type[_T]]:
    res: list[Type[_T]] = [cls]
    for subclass in cls.__subclasses__():
        res.extend(get_subclasses(subclass))
    return res
