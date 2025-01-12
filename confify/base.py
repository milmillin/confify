import warnings
from dataclasses import dataclass


class ConfifyError(Exception):
    pass


class ConfifyParseError(ConfifyError):
    pass


class ConfifyTypeError(ConfifyError):
    pass


class ConfifyWarning(UserWarning):
    pass


class ConfifyCLIConfig:
    prefix: str = "--"
    yaml_prefix: str = "---"


def _warning(msg: str) -> None:
    warnings.warn(msg, ConfifyWarning)
