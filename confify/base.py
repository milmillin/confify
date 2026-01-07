from dataclasses import dataclass
import sys


class ConfifyError(Exception):
    pass


class ConfifyParseError(ConfifyError):
    pass


class ConfifyTypeError(ConfifyError):
    pass


class ConfifyBuilderError(ConfifyError):
    pass


@dataclass
class ConfifyOptions:
    prefix: str = "--"
    yaml_prefix: str = "---"
    type_key: str = "$type"
    ignore_extra_fields: bool = False

    @classmethod
    def get_default(cls) -> "ConfifyOptions":
        return _default_options

    @classmethod
    def set_default(cls, value: "ConfifyOptions"):
        global _default_options
        _default_options = value


_default_options = ConfifyOptions()


def _warning(msg: str) -> None:
    print(f"ConfifyWarning: {msg}", file=sys.stderr)
