from typing import Any, Union, Type, Optional, List, Tuple, Dict, get_origin, Literal, get_args
from enum import Enum
import yaml
from pathlib import Path
from dataclasses import fields, is_dataclass

from .base import ConfifyOptions


class ConfifyDumper(yaml.SafeDumper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def represent_list(self, data: list) -> Any:
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    def represent_tuple(self, data: tuple) -> Any:
        return self.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)

    def represent_enum(self, data: Enum) -> Any:
        return self.represent_scalar("tag:yaml.org,2002:str", data.name)

    def represent_str(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", data)

    def represent_path(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", str(data))


ConfifyDumper.add_multi_representer(Enum, ConfifyDumper.represent_enum)
ConfifyDumper.add_representer(tuple, ConfifyDumper.represent_tuple)
ConfifyDumper.add_representer(str, ConfifyDumper.represent_str)
ConfifyDumper.add_multi_representer(Path, ConfifyDumper.represent_path)
ConfifyDumper.add_representer(list, ConfifyDumper.represent_list)


class ConfifyLoader(yaml.FullLoader):
    def construct_path(self, tag_suffix, node):
        return Path(*map(str, self.construct_sequence(node)))


ConfifyLoader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:pathlib", ConfifyLoader.construct_path)
