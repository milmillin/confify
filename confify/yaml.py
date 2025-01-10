from typing import Any, Union, Type, Optional, List, Tuple, Dict, get_origin, Literal, get_args
from enum import Enum
import yaml
from pathlib import Path


class ConfifyDumper(yaml.Dumper):
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


class ConfifyLoader(yaml.Loader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def construct_enum(self, node):
        return self.construct_scalar(node)

    def construct_path(self, node):
        return Path(self.construct_scalar(node))
