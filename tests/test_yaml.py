import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union, Literal
from enum import Enum

from confify.parser import config_dump_yaml, parse_yaml, read_yaml


class Animal(Enum):
    ANT = 1
    BIRD = 2
    CAT = 3
    null = 4


@dataclass
class A:
    x1: int
    x2: float
    x3: bool
    x4: str
    x5: None
    x6: Path
    x7: list[int]
    x8: list[str]
    x9: tuple[int, str]
    x10: tuple[int, str, bool]
    x11: tuple[int, ...]
    x12: tuple[str, ...]
    x13: list[list[int]]
    x14: tuple[int, tuple[str, bool]]
    x15: dict[str, int]
    x16: dict[str, str]
    x17: dict[str, list[int]]
    x18: Literal["ant", "bird", "cat"]
    x19: Union[int, str, bool, None]
    x20: Animal
    x21: Union[int, str, bool, None, Animal]
    x22: Union[int, str, bool, None, Animal]
    x23: Union[int, str, bool, None, Animal]
    x24: Union[int, str, bool, None, Animal]
    x25: str


@dataclass
class B:
    x1: int
    x2: str
    x3: A


def assert_identity(data: Any):
    tmp = Path("tmp.yaml")
    config_dump_yaml(data, tmp)
    recon = parse_yaml(tmp, type(data))
    if tmp.exists():
        tmp.unlink()
    assert recon == data


def test_yaml():
    a = A(
        1,
        1.1,
        True,
        "a",
        None,
        Path("a"),
        [1, 2, 3],
        ["a", "b", "c"],
        (1, "null"),
        (-1, "a", False),
        (1, 2),
        ("a", "b"),
        [[1, 2, 3], [4, 5], [6]],
        (234, ("True", True)),
        {"a": 1, "b": -123},
        {"a": "a", "b": "BB"},
        {"a": [1, 2, 3], "b": [1, 5, 10, 3]},
        "ant",
        123,
        Animal.null,
        None,
        "False",
        "1",
        "~",
        "hello\nworld",
    )
    assert_identity(a)
    assert_identity(B(1, "a", a))


@dataclass
class Fail:
    x1: Union[Animal, str]
    x2: Union[Animal, str]


def test_known_issues():
    with pytest.raises(AssertionError):
        # Saving enum as string. No way to distinguish between Animal.ANT and "ANT"
        assert_identity(
            Fail(
                Animal.ANT,
                "ANT",
            )
        )


_PATH = Path(__file__).parent


def test_read_path():
    x = read_yaml(_PATH / "yaml_files" / "path.yaml")
    assert x["path"] == Path("asdf", "bbbb", "5123")
