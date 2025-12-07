import pytest
from dataclasses import dataclass
from typing import Type, TypeVar, Union, Optional

from confify.cli import SetType, As, ConfigDuckTyped, SetTypeRecord
from confify.schema import Schema
from confify.base import ConfifyBuilderError

T1 = TypeVar("T1")


def create_duck_typed(T: Type[T1]) -> T1:
    schema = Schema.from_typeform(T)
    duck_typed = ConfigDuckTyped(schema, [])
    return duck_typed  # type: ignore


# Test fixtures
@dataclass
class Animal:
    name: str


@dataclass
class Dog(Animal):
    breed: str


@dataclass
class Cat(Animal):
    indoor: bool


@dataclass
class Bird:
    wingspan: int


@dataclass
class Container:
    pet: Animal


@dataclass
class UnionContainer:
    pet: Union[Dog, Cat]
    optional_dog: Optional[Dog]
    optional_cat: Optional[Cat]


# Tests
def test_settype_with_dict_schema():
    """SetType works with a simple DictSchema (dataclass)"""
    duck = create_duck_typed(Container)

    record = SetType(duck.pet)(As(Dog))

    assert isinstance(record, SetTypeRecord)
    assert record.to_type is Dog


def test_settype_with_union_schema():
    """SetType works with UnionSchema"""
    duck = create_duck_typed(UnionContainer)

    # Dog is a valid member of Union[Dog, Cat]
    record = SetType(duck.pet)(As(Dog))
    assert isinstance(record, SetTypeRecord)
    assert record.to_type is Dog

    # Cat is also valid
    record = SetType(duck.pet)(As(Cat))
    assert record.to_type is Cat

    # Optional[Cat] is also valid
    record = SetType(duck.optional_cat)(As(Cat))
    assert record.to_type is Cat

    # Optional[Dog] is also valid
    record = SetType(duck.optional_dog)(As(Dog))
    assert record.to_type is Dog


def test_settype_on_primitive():
    """SetType works on primitive types"""

    @dataclass
    class HasInt:
        value: int

    duck = create_duck_typed(HasInt)
    record = SetType(duck.value)(As(int))

    assert record.to_type is int


def test_settype_error_incompatible_type():
    """SetType raises error when type is not compatible with any union member"""
    duck = create_duck_typed(UnionContainer)

    # Bird is not a subclass of Dog or Cat
    with pytest.raises(ConfifyBuilderError, match="is not assignable"):
        SetType(duck.pet)(As(Bird))  # type: ignore
