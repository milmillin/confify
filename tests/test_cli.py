import pytest
from dataclasses import dataclass
from typing import Type, TypeVar, Union, Optional

from confify.cli import (
    SetType,
    As,
    ConfigDuckTyped,
    SetTypeRecord,
    Variable,
    SetVariable,
    Set,
    SetRecord,
    execute,
    flatten,
    Sweep,
    ConfifyCLIError,
)
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


# Variable Tests
def test_variable_creation():
    """Variable stores type correctly"""
    v = Variable(int)
    assert v.T is int

    v2 = Variable(str)
    assert v2.T is str


def test_variable_repr():
    """Variable repr includes type name"""
    v = Variable(int)
    assert "int" in repr(v)

    v2 = Variable(str)
    assert "str" in repr(v2)


def test_set_variable_returns_setvariable():
    """Set(variable).to(value) returns SetVariable"""
    v = Variable(int)
    result = Set(v).to(42)

    assert isinstance(result, SetVariable)
    assert result.variable is v
    assert result.value == 42


def test_set_field_with_variable_value():
    """Set(field).to(variable) stores variable as value in SetRecord"""

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)
    v = Variable(int)

    result = Set(duck.value).to(v)

    assert isinstance(result, SetRecord)
    assert result.value is v


def test_execute_resolves_variable():
    """execute() resolves variables to their values"""

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)
    v = Variable(int)

    stmts = [
        SetVariable(v, 42),
        Set(duck.value).to(v),
    ]

    result = execute(stmts)

    assert len(result) == 1
    assert isinstance(result[0], SetRecord)
    assert result[0].value == 42


def test_execute_undefined_variable_error():
    """execute() raises error for undefined variable"""

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)
    v = Variable(int)

    stmts = [
        Set(duck.value).to(v),  # v is never defined
    ]

    with pytest.raises(ConfifyCLIError, match="not defined"):
        execute(stmts)


def test_execute_redefined_variable_error():
    """execute() raises error when variable is redefined without allow_override"""
    v = Variable(int)

    stmts = [
        SetVariable(v, 42),
        SetVariable(v, 100),  # redefining v
    ]

    with pytest.raises(ConfifyCLIError, match="already defined"):
        execute(stmts)


def test_variable_allow_override():
    """Variable with allow_override=True can be redefined"""
    v = Variable(int, allow_override=True)

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)

    stmts = [
        SetVariable(v, 42),
        SetVariable(v, 100),  # allowed because allow_override=True
        Set(duck.value).to(v),
    ]

    result = execute(stmts)

    assert len(result) == 1
    assert result[0].value == 100  # last assigned value  # type: ignore


def test_variable_allow_override_false_by_default():
    """Variable has allow_override=False by default"""
    v = Variable(int)
    assert v.allow_override is False


def test_variable_allow_override_with_sweep():
    """allow_override works correctly across sweep branches"""

    @dataclass
    class Config:
        x: int

    duck = create_duck_typed(Config)
    v = Variable(int, allow_override=True)

    stmts = [
        Set(v).to(0),  # initial value
        Sweep(
            a=[Set(v).to(10)],  # override in branch a
            b=[Set(v).to(20)],  # override in branch b
        ),
        Set(duck.x).to(v),
    ]

    flattened = list(flatten(stmts, "test"))
    assert len(flattened) == 2

    result_a = execute(flattened[0].stmts)
    assert result_a[0].value == 10  # type: ignore

    result_b = execute(flattened[1].stmts)
    assert result_b[0].value == 20  # type: ignore


def test_variable_with_sweep():
    """Variable works with Sweep - defined in one branch, used later"""

    @dataclass
    class Config:
        x: int
        y: int

    duck = create_duck_typed(Config)
    v = Variable(int)

    stmts = [
        Sweep(
            a=[Set(v).to(10)],
            b=[Set(v).to(20)],
        ),
        Set(duck.x).to(v),
        Set(duck.y).to(v),
    ]

    # Flatten expands the sweep
    flattened = list(flatten(stmts, "test"))

    assert len(flattened) == 2  # two branches: a and b

    # Execute each branch
    result_a = execute(flattened[0].stmts)
    assert len(result_a) == 2
    assert result_a[0].value == 10  # type: ignore
    assert result_a[1].value == 10  # type: ignore

    result_b = execute(flattened[1].stmts)
    assert len(result_b) == 2
    assert result_b[0].value == 20  # type: ignore
    assert result_b[1].value == 20  # type: ignore


# Variable-to-Variable Tests
def test_variable_to_variable_assignment():
    """Set(var1).to(var2) assigns var2's value to var1"""

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)
    a = Variable(int)
    b = Variable(int)

    stmts = [
        Set(a).to(10),
        Set(b).to(a),  # b gets value from a
        Set(duck.value).to(b),
    ]

    result = execute(stmts)

    assert len(result) == 1
    assert result[0].value == 10  # type: ignore


def test_variable_chain():
    """Chained variable assignments: a=10, b=a, c=b → c should be 10"""

    @dataclass
    class Simple:
        value: int

    duck = create_duck_typed(Simple)
    a = Variable(int)
    b = Variable(int)
    c = Variable(int)

    stmts = [
        Set(a).to(42),
        Set(b).to(a),
        Set(c).to(b),
        Set(duck.value).to(c),
    ]

    result = execute(stmts)

    assert len(result) == 1
    assert result[0].value == 42  # type: ignore


def test_variable_cycle_undefined_error():
    """Cycle of length 2: Set(a).to(b), Set(b).to(a) → first fails (b not defined)"""
    a = Variable(int)
    b = Variable(int)

    stmts = [
        Set(a).to(b),  # b is not yet defined
        Set(b).to(a),
    ]

    with pytest.raises(ConfifyCLIError, match="not defined"):
        execute(stmts)


def test_variable_self_reference_error():
    """Self-reference: Set(a).to(a) → error (a not defined)"""
    a = Variable(int)

    stmts = [
        Set(a).to(a),  # a references itself before being defined
    ]

    with pytest.raises(ConfifyCLIError, match="not defined"):
        execute(stmts)


def test_variable_larger_cycle_error():
    """Cycle of length 3: Set(a).to(b), Set(b).to(c), Set(c).to(a) → first fails"""
    a = Variable(int)
    b = Variable(int)
    c = Variable(int)

    stmts = [
        Set(a).to(b),  # b is not yet defined
        Set(b).to(c),
        Set(c).to(a),
    ]

    with pytest.raises(ConfifyCLIError, match="not defined"):
        execute(stmts)


def test_variable_override_from_variable():
    """With allow_override: a=10, b=a, a=b → a gets updated from b's value"""

    @dataclass
    class Simple:
        x: int
        y: int

    duck = create_duck_typed(Simple)
    a = Variable(int, allow_override=True)
    b = Variable(int)

    stmts = [
        Set(a).to(10),
        Set(b).to(a),  # b = 10
        Set(a).to(b),  # a = 10 (from b)
        Set(duck.x).to(a),
        Set(duck.y).to(b),
    ]

    result = execute(stmts)

    assert len(result) == 2
    assert result[0].value == 10  # type: ignore
    assert result[1].value == 10  # type: ignore
