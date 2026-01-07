import pytest
from dataclasses import dataclass
from typing import Union

from confify.parser import _parse, ConfifyParseError


################################################################################
# Dataclasses with non-overlapping fields
################################################################################


@dataclass
class A:
    a_field: str


@dataclass
class B:
    b_field: int


@dataclass
class C:
    c_field: bool


def test_union_non_overlapping_parse_A():
    result = _parse(
        {"$type": "test_union_dataclass.A", "a_field": "hello"},
        Union[A, B, C],
    )
    assert result == A(a_field="hello")
    assert isinstance(result, A)


def test_union_non_overlapping_parse_B():
    result = _parse(
        {"$type": "test_union_dataclass.B", "b_field": 42},
        Union[A, B, C],
    )
    assert result == B(b_field=42)
    assert isinstance(result, B)


def test_union_non_overlapping_parse_C():
    result = _parse(
        {"$type": "test_union_dataclass.C", "c_field": True},
        Union[A, B, C],
    )
    assert result == C(c_field=True)
    assert isinstance(result, C)


################################################################################
# Dataclasses with overlapping fields
################################################################################


@dataclass
class Dog:
    name: str
    breed: str


@dataclass
class Cat:
    name: str
    color: str


@dataclass
class Bird:
    name: str
    can_fly: bool


def test_union_overlapping_parse_Dog():
    result = _parse(
        {"$type": "test_union_dataclass.Dog", "name": "Buddy", "breed": "Labrador"},
        Union[Dog, Cat, Bird],
    )
    assert result == Dog(name="Buddy", breed="Labrador")
    assert isinstance(result, Dog)


def test_union_overlapping_parse_Cat():
    result = _parse(
        {"$type": "test_union_dataclass.Cat", "name": "Whiskers", "color": "orange"},
        Union[Dog, Cat, Bird],
    )
    assert result == Cat(name="Whiskers", color="orange")
    assert isinstance(result, Cat)


def test_union_overlapping_parse_Bird():
    result = _parse(
        {"$type": "test_union_dataclass.Bird", "name": "Tweety", "can_fly": True},
        Union[Dog, Cat, Bird],
    )
    assert result == Bird(name="Tweety", can_fly=True)
    assert isinstance(result, Bird)


################################################################################
# Dataclasses with identical fields (requires $type for disambiguation)
################################################################################


@dataclass
class ConfigA:
    value: int
    label: str


@dataclass
class ConfigB:
    value: int
    label: str


def test_union_identical_fields_with_type_key_ConfigA():
    result = _parse(
        {"$type": "test_union_dataclass.ConfigA", "value": 100, "label": "alpha"},
        Union[ConfigA, ConfigB],
    )
    assert result == ConfigA(value=100, label="alpha")
    assert isinstance(result, ConfigA)


def test_union_identical_fields_with_type_key_ConfigB():
    result = _parse(
        {"$type": "test_union_dataclass.ConfigB", "value": 200, "label": "beta"},
        Union[ConfigA, ConfigB],
    )
    assert result == ConfigB(value=200, label="beta")
    assert isinstance(result, ConfigB)


def test_union_identical_fields_without_type_key():
    """Without $type, the first matching type in the union is returned (ambiguous)."""
    result = _parse(
        {"value": 100, "label": "test"},
        Union[ConfigA, ConfigB],
    )
    # Without $type, ConfigA is returned as it's first in the union
    assert isinstance(result, ConfigA)
    assert result.value == 100
    assert result.label == "test"


################################################################################
# Edge cases and error handling
################################################################################


@dataclass
class X:
    x_val: int


@dataclass
class Y:
    y_val: str


@dataclass
class Z:
    z_val: float


def test_union_type_not_in_union():
    """Test error when $type specifies a class not in the union."""
    with pytest.raises(ConfifyParseError):
        _parse(
            {"$type": "test_union_dataclass.Z", "z_val": 3.14},
            Union[X, Y],  # Z is not in this union
        )


def test_union_invalid_type_key():
    """Test error when $type specifies a non-existent class."""
    with pytest.raises(ImportError):
        _parse(
            {"$type": "test_union_dataclass.NonExistent", "field": "value"},
            Union[X, Y, Z],
        )


def test_union_missing_required_field():
    """Test error when required field is missing."""
    with pytest.raises(ConfifyParseError):
        _parse(
            {"$type": "test_union_dataclass.A"},  # missing a_field
            Union[A, B, C],
        )


def test_union_wrong_field_type():
    """Test error when field has wrong type."""
    with pytest.raises(ConfifyParseError):
        _parse(
            {"$type": "test_union_dataclass.B", "b_field": "not_an_int"},
            Union[A, B, C],
        )


################################################################################
# Multiple field dataclasses
################################################################################


@dataclass
class Person:
    name: str
    age: int
    active: bool


@dataclass
class Company:
    name: str
    employee_count: int
    public: bool


@dataclass
class Product:
    name: str
    price: float
    available: bool


def test_union_multi_field_Person():
    result = _parse(
        {"$type": "test_union_dataclass.Person", "name": "Alice", "age": 30, "active": True},
        Union[Person, Company, Product],
    )
    assert result == Person(name="Alice", age=30, active=True)
    assert isinstance(result, Person)


def test_union_multi_field_Company():
    result = _parse(
        {"$type": "test_union_dataclass.Company", "name": "Acme Inc", "employee_count": 500, "public": False},
        Union[Person, Company, Product],
    )
    assert result == Company(name="Acme Inc", employee_count=500, public=False)
    assert isinstance(result, Company)


def test_union_multi_field_Product():
    result = _parse(
        {"$type": "test_union_dataclass.Product", "name": "Widget", "price": 19.99, "available": True},
        Union[Person, Company, Product],
    )
    assert result == Product(name="Widget", price=19.99, available=True)
    assert isinstance(result, Product)


################################################################################
# Dataclasses with optional fields
################################################################################


@dataclass
class OptionalA:
    required_field: str
    optional_field: int = 0


@dataclass
class OptionalB:
    required_field: str
    optional_field: str = "default"


def test_union_optional_with_default():
    """Test parsing with optional fields using defaults."""
    result = _parse(
        {"$type": "test_union_dataclass.OptionalA", "required_field": "test"},
        Union[OptionalA, OptionalB],
    )
    assert result == OptionalA(required_field="test", optional_field=0)
    assert isinstance(result, OptionalA)


def test_union_optional_with_value():
    """Test parsing with optional fields providing value."""
    result = _parse(
        {"$type": "test_union_dataclass.OptionalB", "required_field": "test", "optional_field": "custom"},
        Union[OptionalA, OptionalB],
    )
    assert result == OptionalB(required_field="test", optional_field="custom")
    assert isinstance(result, OptionalB)
