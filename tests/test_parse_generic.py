from typing import Generic, TypeVar, Any
from dataclasses import dataclass, field
import pytest

from confify.parser import _parse, ConfifyParseError, UnresolvedString
from confify.schema import (
    Schema,
    DictSchema,
    IntSchema,
    StrSchema,
    AnySchema,
    ListSchema,
)


# Helper function from test_parse.py
def U(s: str):
    return UnresolvedString(s)


# Type variables
T = TypeVar("T")
U_TV = TypeVar("U_TV")  # Renamed to avoid conflict with U() helper
V = TypeVar("V")


# Test dataclasses from examples
@dataclass
class AA(Generic[T]):
    x1: T


@dataclass
class BB(AA[V], Generic[V]):
    x3: int


@dataclass
class A(Generic[T]):
    x: T


@dataclass
class B(A[U_TV], Generic[U_TV, T]):
    y: T


@dataclass
class C(A[U_TV], Generic[U_TV, T]):
    x: T  # type: ignore
    y: T


@dataclass
class D:
    a: int
    b: str


@dataclass
class E(A[int]):
    c: str


# Additional test dataclasses for optional fields
@dataclass
class OptionalFields(Generic[T]):
    required: T
    optional: int = 5


@dataclass
class MultiOptional(Generic[T, U_TV]):
    req1: T
    req2: U_TV
    opt1: str = "default"
    opt2: list[int] = field(default_factory=list)


# Additional test dataclasses for edge cases
@dataclass
class Container(Generic[T]):
    items: list[T]


@dataclass
class Mapping(Generic[T]):
    data: dict[str, T]


# ============================================================================
# Tests for Schema.from_typeform()
# ============================================================================


def test_schema_aa_parameterized():
    """Test AA[int] schema structure"""
    schema = Schema.from_typeform(AA[int])
    expected = DictSchema(
        annotation=AA[int],
        required_fields={"x1": IntSchema(int)},
        optional_fields={},
        BaseClass=AA[int],  # BaseClass is the parameterized type
        type_args=(int,),
    )
    assert schema.equals(expected)


def test_schema_aa_unparameterized():
    """Test AA without type parameters - should use Any"""
    schema = Schema.from_typeform(AA)
    # Should have x1: Any (type_args auto-filled with Any)
    expected = DictSchema(
        annotation=AA,
        required_fields={"x1": AnySchema(Any)},
        optional_fields={},
        BaseClass=AA,
        type_args=(Any,),
    )
    assert schema.equals(expected)


def test_schema_bb_parameterized():
    """Test BB[str] schema structure"""
    schema = Schema.from_typeform(BB[str])
    # BB[str] inherits from AA[V] where V=str, so x1: str
    # BB also has its own field x3: int
    expected = DictSchema(
        annotation=BB[str],
        required_fields={"x1": StrSchema(str), "x3": IntSchema(int)},
        optional_fields={},
        BaseClass=BB[str],  # BaseClass is the parameterized type
        type_args=(str,),
    )
    assert schema.equals(expected)


def test_schema_bb_unparameterized():
    """Test BB without type parameters - should use Any"""
    schema = Schema.from_typeform(BB)
    # BB inherits from AA[V] where V=Any, and has x3: int
    expected = DictSchema(
        annotation=BB,
        required_fields={"x1": AnySchema(Any), "x3": IntSchema(int)},
        optional_fields={},
        BaseClass=BB,
        type_args=(Any,),
    )
    assert schema.equals(expected)


def test_schema_a_parameterized():
    """Test A[int] schema structure"""
    schema = Schema.from_typeform(A[int])
    expected = DictSchema(
        annotation=A[int],
        required_fields={"x": IntSchema(int)},
        optional_fields={},
        BaseClass=A[int],  # BaseClass is the parameterized type
        type_args=(int,),
    )
    assert schema.equals(expected)


def test_schema_b_two_params():
    """Test B[int, str] schema structure"""
    schema = Schema.from_typeform(B[int, str])
    # B[int, str] inherits from A[U_TV] where U_TV=int, so x: int
    # B has its own field y: T where T=str
    expected = DictSchema(
        annotation=B[int, str],
        required_fields={"x": IntSchema(int), "y": StrSchema(str)},
        optional_fields={},
        BaseClass=B[int, str],  # BaseClass is the parameterized type
        type_args=(int, str),
    )
    assert schema.equals(expected)


def test_schema_b_nested_generic():
    """Test B[AA[int], str] with nested generic"""
    schema = Schema.from_typeform(B[AA[int], str])
    # B[AA[int], str] has x: AA[int] (nested DictSchema) and y: str
    expected = DictSchema(
        annotation=B[AA[int], str],
        required_fields={
            "x": DictSchema(
                annotation=AA[int],
                required_fields={"x1": IntSchema(int)},
                optional_fields={},
                BaseClass=AA[int],  # Nested BaseClass is also parameterized
                type_args=(int,),
            ),
            "y": StrSchema(str),
        },
        optional_fields={},
        BaseClass=B[AA[int], str],  # BaseClass is the parameterized type
        type_args=(AA[int], str),
    )
    assert schema.equals(expected)


def test_schema_c_field_override():
    """Test C[int, str] where child overrides parent field"""
    schema = Schema.from_typeform(C[int, str])
    # C[int, str] inherits from A[U_TV] where U_TV=int, but overrides x with x: T where T=str
    # So both x and y are str, not int
    expected = DictSchema(
        annotation=C[int, str],
        required_fields={"x": StrSchema(str), "y": StrSchema(str)},
        optional_fields={},
        BaseClass=C[int, str],  # BaseClass is the parameterized type
        type_args=(int, str),
    )
    assert schema.equals(expected)


def test_schema_d_non_generic():
    """Test non-generic dataclass D"""
    schema = Schema.from_typeform(D)
    # D is a simple non-generic dataclass with a: int and b: str
    expected = DictSchema(
        annotation=D,
        required_fields={"a": IntSchema(int), "b": StrSchema(str)},
        optional_fields={},
        BaseClass=D,
        type_args=(),
    )
    assert schema.equals(expected)


def test_schema_e_concrete_parent():
    """Test E inheriting from A[int]"""
    schema = Schema.from_typeform(E)
    # E inherits from A[int], so x: int, and adds its own field c: str
    expected = DictSchema(
        annotation=E,
        required_fields={"x": IntSchema(int), "c": StrSchema(str)},
        optional_fields={},
        BaseClass=E,
        type_args=(),
    )
    assert schema.equals(expected)


def test_schema_different_params_not_equal():
    """Test that AA[int] != AA[str]"""
    schema_int = Schema.from_typeform(AA[int])
    schema_str = Schema.from_typeform(AA[str])
    assert not schema_int.equals(schema_str)


def test_schema_different_nested_params_not_equal():
    """Test that B[int, str] != B[str, int]"""
    schema1 = Schema.from_typeform(B[int, str])
    schema2 = Schema.from_typeform(B[str, int])
    assert not schema1.equals(schema2)


# ============================================================================
# Tests for _parse() with valid inputs
# ============================================================================


def test_parse_aa_int():
    """Test parsing dict into AA[int]"""
    result = _parse({"x1": 42}, AA[int])
    assert result == AA(x1=42)
    assert isinstance(result.x1, int)


def test_parse_aa_str():
    """Test parsing dict into AA[str]"""
    result = _parse({"x1": "hello"}, AA[str])
    assert result == AA(x1="hello")
    assert isinstance(result.x1, str)


def test_parse_aa_unparameterized_accepts_any():
    """Test that unparameterized AA accepts any type"""
    result1 = _parse({"x1": 42}, AA)
    assert result1 == AA(x1=42)

    result2 = _parse({"x1": "hello"}, AA)
    assert result2 == AA(x1="hello")


def test_parse_bb_str():
    """Test parsing dict into BB[str]"""
    result = _parse({"x1": "test", "x3": 99}, BB[str])
    assert result == BB(x1="test", x3=99)
    assert isinstance(result.x1, str)
    assert isinstance(result.x3, int)


def test_parse_bb_unparameterized():
    """Test parsing BB without type parameters"""
    result = _parse({"x1": 123, "x3": 99}, BB)
    assert result == BB(x1=123, x3=99)


def test_parse_a_int():
    """Test parsing A[int]"""
    result = _parse({"x": 42}, A[int])
    assert result == A(x=42)


def test_parse_b_two_params():
    """Test parsing B[int, str]"""
    result = _parse({"x": 42, "y": "hello"}, B[int, str])
    assert result == B(x=42, y="hello")
    assert isinstance(result.x, int)
    assert isinstance(result.y, str)


def test_parse_b_nested_generic():
    """Test parsing B[AA[int], str] with nested generic"""
    result = _parse({"x": {"x1": 42}, "y": "hello"}, B[AA[int], str])
    assert result == B(x=AA(x1=42), y="hello")
    assert isinstance(result.x, AA)
    assert isinstance(result.x.x1, int)


def test_parse_c_field_override():
    """Test parsing C[int, str] where child overrides parent field"""
    result = _parse({"x": "override", "y": "value"}, C[int, str])
    assert result == C(x="override", y="value")
    assert isinstance(result.x, str)
    assert isinstance(result.y, str)


def test_parse_d_non_generic():
    """Test parsing non-generic dataclass D"""
    result = _parse({"a": 10, "b": "test"}, D)
    assert result == D(a=10, b="test")


def test_parse_e_concrete_parent():
    """Test parsing E with concrete parent A[int]"""
    result = _parse({"x": 42, "c": "test"}, E)
    assert result == E(x=42, c="test")
    assert isinstance(result.x, int)
    assert isinstance(result.c, str)


# ============================================================================
# Tests for _parse() with UnresolvedString
# ============================================================================


def test_parse_aa_with_unresolved_string():
    """Test parsing AA[int] with UnresolvedString"""
    result = _parse({"x1": U("42")}, AA[int])
    assert result == AA(x1=42)
    assert isinstance(result.x1, int)


def test_parse_bb_with_unresolved_string():
    """Test parsing BB[str] with mixed UnresolvedString"""
    result = _parse({"x1": U("hello"), "x3": U("99")}, BB[str])
    assert result == BB(x1="hello", x3=99)


def test_parse_b_with_unresolved_string():
    """Test parsing B[int, str] with UnresolvedString"""
    result = _parse({"x": U("42"), "y": U("hello")}, B[int, str])
    assert result == B(x=42, y="hello")


def test_parse_nested_with_unresolved_string():
    """Test parsing nested generic with UnresolvedString"""
    result = _parse({"x": {"x1": U("42")}, "y": U("test")}, B[AA[int], str])
    assert result == B(x=AA(x1=42), y="test")


# ============================================================================
# Tests for _parse() error cases
# ============================================================================


def test_parse_aa_wrong_type():
    """Test that AA[int] rejects string"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": "not_an_int"}, AA[int])


def test_parse_aa_wrong_type_unresolved():
    """Test that AA[int] rejects invalid UnresolvedString"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": U("not_an_int")}, AA[int])


def test_parse_bb_wrong_inherited_type():
    """Test that BB[int] rejects string for inherited field"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": "not_an_int", "x3": 99}, BB[int])


def test_parse_bb_wrong_own_type():
    """Test that BB[str] rejects wrong type for own field"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": "valid", "x3": "not_an_int"}, BB[str])


def test_parse_b_wrong_first_param():
    """Test that B[int, str] rejects wrong type for first param"""
    with pytest.raises(ConfifyParseError):
        _parse({"x": "not_an_int", "y": "hello"}, B[int, str])


def test_parse_b_wrong_second_param():
    """Test that B[int, str] rejects wrong type for second param"""
    with pytest.raises(ConfifyParseError):
        _parse({"x": 42, "y": 123}, B[int, str])


def test_parse_missing_required_field():
    """Test that missing required field raises error"""
    with pytest.raises(ConfifyParseError):
        _parse({}, AA[int])


def test_parse_bb_missing_inherited_field():
    """Test that missing inherited required field raises error"""
    with pytest.raises(ConfifyParseError):
        _parse({"x3": 99}, BB[str])


def test_parse_bb_missing_own_field():
    """Test that missing own required field raises error"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": "test"}, BB[str])


def test_parse_extra_field():
    """Test that extra fields raise error"""
    with pytest.raises(ConfifyParseError):
        _parse({"x1": 42, "extra": "field"}, AA[int])


def test_parse_d_extra_field():
    """Test that extra fields raise error on non-generic"""
    with pytest.raises(ConfifyParseError):
        _parse({"a": 10, "b": "test", "c": "extra"}, D)


# ============================================================================
# Tests for optional fields in generic dataclasses
# ============================================================================


def test_parse_optional_field_provided():
    """Test parsing generic with optional field provided"""
    result = _parse({"required": 42, "optional": 10}, OptionalFields[int])
    assert result == OptionalFields(required=42, optional=10)


def test_parse_optional_field_omitted():
    """Test parsing generic with optional field omitted - uses default"""
    result = _parse({"required": 42}, OptionalFields[int])
    assert result == OptionalFields(required=42, optional=5)
    assert result.optional == 5


def test_parse_optional_field_wrong_type():
    """Test that optional field still enforces type"""
    with pytest.raises(ConfifyParseError):
        _parse({"required": 42, "optional": "not_an_int"}, OptionalFields[int])


def test_parse_optional_field_required_missing():
    """Test that required field still required even with optional fields"""
    with pytest.raises(ConfifyParseError):
        _parse({"optional": 10}, OptionalFields[int])


def test_parse_multi_optional_all_provided():
    """Test parsing with multiple optional fields all provided"""
    result = _parse({"req1": 42, "req2": "test", "opt1": "custom", "opt2": [1, 2, 3]}, MultiOptional[int, str])
    assert result == MultiOptional(req1=42, req2="test", opt1="custom", opt2=[1, 2, 3])


def test_parse_multi_optional_some_omitted():
    """Test parsing with some optional fields omitted"""
    result = _parse({"req1": 42, "req2": "test"}, MultiOptional[int, str])
    assert result == MultiOptional(req1=42, req2="test", opt1="default", opt2=[])


def test_parse_multi_optional_one_provided():
    """Test parsing with one optional field provided"""
    result = _parse({"req1": 42, "req2": "test", "opt1": "custom"}, MultiOptional[int, str])
    assert result == MultiOptional(req1=42, req2="test", opt1="custom", opt2=[])


# ============================================================================
# Additional edge cases
# ============================================================================


def test_parse_deeply_nested_generics():
    """Test parsing deeply nested generic structures"""
    # B[AA[AA[int]], str]
    result = _parse({"x": {"x1": {"x1": 42}}, "y": "test"}, B[AA[AA[int]], str])
    assert result == B(x=AA(x1=AA(x1=42)), y="test")
    assert isinstance(result.x.x1.x1, int)


def test_parse_list_of_generics():
    """Test parsing list of generic dataclasses"""
    result = _parse([{"x1": 1}, {"x1": 2}, {"x1": 3}], list[AA[int]])
    assert result == [AA(x1=1), AA(x1=2), AA(x1=3)]


def test_parse_generic_with_list_field():
    """Test generic dataclass containing list"""
    result = _parse({"items": [1, 2, 3]}, Container[int])
    assert result == Container(items=[1, 2, 3])


def test_parse_generic_with_dict_field():
    """Test generic dataclass containing dict"""
    result = _parse({"data": {"a": 1, "b": 2}}, Mapping[int])
    assert result == Mapping(data={"a": 1, "b": 2})


def test_schema_repr_contains_type_params():
    """Test that schema representation includes type parameters"""
    schema = Schema.from_typeform(AA[int])
    repr_str = schema._repr()
    assert "AA[int]" in repr_str


def test_schema_repr_nested_generics():
    """Test schema representation for nested generics"""
    schema = Schema.from_typeform(B[AA[int], str])
    repr_str = schema._repr()
    # Should show nested structure
    assert f"{B.__module__}.B[{AA.__module__}.AA[int], str]" in repr_str
