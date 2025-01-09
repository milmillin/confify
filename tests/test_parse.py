import pytest
from dataclasses import dataclass
import math
from types import NoneType
from pathlib import Path
from typing import List, Union, Optional, Tuple, Literal, TypedDict, NotRequired
from enum import Enum


from confify.base import _parse, ConfifyError, ConfifyWarning


def _assert_bool_equals(x, y: bool):
    assert isinstance(x, bool)
    assert x == y


################################################################################
# Test int
################################################################################


def test_int_from_int():
    assert _parse(124, int) == 124
    assert _parse(-124, int) == -124
    assert _parse(False, int) == 0
    assert _parse(True, int) == 1


def test_int_from_str():
    assert _parse("123", int) == 123
    assert _parse("123  ", int) == 123
    assert _parse("  123", int) == 123
    assert _parse("  123  ", int) == 123
    assert _parse("-123  ", int) == -123
    assert _parse("  -123", int) == -123
    assert _parse("  -123  ", int) == -123


def test_int_fail():
    with pytest.raises(ConfifyError):
        _parse("123.456", int)
    with pytest.raises(ConfifyError):
        _parse("12abc", int)
    with pytest.raises(ConfifyError):
        _parse("-3e5", int)
    with pytest.raises(ConfifyError):
        _parse(None, int)
    with pytest.raises(ConfifyError):
        _parse("'123'", int)
    with pytest.raises(ConfifyError):
        _parse("'-123'", int)


################################################################################
# Test float
################################################################################


def test_float_from_float():
    assert _parse(123.456, float) == 123.456
    assert _parse(-123.456, float) == -123.456
    assert _parse(123.456e-2, float) == 123.456e-2
    assert _parse(-123.456e-2, float) == -123.456e-2
    assert _parse(float("inf"), float) == float("inf")
    assert _parse(float("-inf"), float) == float("-inf")
    assert math.isnan(_parse(float("nan"), float))


def test_float_from_int():
    assert _parse(124, float) == 124
    assert _parse(-124, float) == -124
    assert _parse(False, float) == 0
    assert _parse(True, float) == 1


def test_float_from_str():
    assert _parse("123.456", float) == 123.456
    assert _parse("123.456  ", float) == 123.456
    assert _parse("  123.456", float) == 123.456
    assert _parse("  123.456  ", float) == 123.456
    assert _parse("-123.456  ", float) == -123.456
    assert _parse("  -123.456", float) == -123.456
    assert _parse("  -123.456  ", float) == -123.456
    assert _parse("1e-2", float) == 1e-2
    assert _parse("-1e-2", float) == -1e-2
    assert _parse("1.25e-2", float) == 1.25e-2
    assert _parse("-1.25e-2", float) == -1.25e-2
    assert _parse("1e+2", float) == 1e2
    assert _parse("-1e+2", float) == -1e2
    assert _parse("inf", float) == float("inf")
    assert _parse("-inf", float) == float("-inf")
    assert math.isnan(_parse("nan", float))
    assert math.isnan(_parse("NaN", float))
    assert math.isnan(_parse("NAN", float))


def test_float_fail():
    with pytest.raises(ConfifyError):
        _parse("123.456.789", float)
    with pytest.raises(ConfifyError):
        _parse("123abc", float)
    with pytest.raises(ConfifyError):
        _parse("-3e5f", float)
    with pytest.raises(ConfifyError):
        _parse(None, float)
    with pytest.raises(ConfifyError):
        _parse("'123.456'", float)
    with pytest.raises(ConfifyError):
        _parse("'123'", float)
    with pytest.raises(ConfifyError):
        _parse("'-3e5'", float)


################################################################################
# Test bool
################################################################################


def test_bool_from_bool():
    _assert_bool_equals(_parse(True, bool), True)
    _assert_bool_equals(_parse(False, bool), False)


def test_bool_from_int():
    _assert_bool_equals(_parse(1, bool), True)
    _assert_bool_equals(_parse(0, bool), False)


def test_bool_from_str():
    _assert_bool_equals(_parse("True", bool), True)
    _assert_bool_equals(_parse("true", bool), True)
    _assert_bool_equals(_parse("TRUE", bool), True)
    _assert_bool_equals(_parse("False", bool), False)
    _assert_bool_equals(_parse("false", bool), False)
    _assert_bool_equals(_parse("FALSE", bool), False)
    _assert_bool_equals(_parse("1", bool), True)
    _assert_bool_equals(_parse("0", bool), False)
    _assert_bool_equals(_parse("  True", bool), True)
    _assert_bool_equals(_parse("  true", bool), True)
    _assert_bool_equals(_parse("  TRUE", bool), True)
    _assert_bool_equals(_parse("  False", bool), False)
    _assert_bool_equals(_parse("  false", bool), False)
    _assert_bool_equals(_parse("  FALSE", bool), False)
    _assert_bool_equals(_parse("  1", bool), True)
    _assert_bool_equals(_parse("  0", bool), False)


def test_bool_fail():
    with pytest.raises(ConfifyError):
        _parse("Truee", bool)
    with pytest.raises(ConfifyError):
        _parse("Truee", bool)
    with pytest.raises(ConfifyError):
        _parse("Tru", bool)
    with pytest.raises(ConfifyError):
        _parse("Fa", bool)
    with pytest.raises(ConfifyError):
        _parse("00", bool)
    with pytest.raises(ConfifyError):
        _parse("01", bool)
    with pytest.raises(ConfifyError):
        _parse("10", bool)
    with pytest.raises(ConfifyError):
        _parse("'True'", bool)
    with pytest.raises(ConfifyError):
        _parse("'False'", bool)
    with pytest.raises(ConfifyError):
        _parse(123, bool)
    with pytest.raises(ConfifyError):
        _parse(-123, bool)


################################################################################
# Test str
################################################################################


def test_str_from_str():
    assert _parse("abc", str) == "abc"
    assert _parse("  abc", str) == "  abc"
    assert _parse("  abc  ", str) == "  abc  "
    assert _parse(123, str) == "123"
    assert _parse(123.456, str) == "123.456"
    assert _parse(True, str) == "True"
    assert _parse(False, str) == "False"


def test_str_from_quoted_str():
    assert _parse('"abc"', str) == "abc"
    assert _parse('"ab\\"c"', str) == 'ab"c'
    assert _parse('"ab\\\'c"', str) == "ab'c"
    assert _parse('"ab\\\\c"', str) == "ab\\c"
    assert _parse(' "abc"', str) == ' "abc"'
    assert _parse(' "abc"  ', str) == ' "abc"  '
    assert _parse('"None"', str) == "None"
    assert _parse('"123"', str) == "123"
    assert _parse('"False"', str) == "False"


################################################################################
# Test None
################################################################################


def test_none_from_none():
    assert _parse(None, NoneType) == None


def test_none_from_str():
    assert _parse("null", NoneType) == None


def test_none_fail():
    with pytest.raises(ConfifyError):
        _parse("Nonee", NoneType)
    with pytest.raises(ConfifyError):
        _parse("False", NoneType)
    with pytest.raises(ConfifyError):
        _parse("False", NoneType)
    with pytest.raises(ConfifyError):
        _parse("Nul", NoneType)
    with pytest.raises(ConfifyError):
        _parse("Fa", NoneType)


################################################################################
# Test Path
################################################################################


def test_path_from_path():
    assert _parse(Path("abc"), Path) == Path("abc")
    assert _parse(Path("abc/def"), Path) == Path("abc/def")
    assert _parse(Path("abc/def/"), Path) == Path("abc/def/")
    assert _parse(Path("/abc/def/"), Path) == Path("/abc/def/")
    assert _parse(Path("/abc/def"), Path) == Path("/abc/def")
    assert _parse(Path("/abc/def/"), Path) == Path("/abc/def/")
    assert _parse(Path("/abc/def/ghi"), Path) == Path("/abc/def/ghi")
    assert _parse(Path("/abc/def/ghi/"), Path) == Path("/abc/def/ghi/")


def test_path_from_str():
    assert _parse("abc", Path) == Path("abc")
    assert _parse("abc/def", Path) == Path("abc/def")
    assert _parse("abc/def/", Path) == Path("abc/def/")
    assert _parse("/abc/def/", Path) == Path("/abc/def/")
    assert _parse("/abc/def", Path) == Path("/abc/def")
    assert _parse("/abc/def/", Path) == Path("/abc/def/")
    assert _parse("/abc/def/ghi", Path) == Path("/abc/def/ghi")
    assert _parse("/abc/def/ghi/", Path) == Path("/abc/def/ghi/")


def test_path_fail():
    with pytest.raises(ConfifyError):
        _parse(123, Path)
    with pytest.raises(ConfifyError):
        _parse(123.456, Path)
    with pytest.raises(ConfifyError):
        _parse(True, Path)
    with pytest.raises(ConfifyError):
        _parse(False, Path)
    with pytest.raises(ConfifyError):
        _parse(None, Path)


################################################################################
# Test list
################################################################################


def test_untyped_list_from_list():
    for t in [list, List]:
        assert _parse([], t) == []
        assert _parse([1, 2, 3], t) == [1, 2, 3]
        assert _parse([1, 2, 3, (1, 2, 3)], t) == [1, 2, 3, (1, 2, 3)]
        assert _parse([1, "a", False, None], t) == [1, "a", False, None]
        assert _parse((), t) == []
        assert _parse((1, 2, 3), t) == [1, 2, 3]
        assert _parse((1, 2, 3, (1, 2, 3)), t) == [1, 2, 3, (1, 2, 3)]
        assert _parse((1, "a", False, None), t) == [1, "a", False, None]


def test_untyped_list_from_str():
    for t in [list, List]:
        assert _parse("[]", t) == []
        assert _parse("[1, 2, 3]", t) == ["1", "2", "3"]
        assert _parse("[1, 2, 3, (1, 2, 3)]", t) == ["1", "2", "3", "(1, 2, 3)"]
        assert _parse("[1, a, False, None]", t) == ["1", "a", "False", "None"]
        assert _parse("()", t) == []
        assert _parse("(1, 2, 3)", t) == ["1", "2", "3"]
        assert _parse("(1, 2, 3, (1, 2, 3))", t) == ["1", "2", "3", "(1, 2, 3)"]
        assert _parse("(1, a, False, None)", t) == ["1", "a", "False", "None"]


def test_untyped_tuple_from_list():
    for t in [tuple, Tuple]:
        assert _parse([], t) == ()
        assert _parse([1, 2, 3], t) == (1, 2, 3)
        assert _parse([1, 2, 3, (1, 2, 3)], t) == (1, 2, 3, (1, 2, 3))
        assert _parse([1, "a", False, None], t) == (1, "a", False, None)
        assert _parse((), t) == ()
        assert _parse((1, 2, 3), t) == (1, 2, 3)
        assert _parse((1, 2, 3, (1, 2, 3)), t) == (1, 2, 3, (1, 2, 3))
        assert _parse((1, "a", False, None), t) == (1, "a", False, None)


def test_untyped_tuple_from_str():
    for t in [tuple, Tuple]:
        assert _parse("[]", t) == ()
        assert _parse("[1, 2, 3]", t) == ("1", "2", "3")
        assert _parse("[1, 2, 3, (1, 2, 3)]", t) == ("1", "2", "3", "(1, 2, 3)")
        assert _parse("[1, a, False, None]", t) == ("1", "a", "False", "None")
        assert _parse("()", t) == ()
        assert _parse("(1, 2, 3)", t) == ("1", "2", "3")
        assert _parse("(1, 2, 3, (1, 2, 3))", t) == ("1", "2", "3", "(1, 2, 3)")
        assert _parse("(1, a, False, None)", t) == ("1", "a", "False", "None")


def test_typed_list_from_list():
    for t in [list[int], List[int]]:
        assert _parse([], t) == []
        assert _parse([1], t) == [1]
        assert _parse([1, 2, 3], t) == [1, 2, 3]
        assert _parse([" 1", "2", "3 "], t) == [1, 2, 3]
        with pytest.raises(ConfifyError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyError):
            _parse([1, "a", False, None], t)


def test_typed_list_from_str():
    for t in [list[int], List[int]]:
        assert _parse("[]", t) == []
        assert _parse("[1]", t) == [1]
        assert _parse("[1, 2, 3]", t) == [1, 2, 3]
        assert _parse("[ 1, 2, 3 ]", t) == [1, 2, 3]
        with pytest.raises(ConfifyError):
            _parse("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, a, False, None]", t)


def test_typed_tuple_from_list():
    for t in [tuple[int, ...], Tuple[int, ...]]:
        assert _parse([], t) == ()
        assert _parse([1], t) == (1,)
        assert _parse([1, 2, 3], t) == (1, 2, 3)
        assert _parse([" 1", "2", "3 "], t) == (1, 2, 3)
        with pytest.raises(ConfifyError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyError):
            _parse([1, "a", False, None], t)

    for t in [tuple[int], Tuple[int]]:
        with pytest.raises(ConfifyError):
            _parse([], t)
        assert _parse([1], t) == (1,)
        with pytest.raises(ConfifyError):
            _parse([1, 2, 3], t)
        with pytest.raises(ConfifyError):
            _parse([" 1", "2", "3 "], t)
        with pytest.raises(ConfifyError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyError):
            _parse([1, "a", False, None], t)

    for t in [tuple[int, int, int], Tuple[int, int, int]]:
        with pytest.raises(ConfifyError):
            _parse([], t)
        with pytest.raises(ConfifyError):
            _parse([1], t)
        assert _parse([1, 2, 3], t) == (1, 2, 3)
        assert _parse([" 1", "2", "3 "], t) == (1, 2, 3)
        with pytest.raises(ConfifyError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyError):
            _parse([1, 2, (1, 2, 3)], t)
        with pytest.raises(ConfifyError):
            _parse([1, "a", False, None], t)


def test_typed_tuple_from_str():
    for t in [tuple[int, ...], Tuple[int, ...]]:
        assert _parse("[]", t) == ()
        assert _parse("[1]", t) == (1,)
        assert _parse("[1, 2, 3]", t) == (1, 2, 3)
        assert _parse("[ 1, 2, 3 ]", t) == (1, 2, 3)
        with pytest.raises(ConfifyError):
            _parse("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, a, False, null]", t)

    for t in [tuple[int], Tuple[int]]:
        with pytest.raises(ConfifyError):
            _parse("[]", t)
        assert _parse("[1]", t) == (1,)
        with pytest.raises(ConfifyError):
            _parse("[1, 2, 3]", t)
        with pytest.raises(ConfifyError):
            _parse("[ 1, 2, 3 ]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, a, False, None]", t)

    for t in [tuple[int, int, int], Tuple[int, int, int]]:
        with pytest.raises(ConfifyError):
            _parse("[]", t)
        with pytest.raises(ConfifyError):
            _parse("[1]", t)
        assert _parse("[1, 2, 3]", t) == (1, 2, 3)
        assert _parse("[ 1, 2, 3 ]", t) == (1, 2, 3)
        with pytest.raises(ConfifyError):
            _parse("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, 2, (1, 2, 3)]", t)
        with pytest.raises(ConfifyError):
            _parse("[1, a, False, None]", t)

    assert _parse("[1, a, False, null]", tuple[int, str, bool, None]) == (
        1,
        "a",
        False,
        None,
    )


def test_empty_tuple():
    for t in [tuple[()], Tuple[()]]:
        assert _parse([], t) == ()

        with pytest.raises(ConfifyError):
            _parse([1, 2], t)

    for t in [tuple, Tuple]:
        assert _parse([], t) == ()
        assert _parse([1, 2], t) == (1, 2)


def test_nested_tuple():
    for t in [
        tuple[tuple[int, int], tuple[int, int]],
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        with pytest.raises(ConfifyError):
            _parse([], t)
        assert _parse([(1, 2), (3, 4)], t) == ((1, 2), (3, 4))
        with pytest.raises(ConfifyError):
            _parse([(1, 2), (3, 4), (5, 6)], t)
        assert _parse([" (1, 2)", " (3, 4) "], t) == ((1, 2), (3, 4))
        with pytest.raises(ConfifyError):
            _parse([(1, 2), (3, 4), (5, 6), (7, 8)], t)
        assert _parse("[ (1, 2),  (3, 4) ]", t) == ((1, 2), (3, 4))
    assert _parse("[(1, 2),[2,4],'(3,5']", tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "(3,5",
    )
    assert _parse("[(1, 2),[2,4], asdf]", tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "asdf",
    )
    assert _parse('[(1, 2),[2,4], "asdf"]', tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "asdf",
    )
    assert _parse("[(1, 2),[2,4], asdf]", tuple[str, list[int], str]) == (
        "(1, 2)",
        [2, 4],
        "asdf",
    )
    assert _parse("['(1, 2)',[2,4], asdf]", tuple[str, list[int], str]) == (
        "(1, 2)",
        [2, 4],
        "asdf",
    )
    with pytest.raises(ConfifyError):
        _parse("['(1, 2)',[2,4], asdf]", tuple[tuple[int, int], list[int], str])


def test_list_tuple_fail():
    for t in [
        list,
        List,
        tuple,
        Tuple,
        list[int],
        tuple[int],
        tuple[int, int],
        tuple[int, ...],
        tuple[int, int, int],
    ]:
        with pytest.raises(ConfifyError):
            _parse("123", t)
        with pytest.raises(ConfifyError):
            _parse("123.456", t)
        with pytest.raises(ConfifyError):
            _parse("True", t)
        with pytest.raises(ConfifyError):
            _parse("False", t)
        with pytest.raises(ConfifyError):
            _parse("None", t)
        with pytest.raises(ConfifyError):
            _parse("Nonee", t)
        with pytest.raises(ConfifyError):
            _parse("Nul", t)
        with pytest.raises(ConfifyError):
            _parse("Fa", t)


################################################################################
# Test dict
################################################################################


def test_untyped_dict():
    assert _parse({}, dict) == {}
    assert _parse({"a": 1, "b": 2}, dict) == {"a": 1, "b": 2}
    assert _parse({"a": 1, "b": 2, "c": 3}, dict) == {"a": 1, "b": 2, "c": 3}
    assert _parse({"a": 1, "b": 2, "c": 3, "d": 4}, dict) == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    assert _parse({"a": {"b": 1, "c": 2}, "d": 4}, dict) == {
        "a": {"b": 1, "c": 2},
        "d": 4,
    }


def test_typed_dict():
    assert _parse({}, dict[str, int]) == {}
    assert _parse({"a": 1, "b": 2}, dict[str, int]) == {"a": 1, "b": 2}
    assert _parse({"a": 1, "b": 2, "c": 3}, dict[str, int]) == {"a": 1, "b": 2, "c": 3}
    assert _parse({"a": 1, "b": 2, "c": 3, "d": 4}, dict[str, int]) == {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
    }
    with pytest.raises(ConfifyError):
        _parse({"a": {"b": 1, "c": 2}, "d": 4}, dict[str, int])
    with pytest.raises(ConfifyError):
        _parse({"a": 1, "b": 2, "c": 1.34, "d": 4}, dict[str, int])

    assert _parse({}, dict[str, str]) == {}
    assert _parse({"a": "1", "b": "2"}, dict[str, str]) == {"a": "1", "b": "2"}
    assert _parse({"a": "1", "b": "2", "c": "3"}, dict[str, str]) == {
        "a": "1",
        "b": "2",
        "c": "3",
    }
    assert _parse({"a": "1", "b": "2", "c": "3", "d": "4"}, dict[str, str]) == {
        "a": "1",
        "b": "2",
        "c": "3",
        "d": "4",
    }
    assert _parse({"a": {"b": "1", "c": "2"}, "d": "4"}, dict[str, str]) == {
        "a": str({"b": "1", "c": "2"}),
        "d": "4",
    }


################################################################################
# Test Literal
################################################################################


def test_literal():
    assert _parse("abc", Literal["abc"]) == "abc"
    assert _parse("abc", Literal["abc", "def"]) == "abc"
    assert _parse("def", Literal["abc", "def"]) == "def"
    with pytest.raises(ConfifyError):
        _parse("ghi", Literal["abc", "def"])

    assert _parse(123, Literal[123, "123"]) == 123
    assert _parse("123", Literal[123, "123"]) == 123
    assert _parse('"123"', Literal[123, "123"]) == "123"

    _assert_bool_equals(_parse(True, Literal[True, "True"]), True)
    _assert_bool_equals(_parse("True", Literal[True, "True"]), True)
    assert _parse('"True"', Literal[True, "True"]) == "True"
    _assert_bool_equals(_parse("1", Literal[True, "True"]), True)
    _assert_bool_equals(_parse(1, Literal[True, "True"]), True)

    assert _parse(None, Literal[None, "null"]) == None
    assert _parse("null", Literal[None, "null"]) == None
    assert _parse("'null'", Literal[None, "null"]) == "null"

    assert _parse({"a": 1, "b": 2}, Literal["123", 123, {"a": 1, "b": 2}]) == {
        "a": 1,
        "b": 2,
    }


################################################################################
# Test Union
################################################################################


def test_union():
    for t in [
        Union[int, str, bool, None],
        Optional[Union[int, str, bool]],
        Union[Optional[int], str, bool],
    ]:
        assert _parse(123, t) == 123
        assert _parse("123", t) == 123
        assert _parse('"123"', t) == "123"
        assert _parse(-5, t) == -5

        _assert_bool_equals(_parse(True, t), True)
        _assert_bool_equals(_parse(False, t), False)
        with pytest.warns(ConfifyWarning):
            _assert_bool_equals(_parse(1, t), True)
        with pytest.warns(ConfifyWarning):
            _assert_bool_equals(_parse(0, t), False)
        _assert_bool_equals(_parse("True", t), True)
        _assert_bool_equals(_parse("False", t), False)
        assert _parse('"True"', t) == "True"
        assert _parse('"False"', t) == "False"

    assert _parse({"a": 1, "b": 2}, Union[int, str]) == str({"a": 1, "b": 2})
    assert _parse(123.456, Union[int, str]) == str(123.456)
    assert _parse("123.456", Union[int, str]) == "123.456"
    assert _parse(True, Union[int, str]) == 1


################################################################################
# Test Enum
################################################################################


def test_enum():
    class A(Enum):
        a = 1
        b = 2
        c = 3

    assert _parse("a", A) == A.a
    assert _parse("b", A) == A.b
    assert _parse("c", A) == A.c
    with pytest.raises(ConfifyError):
        _parse('"a"', A)
    with pytest.raises(ConfifyError):
        _parse('"b"', A)
    with pytest.raises(ConfifyError):
        _parse('"c"', A)
    assert _parse(A.a, A) == A.a
    assert _parse(A.b, A) == A.b
    assert _parse(A.c, A) == A.c


###############################################################################
# Test Dataclasses
###############################################################################


@dataclass
class A:
    a: str
    b: int
    c: bool
    d: int = 5


@dataclass
class B:
    x1: int
    x2: str
    x3: A


def test_dataclass():
    assert _parse(A("a", 1, True), A) == A("a", 1, True)
    assert _parse({"a": "a", "b": 1, "c": True}, A) == A("a", 1, True)
    assert _parse({"a": "a", "b": "1", "c": "True"}, A) == A("a", 1, True)
    assert _parse({"a": "a", "b": "1", "c": "True", "d": "6"}, A) == A("a", 1, True, 6)
    # incorrect type
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1", "c": 33}, A)
    # missing field
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1"}, A)
    # extra field
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1", "c": "True", "d": "6", "e": "7"}, A)

    # nested dataclass
    assert _parse({"x1": 1, "x2": "2", "x3": A("a", 1, True)}, B) == B(1, "2", A("a", 1, True))


################################################################################
# Test TypedDict
################################################################################


class C(TypedDict):
    a: str
    b: int
    c: bool
    d: NotRequired[int]


class D(TypedDict):
    x1: int
    x2: str
    x3: C


def test_typeddict():
    assert _parse(C({"a": "a", "b": 1, "c": True}), C) == C({"a": "a", "b": 1, "c": True})
    assert _parse({"a": "a", "b": 1, "c": True}, C) == C({"a": "a", "b": 1, "c": True})
    assert _parse({"a": "a", "b": "1", "c": "True"}, C) == C({"a": "a", "b": 1, "c": True})
    assert _parse({"a": "a", "b": "1", "c": "True", "d": "6"}, C) == C({"a": "a", "b": 1, "c": True, "d": 6})
    # incorrect type
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1", "c": 33}, C)
    # missing field
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1"}, C)
    # extra field
    with pytest.raises(ConfifyError):
        _parse({"a": "a", "b": "1", "c": "True", "d": "6", "e": "7"}, C)

    # nested dataclass
    assert _parse({"x1": 1, "x2": "2", "x3": C({"a": "a", "b": 1, "c": True})}, D) == D(
        {"x1": 1, "x2": "2", "x3": C({"a": "a", "b": 1, "c": True})}
    )
