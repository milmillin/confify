import pytest
from dataclasses import dataclass
import math
from types import NoneType
from pathlib import Path
from typing import Any, List, Union, Optional, Tuple, Literal, TypedDict, NotRequired
from enum import Enum


from confify.parser import _parse, ConfifyParseError, UnresolvedString


def _assert_bool_equals(x, y: bool):
    assert isinstance(x, bool)
    assert x == y


def _parseu(s: str, cls):
    return _parse(UnresolvedString(s), cls)


def U(s: str):
    return UnresolvedString(s)


################################################################################
# Test int
################################################################################


def test_int_from_int():
    assert _parse(124, int) == 124
    assert _parse(-124, int) == -124


def test_int_from_str():
    assert _parseu("123", int) == 123
    assert _parseu("123  ", int) == 123
    assert _parseu("  123", int) == 123
    assert _parseu("  123  ", int) == 123
    assert _parseu("-123  ", int) == -123
    assert _parseu("  -123", int) == -123
    assert _parseu("  -123  ", int) == -123


def test_int_fail():
    with pytest.raises(ConfifyParseError):
        _parseu("123.456", int)
    with pytest.raises(ConfifyParseError):
        _parseu("12abc", int)
    with pytest.raises(ConfifyParseError):
        _parseu("-3e5", int)
    with pytest.raises(ConfifyParseError):
        _parse(None, int)
    with pytest.raises(ConfifyParseError):
        _parse(True, int)
    with pytest.raises(ConfifyParseError):
        _parse(False, int)
    with pytest.raises(ConfifyParseError):
        _parseu("'123'", int)
    with pytest.raises(ConfifyParseError):
        _parseu("'-123'", int)


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
    assert _parseu("123.456", float) == 123.456
    assert _parseu("123.456  ", float) == 123.456
    assert _parseu("  123.456", float) == 123.456
    assert _parseu("  123.456  ", float) == 123.456
    assert _parseu("-123.456  ", float) == -123.456
    assert _parseu("  -123.456", float) == -123.456
    assert _parseu("  -123.456  ", float) == -123.456
    assert _parseu("1e-2", float) == 1e-2
    assert _parseu("-1e-2", float) == -1e-2
    assert _parseu("1.25e-2", float) == 1.25e-2
    assert _parseu("-1.25e-2", float) == -1.25e-2
    assert _parseu("1e+2", float) == 1e2
    assert _parseu("-1e+2", float) == -1e2
    assert _parseu("inf", float) == float("inf")
    assert _parseu("-inf", float) == float("-inf")
    assert math.isnan(_parseu("nan", float))
    assert math.isnan(_parseu("NaN", float))
    assert math.isnan(_parseu("NAN", float))


def test_float_fail():
    with pytest.raises(ConfifyParseError):
        _parseu("123.456.789", float)
    with pytest.raises(ConfifyParseError):
        _parseu("123abc", float)
    with pytest.raises(ConfifyParseError):
        _parseu("-3e5f", float)
    with pytest.raises(ConfifyParseError):
        _parse(None, float)
    with pytest.raises(ConfifyParseError):
        _parseu("'123.456'", float)
    with pytest.raises(ConfifyParseError):
        _parseu("'123'", float)
    with pytest.raises(ConfifyParseError):
        _parseu("'-3e5'", float)


################################################################################
# Test bool
################################################################################


def test_bool_from_bool():
    _assert_bool_equals(_parse(True, bool), True)
    _assert_bool_equals(_parse(False, bool), False)


def test_bool_from_str():
    _assert_bool_equals(_parseu("True", bool), True)
    _assert_bool_equals(_parseu("true", bool), True)
    _assert_bool_equals(_parseu("TRUE", bool), True)
    _assert_bool_equals(_parseu("False", bool), False)
    _assert_bool_equals(_parseu("false", bool), False)
    _assert_bool_equals(_parseu("FALSE", bool), False)
    _assert_bool_equals(_parseu("NO", bool), False)
    _assert_bool_equals(_parseu("no", bool), False)
    _assert_bool_equals(_parseu("Yes", bool), True)
    _assert_bool_equals(_parseu("yes", bool), True)
    _assert_bool_equals(_parseu("on", bool), True)
    _assert_bool_equals(_parseu("off", bool), False)
    _assert_bool_equals(_parseu("  True", bool), True)
    _assert_bool_equals(_parseu("  true", bool), True)
    _assert_bool_equals(_parseu("  TRUE", bool), True)
    _assert_bool_equals(_parseu("  False", bool), False)
    _assert_bool_equals(_parseu("  false", bool), False)
    _assert_bool_equals(_parseu("  FALSE", bool), False)


def test_bool_fail():
    with pytest.raises(ConfifyParseError):
        _parseu("Truee", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("Truee", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("Tru", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("Fa", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("00", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("01", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("10", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("'True'", bool)
    with pytest.raises(ConfifyParseError):
        _parseu("'False'", bool)
    with pytest.raises(ConfifyParseError):
        _parse(123, bool)
    with pytest.raises(ConfifyParseError):
        _parse(-123, bool)


################################################################################
# Test str
################################################################################


def test_str_from_str():
    assert _parse("abc", str) == "abc"
    assert _parse("  abc", str) == "  abc"
    assert _parse("  abc  ", str) == "  abc  "
    assert _parseu("abc", str) == "abc"
    assert _parseu("  abc", str) == "  abc"
    assert _parseu("  abc  ", str) == "  abc  "
    assert _parseu('"  abc  "', str) == "  abc  "
    with pytest.raises(ConfifyParseError):
        assert _parse(123, str) == "123"
    with pytest.raises(ConfifyParseError):
        assert _parse(123.456, str) == "123.456"
    with pytest.raises(ConfifyParseError):
        assert _parse(True, str) == "True"
    with pytest.raises(ConfifyParseError):
        assert _parse(False, str) == "False"


def test_str_from_quoted_str():
    assert _parseu('"abc"', str) == "abc"
    assert _parseu('"ab\\"c"', str) == 'ab"c'
    assert _parseu('"ab\\\'c"', str) == "ab'c"
    assert _parseu('"ab\\\\c"', str) == "ab\\c"
    assert _parseu(' "abc"', str) == ' "abc"'
    assert _parseu(' "abc"  ', str) == ' "abc"  '
    assert _parseu('"None"', str) == "None"
    assert _parseu('"123"', str) == "123"
    assert _parseu('"False"', str) == "False"
    # regular str
    assert _parse('"abc"', str) == '"abc"'
    assert _parse('"ab\\"c"', str) == '"ab\\"c"'
    assert _parse('"ab\\\'c"', str) == '"ab\\\'c"'
    assert _parse('"ab\\\\c"', str) == '"ab\\\\c"'
    assert _parse(' "abc"', str) == ' "abc"'
    assert _parse(' "abc"  ', str) == ' "abc"  '
    assert _parse('"None"', str) == '"None"'
    assert _parse('"123"', str) == '"123"'
    assert _parse('"False"', str) == '"False"'


################################################################################
# Test None
################################################################################


def test_none_from_none():
    assert _parse(None, NoneType) == None


def test_none_from_str():
    assert _parseu("null", NoneType) == None
    assert _parseu("~", NoneType) == None
    assert _parseu("Null", NoneType) == None
    assert _parseu("None", NoneType) == None
    assert _parseu("none", NoneType) == None
    # regular str
    with pytest.raises(ConfifyParseError):
        _parse("null", NoneType)
    with pytest.raises(ConfifyParseError):
        _parse("~", NoneType)
    with pytest.raises(ConfifyParseError):
        _parse("none", NoneType)


def test_none_fail():
    with pytest.raises(ConfifyParseError):
        _parseu("Nonee", NoneType)
    with pytest.raises(ConfifyParseError):
        _parseu("False", NoneType)
    with pytest.raises(ConfifyParseError):
        _parseu("False", NoneType)
    with pytest.raises(ConfifyParseError):
        _parseu("Nul", NoneType)
    with pytest.raises(ConfifyParseError):
        _parseu("Fa", NoneType)


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
    assert _parseu("abc", Path) == Path("abc")
    assert _parseu("abc/def", Path) == Path("abc/def")
    assert _parseu("abc/def/", Path) == Path("abc/def/")
    assert _parseu("/abc/def/", Path) == Path("/abc/def/")
    assert _parseu("/abc/def", Path) == Path("/abc/def")
    assert _parseu("/abc/def/", Path) == Path("/abc/def/")
    assert _parseu("/abc/def/ghi", Path) == Path("/abc/def/ghi")
    assert _parseu("/abc/def/ghi/", Path) == Path("/abc/def/ghi/")
    # str
    assert _parse("abc", Path) == Path("abc")
    assert _parse("abc/def", Path) == Path("abc/def")
    assert _parse("abc/def/", Path) == Path("abc/def/")
    assert _parse("/abc/def/", Path) == Path("/abc/def/")
    assert _parse("/abc/def", Path) == Path("/abc/def")
    assert _parse("/abc/def/", Path) == Path("/abc/def/")
    assert _parse("/abc/def/ghi", Path) == Path("/abc/def/ghi")
    assert _parse("/abc/def/ghi/", Path) == Path("/abc/def/ghi/")


def test_path_fail():
    with pytest.raises(ConfifyParseError):
        _parse(123, Path)
    with pytest.raises(ConfifyParseError):
        _parse(123.456, Path)
    with pytest.raises(ConfifyParseError):
        _parse(True, Path)
    with pytest.raises(ConfifyParseError):
        _parse(False, Path)
    with pytest.raises(ConfifyParseError):
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
        assert _parseu("[]", t) == []
        assert _parseu("[1, 2, 3]", t) == [1, 2, 3]
        assert _parseu("[1, 2, 3, (1, 2, 3)]", t) == [1, 2, 3, (1, 2, 3)]
        assert _parseu("[1, a, False, None]", t) == [1, "a", False, None]
        assert _parseu("()", t) == []
        assert _parseu("(1, 2, 3)", t) == [1, 2, 3]
        assert _parseu("(1, 2, 3, (1, 2, 3))", t) == [1, 2, 3, (1, 2, 3)]
        assert _parseu("(1, 2, 3, [1, 2, 3])", t) == [1, 2, 3, [1, 2, 3]]
        assert _parseu("(1, a, False, None)", t) == [1, "a", False, None]


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
        assert _parseu("[]", t) == ()
        assert _parseu("[1, 2, 3]", t) == (1, 2, 3)
        assert _parseu("[1, 2, 3, (1, 2, 3)]", t) == (1, 2, 3, (1, 2, 3))
        assert _parseu("[1, a, False, None]", t) == (1, "a", False, None)
        assert _parseu("()", t) == ()
        assert _parseu("(1, 2, 3)", t) == (1, 2, 3)
        assert _parseu("(1, 2, 3, (1, 2, 3))", t) == (1, 2, 3, (1, 2, 3))
        assert _parseu("(1, 2, 3, [1, 2, 3])", t) == (1, 2, 3, [1, 2, 3])
        assert _parseu("(1, a, False, None)", t) == (1, "a", False, None)


def test_typed_list_from_list():
    for t in [list[int], List[int]]:
        assert _parse([], t) == []
        assert _parse([1], t) == [1]
        assert _parse([1, 2, 3], t) == [1, 2, 3]
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, "a", False, None], t)


def test_typed_list_from_str():
    for t in [list[int], List[int]]:
        assert _parseu("[]", t) == []
        assert _parseu("[1]", t) == [1]
        assert _parseu("[1, 2, 3]", t) == [1, 2, 3]
        assert _parseu("[ 1, 2, 3 ]", t) == [1, 2, 3]
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, a, False, None]", t)


def test_typed_tuple_from_list():
    for t in [tuple[int, ...], Tuple[int, ...]]:
        assert _parse([], t) == ()
        assert _parse([1], t) == (1,)
        assert _parse([1, 2, 3], t) == (1, 2, 3)
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, "a", False, None], t)

    for t in [tuple[int], Tuple[int]]:
        with pytest.raises(ConfifyParseError):
            _parse([], t)
        assert _parse([1], t) == (1,)
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, 3], t)
        with pytest.raises(ConfifyParseError):
            _parse([" 1", "2", "3 "], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, "a", False, None], t)

    for t in [tuple[int, int, int], Tuple[int, int, int]]:
        with pytest.raises(ConfifyParseError):
            _parse([], t)
        with pytest.raises(ConfifyParseError):
            _parse([1], t)
        assert _parse([1, 2, 3], t) == (1, 2, 3)
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, 3, (1, 2, 3)], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, 2, (1, 2, 3)], t)
        with pytest.raises(ConfifyParseError):
            _parse([1, "a", False, None], t)


def test_typed_tuple_from_str():
    for t in [tuple[int, ...], Tuple[int, ...]]:
        assert _parseu("[]", t) == ()
        assert _parseu("[1]", t) == (1,)
        assert _parseu("[1, 2, 3]", t) == (1, 2, 3)
        assert _parseu("[ 1, 2, 3 ]", t) == (1, 2, 3)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, a, False, null]", t)

    for t in [tuple[int], Tuple[int]]:
        with pytest.raises(ConfifyParseError):
            _parseu("[]", t)
        assert _parseu("[1]", t) == (1,)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, 3]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[ 1, 2, 3 ]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, a, False, None]", t)

    for t in [tuple[int, int, int], Tuple[int, int, int]]:
        with pytest.raises(ConfifyParseError):
            _parseu("[]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1]", t)
        assert _parseu("[1, 2, 3]", t) == (1, 2, 3)
        assert _parseu("[ 1, 2, 3 ]", t) == (1, 2, 3)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, 3, (1, 2, 3)]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, 2, (1, 2, 3)]", t)
        with pytest.raises(ConfifyParseError):
            _parseu("[1, a, False, None]", t)

    assert _parseu("[1, a, False, null]", tuple[int, str, bool, None]) == (
        1,
        "a",
        False,
        None,
    )


def test_empty_tuple():
    for t in [tuple[()], Tuple[()]]:
        assert _parse([], t) == ()

        with pytest.raises(ConfifyParseError):
            _parse([1, 2], t)

    for t in [tuple, Tuple]:
        assert _parse([], t) == ()
        assert _parse([1, 2], t) == (1, 2)


def test_nested_tuple():
    for t in [
        tuple[tuple[int, int], tuple[int, int]],
        Tuple[Tuple[int, int], Tuple[int, int]],
    ]:
        with pytest.raises(ConfifyParseError):
            _parse([], t)
        assert _parse([(1, 2), (3, 4)], t) == ((1, 2), (3, 4))
        with pytest.raises(ConfifyParseError):
            _parse([(1, 2), (3, 4), (5, 6)], t)
        assert _parseu("[ (1, 2),  (3, 4) ]", t) == ((1, 2), (3, 4))
        with pytest.raises(ConfifyParseError):
            _parse([(1, 2), (3, 4), (5, 6), (7, 8)], t)
        assert _parseu("[ (1, 2),  (3, 4) ]", t) == ((1, 2), (3, 4))
    assert _parseu("[(1, 2),[2,4],'(3,5']", tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "(3,5",
    )
    assert _parseu("[(1, 2),[2,4], asdf]", tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "asdf",
    )
    assert _parseu('[(1, 2),[2,4], "asdf"]', tuple[tuple[int, int], list[int], str]) == (
        (1, 2),
        [2, 4],
        "asdf",
    )
    assert _parseu("[(1, 2),[2,4], asdf]", tuple[str, list[int], str]) == (
        "(1, 2)",
        [2, 4],
        "asdf",
    )
    assert _parseu("['(1, 2)',[2,4], asdf]", tuple[str, list[int], str]) == (
        "(1, 2)",
        [2, 4],
        "asdf",
    )
    with pytest.raises(ConfifyParseError):
        _parseu("['(1, 2)',[2,4], asdf]", tuple[tuple[int, int], list[int], str])


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
        with pytest.raises(ConfifyParseError):
            _parseu("123", t)
        with pytest.raises(ConfifyParseError):
            _parseu("123.456", t)
        with pytest.raises(ConfifyParseError):
            _parseu("True", t)
        with pytest.raises(ConfifyParseError):
            _parseu("False", t)
        with pytest.raises(ConfifyParseError):
            _parseu("None", t)
        with pytest.raises(ConfifyParseError):
            _parseu("Nonee", t)
        with pytest.raises(ConfifyParseError):
            _parseu("Nul", t)
        with pytest.raises(ConfifyParseError):
            _parseu("Fa", t)


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
    with pytest.raises(ConfifyParseError):
        _parse({"a": {"b": 1, "c": 2}, "d": 4}, dict[str, int])
    with pytest.raises(ConfifyParseError):
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
    assert _parse({"a": str({"b": "1", "c": "2"}), "d": "4"}, dict[str, str]) == {
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
    with pytest.raises(ConfifyParseError):
        _parse("ghi", Literal["abc", "def"])

    assert _parse(123, Literal[123, "123"]) == 123
    assert _parseu("123", Literal[123, "123"]) == 123
    assert _parseu('"123"', Literal[123, "123"]) == "123"

    _assert_bool_equals(_parse(True, Literal[True, "True"]), True)
    _assert_bool_equals(_parseu("True", Literal[True, "True"]), True)
    assert _parseu('"True"', Literal[True, "True"]) == "True"

    assert _parse(None, Literal[None, "null"]) == None
    assert _parseu("null", Literal[None, "null"]) == None
    assert _parseu("'null'", Literal[None, "null"]) == "null"

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
        assert _parseu("123", t) == 123
        assert _parseu('"123"', t) == "123"
        assert _parse(-5, t) == -5

        _assert_bool_equals(_parse(True, t), True)
        _assert_bool_equals(_parse(False, t), False)
        _assert_bool_equals(_parseu("True", t), True)
        _assert_bool_equals(_parseu("False", t), False)
        assert _parseu('"True"', t) == "True"
        assert _parseu('"False"', t) == "False"

    with pytest.raises(ConfifyParseError):
        _parse({"a": 1, "b": 2}, Union[int, str])
    with pytest.raises(ConfifyParseError):
        _parse(123.456, Union[int, str])
    assert _parseu("123.456", Union[int, str]) == "123.456"
    with pytest.raises(ConfifyParseError):
        _parse(True, Union[int, str])


def test_union_syntactic_sugar():
    for t in [
        int | str | bool | None,
        int | str | bool | None,  # equivalent to Optional[Union[int, str, bool]]
        int | None | str | bool,  # equivalent to Union[Optional[int], str, bool]
    ]:
        assert _parse(123, t) == 123
        assert _parseu("123", t) == 123
        assert _parseu('"123"', t) == "123"
        assert _parse(-5, t) == -5

        _assert_bool_equals(_parse(True, t), True)
        _assert_bool_equals(_parse(False, t), False)
        _assert_bool_equals(_parseu("True", t), True)
        _assert_bool_equals(_parseu("False", t), False)
        assert _parseu('"True"', t) == "True"
        assert _parseu('"False"', t) == "False"

    with pytest.raises(ConfifyParseError):
        _parse({"a": 1, "b": 2}, int | str)
    with pytest.raises(ConfifyParseError):
        _parse(123.456, int | str)
    assert _parseu("123.456", int | str) == "123.456"
    with pytest.raises(ConfifyParseError):
        _parse(True, int | str)


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
    assert _parseu("a", A) == A.a
    assert _parseu("b", A) == A.b
    assert _parseu("c", A) == A.c
    with pytest.raises(ConfifyParseError):
        _parse('"a"', A)
    with pytest.raises(ConfifyParseError):
        _parse('"b"', A)
    with pytest.raises(ConfifyParseError):
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
    assert _parse({"a": "a", "b": U("1"), "c": U("True")}, A) == A("a", 1, True)
    assert _parse({"a": "a", "b": U("1"), "c": U("True"), "d": U("6")}, A) == A("a", 1, True, 6)
    # incorrect type
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1"), "c": 33}, A)
    # missing field
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1")}, A)
    # extra field
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1"), "c": "True", "d": U("6"), "e": U("7")}, A)

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
    assert _parse({"a": "a", "b": U("1"), "c": U("True")}, C) == C({"a": "a", "b": 1, "c": True})
    assert _parse({"a": "a", "b": U("1"), "c": U("True"), "d": U("6")}, C) == C({"a": "a", "b": 1, "c": True, "d": 6})
    # incorrect type
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1"), "c": 33}, C)
    # missing field
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1")}, C)
    # extra field
    with pytest.raises(ConfifyParseError):
        _parse({"a": "a", "b": U("1"), "c": U("True"), "d": U("6"), "e": U("7")}, C)

    # nested dataclass
    assert _parse({"x1": 1, "x2": "2", "x3": C({"a": "a", "b": 1, "c": True})}, D) == D(
        {"x1": 1, "x2": "2", "x3": C({"a": "a", "b": 1, "c": True})}
    )


################################################################################
# Test Any Type
################################################################################


def test_any_basic_int_inference():
    """Test integer type inference from UnresolvedString with Any."""
    assert _parseu("42", Any) == 42
    assert _parseu("-42", Any) == -42
    assert _parseu("  42  ", Any) == 42
    assert _parseu("0", Any) == 0
    assert _parseu("-0", Any) == 0
    assert _parseu("007", Any) == 7
    assert _parseu("123456789", Any) == 123456789


def test_any_basic_float_inference():
    """Test float type inference from UnresolvedString with Any."""
    assert _parseu("3.14", Any) == 3.14
    assert _parseu("-3.14", Any) == -3.14
    assert _parseu("0.0", Any) == 0.0
    assert _parseu("-0.0", Any) == -0.0
    assert _parseu("1e-5", Any) == 1e-5
    assert _parseu("1.5e+10", Any) == 1.5e10
    assert _parseu("1e5", Any) == 100000.0
    assert _parseu("1.5e-3", Any) == 0.0015
    assert _parseu("-1e-2", Any) == -1e-2
    assert _parseu("1.25e-2", Any) == 1.25e-2


def test_any_special_float_values():
    """Test special float values (inf, nan) with Any."""
    assert _parseu("inf", Any) == float("inf")
    assert _parseu("-inf", Any) == float("-inf")
    assert math.isnan(_parseu("nan", Any))
    assert math.isnan(_parseu("NaN", Any))
    assert math.isnan(_parseu("NAN", Any))


def test_any_basic_bool_inference():
    """Test bool type inference from UnresolvedString with Any."""
    _assert_bool_equals(_parseu("True", Any), True)
    _assert_bool_equals(_parseu("true", Any), True)
    _assert_bool_equals(_parseu("TRUE", Any), True)
    _assert_bool_equals(_parseu("yes", Any), True)
    _assert_bool_equals(_parseu("Yes", Any), True)
    _assert_bool_equals(_parseu("on", Any), True)
    _assert_bool_equals(_parseu("False", Any), False)
    _assert_bool_equals(_parseu("false", Any), False)
    _assert_bool_equals(_parseu("FALSE", Any), False)
    _assert_bool_equals(_parseu("no", Any), False)
    _assert_bool_equals(_parseu("NO", Any), False)
    _assert_bool_equals(_parseu("off", Any), False)


def test_any_basic_none_inference():
    """Test None type inference from UnresolvedString with Any."""
    assert _parseu("null", Any) is None
    assert _parseu("~", Any) is None
    assert _parseu("None", Any) is None
    assert _parseu("none", Any) is None
    assert _parseu("Null", Any) is None
    assert _parseu("NULL", Any) is None


def test_any_string_inference_quoted():
    """Test string inference with quoted strings for Any."""
    assert _parseu('"hello"', Any) == "hello"
    assert _parseu("'hello'", Any) == "hello"
    assert _parseu('"42"', Any) == "42"  # Force string
    assert _parseu('"True"', Any) == "True"  # Force string
    assert _parseu('"null"', Any) == "null"  # Force string
    assert _parseu('"false"', Any) == "false"  # Force string
    assert _parseu('""', Any) == ""  # Empty string
    assert _parseu("''", Any) == ""  # Empty string
    assert _parseu('" "', Any) == " "  # Space string
    assert _parseu('"  abc  "', Any) == "  abc  "  # Preserve whitespace


def test_any_string_inference_unquoted():
    """Test string inference with unquoted strings (fallback) for Any."""
    assert _parseu("hello", Any) == "hello"
    assert _parseu("abc123", Any) == "abc123"
    assert _parseu("True123", Any) == "True123"  # Not a valid bool
    assert _parseu("trueish", Any) == "trueish"  # Not "true"
    assert _parseu("nullify", Any) == "nullify"  # Not "null"
    assert _parseu("123abc", Any) == "123abc"  # Not valid int


def test_any_string_escape_sequences():
    """Test escape sequences in quoted strings for Any."""
    assert _parseu('"hello\\nworld"', Any) == "hello\nworld"
    assert _parseu('"a\\"b"', Any) == 'a"b'
    assert _parseu('"a\\\\b"', Any) == "a\\b"
    assert _parseu("'a\\'b'", Any) == "a'b"


def test_any_sequences_list():
    """Test list sequence inference from UnresolvedString with Any."""
    assert _parseu("[]", Any) == []
    assert _parseu("[1, 2, 3]", Any) == [1, 2, 3]
    assert _parseu("[1, abc, True, null]", Any) == [1, "abc", True, None]
    assert _parseu("[  ]", Any) == []  # Empty with whitespace
    assert _parseu("[ 1 , 2 , 3 ]", Any) == [1, 2, 3]  # Whitespace handling


def test_any_sequences_tuple():
    """Test tuple sequence inference from UnresolvedString with Any."""
    assert _parseu("()", Any) == ()
    assert _parseu("(1, 2, 3)", Any) == (1, 2, 3)
    assert _parseu("(1, abc, True, null)", Any) == (1, "abc", True, None)
    assert _parseu("(  )", Any) == ()  # Empty with whitespace
    assert _parseu("( 1 , 2 , 3 )", Any) == (1, 2, 3)  # Whitespace handling


def test_any_sequences_nested():
    """Test nested sequence inference with Any."""
    assert _parseu("[[1, 2], [3, 4]]", Any) == [[1, 2], [3, 4]]
    assert _parseu("[(1, 2), (3, 4)]", Any) == [(1, 2), (3, 4)]
    assert _parseu("([1, 2], [3, 4])", Any) == ([1, 2], [3, 4])
    assert _parseu("[[[1]]]", Any) == [[[1]]]
    assert _parseu("[(1, [2, 3]), [4, (5, 6)]]", Any) == [(1, [2, 3]), [4, (5, 6)]]


def test_any_sequences_mixed_types():
    """Test sequences with mixed types inferred from strings for Any."""
    assert _parseu("[1, 2.5, true, null, abc]", Any) == [1, 2.5, True, None, "abc"]
    assert _parseu('["quoted", unquoted, 42]', Any) == ["quoted", "unquoted", 42]
    assert _parseu("[1, 2.5, True, False, null, abc, [1, 2]]", Any) == [
        1,
        2.5,
        True,
        False,
        None,
        "abc",
        [1, 2],
    ]
    assert _parseu("(1, 2.5, true, null, abc, (1, 2))", Any) == (
        1,
        2.5,
        True,
        None,
        "abc",
        (1, 2),
    )


def test_any_from_direct_values():
    """Test Any with already-resolved values (pass-through)."""
    assert _parse(42, Any) == 42
    assert _parse(3.14, Any) == 3.14
    _assert_bool_equals(_parse(True, Any), True)
    _assert_bool_equals(_parse(False, Any), False)
    assert _parse(None, Any) is None
    assert _parse("hello", Any) == "hello"
    assert _parse([1, 2, 3], Any) == [1, 2, 3]
    assert _parse((1, 2, 3), Any) == (1, 2, 3)
    assert _parse({"a": 1}, Any) == {"a": 1}
    assert _parse(float("inf"), Any) == float("inf")
    assert math.isnan(_parse(float("nan"), Any))


def test_any_nested_dict_with_unresolved():
    """Test Any with nested dict containing UnresolvedString."""
    assert _parse({"a": U("42"), "b": U("true")}, Any) == {"a": 42, "b": True}
    assert _parse({"a": U("42"), "b": U("3.14"), "c": U("null")}, Any) == {
        "a": 42,
        "b": 3.14,
        "c": None,
    }
    assert _parse({"a": U("1"), "b": U("false"), "c": U("abc"), "d": U('"quoted"')}, Any) == {
        "a": 1,
        "b": False,
        "c": "abc",
        "d": "quoted",
    }


def test_any_nested_list_with_unresolved():
    """Test Any with nested list containing UnresolvedString."""
    assert _parse([U("42"), U("true"), U("null")], Any) == [42, True, None]
    assert _parse([U("1"), U("2.5"), U("false"), U("abc")], Any) == [
        1,
        2.5,
        False,
        "abc",
    ]


def test_any_nested_tuple_with_unresolved():
    """Test Any with nested tuple containing UnresolvedString."""
    assert _parse((U("42"), U("true"), U("null")), Any) == (42, True, None)
    assert _parse((U("1"), U("2.5"), U("false"), U("abc")), Any) == (
        1,
        2.5,
        False,
        "abc",
    )


def test_any_deeply_nested_structures():
    """Test Any with deeply nested structures mixing UnresolvedString and resolved values."""
    assert _parse({"a": [U("1"), U("2")], "b": {"c": U("true")}}, Any) == {
        "a": [1, 2],
        "b": {"c": True},
    }
    assert _parse({"outer": {"inner": [U("1"), U("2"), (U("3"), U("4"))]}}, Any) == {"outer": {"inner": [1, 2, (3, 4)]}}
    assert _parse([U("1"), {"a": U("2"), "b": [U("3"), U("4")]}, (U("5"), U("6"))], Any) == [
        1,
        {"a": 2, "b": [3, 4]},
        (5, 6),
    ]


def test_any_edge_case_empty():
    """Test Any with empty values."""
    assert _parseu("", Any) == ""
    assert _parse([], Any) == []
    assert _parse((), Any) == ()
    assert _parse({}, Any) == {}


def test_any_edge_case_whitespace():
    """Test Any with whitespace handling."""
    assert _parseu("  ", Any) == ""  # Strips to empty string
    assert _parseu("  123  ", Any) == 123  # Strips then parses
    assert _parseu("  true  ", Any) is True  # Strips then parses
    assert _parseu("  null  ", Any) is None  # Strips then parses


def test_any_edge_case_all_same_type():
    """Test Any with sequences of all same type."""
    assert _parseu("[null, null, null]", Any) == [None, None, None]
    assert _parseu("[true, false, true]", Any) == [True, False, True]
    assert _parseu("[1, 2, 3, 4, 5]", Any) == [1, 2, 3, 4, 5]
    assert _parseu("[1.1, 2.2, 3.3]", Any) == [1.1, 2.2, 3.3]
    assert _parseu("[a, b, c]", Any) == ["a", "b", "c"]


def test_any_type_priority():
    """Test type inference priority for Any."""
    # Priority: sequence > quoted string > None > bool > int > float > str
    assert _parseu("null", Any) is None  # Not "null"
    assert _parseu('"null"', Any) == "null"  # Quoted forces string
    _assert_bool_equals(_parseu("true", Any), True)  # Not "true"
    assert _parseu('"true"', Any) == "true"  # Quoted forces string
    assert _parseu("123", Any) == 123  # Not "123"
    assert _parseu('"123"', Any) == "123"  # Quoted forces string
    assert _parseu("3.14", Any) == 3.14  # Not "3.14"
    assert _parseu('"3.14"', Any) == "3.14"  # Quoted forces string


def test_any_ambiguous_values():
    """Test Any with values that look like types but aren't."""
    assert _parseu("trueish", Any) == "trueish"  # Not "true"
    assert _parseu("nullify", Any) == "nullify"  # Not "null"
    assert _parseu("123abc", Any) == "123abc"  # Not valid int
    assert _parseu("12.34.56", Any) == "12.34.56"  # Not valid float
    assert _parseu("Tru", Any) == "Tru"  # Not "True"
    assert _parseu("Nul", Any) == "Nul"  # Not "null"


def test_any_trailing_comma():
    """Test Any with trailing comma produces empty string."""
    assert _parseu("(1,)", Any) == (1, "")
    assert _parseu("(abc,)", Any) == ("abc", "")


################################################################################
# Test Unparameterized Collections - Advanced
################################################################################


def test_untyped_list_mixed_inference():
    """Test unparameterized list with mixed types inferred from strings."""
    assert _parseu("[1, 2.5, true, abc, [1,2]]", list) == [1, 2.5, True, "abc", [1, 2]]
    assert _parseu("[null, false, 123, abc]", list) == [None, False, 123, "abc"]
    assert _parseu('["quoted", unquoted, 42, true]', list) == [
        "quoted",
        "unquoted",
        42,
        True,
    ]


def test_untyped_list_all_unresolved():
    """Test unparameterized list with all UnresolvedString elements."""
    assert _parse([U("1"), U("2"), U("3")], list) == [1, 2, 3]
    assert _parse([U("true"), U("false"), U("null")], list) == [True, False, None]
    assert _parse([U("1"), U("2.5"), U("abc")], list) == [1, 2.5, "abc"]


def test_untyped_tuple_mixed_inference():
    """Test unparameterized tuple with mixed types inferred from strings."""
    assert _parseu("(1, 2.5, true, abc, (1,2))", tuple) == (1, 2.5, True, "abc", (1, 2))
    assert _parseu("(null, false, 123, abc)", tuple) == (None, False, 123, "abc")
    assert _parseu('("quoted", unquoted, 42, true)', tuple) == (
        "quoted",
        "unquoted",
        42,
        True,
    )


def test_untyped_tuple_trailing_comma():
    """Test unparameterized tuple with trailing comma produces empty string."""
    assert _parseu("(1,)", tuple) == (1, "")
    assert _parseu("(abc,)", tuple) == ("abc", "")


def test_untyped_tuple_single_element():
    """Test unparameterized tuple with single element from list."""
    assert _parse([1], tuple) == (1,)
    assert _parse([U("42")], tuple) == (42,)


def test_untyped_tuple_from_list_conversion():
    """Test unparameterized tuple conversion from list."""
    assert _parse([1, 2, 3], tuple) == (1, 2, 3)
    assert _parse([], tuple) == ()
    assert _parse([U("1"), U("2.5"), U("true")], tuple) == (1, 2.5, True)


def test_untyped_dict_with_unresolved_values():
    """Test unparameterized dict with UnresolvedString values (type inference)."""
    assert _parse({"a": U("42"), "b": U("true"), "c": U("null")}, dict) == {
        "a": 42,
        "b": True,
        "c": None,
    }
    assert _parse({"x": U("1"), "y": U("2.5"), "z": U("abc")}, dict) == {
        "x": 1,
        "y": 2.5,
        "z": "abc",
    }
    assert _parse(
        {"int": U("123"), "float": U("3.14"), "bool": U("true"), "str": U('"text"')},
        dict,
    ) == {"int": 123, "float": 3.14, "bool": True, "str": "text"}


def test_untyped_dict_nested():
    """Test unparameterized dict with nested structures."""
    assert _parse({"a": 1, "b": "text", "c": True, "d": None, "e": [1, 2]}, dict) == {
        "a": 1,
        "b": "text",
        "c": True,
        "d": None,
        "e": [1, 2],
    }
    assert _parse({"a": {"b": {"c": U("123")}}}, dict) == {"a": {"b": {"c": 123}}}
    assert _parse(
        {
            "ints": [U("1"), U("2"), U("3")],
            "strs": [U("a"), U("b")],
            "nested": {"x": U("true"), "y": U("3.14")},
        },
        dict,
    ) == {
        "ints": [1, 2, 3],
        "strs": ["a", "b"],
        "nested": {"x": True, "y": 3.14},
    }


def test_untyped_dict_complex_values():
    """Test unparameterized dict with complex value types."""
    assert _parse(
        {
            "list": [1, 2, 3],
            "tuple": (1, 2),
            "dict": {"nested": U("42")},
            "mixed": [U("1"), U("abc"), U("true")],
        },
        dict,
    ) == {
        "list": [1, 2, 3],
        "tuple": (1, 2),
        "dict": {"nested": 42},
        "mixed": [1, "abc", True],
    }


def test_untyped_collections_nested_mixed():
    """Test deeply nested unparameterized collections with mixed types."""
    assert _parseu("[(1, 2), [3, 4], (5, [6, 7])]", list) == [(1, 2), [3, 4], (5, [6, 7])]
    assert _parse([[U("1"), U("2")], (U("3"), U("4")), [U("true"), U("false")]], list) == [
        [1, 2],
        (3, 4),
        [True, False],
    ]
    assert _parse(([U("1"), U("2")], [U("3"), U("4")], (U("true"), U("false"))), tuple) == (
        [1, 2],
        [3, 4],
        (True, False),
    )


def test_untyped_collections_conversion():
    """Test conversion between unparameterized list and tuple."""
    # List from tuple
    assert _parse((1, 2, 3), list) == [1, 2, 3]
    assert _parse((U("1"), U("2"), U("3")), list) == [1, 2, 3]

    # Tuple from list
    assert _parse([1, 2, 3], tuple) == (1, 2, 3)
    assert _parse([U("1"), U("2"), U("3")], tuple) == (1, 2, 3)
