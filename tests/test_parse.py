import pytest
from dataclasses import dataclass
import math
from types import NoneType
from pathlib import Path
from typing import List, Union, Optional, Tuple, Literal, TypedDict, NotRequired
from enum import Enum


from confify.parser import _parse, ConfifyParseError, _UnresolvedString


def _assert_bool_equals(x, y: bool):
    assert isinstance(x, bool)
    assert x == y


def _parseu(s: str, cls):
    return _parse(_UnresolvedString(s), cls)


def U(s: str):
    return _UnresolvedString(s)


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
        assert _parseu("[1, 2, 3]", t) == ["1", "2", "3"]
        assert _parseu("[1, 2, 3, (1, 2, 3)]", t) == ["1", "2", "3", "(1, 2, 3)"]
        assert _parseu("[1, a, False, None]", t) == ["1", "a", "False", "None"]
        assert _parseu("()", t) == []
        assert _parseu("(1, 2, 3)", t) == ["1", "2", "3"]
        assert _parseu("(1, 2, 3, (1, 2, 3))", t) == ["1", "2", "3", "(1, 2, 3)"]
        assert _parseu("(1, a, False, None)", t) == ["1", "a", "False", "None"]


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
        assert _parseu("[1, 2, 3]", t) == ("1", "2", "3")
        assert _parseu("[1, 2, 3, (1, 2, 3)]", t) == ("1", "2", "3", "(1, 2, 3)")
        assert _parseu("[1, a, False, None]", t) == ("1", "a", "False", "None")
        assert _parseu("()", t) == ()
        assert _parseu("(1, 2, 3)", t) == ("1", "2", "3")
        assert _parseu("(1, 2, 3, (1, 2, 3))", t) == ("1", "2", "3", "(1, 2, 3)")
        assert _parseu("(1, a, False, None)", t) == ("1", "a", "False", "None")


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
