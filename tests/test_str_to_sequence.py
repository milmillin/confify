import pytest

from confify.base import _UnresolvedString


def _str_to_sequence(s: str) -> list[str]:
    return [v.value for v in _UnresolvedString(s).resolve_as_sequence()]


def test():
    assert _str_to_sequence("[1, 2, 3]") == ["1", "2", "3"]
    assert _str_to_sequence("(1, 2, 3)") == ["1", "2", "3"]
    assert _str_to_sequence("((1, 2, 3))") == ["(1, 2, 3)"]
    assert _str_to_sequence("[(1, 2, 3)]") == ["(1, 2, 3)"]
    assert _str_to_sequence("[[1, 2, 3]]") == ["[1, 2, 3]"]
    assert _str_to_sequence("[[1, 2, 3], [4, 5, 6]]") == ["[1, 2, 3]", "[4, 5, 6]"]
    assert _str_to_sequence("[[1, 2, 3], [4, 5, 6], [7, 8, 9]]") == [
        "[1, 2, 3]",
        "[4, 5, 6]",
        "[7, 8, 9]",
    ]
    assert _str_to_sequence("[]") == []
    assert _str_to_sequence("()") == []
    assert _str_to_sequence("[  ]") == []
    assert _str_to_sequence("(  )") == []
    assert _str_to_sequence("( , )") == ["", ""]
    assert _str_to_sequence("( ,, )") == ["", "", ""]
    assert _str_to_sequence("( ( ) )") == ["( )"]


def test_quotes():
    assert _str_to_sequence('["a", "b", "c"]') == ['"a"', '"b"', '"c"']
    assert _str_to_sequence('["a]", "b[a, b, c]", "c"]') == [
        '"a]"',
        '"b[a, b, c]"',
        '"c"',
    ]
    with pytest.raises(ValueError):
        _str_to_sequence('["a]"[b, c]", "b[a, b, c]", "c"]')

    assert _str_to_sequence('["a\\"bc", "b[a, b, c]", "c"]') == [
        '"a\\"bc"',
        '"b[a, b, c]"',
        '"c"',
    ]
    assert _str_to_sequence('["a\\"bc", "b[a, b, c]", "c"]') == [
        '"a\\"bc"',
        '"b[a, b, c]"',
        '"c"',
    ]
    assert _str_to_sequence('["a\\xbc", "b[a, b, c]", "c"]') == [
        '"a\\xbc"',
        '"b[a, b, c]"',
        '"c"',
    ]
    assert _str_to_sequence('["a[", "]b",, "c"]') == ['"a["', '"]b"', "", '"c"']
    assert _str_to_sequence("[a,[b,[c,[d]]]]") == ["a", "[b,[c,[d]]]"]
    assert _str_to_sequence("[a,[b,[c,[d]]],[e,[f,g]],h]") == [
        "a",
        "[b,[c,[d]]]",
        "[e,[f,g]]",
        "h",
    ]


def test_fail():
    with pytest.raises(ValueError):
        _str_to_sequence("[1, 2, 3")
    with pytest.raises(ValueError):
        _str_to_sequence("[1, (], 3]")
    with pytest.raises(ValueError):
        _str_to_sequence("[1, 2, 3]]")
    with pytest.raises(ValueError):
        _str_to_sequence("[[1, 2, 3]")
