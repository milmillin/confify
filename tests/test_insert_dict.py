import pytest

from confify.base import _insert_dict


def test1():
    d = {}
    _insert_dict(d, ["a", "b", "c"], 1)
    assert d == {"a": {"b": {"c": 1}}}


def test2():
    d = {"a": {"b": {}}}
    _insert_dict(d, ["a", "b", "c"], 1)
    assert d == {"a": {"b": {"c": 1}}}


def test3():
    d = {"a": {"b": {}}}
    _insert_dict(d, ["a", "b"], 1)
    assert d == {"a": {"b": 1}}


def test4():
    d = {}
    _insert_dict(d, ["a", "b"], {"c": 1, "d": 2, "e": {"f": 3, "h": 7}})
    _insert_dict(d, ["a", "b", "d"], 4)
    _insert_dict(d, ["a", "b", "e"], {"f": 5, "g": 6})
    assert d == {"a": {"b": {"c": 1, "d": 4, "e": {"f": 5, "g": 6, "h": 7}}}}
