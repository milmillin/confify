# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for SetType/As/generator patterns (mirrors examples/ex_generator.py)."""

from typing import assert_type
from dataclasses import dataclass

from confify import Confify, Set, SetType, As, Sweep, L, ConfigStatements
from confify.cli import SetRecord, SetTypeRecord, SetTypeRecordWithStatements, AsWithStatements


@dataclass
class Encoder:
    depth: int


@dataclass
class HeheEncoder(Encoder):
    ch_mult: tuple[int, ...]


@dataclass
class HuhuEncoder(Encoder):
    layers: int


@dataclass
class Config:
    name: str
    value: int
    dims: tuple[int, ...]
    encoder: Encoder


# --- As[T] construction ---
assert_type(As(HeheEncoder), As[HeheEncoder])
assert_type(As(HuhuEncoder), As[HuhuEncoder])

# --- As.then() returns AsWithStatements[T] ---
as_hehe_with = As(HeheEncoder).then(lambda e: [Set(e.ch_mult).to((3, 4))])
assert_type(as_hehe_with, AsWithStatements[HeheEncoder])


def check_types(_: Config) -> None:
    # --- SetType(_.field)(As(SubClass)) return type ---
    result = SetType(_.encoder)(As(HeheEncoder))
    assert_type(result, SetTypeRecord[Encoder] | SetTypeRecordWithStatements[Encoder])

    result_with = SetType(_.encoder)(
        As(HeheEncoder).then(
            lambda e: [
                Set(e.ch_mult).to((3, 4)),
                Set(e.depth).to(1),
            ]
        )
    )
    assert_type(result_with, SetTypeRecord[Encoder] | SetTypeRecordWithStatements[Encoder])

    # --- Lambda parameter narrowing: HeheEncoder-specific fields ---
    # The fact that Set(e.ch_mult) compiles proves e is narrowed to HeheEncoder
    SetType(_.encoder)(
        As(HeheEncoder).then(
            lambda e: [
                Set(e.ch_mult).to((1, 2, 3)),
                Set(e.depth).to(1),
            ]
        )
    )

    # HuhuEncoder-specific fields
    SetType(_.encoder)(
        As(HuhuEncoder).then(
            lambda e: [
                Set(e.layers).to(2),
                Set(e.depth).to(1),
            ]
        )
    )

    # --- Basic Set types within generators ---
    assert_type(Set(_.name).to(L("{name}")), SetRecord[str])
    assert_type(Set(_.value).to(1), SetRecord[int])
    assert_type(Set(_.dims).to((1, 2, 3)), SetRecord[tuple[int, ...]])

    # --- Negative: wrong types ---
    Set(_.value).to("hello")  # type: ignore  -- str into int field

    # HeheEncoder doesn't have 'layers'
    As(HeheEncoder).then(
        lambda e: [
            Set(e.layers).to(2),  # type: ignore  -- HeheEncoder has no 'layers'
        ]
    )

    # HuhuEncoder doesn't have 'ch_mult'
    As(HuhuEncoder).then(
        lambda e: [
            Set(e.ch_mult).to((1, 2)),  # type: ignore  -- HuhuEncoder has no 'ch_mult'
        ]
    )

    # Wrong type for tuple field
    Set(_.dims).to(42)  # type: ignore  -- int into tuple field
