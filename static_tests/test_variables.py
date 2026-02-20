# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for Variable usage (mirrors examples/ex_variables.py)."""

from typing import assert_type, Optional
from dataclasses import dataclass

from confify import Set, Sweep, SetType, As, L, Variable, ConfigStatements
from confify.cli import SetRecord, IExpression


@dataclass
class Encoder:
    depth: int
    dim: int


@dataclass
class Decoder:
    depth: int
    dim: int


@dataclass
class Config:
    name: str
    value: int
    dims: tuple[int, ...]
    encoder: Optional[Encoder] = None
    decoder: Optional[Decoder] = None


# --- Variable[T] construction ---
dim = Variable(int)
assert_type(dim, Variable[int])

# --- Variable[T] is assignable to IExpression[T] ---
dim_as_iexpr: IExpression[int] = dim  # proves Variable[int] <: IExpression[int]

# --- Set(variable).to(value) type propagation ---
assert_type(Set(dim).to(32), SetRecord[int])
assert_type(Set(dim).to(64), SetRecord[int])


def check_types(_: Config) -> None:
    dim = Variable(int)

    # --- Variable as expression value in Set(_.field).to(dim) ---
    # Variable[int] is IExpression[int], accepted by Set[int].to(IExpression[int])
    assert_type(Set(_.value).to(dim), SetRecord[int])

    # --- Set.to() with literal values on variable-consuming fields ---
    assert_type(Set(_.name).to(L("{name}")), SetRecord[str])
    assert_type(Set(_.dims).to((1, 2, 3)), SetRecord[tuple[int, ...]])

    # --- Negative: type mismatches between Variable[T] and field types ---
    str_var = Variable(str)
    Set(_.value).to(str_var)  # type: ignore  -- Variable[str] into int field
