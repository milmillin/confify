# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for Use/Expression (mirrors examples/ex_expressions.py)."""

from typing import assert_type, Optional
from dataclasses import dataclass

from confify import Set, SetType, As, Variable, Use
from confify.cli import SetRecord, Expression, IExpression, _Use


@dataclass
class Encoder:
    depth: int
    dim: int


@dataclass
class Config:
    name: str
    value: int
    area: int
    encoder: Optional[Encoder] = None


# --- Use overloads ---
dim = Variable(int)
width = Variable(int)
height = Variable(int)
str_var = Variable(str)

# Single-variable Use
assert_type(Use(dim), _Use[int])

# Two-variable Use
assert_type(Use(width, height), _Use[int, int])

# Three-variable Use
assert_type(Use(dim, width, height), _Use[int, int, int])

# Mixed-type Use
assert_type(Use(dim, str_var), _Use[int, str])

# --- Expression[T] from Use ---
expr_int = Use(dim)(lambda d: d * 2)
assert_type(expr_int, Expression[int])

expr_int2 = Use(width, height)(lambda w, h: w * h)
assert_type(expr_int2, Expression[int])

expr_str = Use(dim)(lambda d: f"dim={d}")
assert_type(expr_str, Expression[str])

# --- Expression[T] is assignable to IExpression[T] ---
expr_as_iexpr: IExpression[int] = expr_int  # proves Expression[int] <: IExpression[int]


def check_types(_: Config) -> None:
    dim = Variable(int)
    width = Variable(int)
    height = Variable(int)

    # --- Passing expressions to Set.to() for matching fields ---
    expr_int = Use(dim)(lambda d: d * 2)
    assert_type(Set(_.value).to(expr_int), SetRecord[int])
    assert_type(Set(_.area).to(Use(width, height)(lambda w, h: w * h)), SetRecord[int])

    # --- Expressions inside SetType/As.then ---
    SetType(_.encoder)(
        As(Encoder).then(
            lambda e: [
                Set(e.depth).to(1),
                Set(e.dim).to(Use(dim)(lambda d: d * 2)),
            ]
        )
    )

    # --- Negative: Expression[str] into int field ---
    expr_str = Use(dim)(lambda d: f"dim={d}")
    Set(_.value).to(expr_str)  # type: ignore  -- Expression[str] into int field

    # --- Negative: Expression[float] into int field ---
    expr_float = Use(dim)(lambda d: d * 0.5)
    Set(_.value).to(expr_float)  # type: ignore  -- Expression[float] into int field
