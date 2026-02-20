# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for Sweep patterns (mirrors examples/ex_sweep_patterns.py)."""

from typing import assert_type
from dataclasses import dataclass

from confify import Set, Sweep, L, ConfigStatements
from confify.cli import SetRecord


@dataclass
class Config:
    experiment_name: str
    learning_rate: float
    batch_size: int
    optimizer: str
    dropout: float
    num_layers: int


def check_types(_: Config) -> None:
    # --- Sweep construction with named kwargs ---
    sweep = Sweep(
        _lr_small=[Set(_.learning_rate).to(0.0001)],
        _lr_medium=[Set(_.learning_rate).to(0.001)],
        _lr_large=[Set(_.learning_rate).to(0.01)],
    )
    assert_type(sweep, Sweep)

    # --- Sweep construction with anonymous positional argument ---
    anon_sweep = Sweep(
        [Set(_.learning_rate).to(0.001)],
        _variant2=[Set(_.learning_rate).to(0.01)],
    )
    assert_type(anon_sweep, Sweep)

    # --- ConfigStatements type alias accepts all valid statement types ---
    stmts: ConfigStatements = [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.batch_size).to(128),
        Set(_.optimizer).to("adam"),
        Set(_.dropout).to(0.1),
        Set(_.num_layers).to(6),
        Sweep(
            _lr_small=[Set(_.learning_rate).to(0.0001)],
            _lr_large=[Set(_.learning_rate).to(0.01)],
        ),
        None,  # None is valid in ConfigStatements
    ]

    # --- Nested sweeps ---
    nested: ConfigStatements = [
        Sweep(
            _adam=[
                Set(_.optimizer).to("adam"),
                Sweep(
                    _lr1=[Set(_.learning_rate).to(0.0001)],
                    _lr2=[Set(_.learning_rate).to(0.001)],
                ),
            ],
            _sgd=[
                Set(_.optimizer).to("sgd"),
                Sweep(
                    _lr1=[Set(_.learning_rate).to(0.01)],
                    _lr2=[Set(_.learning_rate).to(0.1)],
                ),
            ],
        ),
    ]

    # --- Negative: wrong field types within sweep variants ---
    Sweep(
        _bad=[Set(_.learning_rate).to("fast")],  # type: ignore  -- str into float
    )
    Sweep(
        _bad=[Set(_.num_layers).to(3.5)],  # type: ignore  -- float into int
    )
