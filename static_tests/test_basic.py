# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for basic Confify usage (mirrors examples/ex_basic.py)."""

from typing import assert_type, Optional, Literal
from dataclasses import dataclass
from pathlib import Path

from confify import Confify, Set, L
from confify.cli import SetRecord


@dataclass
class TrainingConfig:
    data_path: Path
    dataset: Literal["imagenet", "cifar10", "cifar100"]
    epochs: int
    batch_size: int
    learning_rate: float
    model_name: str
    output_dir: Path
    weight_decay: float = 0.0001
    pretrained: bool = False
    experiment_name: Optional[str] = None


# --- Confify[T] generic ---
c = Confify(TrainingConfig)
assert_type(c, Confify[TrainingConfig])


def check_types(_: TrainingConfig) -> None:
    # --- Set.to() for primitive types ---
    assert_type(Set(_.epochs).to(42), SetRecord[int])
    assert_type(Set(_.batch_size).to(128), SetRecord[int])
    assert_type(Set(_.learning_rate).to(0.01), SetRecord[float])
    assert_type(Set(_.weight_decay).to(0.0001), SetRecord[float])
    assert_type(Set(_.model_name).to("resnet18"), SetRecord[str])
    assert_type(Set(_.pretrained).to(True), SetRecord[bool])

    # --- Optional[str] field ---
    assert_type(Set(_.experiment_name).to("test"), SetRecord[str | None])
    assert_type(Set(_.experiment_name).to(None), SetRecord[str | None])

    # --- Path overloads: all three input types produce SetRecord[Path] ---
    assert_type(Set(_.data_path).to(Path("/data")), SetRecord[Path])
    assert_type(Set(_.data_path).to(L("/data/{name}")), SetRecord[Path])
    assert_type(Set(_.data_path).to("/data/raw"), SetRecord[Path])
    assert_type(Set(_.output_dir).to(Path("./out")), SetRecord[Path])
    assert_type(Set(_.output_dir).to(L("./out/{name}")), SetRecord[Path])
    assert_type(Set(_.output_dir).to("./out"), SetRecord[Path])

    # --- Negative: wrong types ---
    Set(_.epochs).to("hello")  # type: ignore
    Set(_.learning_rate).to("hello")  # type: ignore
    Set(_.pretrained).to("yes")  # type: ignore
    Set(_.batch_size).to(3.14)  # type: ignore
