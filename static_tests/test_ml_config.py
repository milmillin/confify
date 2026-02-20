# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static type tests for ML config patterns (mirrors examples/ex_ml_config.py)."""

from typing import assert_type, Literal, Optional
from dataclasses import dataclass
from pathlib import Path

from confify import Confify, Set, Sweep, SetType, As, L, ConfigStatements
from confify.cli import SetRecord, SetTypeRecord, SetTypeRecordWithStatements, AsWithStatements


# --- Full polymorphic hierarchy ---


@dataclass
class DataConfig:
    dataset: Literal["imagenet", "cifar10", "cifar100"]
    data_root: Path
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    random_crop: bool = True
    random_flip: bool = True
    color_jitter: float = 0.0
    auto_augment: Optional[str] = None


@dataclass
class ModelConfig:
    num_classes: int
    pretrained: bool = False


@dataclass
class ResNetConfig(ModelConfig):
    depth: Literal[18, 34, 50, 101, 152] = 18
    width_multiplier: float = 1.0


@dataclass
class ViTConfig(ModelConfig):
    patch_size: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0


@dataclass
class OptimizerConfig:
    learning_rate: float
    weight_decay: float = 0.0


@dataclass
class SGDConfig(OptimizerConfig):
    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class AdamWConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    pass


@dataclass
class CosineScheduler(SchedulerConfig):
    T_max: int = 100
    eta_min: float = 0.0


@dataclass
class TrainingConfig:
    experiment_name: str
    output_dir: Path
    data: DataConfig
    augmentation: AugmentationConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    num_epochs: int
    batch_size: int
    seed: int = 42
    gradient_clip: Optional[float] = None
    eval_frequency: int = 1
    save_frequency: int = 10
    log_frequency: int = 100
    wandb_project: Optional[str] = None


c = Confify(TrainingConfig)
assert_type(c, Confify[TrainingConfig])


def check_types(_: TrainingConfig) -> None:
    # --- SetType/As.then() narrowing with subclass-specific fields ---
    result_resnet = SetType(_.model)(
        As(ResNetConfig).then(
            lambda m: [
                Set(m.num_classes).to(10),
                Set(m.depth).to(18),
                Set(m.pretrained).to(False),
                Set(m.width_multiplier).to(1.0),
            ]
        )
    )
    assert_type(result_resnet, SetTypeRecord[ModelConfig] | SetTypeRecordWithStatements[ModelConfig])

    result_vit = SetType(_.model)(
        As(ViTConfig).then(
            lambda m: [
                Set(m.num_classes).to(1000),
                Set(m.patch_size).to(16),
                Set(m.embed_dim).to(384),
                Set(m.num_heads).to(6),
                Set(m.num_layers).to(12),
            ]
        )
    )
    assert_type(result_vit, SetTypeRecord[ModelConfig] | SetTypeRecordWithStatements[ModelConfig])

    # --- Optimizer hierarchy ---
    SetType(_.optimizer)(
        As(SGDConfig).then(
            lambda opt: [
                Set(opt.weight_decay).to(5e-4),
                Set(opt.momentum).to(0.9),
                Set(opt.learning_rate).to(0.1),
            ]
        )
    )

    SetType(_.optimizer)(
        As(AdamWConfig).then(
            lambda opt: [
                Set(opt.weight_decay).to(0.01),
                Set(opt.beta1).to(0.9),
                Set(opt.beta2).to(0.999),
                Set(opt.learning_rate).to(0.001),
            ]
        )
    )

    # --- Optional[Path] overload ---
    assert_type(Set(_.output_dir).to(Path("./out")), SetRecord[Path])
    assert_type(Set(_.output_dir).to(L("./out/{name}")), SetRecord[Path])
    assert_type(Set(_.output_dir).to("./out"), SetRecord[Path])

    # --- Optional[float] field ---
    assert_type(Set(_.gradient_clip).to(1.0), SetRecord[float | None])
    assert_type(Set(_.gradient_clip).to(None), SetRecord[float | None])

    # --- Optional[str] field ---
    assert_type(Set(_.wandb_project).to("my-project"), SetRecord[str | None])
    assert_type(Set(_.wandb_project).to(None), SetRecord[str | None])

    # --- Scheduler as direct value ---
    assert_type(Set(_.scheduler).to(CosineScheduler(T_max=200)), SetRecord[SchedulerConfig])

    # --- As[T] for the full hierarchy ---
    assert_type(As(ResNetConfig), As[ResNetConfig])
    assert_type(As(ViTConfig), As[ViTConfig])
    assert_type(As(SGDConfig), As[SGDConfig])
    assert_type(As(AdamWConfig), As[AdamWConfig])
    assert_type(
        As(ResNetConfig).then(lambda m: [Set(m.depth).to(18)]),
        AsWithStatements[ResNetConfig],
    )

    # --- Negative: ResNet doesn't have ViT fields ---
    As(ResNetConfig).then(
        lambda m: [
            Set(m.embed_dim).to(768),  # type: ignore  -- ResNetConfig has no embed_dim
        ]
    )

    # --- Negative: wrong type for float field ---
    Set(_.gradient_clip).to("fast")  # type: ignore  -- str into Optional[float]

    # --- Negative: ViTConfig doesn't have depth (Literal) ---
    As(ViTConfig).then(
        lambda m: [
            Set(m.width_multiplier).to(1.0),  # type: ignore  -- ViTConfig has no width_multiplier
        ]
    )
