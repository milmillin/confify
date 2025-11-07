"""
Machine Learning Training Configuration Example

A realistic example demonstrating how to use Confify for ML experiments.
Shows:
- Polymorphic model configurations using SetType
- Optimizer and scheduler configurations
- Data augmentation settings
- Comprehensive hyperparameter sweeps
- Production-ready configuration management
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L


# ============================================================================
# Data Configuration
# ============================================================================


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


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class ModelConfig:
    """Base model configuration."""

    num_classes: int
    pretrained: bool = False


@dataclass
class ResNetConfig(ModelConfig):
    """ResNet model configuration."""

    depth: Literal[18, 34, 50, 101, 152] = 18
    width_multiplier: float = 1.0


@dataclass
class ViTConfig(ModelConfig):
    """Vision Transformer configuration."""

    patch_size: int = 16
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0


@dataclass
class EfficientNetConfig(ModelConfig):
    """EfficientNet configuration."""

    variant: Literal["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"] = "b0"
    dropout_rate: float = 0.2


# ============================================================================
# Optimizer Configuration
# ============================================================================


@dataclass
class OptimizerConfig:
    """Base optimizer configuration."""

    learning_rate: float
    weight_decay: float = 0.0


@dataclass
class SGDConfig(OptimizerConfig):
    """SGD optimizer."""

    momentum: float = 0.9
    nesterov: bool = False


@dataclass
class AdamConfig(OptimizerConfig):
    """Adam optimizer."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class AdamWConfig(OptimizerConfig):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


# ============================================================================
# Scheduler Configuration
# ============================================================================


@dataclass
class SchedulerConfig:
    """Base scheduler configuration."""

    pass


@dataclass
class CosineScheduler(SchedulerConfig):
    """Cosine annealing scheduler."""

    T_max: int = 100
    eta_min: float = 0.0


@dataclass
class StepScheduler(SchedulerConfig):
    """Step decay scheduler."""

    step_size: int = 30
    gamma: float = 0.1


@dataclass
class MultiStepScheduler(SchedulerConfig):
    """Multi-step decay scheduler."""

    milestones: tuple[int, ...] = (30, 60, 90)
    gamma: float = 0.1


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    # Experiment metadata
    experiment_name: str
    output_dir: Path

    # Data
    data: DataConfig
    augmentation: AugmentationConfig

    # Model
    model: ModelConfig

    # Optimization
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    # Training
    num_epochs: int
    batch_size: int

    # Optional fields with defaults
    seed: int = 42
    gradient_clip: Optional[float] = None
    eval_frequency: int = 1
    save_frequency: int = 10
    log_frequency: int = 100
    wandb_project: Optional[str] = None


# ============================================================================
# Confify Setup and Generators
# ============================================================================

c = Confify(TrainingConfig)


@c.generator("resnet_baseline")
def resnet_baseline(_: TrainingConfig) -> ConfigStatements:
    """
    ResNet baseline experiments on CIFAR-10.

    Generates 6 configs:
    - ResNet-18/50 × SGD/AdamW × 2 learning rates each
    """
    return [
        # Metadata
        Set(_.experiment_name).to(L("resnet_baseline_{name}")),
        Set(_.output_dir).to(L("./checkpoints/{name}")),
        Set(_.seed).to(42),
        # Data
        Set(_.data).to(
            DataConfig(
                dataset="cifar10",
                data_root=Path("/data/cifar10"),
                num_workers=8,
            )
        ),
        Set(_.augmentation).to(
            AugmentationConfig(
                random_crop=True,
                random_flip=True,
                color_jitter=0.1,
            )
        ),
        # Training
        Set(_.num_epochs).to(200),
        Set(_.batch_size).to(128),
        Set(_.gradient_clip).to(1.0),
        # Evaluation
        Set(_.eval_frequency).to(1),
        Set(_.save_frequency).to(50),
        # Logging
        Set(_.log_frequency).to(50),
        Set(_.wandb_project).to("cifar10-baselines"),
        # Scheduler
        Set(_.scheduler).to(CosineScheduler(T_max=200, eta_min=1e-6)),
        # Model sweep
        Sweep(
            _resnet18=[
                SetType(_.model)(
                    As(ResNetConfig).then(
                        lambda m: [
                            Set(m.num_classes).to(10),
                            Set(m.depth).to(18),
                            Set(m.pretrained).to(False),
                        ]
                    )
                )
            ],
            _resnet50=[
                SetType(_.model)(
                    As(ResNetConfig).then(
                        lambda m: [
                            Set(m.num_classes).to(10),
                            Set(m.depth).to(50),
                            Set(m.pretrained).to(False),
                        ]
                    )
                )
            ],
        ),
        # Optimizer sweep
        Sweep(
            _sgd=[
                SetType(_.optimizer)(
                    As(SGDConfig).then(
                        lambda opt: [
                            Set(opt.weight_decay).to(5e-4),
                            Set(opt.momentum).to(0.9),
                            Sweep(
                                _lr1=[Set(opt.learning_rate).to(0.1)],
                                _lr2=[Set(opt.learning_rate).to(0.05)],
                            ),
                        ]
                    )
                )
            ],
            _adamw=[
                SetType(_.optimizer)(
                    As(AdamWConfig).then(
                        lambda opt: [
                            Set(opt.weight_decay).to(0.01),
                            Sweep(
                                _lr1=[Set(opt.learning_rate).to(0.001)],
                                _lr2=[Set(opt.learning_rate).to(0.0005)],
                            ),
                        ]
                    )
                )
            ],
        ),
    ]


@c.generator("vit_imagenet")
def vit_imagenet(_: TrainingConfig) -> ConfigStatements:
    """
    Vision Transformer experiments on ImageNet.

    Generates 4 configs:
    - ViT-Small/Base × 2 learning rates each
    """
    return [
        # Metadata
        Set(_.experiment_name).to(L("vit_{name}")),
        Set(_.output_dir).to(L("./checkpoints/imagenet/{name}")),
        Set(_.seed).to(42),
        # Data
        Set(_.data).to(
            DataConfig(
                dataset="imagenet",
                data_root=Path("/data/imagenet"),
                num_workers=16,
                pin_memory=True,
            )
        ),
        Set(_.augmentation).to(
            AugmentationConfig(
                random_crop=True,
                random_flip=True,
                auto_augment="rand-m9-mstd0.5-inc1",
            )
        ),
        # Training
        Set(_.num_epochs).to(300),
        Set(_.batch_size).to(256),
        Set(_.gradient_clip).to(1.0),
        # Evaluation
        Set(_.eval_frequency).to(5),
        Set(_.save_frequency).to(25),
        # Logging
        Set(_.log_frequency).to(100),
        Set(_.wandb_project).to("imagenet-vit"),
        # Optimizer
        SetType(_.optimizer)(
            As(AdamWConfig).then(
                lambda opt: [
                    Set(opt.weight_decay).to(0.05),
                    Set(opt.beta1).to(0.9),
                    Set(opt.beta2).to(0.999),
                    Sweep(
                        _lr1=[Set(opt.learning_rate).to(0.001)],
                        _lr2=[Set(opt.learning_rate).to(0.0005)],
                    ),
                ]
            )
        ),
        # Scheduler
        Set(_.scheduler).to(CosineScheduler(T_max=300, eta_min=1e-5)),
        # Model sweep
        Sweep(
            _small=[
                SetType(_.model)(
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
            ],
            _base=[
                SetType(_.model)(
                    As(ViTConfig).then(
                        lambda m: [
                            Set(m.num_classes).to(1000),
                            Set(m.patch_size).to(16),
                            Set(m.embed_dim).to(768),
                            Set(m.num_heads).to(12),
                            Set(m.num_layers).to(12),
                        ]
                    )
                )
            ],
        ),
    ]


@c.generator("ablation")
def ablation_study(_: TrainingConfig) -> ConfigStatements:
    """
    Ablation study for data augmentation on CIFAR-10.

    Generates 4 configs testing different augmentation strategies.
    """
    return [
        # Metadata
        Set(_.experiment_name).to(L("ablation_{name}")),
        Set(_.output_dir).to(L("./checkpoints/ablation/{name}")),
        Set(_.seed).to(42),
        # Data
        Set(_.data).to(
            DataConfig(
                dataset="cifar10",
                data_root=Path("/data/cifar10"),
                num_workers=8,
            )
        ),
        # Model (fixed)
        Set(_.model).to(
            ResNetConfig(
                num_classes=10,
                depth=18,
                pretrained=False,
            )
        ),
        # Optimizer (fixed)
        Set(_.optimizer).to(
            SGDConfig(
                learning_rate=0.1,
                momentum=0.9,
                weight_decay=5e-4,
            )
        ),
        # Scheduler
        Set(_.scheduler).to(CosineScheduler(T_max=200)),
        # Training
        Set(_.num_epochs).to(200),
        Set(_.batch_size).to(128),
        # Logging
        Set(_.log_frequency).to(50),
        Set(_.wandb_project).to("cifar10-ablation"),
        # Augmentation sweep
        Sweep(
            _none=[
                Set(_.augmentation).to(
                    AugmentationConfig(
                        random_crop=False,
                        random_flip=False,
                        color_jitter=0.0,
                    )
                )
            ],
            _basic=[
                Set(_.augmentation).to(
                    AugmentationConfig(
                        random_crop=True,
                        random_flip=True,
                        color_jitter=0.0,
                    )
                )
            ],
            _colorjitter=[
                Set(_.augmentation).to(
                    AugmentationConfig(
                        random_crop=True,
                        random_flip=True,
                        color_jitter=0.2,
                    )
                )
            ],
            _autoaug=[
                Set(_.augmentation).to(
                    AugmentationConfig(
                        random_crop=True,
                        random_flip=True,
                        auto_augment="cifar10",
                    )
                )
            ],
        ),
    ]


@c.main()
def main(config: TrainingConfig):
    """Main training function."""
    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print("=" * 80)

    print(f"\nData:")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Root: {config.data.data_root}")
    print(f"  Workers: {config.data.num_workers}")

    print(f"\nAugmentation:")
    print(f"  Random crop: {config.augmentation.random_crop}")
    print(f"  Random flip: {config.augmentation.random_flip}")
    print(f"  Color jitter: {config.augmentation.color_jitter}")
    print(f"  Auto augment: {config.augmentation.auto_augment}")

    print(f"\nModel: {config.model.__class__.__name__}")
    if isinstance(config.model, ResNetConfig):
        print(f"  Depth: {config.model.depth}")
        print(f"  Width multiplier: {config.model.width_multiplier}")
    elif isinstance(config.model, ViTConfig):
        print(f"  Patch size: {config.model.patch_size}")
        print(f"  Embed dim: {config.model.embed_dim}")
        print(f"  Num heads: {config.model.num_heads}")
        print(f"  Num layers: {config.model.num_layers}")

    print(f"\nOptimizer: {config.optimizer.__class__.__name__}")
    print(f"  Learning rate: {config.optimizer.learning_rate}")
    print(f"  Weight decay: {config.optimizer.weight_decay}")
    if isinstance(config.optimizer, SGDConfig):
        print(f"  Momentum: {config.optimizer.momentum}")

    print(f"\nScheduler: {config.scheduler.__class__.__name__}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient clip: {config.gradient_clip}")

    print(f"\nOutput:")
    print(f"  Directory: {config.output_dir}")
    print(f"  WandB project: {config.wandb_project}")

    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    # Your training code would go here
    # train(config)


if __name__ == "__main__":
    main()


# Example usage:
#
# List all configs from a generator:
# python ex_ml_config.py list resnet_baseline
# python ex_ml_config.py list vit_imagenet ablation
#
# Generate shell scripts:
# python ex_ml_config.py generate shell resnet_baseline
# python ex_ml_config.py generate shell vit_imagenet
# python ex_ml_config.py generate shell ablation
#
# Run a specific config:
# python ex_ml_config.py run resnet_baseline_resnet18_sgd_lr1
# python ex_ml_config.py run vit_small_lr1
# python ex_ml_config.py run ablation_autoaug
#
# The generated scripts will be in:
# _generated/ex_ml_config_resnet_baseline/
# _generated/ex_ml_config_vit_imagenet/
# _generated/ex_ml_config_ablation/
