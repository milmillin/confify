"""
Basic Confify CLI Usage Example

This example demonstrates simple usage of Confify's main function
without generators. Perfect for straightforward CLI applications.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from confify import Confify


@dataclass
class TrainingConfig:
    # Data
    data_path: Path
    dataset: Literal["imagenet", "cifar10", "cifar100"]

    # Training hyperparameters
    epochs: int
    batch_size: int
    learning_rate: float

    # Model
    model_name: str

    # Output
    output_dir: Path

    # Optional fields with defaults
    weight_decay: float = 0.0001
    pretrained: bool = False
    experiment_name: Optional[str] = None


# Create Confify instance
c = Confify(TrainingConfig)


@c.main()
def main(config: TrainingConfig):
    """Main training function."""
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)

    print(f"\nData:")
    print(f"  Dataset: {config.dataset}")
    print(f"  Path: {config.data_path}")

    print(f"\nModel:")
    print(f"  Name: {config.model_name}")
    print(f"  Pretrained: {config.pretrained}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")

    print(f"\nOutput:")
    print(f"  Directory: {config.output_dir}")
    print(f"  Experiment: {config.experiment_name or 'default'}")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Your training code would go here
    # train_model(config)


if __name__ == "__main__":
    main()


# Example usage:
#
# python ex_basic.py \
#     --data_path /data/cifar10 \
#     --dataset cifar10 \
#     --epochs 100 \
#     --batch_size 128 \
#     --learning_rate 0.1 \
#     --model_name resnet18 \
#     --pretrained True \
#     --output_dir ./checkpoints \
#     --experiment_name baseline
#
# Or with YAML:
#
# python ex_basic.py ---config.yaml
#
# Where config.yaml contains:
# data_path: /data/cifar10
# dataset: cifar10
# epochs: 100
# batch_size: 128
# learning_rate: 0.1
# model_name: resnet18
# pretrained: true
# output_dir: ./checkpoints
# experiment_name: baseline
