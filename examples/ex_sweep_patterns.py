"""
Sweep Patterns Example

This example demonstrates different ways to use Sweep for creating
configuration variants. Shows simple sweeps, nested sweeps, and
anonymous sweeps.
"""

from dataclasses import dataclass
from confify import Confify, ConfigStatements, Set, Sweep, L


@dataclass
class Config:
    experiment_name: str
    learning_rate: float
    batch_size: int
    optimizer: str
    dropout: float
    num_layers: int


c = Confify(Config)


@c.generator("simple_sweep")
def simple_sweep(_: Config) -> ConfigStatements:
    """
    Simple sweep: One parameter varies.

    Generates 3 configs:
    - simple_sweep_lr_small
    - simple_sweep_lr_medium
    - simple_sweep_lr_large
    """
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.batch_size).to(128),
        Set(_.optimizer).to("adam"),
        Set(_.dropout).to(0.1),
        Set(_.num_layers).to(6),
        Sweep(
            _lr_small=[Set(_.learning_rate).to(0.0001)],
            _lr_medium=[Set(_.learning_rate).to(0.001)],
            _lr_large=[Set(_.learning_rate).to(0.01)],
        ),
    ]


@c.generator("grid_sweep")
def grid_sweep(_: Config) -> ConfigStatements:
    """
    Grid sweep: Multiple parameters vary (cartesian product).

    Generates 6 configs:
    - grid_sweep_lr_small_bs64
    - grid_sweep_lr_small_bs128
    - grid_sweep_lr_medium_bs64
    - grid_sweep_lr_medium_bs128
    - grid_sweep_lr_large_bs64
    - grid_sweep_lr_large_bs128
    """
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.optimizer).to("adam"),
        Set(_.dropout).to(0.1),
        Set(_.num_layers).to(6),
        Sweep(
            _lr_small=[Set(_.learning_rate).to(0.0001)],
            _lr_medium=[Set(_.learning_rate).to(0.001)],
            _lr_large=[Set(_.learning_rate).to(0.01)],
        ),
        Sweep(
            _bs64=[Set(_.batch_size).to(64)],
            _bs128=[Set(_.batch_size).to(128)],
        ),
    ]


@c.generator("nested_sweep")
def nested_sweep(_: Config) -> ConfigStatements:
    """
    Nested sweep: Different sweeps for different variants.

    Generates 4 configs:
    - nested_sweep_adam_lr1
    - nested_sweep_adam_lr2
    - nested_sweep_sgd_lr1
    - nested_sweep_sgd_lr2

    Note: Different optimizers get different learning rates
    """
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.batch_size).to(128),
        Set(_.dropout).to(0.1),
        Set(_.num_layers).to(6),
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


@c.generator("anonymous_sweep")
def anonymous_sweep(_: Config) -> ConfigStatements:
    """
    Anonymous sweep: Unnamed variants using positional argument.

    Generates 3 configs:
    - anonymous_sweep (from the positional argument)
    - anonymous_sweep_variant2
    - anonymous_sweep_variant3

    Useful when you want a "default" variant without a suffix.
    """
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.batch_size).to(128),
        Set(_.optimizer).to("adam"),
        Set(_.dropout).to(0.1),
        Set(_.num_layers).to(6),
        Sweep(
            # First positional argument (anonymous)
            [Set(_.learning_rate).to(0.001)],
            # Named variants
            _variant2=[Set(_.learning_rate).to(0.01)],
            _variant3=[Set(_.learning_rate).to(0.1)],
        ),
    ]


@c.generator("complex_sweep")
def complex_sweep(_: Config) -> ConfigStatements:
    """
    Complex sweep: Combines multiple sweep patterns.

    Demonstrates:
    - Multiple independent sweeps (cartesian product)
    - Sweeps with multiple statements per variant
    - Realistic hyperparameter search

    Generates 12 configs (2 optimizers × 2 architectures × 3 learning rates)
    """
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        # Optimizer sweep
        Sweep(
            _adam=[Set(_.optimizer).to("adam")],
            _sgd=[Set(_.optimizer).to("sgd")],
        ),
        # Architecture sweep (multiple parameters together)
        Sweep(
            _small=[
                Set(_.num_layers).to(4),
                Set(_.dropout).to(0.1),
            ],
            _large=[
                Set(_.num_layers).to(12),
                Set(_.dropout).to(0.2),
            ],
        ),
        # Learning rate sweep
        Sweep(
            _lr1=[Set(_.learning_rate).to(0.0001)],
            _lr2=[Set(_.learning_rate).to(0.001)],
            _lr3=[Set(_.learning_rate).to(0.01)],
        ),
        # Batch size is fixed
        Set(_.batch_size).to(128),
    ]


@c.main()
def main(config: Config):
    """Print the configuration."""
    print(f"Experiment: {config.experiment_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Dropout: {config.dropout}")
    print(f"Num layers: {config.num_layers}")


if __name__ == "__main__":
    main()


# Example usage:
#
# List all configs from a generator:
# python ex_sweep_patterns.py list simple_sweep
# python ex_sweep_patterns.py list grid_sweep
# python ex_sweep_patterns.py list complex_sweep
#
# Generate shell scripts:
# python ex_sweep_patterns.py generate shell simple_sweep
# python ex_sweep_patterns.py generate shell grid_sweep
#
# Run a specific config:
# python ex_sweep_patterns.py run grid_sweep_lr_medium_bs128
#
# The generated scripts will be in:
# _generated/ex_sweep_patterns_simple_sweep/
# _generated/ex_sweep_patterns_grid_sweep/
# etc.
