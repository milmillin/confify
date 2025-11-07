# Confify CLI System

The Confify CLI system provides a powerful framework for building command-line applications with configuration management, experiment generation, and sweep capabilities.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Main Function Usage](#main-function-usage)
- [Generator System](#generator-system)
- [DSL Reference](#dsl-reference)
- [L String Format Variables](#l-string-format-variables)
- [CLI Commands](#cli-commands)
- [Exporter System](#exporter-system)
- [Best Practices](#best-practices)

## Overview

The Confify CLI system consists of several key components:

- **`Confify` class**: Orchestrates CLI workflow with decorators for main and generator functions
- **DSL (Domain-Specific Language)**: Provides `Set`, `Sweep`, `SetType`, `As`, and `L` for building configurations
- **Generators**: Functions that produce multiple configuration variants
- **Exporters**: Convert generated configurations to various formats (shell scripts, JSON, etc.)
- **CLI Commands**: Built-in commands for listing, generating, and running configurations

## Getting Started

### Basic Setup

```python
from dataclasses import dataclass
from confify import Confify

@dataclass
class Config:
    learning_rate: float
    batch_size: int
    model_name: str

# Create Confify instance
c = Confify(Config)

@c.main()
def main(config: Config):
    print(f"Training with lr={config.learning_rate}, batch={config.batch_size}")
    # Your training code here

if __name__ == "__main__":
    main()
```

Run your script:

```bash
# Run with CLI arguments
python train.py --learning_rate 0.001 --batch_size 32 --model_name resnet50

# Run with YAML config
python train.py ---config.yaml
```

## Main Function Usage

The `@c.main()` decorator converts your function into a CLI entrypoint that automatically parses configuration from command-line arguments.

### Simple Example

```python
from dataclasses import dataclass
from pathlib import Path
from confify import Confify

@dataclass
class TrainingConfig:
    data_path: Path
    epochs: int
    learning_rate: float
    save_dir: Path

c = Confify(TrainingConfig)

@c.main()
def main(config: TrainingConfig):
    print(f"Loading data from {config.data_path}")
    print(f"Training for {config.epochs} epochs with lr={config.learning_rate}")
    print(f"Saving to {config.save_dir}")
    # Your training logic here

if __name__ == "__main__":
    main()
```

Usage:

```bash
python train.py \
    --data_path /data/imagenet \
    --epochs 100 \
    --learning_rate 0.001 \
    --save_dir ./checkpoints
```

### With Configuration Options

```python
from confify import Confify, ConfifyOptions

# Customize parsing behavior
options = ConfifyOptions(
    prefix="--",
    yaml_prefix="---",
    ignore_extra_fields=True
)

c = Confify(Config, options=options)
```

## Generator System

Generators allow you to programmatically create multiple configuration variants, perfect for hyperparameter sweeps and experiment management.

### Basic Generator

```python
from confify import Confify, ConfigStatements, Set

@dataclass
class Config:
    learning_rate: float
    batch_size: int

c = Confify(Config)

@c.generator()
def experiments(_: Config) -> ConfigStatements:
    return [
        Set(_.learning_rate).to(0.001),
        Set(_.batch_size).to(32),
    ]

@c.main()
def main(config: Config):
    print(config)

if __name__ == "__main__":
    main()
```

Usage:

```bash
# List generated configs
python train.py list experiments

# Generate shell scripts
python train.py generate shell experiments

# Run a specific generated config
python train.py run experiments
```

### Named Generators

You can register multiple generators with custom names:

```python
@c.generator("lr_sweep")
def learning_rate_sweep(_: Config) -> ConfigStatements:
    return [
        Sweep(
            _small=[Set(_.learning_rate).to(0.0001)],
            _medium=[Set(_.learning_rate).to(0.001)],
            _large=[Set(_.learning_rate).to(0.01)],
        )
    ]

@c.generator("batch_sweep")
def batch_size_sweep(_: Config) -> ConfigStatements:
    return [
        Sweep(
            _bs16=[Set(_.batch_size).to(16)],
            _bs32=[Set(_.batch_size).to(32)],
            _bs64=[Set(_.batch_size).to(64)],
        )
    ]
```

## DSL Reference

The Confify DSL provides a type-safe way to build configurations programmatically.

### Set

Set field values in your configuration.

```python
# Set a simple value
Set(_.learning_rate).to(0.001)

# Set a complex value
Set(_.hidden_dims).to((128, 256, 512))

# Load from YAML file
Set(_.encoder).from_yaml(Path("encoder_config.yaml"))
```

### L (Template Strings)

Create template strings with placeholders that get filled during generation.

```python
# Use {name} placeholder for the configuration name
Set(_.experiment_name).to(L("exp_{name}"))
Set(_.save_path).to(L("/checkpoints/{name}"))

# Results in configs like:
# - experiment_name: "exp_lr_sweep_small"
# - save_path: "/checkpoints/lr_sweep_small"
```

### Sweep

Create multiple configuration variants from parameter grids.

```python
# Simple sweep
Sweep(
    _variant1=[Set(_.param).to(value1)],
    _variant2=[Set(_.param).to(value2)],
)

# Nested sweeps (creates cartesian product)
[
    Set(_.model).to("resnet"),
    Sweep(
        _small=[Set(_.depth).to(18)],
        _large=[Set(_.depth).to(50)],
    ),
    Sweep(
        _sgd=[Set(_.optimizer).to("sgd")],
        _adam=[Set(_.optimizer).to("adam")],
    ),
]
# Generates: resnet_small_sgd, resnet_small_adam, resnet_large_sgd, resnet_large_adam
```

**Anonymous Sweep:**

You can use an anonymous sweep by passing it as the first positional argument:

```python
Sweep(
    [Set(_.learning_rate).to(0.001)],  # First positional arg (anonymous)
    _v2=[Set(_.learning_rate).to(0.01)],  # Named variant
)
# Generates: <base_name>, <base_name>_v2
```

### SetType

Set polymorphic types using dataclass subclassing.

```python
from confify import SetType, As

@dataclass
class Encoder:
    pass

@dataclass
class ConvEncoder(Encoder):
    num_layers: int

@dataclass
class TransformerEncoder(Encoder):
    num_heads: int

@dataclass
class Config:
    encoder: Encoder

# Basic type setting
SetType(_.encoder)(As(ConvEncoder))

# With nested configuration
SetType(_.encoder)(
    As(ConvEncoder).then(lambda e: [
        Set(e.num_layers).to(6),
    ])
)

# Alternative: direct object assignment
Set(_.encoder).to(ConvEncoder(num_layers=6))
```

### As

Specify type for `SetType` operations.

```python
# Simple type specification
As(ConvEncoder)

# With nested statements
As(ConvEncoder).then(lambda e: [
    Set(e.num_layers).to(6),
    Sweep(
        _small=[Set(e.depth).to(2)],
        _large=[Set(e.depth).to(4)],
    ),
])
```

### Complete Example

```python
from dataclasses import dataclass
from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L

@dataclass
class Optimizer:
    learning_rate: float

@dataclass
class SGD(Optimizer):
    momentum: float

@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999

@dataclass
class Config:
    experiment_name: str
    batch_size: int
    optimizer: Optimizer

c = Confify(Config)

@c.generator()
def sweep(_: Config) -> ConfigStatements:
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Sweep(
            _bs32=[Set(_.batch_size).to(32)],
            _bs64=[Set(_.batch_size).to(64)],
        ),
        Sweep(
            _sgd=[
                SetType(_.optimizer)(
                    As(SGD).then(lambda opt: [
                        Set(opt.learning_rate).to(0.01),
                        Set(opt.momentum).to(0.9),
                    ])
                )
            ],
            _adam=[
                SetType(_.optimizer)(
                    As(Adam).then(lambda opt: [
                        Set(opt.learning_rate).to(0.001),
                    ])
                )
            ],
        ),
    ]

@c.main()
def main(config: Config):
    print(config)

if __name__ == "__main__":
    main()
```

This generates 4 configurations:
- `sweep_bs32_sgd`
- `sweep_bs32_adam`
- `sweep_bs64_sgd`
- `sweep_bs64_adam`

## L String Format Variables

Template strings created with `L()` support variable interpolation using the format `{variable_name}`. These variables are automatically populated by Confify and are available in both configuration statements and custom exporters.

### Available Variables

#### Core Variables

- **`{name}`**: Full name of the generated configuration
  - Example: `"experiments_resnet_bs32_lr1"`
  - Most commonly used for unique experiment names and output paths

- **`{generator_name}`**: Name of the generator function
  - Example: `"experiments"`, `"ablation_study"`
  - Useful for organizing outputs by generator

- **`{script_name}`**: Filename (without extension) of the script
  - Example: `"train"` from `train.py`
  - Useful for creating consistent directory structures

#### Datetime Variables

All datetime variables are populated when the generator runs:

- **`{Y}`**: Year (4 digits)
  - Example: `"2025"`

- **`{m}`**: Month (2 digits, zero-padded)
  - Example: `"01"` for January, `"12"` for December

- **`{d}`**: Day of month (2 digits, zero-padded)
  - Example: `"05"`, `"23"`

- **`{H}`**: Hour (2 digits, 24-hour format, zero-padded)
  - Example: `"09"`, `"14"`, `"23"`

- **`{M}`**: Minute (2 digits, zero-padded)
  - Example: `"05"`, `"30"`

- **`{s}`**: Second (2 digits, zero-padded)
  - Example: `"07"`, `"42"`

### Usage Examples

#### Basic Usage

```python
# Unique experiment names
Set(_.experiment_name).to(L("exp_{name}"))
# Result: "exp_experiments_resnet_bs32_lr1"

# Output directories
Set(_.output_dir).to(L("/checkpoints/{name}"))
# Result: "/checkpoints/experiments_resnet_bs32_lr1"
```

#### With Generator Name

```python
# Organize by generator
Set(_.save_path).to(L("/results/{generator_name}/{name}"))
# Result: "/results/experiments/experiments_resnet_bs32_lr1"
```

#### With Script Name

```python
# Script-specific paths
Set(_.log_dir).to(L("/logs/{script_name}/{name}"))
# Result: "/logs/train/experiments_resnet_bs32_lr1"
```

#### Timestamped Outputs

```python
# Daily experiment directories
Set(_.output_dir).to(L("/experiments/{Y}-{m}-{d}/{name}"))
# Result: "/experiments/2025-01-15/experiments_resnet_bs32_lr1"

# Full timestamp
Set(_.run_id).to(L("{Y}{m}{d}_{H}{M}{s}_{name}"))
# Result: "20250115_143052_experiments_resnet_bs32_lr1"

# Organized by date hierarchy
Set(_.checkpoint_dir).to(L("/checkpoints/{Y}/{m}/{d}/{name}"))
# Result: "/checkpoints/2025/01/15/experiments_resnet_bs32_lr1"
```

#### Complex Paths

```python
# Combining multiple variables
Set(_.wandb_run_name).to(L("{script_name}_{generator_name}_{name}"))
# Result: "train_experiments_experiments_resnet_bs32_lr1"

Set(_.output_dir).to(L("/data/{script_name}/{Y}-{m}/{name}"))
# Result: "/data/train/2025-01/experiments_resnet_bs32_lr1"
```

### Custom Variables in Exporters

When creating custom exporters, you can add your own variables by returning them from `pre_run()`:

```python
class MyExporter(ConfifyExporter):
    def pre_run(self, lstr_kwargs: dict[str, str]) -> dict[str, str]:
        # Add custom variables
        return {
            "user": "john",
            "project": "vision",
            "version": "v1.0",
        }

    def run(self, args: list[str], lstr_kwargs: dict[str, str]):
        # Now {user}, {project}, {version} are available
        # along with all standard variables
        output_path = "{project}/{user}/{name}".format(**lstr_kwargs)
        # Result: "vision/john/experiments_resnet_bs32_lr1"
```

### Best Practices

1. **Always use `{name}` for uniqueness**: Ensures each generated config has a unique identifier
   ```python
   Set(_.experiment_name).to(L("exp_{name}"))
   ```

2. **Use timestamps for versioning**: Helpful for tracking when experiments were configured
   ```python
   Set(_.version).to(L("{Y}{m}{d}"))
   ```

3. **Create hierarchical paths**: Makes outputs easier to organize and find
   ```python
   Set(_.output_dir).to(L("/results/{generator_name}/{Y}-{m}/{name}"))
   ```

4. **Avoid redundant generator names**: Since `{name}` already includes the generator name prefix
   ```python
   # Good
   Set(_.path).to(L("/data/{name}"))

   # Redundant
   Set(_.path).to(L("/data/{generator_name}_{name}"))
   ```

## CLI Commands

The Confify CLI supports several built-in commands:

### Default (No Command)

Run the main function with parsed configuration from CLI arguments.

```bash
python script.py --param1 value1 --param2 value2
```

### list / ls / l

List all configuration names from specified generators.

```bash
# List configs from one generator
python script.py list experiments

# List configs from multiple generators
python script.py list exp1 exp2 exp3
```

Output:
```
exp1_variant1
exp1_variant2
exp2_config1
exp2_config2
```

### generate / gen / g

Generate configuration files using an exporter.

```bash
# Generate shell scripts for all configs in a generator
python script.py generate shell experiments
```

The `shell` exporter creates executable bash scripts in `_generated/{script}_{generator_name}/` directory:

```bash
_generated/
└── train_experiments/
    ├── experiments_variant1.sh
    └── experiments_variant2.sh
```

Each generated script contains the CLI arguments to reproduce that configuration:

```bash
#!/bin/bash

python train.py \
    --learning_rate 0.001 \
    --batch_size 32 \
    --model_name resnet50
```

### run / r

Run the main function with a specific generated configuration.

```bash
# Run a specific config by name
python script.py run experiments_variant1
```

This is equivalent to running the generated shell script, but executes directly without creating a file.

## Exporter System

Exporters convert generated configurations into various formats. Confify includes a built-in `ShellExporter` and supports custom exporters.

### Built-in ShellExporter

The `ShellExporter` creates executable bash scripts for each configuration.

```python
c = Confify(Config)

# ShellExporter is registered by default as "shell"
# No additional setup needed
```

Generated scripts:

```bash
#!/bin/bash

python your_script.py \
    --param1 value1 \
    --param2 value2 \
    --param3 value3
```

### Custom Exporter

Create custom exporters by subclassing `ConfifyExporter`:

```python
from confify.cli import ConfifyExporter, ConfifyExporterConfig
from pathlib import Path
from typing import ClassVar
import json

class JsonExporter(ConfifyExporter):
    config: ClassVar[ConfifyExporterConfig] = ConfifyExporterConfig(shell_escape=False)

    def pre_run(self, lstr_kwargs: dict[str, str]) -> dict[str, str]:
        """Called once before generating all configs"""
        generator_name = lstr_kwargs["generator_name"]
        script_name = lstr_kwargs["script_name"]

        output_dir = Path("configs") / f"{script_name}_{generator_name}"
        output_dir.mkdir(exist_ok=True, parents=True)
        return {"output_dir": str(output_dir)}

    def run(self, args: list[str], lstr_kwargs: dict[str, str]):
        """Called for each generated config"""
        config_name = lstr_kwargs["name"]

        # Convert args to dict
        config_dict = {}
        for i in range(0, len(args), 2):
            key = args[i].replace("--", "")
            value = args[i + 1]
            config_dict[key] = value

        # Write JSON file
        output_dir = Path(lstr_kwargs["output_dir"])
        output_file = output_dir / f"{config_name}.json"
        output_file.write_text(json.dumps(config_dict, indent=2))
        print(f"Generated {output_file}")

    def post_run(self, lstr_kwargs: dict[str, str]):
        """Called once after generating all configs"""
        generator_name = lstr_kwargs["generator_name"]
        print(f"Finished generating configs for {generator_name}")

# Register custom exporter
c = Confify(Config)
c.register_exporter("json", JsonExporter())

# Use it
# python script.py generate json experiments
```

### Exporter Methods

All exporter methods receive `lstr_kwargs`, a dictionary containing format variables (see [L String Format Variables](#l-string-format-variables) for full list):

- **`pre_run(lstr_kwargs: dict[str, str]) -> dict[str, str]`**: Called once before generating all configs.
  - **Receives**: `lstr_kwargs` with `script_name`, `generator_name`, and datetime variables
  - **Returns**: Dict of additional variables to add to `lstr_kwargs` for subsequent calls
  - **Use for**: Creating directories, initializing resources, adding custom format variables

- **`run(args: list[str], lstr_kwargs: dict[str, str])`**: Called once for each generated config.
  - **Parameters**:
    - `args`: List of CLI arguments in alternating key-value pairs (e.g., `["--lr", "0.001", "--batch", "32"]`)
    - `lstr_kwargs`: All format variables including `name` (the full config name), custom variables from `pre_run()`, and standard variables
  - **Use for**: Exporting individual configs to files, databases, APIs, etc.

- **`post_run(lstr_kwargs: dict[str, str])`**: Called once after all configs are generated.
  - **Receives**: `lstr_kwargs` with all variables
  - **Use for**: Cleanup, writing summary files, finalizing resources

**Accessing Variables:**

All context information is available through `lstr_kwargs`:

```python
def run(self, args: list[str], lstr_kwargs: dict[str, str]):
    # Standard variables
    config_name = lstr_kwargs["name"]              # Full config name
    generator_name = lstr_kwargs["generator_name"] # Generator function name
    script_name = lstr_kwargs["script_name"]       # Script filename (no ext)

    # Datetime variables
    timestamp = f"{lstr_kwargs['Y']}{lstr_kwargs['m']}{lstr_kwargs['d']}"

    # Custom variables from pre_run()
    output_dir = lstr_kwargs["output_dir"]
```

### Exporter Configuration

Control exporter behavior with `ConfifyExporterConfig`:

```python
class MyExporter(ConfifyExporter):
    config = ConfifyExporterConfig(
        shell_escape=False  # Disable shell escaping for args
    )
```

Options:
- **`shell_escape`** (default: `True`): Whether to escape arguments for shell safety

## Best Practices

### 1. Use Template Strings for Unique Names

```python
Set(_.experiment_name).to(L("exp_{name}"))
Set(_.output_dir).to(L("/results/{name}"))
```

This ensures each generated config has a unique, descriptive name.

### 2. Organize Sweeps Hierarchically

```python
[
    # Fixed parameters first
    Set(_.dataset).to("imagenet"),

    # Model architecture sweep
    Sweep(
        _resnet=[Set(_.model).to("resnet")],
        _vit=[Set(_.model).to("vit")],
    ),

    # Hyperparameter sweep (applies to all models)
    Sweep(
        _lr1=[Set(_.learning_rate).to(0.001)],
        _lr2=[Set(_.learning_rate).to(0.01)],
    ),
]
```

### 3. Use Descriptive Sweep Names

```python
# Good: Clear what the variant is
Sweep(
    _lr_small=[Set(_.learning_rate).to(0.0001)],
    _lr_medium=[Set(_.learning_rate).to(0.001)],
    _lr_large=[Set(_.learning_rate).to(0.01)],
)

# Avoid: Unclear variant names
Sweep(
    _v1=[Set(_.learning_rate).to(0.0001)],
    _v2=[Set(_.learning_rate).to(0.001)],
    _v3=[Set(_.learning_rate).to(0.01)],
)
```

### 4. Validate Configs in Main Function

```python
@c.main()
def main(config: Config):
    # Validate config before expensive operations
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")

    if not config.data_path.exists():
        raise FileNotFoundError(f"Data path not found: {config.data_path}")

    # Your main logic here
```

### 5. Use Multiple Generators for Different Experiment Types

```python
@c.generator("ablation")
def ablation_study(_: Config) -> ConfigStatements:
    # Systematic component ablations
    return [...]

@c.generator("hyperparameter")
def hyperparameter_search(_: Config) -> ConfigStatements:
    # Broad hyperparameter sweep
    return [...]

@c.generator("final")
def final_runs(_: Config) -> ConfigStatements:
    # Final evaluation configs
    return [...]
```

### 6. Combine Direct Assignment with SetType

```python
# When config is simple, use direct assignment
Set(_.optimizer).to(Adam(learning_rate=0.001))

# When config needs sweeps, use SetType
SetType(_.optimizer)(
    As(Adam).then(lambda opt: [
        Sweep(
            _lr1=[Set(opt.learning_rate).to(0.0001)],
            _lr2=[Set(opt.learning_rate).to(0.001)],
        ),
    ])
)
```

### 7. Test Generators Before Full Generation

```bash
# First, list configs to verify names
python script.py list experiments

# Then generate to a test directory
python script.py generate shell experiments

# Review generated scripts before running
cat _generated/script_experiments/experiments_variant1.sh
```

### 8. Use YAML for Complex Nested Configs

```python
# For deeply nested or large configs, use YAML
Set(_.encoder).from_yaml(Path("configs/encoder_large.yaml"))

# Instead of verbose inline config
Set(_.encoder).to(ComplexEncoder(
    num_layers=12,
    hidden_dim=768,
    num_heads=12,
    # ... many more params
))
```

### 9. Document Your Generators

```python
@c.generator("ablation")
def ablation_study(_: Config) -> ConfigStatements:
    """
    Ablation study for model components.

    Tests removing each component individually:
    - attention mechanism
    - residual connections
    - layer normalization

    Generates 4 configs: baseline + 3 ablations
    """
    return [...]
```

### 10. Version Control Generated Scripts

Consider committing generated scripts to track experiment configurations:

```bash
# Generate scripts
python train.py generate shell experiments

# Review and commit
git add _generated/
git commit -m "Add experiment configs for v1.2"
```

This creates a permanent record of exact configurations used in experiments.
