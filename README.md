# Confify

Confify is a fully typed, plug-and-play configuration library for Python.

**Key features:**

- Uses type annotations from `dataclass` and `TypedDict`.
- Uses dotlist notations for CLI arguments. (e.g., `--encoder.depth 6`, `--model.hidden_dims '(10, 20)'`)
- Loads partial configurations from YAML in CLI arguments. (e.g., `---encoder encoder.yaml`)
- Supports subclassing of `dataclass` by specifying the classname. (e.g., `--encoder.\$type mymodule.MyEncoder`)
- Supports `Optional`, `Union`, `Literal`.
- Has minimal dependencies (only `PyYaml`).
- Supports **static-type-safe** configuration sweeps for hyperparameter search. (see [Configuration Sweeps](#configuration-sweeps))

## Installation

(TODO) Upload to PyPI

```bash
pip install git+https://github.com/milmillin/confify.git
```

## Usage

### Example Usage

```python
# example.py

from dataclasses import dataclass
from confify import Confify, config_dump_yaml

@dataclass
class EncoderConfig:
    depth: int
    ch_mult: tuple[int, ...]
    activation_fn: Literal["relu", "silu", "gelu"]
    augment_type: Optional[Literal["cutmix", "mixup"]] = "cutmix"

@dataclass
class Config:
    save_path: Path
    run_id: Optional[str]
    encoder: EncoderConfig

c = Confify(Config)

@c.main()
def main(config: Config):
    # config is properly typed
    assert reveal_type(config) == Config
    # dumping config to yaml
    config_dump_yaml("config.yaml", config)

if __name__ == "__main__":
    main()
```

### Using the CLI

We use dotlist notations for CLI arguments. See examples below.

```bash
python example.py \
    --encoder.depth 6 \
    --encoder.ch_mult '(3,4)' \
    --encoder.activation_fn silu \
    --encoder.augment_type None \
    --save_path ~/experiments/exp1 \
    --run_id exp1
```

We **do not** support equal signs in dotlist notations (for now). For example, `--encoder.depth=6` will not work.

For advanced CLI features including configuration generators, sweeps, and custom exporters, see the **[CLI Documentation](docs/cli.md)**.

### Loading partial configurations from YAML

We support loading partial configurations from YAML files. This ensures modularity and reusability of configurations.

Suppose you have the following `encoder.yaml`:

```yaml
depth: 6
ch_mult: [3, 4]
activation_fn: silu
augment_type: None
```

You can use a triple dash `---` prefix to load from the YAML file.

```bash
python example.py \
    ---encoder encoder.yaml \
    --save_path ~/experiments/exp1 \
    --run_id exp1
```

You can use multiple YAML files on the same key. For example if you have the following `augment_mixup.yaml`:

```yaml
augment_type: mixup
```

Then you can override some fields by specifying additional YAML file on the same key. Note that the order matters.

```bash
python example.py \
    ---encoder encoder.yaml \
    ---encoder augment_mixup.yaml \
    --save_path ~/experiments/exp1 \
    --run_id exp1
```

Running the above command will result in the following configuration:

```yaml
encoder:
  depth: 6
  ch_mult: [3, 4]
  activation_fn: silu
  augment_type: mixup
save_path: ~/experiments/exp1
run_id: exp1
```

You can load the entire configuration from a YAML file by specifying the key as empty string.

```bash
python example.py --- base_config.yaml
```

### Configuration Options

Confify behavior can be customized using `ConfifyOptions`. You can either pass options to individual functions or set global defaults.

```python
from confify import ConfifyOptions, parse, read_config_from_cli

# Method 1: Pass options to individual functions
options = ConfifyOptions(
    ignore_extra_fields=True,
    strict_subclass_check=True
)
config = parse(data, MyConfig, options=options)

# Method 2: Set global defaults for all subsequent calls
ConfifyOptions.set_default(ConfifyOptions(
    ignore_extra_fields=True,
    strict_subclass_check=False
))
config = read_config_from_cli(MyConfig)  # Uses global defaults
```

#### Option Reference

- **`prefix`** (default: `"--"`): Prefix for CLI arguments. Keys starting with this prefix are treated as configuration fields. Example: `--model.name value`

- **`yaml_prefix`** (default: `"---"`): Prefix for loading YAML files from CLI. Keys starting with this prefix treat the value as a file path to load. Example: `---encoder config.yaml`

- **`type_key`** (default: `"$type"`): Special dictionary key for polymorphic type resolution. When present in input data, specifies the actual class to use (must be a fully qualified name like `module.ClassName`). See [Dataclasses Subclassing](#dataclasses-subclassing) for details.

- **`ignore_extra_fields`** (default: `False`): Controls behavior when extra/unknown fields are present in dataclasses or TypedDict:

  - `False`: Raises an error if extra fields are found (strict validation)
  - `True`: Issues a warning and ignores extra fields (lenient validation)

- **`strict_subclass_check`** (default: `False`): Controls validation when using `type_key` to specify a different class:
  - `False`: Issues a warning if the specified class is not a subclass, but continues parsing
  - `True`: Raises an error if the specified class is not a proper subclass

### Type Resolution

Arguments from CLI are converted to the annotated type using the following rule:

- `int`, `float`, `Path`: we use default constructors.
- `bool`: we convert case-insensitive `true`, `on`, `yes` to `True`, and `false`, `off`, `no` to `False`.
- `str`: if the value is surrounded by quotes, we use `ast.literal_eval` to convert it to a string. Otherwise, we use the value as is.
- `None`: we convert case-insensitive `null`, `~`, `none` to `None`.
- `Enum`: we find the enum entry with the same name.

#### Handling `Union`, `Optional`, `Literal` Type

There may be ambiguity when the type is `Optional`, `Union` or `Literal`. For example, if a user input `null` for type `Union[str, None]`, it can be interpreted as either `None` or a string `"null"`. We follow the following order and returns the first that succeeds:

1. Types not listed below (from left to right)
2. `None`
3. `bool`
4. `int`
5. `float`
6. `str`

So `null` for type `Union[str, None]` will be parsed as `None`. To get a string `"null"`, the user needs to explicitly surround it with quotes (e.g., `python example.py '"null"'`).

#### Handling `Any` Type

When a field has type `Any` or when unparameterized collections are used (see below), confify automatically infers the appropriate type from CLI string inputs using the following order:

1. **Sequences**: `[1,2,3]` → `list`, `(1,2,3)` → `tuple` (recursively inferred)
2. **Quoted strings**: `"abc"` or `'abc'` → `str` (quotes removed via `ast.literal_eval`)
3. **None**: case-insensitive `null`, `~`, `none` → `None`
4. **bool**: case-insensitive `true`, `on`, `yes` → `True`; `false`, `off`, `no` → `False`
5. **int**: numeric values without decimals → `int`
6. **float**: numeric values with decimals or scientific notation → `float`
7. **str**: anything else → `str`

Example:

```python
@dataclass
class Config:
    value: Any
```

```bash
# Inferred as int
python example.py --value 42  # → 42 (int)

# Inferred as list of ints
python example.py --value '[1,2,3]'  # → [1, 2, 3] (list[int])

# Inferred as bool
python example.py --value true  # → True (bool)

# Force string with quotes
python example.py --value '"42"'  # → "42" (str)
```

#### Unparameterized Collection Types

Collections without type parameters automatically use `Any` for their elements:

- `list` is equivalent to `list[Any]`
- `tuple` is equivalent to `tuple[Any, ...]` (variable-length)
- `dict` is equivalent to `dict[str, Any]`

Elements in these collections benefit from automatic type inference:

```python
@dataclass
class Config:
    items: list  # unparameterized list
    values: tuple  # unparameterized tuple
```

```bash
# Each element is automatically inferred
python example.py --items '[1,abc,True,null]'  # → [1, "abc", True, None]

# Nested sequences preserve their types
python example.py --items '[1,[2,3],abc]'  # → [1, [2, 3], "abc"]

# Tuples work the same way
python example.py --values '(1,2,abc)'  # → (1, 2, "abc")
```

#### Dataclasses Subclassing

Confify supports polymorphic types through the `type_key` field (default: `"$type"`). This allows you to specify a different class than the one declared in the type annotation, enabling runtime polymorphism.

**How it works:**

1. When parsing a dictionary into a dataclass or TypedDict, confify checks for the `type_key` field
2. If present, the value must be a fully qualified class name (e.g., `"my.module.MyClass"`)
3. The specified class is dynamically loaded and used instead of the annotated type
4. The behavior depends on the `strict_subclass_check` option:
   - **`strict_subclass_check=False`** (default): If the specified class is not a subclass, issues a warning but continues
   - **`strict_subclass_check=True`**: If the specified class is not a subclass, raises an error

**Example:**

```python
from dataclasses import dataclass
from confify import read_config_from_cli, ConfifyOptions

@dataclass
class BaseEncoder:
    depth: int

@dataclass
class TransformerEncoder(BaseEncoder):
    num_heads: int

@dataclass
class Config:
    encoder: BaseEncoder

# In YAML or CLI, specify the actual type:
# encoder:
#   $type: my.module.TransformerEncoder
#   depth: 6
#   num_heads: 8

# With strict checking enabled
options = ConfifyOptions(strict_subclass_check=True)
config = read_config_from_cli(Config, options=options)
# Now config.encoder is a TransformerEncoder instance
```

**Command-line usage:**

```bash
# Using YAML file with $type
python example.py ---encoder encoder.yaml

# Or inline (though less practical):
python example.py --encoder.\$type my.module.TransformerEncoder --encoder.depth 6 --encoder.num_heads 8
```

**Note:** When dumping configs with `config_dump_yaml()`, the `$type` field is automatically added to preserve the actual class information.

#### Handling Extra Fields

By default, confify performs strict validation and raises an error if the input contains fields that don't exist in the dataclass or TypedDict definition. You can control this behavior using the `ignore_extra_fields` option.

**Example:**

```python
from dataclasses import dataclass
from confify import parse, ConfifyOptions

@dataclass
class Config:
    name: str
    value: int

# This will raise an error due to extra field 'extra'
data = {"name": "test", "value": 42, "extra": "field"}

# Option 1: Raises ConfifyParseError
config = parse(data, Config)  # Error: Got extra fields: extra

# Option 2: Issues warning and ignores extra field
options = ConfifyOptions(ignore_extra_fields=True)
config = parse(data, Config, options=options)  # Warning issued, but succeeds
# config = Config(name="test", value=42)
```

This applies to both dataclasses and TypedDict. Missing required fields always raise an error regardless of this option.

#### YAML

Values loaded from YAML usually have the correct type, except for type `Union[Enum, str]` where the value will always be converted to a matching `Enum`.

`config_dump_yaml` will dump the config to a YAML file without special tags. We add `$type` field to the dataclasses to indicate the type of the dataclass.

### Supported types

`int`, `float`, `bool`, `str`, `None`, `Path`, `Any`, `list`, `tuple`, `dict`, `Iterable`, `Sequence`, `Enum`, `dataclass`, `Union`, `Literal`, `TypedDict`

## Configuration Sweeps

Confify supports **type-safe configuration sweeps** for hyperparameter search and experiment management. Use generators to programmatically create multiple configuration variants from parameter grids.

### Basic Example

```python
from dataclasses import dataclass
from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L

@dataclass
class Optimizer:
    learning_rate: float

@dataclass
class SGD(Optimizer):
    momentum: float = 0.9

@dataclass
class Adam(Optimizer):
    beta1: float = 0.9

@dataclass
class Config:
    experiment_name: str
    model: str
    batch_size: int
    optimizer: Optimizer

c = Confify(Config)

@c.generator()
def experiments(_: Config) -> ConfigStatements:
    return [
        Set(_.experiment_name).to(L("exp_{name}")),
        Set(_.model).to("resnet50"),
        Sweep(
            _bs32=[Set(_.batch_size).to(32)],
            _bs64=[Set(_.batch_size).to(64)],
        ),
        Sweep(
            _sgd=[
                Set(_.optimizer.learning_rate).to(0.1),
                SetType(_.optimizer)(
                    As(SGD).then(lambda opt: [
                        Set(opt.momentum).to(0.8),
                    ])
                )
            ],
            _adam=[
                Set(_.optimizer.learning_rate).to(0.001),
                SetType(_.optimizer)(
                    As(Adam).then(lambda opt: [
                        Set(opt.beta1).to(0.99),
                    ])
                )
            ],
        ),
    ]

@c.main()
def main(config: Config):
    print(f"Running {config.experiment_name}")
    print(f"  Model: {config.model}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Optimizer: {config.optimizer}")

if __name__ == "__main__":
    main()
```

This generates **4 configurations** (2 batch sizes × 2 optimizers):

- `experiments_bs32_sgd`
- `experiments_bs32_adam`
- `experiments_bs64_sgd`
- `experiments_bs64_adam`

### Usage

```bash
# List all generated configs
python train.py list experiments

# Generate shell scripts for each config
python train.py generate shell experiments
# Creates: _generated/train_experiments/experiments_bs32_sgd.sh, etc.

# Run a specific config directly
python train.py run experiments_bs32_sgd
```

### Key Components

- **`Set(_.field).to(value)`**: Set a field value
- **`Sweep(_variant1=[...], _variant2=[...])`**: Create multiple variants from parameter combinations (cartesian product)
- **`SetType(_.field)(As(Type).then(lambda x: [...]))`**: Set polymorphic types for dataclass subclassing
- **`L("{name}")`**: Template strings for unique experiment names

See **[CLI Documentation](docs/cli.md)** for comprehensive guides on generators, sweeps, custom exporters, and best practices.

## Examples

The `examples/` directory contains several examples demonstrating different Confify features:

- **`ex_basic.py`** - Simple CLI application with main function
- **`ex_generator.py`** - Configuration generators with polymorphic types
- **`ex_sweep_patterns.py`** - Different sweep patterns for hyperparameter searches
- **`ex_ml_config.py`** - Realistic ML training configuration example

## Limitations and Known Issues

1. Lists and tuples of dataclasses are not supported. There is currently no way to input a list of dataclasses in CLI arguments.

2. Default values defined in nested dataclass default constructor will be overwritten by the CLI arguments. For example:

```python
@dataclass
class A:
    v1: int = 1
    v2: bool = True

@dataclass
class Config:
    a: A = field(default_factory=lambda: A(v2=False))
```

Running `python example.py --a.v1 2` will result in `Config(a=A(v1=2, v2=True))`. Notice `a.v2` gets the default value from the definition of `A` bypassing the default factory. Running `python example.py` will result in `Config(a=A(v1=1, v2=False))` using the default factory.

3. For type `Union[Enum, str]`, the value from YAML will always be converted to a matching `Enum` without a way of forcing it to `str`. In CLI arguments, enclosing the value in quotes will still force it to `str`.

## License

Confify is licensed under the MIT License.
