# Confify

Confify is a fully typed, plug-and-play configuration library for Python.

**Key features:**

- Uses type annotations from `dataclass` and `TypedDict`.
- Uses dotlist notations for CLI arguments. (e.g., `--encoder.depth 6`, `--model.hidden_dims '(10, 20)'`)
- Loads partial configurations from YAML in CLI arguments. (e.g., `---encoder encoder.yaml`)
- Supports subclassing of `dataclass` by specifying the classname. (e.g., `--encoder.$type mymodule.MyEncoder`)
- Supports `Optional`, `Union`, `Literal`.
- Has minimal dependencies (only `PyYaml`).

## Installation

(TODO) Upload to PyPI
```bash
pip install git+https://github.com/milmillin/confify.git
```

## Usage

### Example Usage

```python
# example.py

from confify import read_config_from_cli, config_dump_yaml

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

if __name__ == "__main__":
    # parsing config from sys.argv
    config = read_config_from_cli(Config)
    # config is properly typed
    assert reveal_type(config) == Config
    # dumping config to yaml
    config_dump_yaml("config.yaml", config)
```

### Using the CLI

We use dotlist notations for CLI arguments. See examples below.

```bash
python example.py \
    --encoder.depth 6 \
    --encoder.ch_mult \(3,4\) \
    --encoder.activation_fn silu \
    --encoder.augment_type None \
    --save_path ~/experiments/exp1 \
    --run_id exp1
```

We **do not** support equal signs in dotlist notations (for now). For example, `--encoder.depth=6` will not work.

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

### Type Resolution

Arguments from CLI are converted to the annotated type using the following rule:

- `int`, `float`, `Path`: we use default constructors.
- `bool`: we convert case-insensitive `"true"`, `"on"`, `"yes"` to `True`, and `"false"`, `"off"`, `"no"` to `False`.
- `str`: if the value is surrounded by quotes, we use `ast.literal_eval` to convert it to a string. Otherwise, we use the value as is.
- `None`: we convert case-insensitive `"null"`, `"~"`, `"none"` to `None`.
- `Enum`: we find the enum entry with the same name.

There may be ambiguity when the type is `Optional`, `Union` or `Literal`. For example, if a user input `"null"` for type `Union[str, None]`, it can be interpreted as either `None` or a string `"null"`. We follow the following order and returns the first that succeeds:

1. Types not listed below (from left to right)
2. `None`
3. `bool`
4. `int`
5. `float`
6. `str`

So `"null"` for type `Union[str, None]` will be parsed as `None`. To get a string `"null"`, the user needs to input `"\"null\""` with quotes. 

#### Dataclasses Subclassing
If the type incidated by `$type` is a subclass of the annotated type, we use the constructor of `$type`. Otherwise, a warning will be issued, and the object will be constructed using the annotated type. We raise an exception if the value contains extra fields or missing fields (of the constructor).

#### YAML
Values loaded from YAML usually have the correct type, except for type `Union[Enum, str]` where the value will always be converted to a matching `Enum`.

`config_dump_yaml` will dump the config to a YAML file without special tags. We add `$type` field to the dataclasses to indicate the type of the dataclass.


### Supported types

`int`, `float`, `bool`, `str`, `None`, `Path`, `list`, `tuple`, `dict`, `Enum`, `dataclass`, `Union`, `Literal`, `TypedDict`

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