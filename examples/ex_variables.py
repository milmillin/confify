from dataclasses import dataclass
from typing import Optional

from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L, Variable


@dataclass
class Encoder:
    depth: int
    dim: int


@dataclass
class Decoder:
    depth: int
    dim: int


@dataclass
class Config:
    name: str
    value: int
    dims: tuple[int, ...]
    encoder: Optional[Encoder] = None
    decoder: Optional[Decoder] = None


c = Confify(Config)


@c.generator()
def v1(_: Config) -> ConfigStatements:
    dim = Variable(int)
    return [
        Set(_.name).to(L("{name}")),
        Set(_.value).to(1),
        Set(_.dims).to((1, 2, 3)),
        Sweep(
            _z32=[
                Set(dim).to(32),
            ],
            _z64=[
                Set(dim).to(64),
            ],
        ),
        Sweep(
            _enc_only=[
                SetType(_.encoder)(
                    As(Encoder).then(
                        lambda e: [
                            Set(e.depth).to(1),
                            Set(e.dim).to(dim),
                        ]
                    )
                ),
            ],
            _dec_only=[
                SetType(_.decoder)(
                    As(Decoder).then(
                        lambda e: [
                            Set(e.depth).to(1),
                            Set(e.dim).to(dim),
                        ]
                    )
                ),
            ],
        ),
    ]


@c.main()
def main(config: Config):
    print("Main")
    print(config)


if __name__ == "__main__":
    main()
