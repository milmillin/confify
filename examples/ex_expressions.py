from dataclasses import dataclass
from typing import Optional

from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L, Variable, Use


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
    area: int
    encoder: Optional[Encoder] = None
    decoder: Optional[Decoder] = None


c = Confify(Config)


@c.generator()
def v1(_: Config) -> ConfigStatements:
    dim = Variable(int)
    width = Variable(int)
    height = Variable(int)

    return [
        Set(_.name).to(L("{name}")),
        # Single-variable expression: compute value as dim * 2
        Sweep(
            _z32=[
                Set(dim).to(32),
                Set(width).to(4),
                Set(height).to(8),
            ],
            _z64=[
                Set(dim).to(64),
                Set(width).to(8),
                Set(height).to(16),
            ],
        ),
        # Use a single-variable expression
        Set(_.value).to(Use(dim)(lambda d: d * 2)),
        # Use a multi-variable expression
        Set(_.area).to(Use(width, height)(lambda w, h: w * h)),
        # Expressions work inside SetType/As.then as well
        Sweep(
            _enc_only=[
                SetType(_.encoder)(
                    As(Encoder).then(
                        lambda e: [
                            Set(e.depth).to(1),
                            Set(e.dim).to(Use(dim)(lambda d: d * 2)),
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
