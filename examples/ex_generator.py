from dataclasses import dataclass

from confify import Confify, ConfigStatements, Set, Sweep, SetType, As


@dataclass
class Encoder:
    depth: int


@dataclass
class HeheEncoder(Encoder):
    ch_mult: tuple[int, ...]


@dataclass
class HuhuEncoder:
    ch_mult: tuple[int, ...]


@dataclass
class Config:
    name: str
    value: int
    dims: tuple[int, ...]
    encoder: Encoder


c = Confify(Config)


@c.gen()
def v1(_: Config) -> ConfigStatements:
    return [
        Set(_.name).to("hello"),
        Set(_.value).to(1),
        Sweep(
            _small=[
                Set(_.dims).to((1, 2, 3)),
            ],
            _large=[
                Set(_.dims).to((4, 5, 6)),
                Sweep(
                    [
                        Set(_.value).to(2),
                    ],
                    _=[
                        Set(_.value).to(3),
                    ],
                ),
            ],
        ),
        SetType(_.encoder)(As(Encoder)),
        SetType(_.encoder)(
            As(HeheEncoder).then(
                lambda e: [
                    Set(e.ch_mult).to((3, 4)),
                    Sweep(
                        _X=[
                            Set(e.depth).to(1),
                        ],
                        _Y=[
                            Set(e.depth).to(2),
                        ],
                    ),
                ]
            )
        ),
    ]


@c.main()
def main(config: Config):
    print(config)


if __name__ == "__main__":
    main()
