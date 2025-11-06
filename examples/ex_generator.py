from dataclasses import dataclass

from confify import Confify, ConfigStatements, Set, Sweep, SetType, As, L


@dataclass
class Encoder:
    depth: int


@dataclass
class HeheEncoder(Encoder):
    ch_mult: tuple[int, ...]


@dataclass
class HuhuEncoder(Encoder):
    layers: int


@dataclass
class Config:
    name: str
    value: int
    dims: tuple[int, ...]
    encoder: Encoder


c = Confify(Config)


@c.generator()
def v1(_: Config) -> ConfigStatements:
    return [
        Set(_.name).to(L("{name}")),
        Set(_.value).to(1),
        Sweep(
            _small=[
                Set(_.dims).to((1, 2, 3)),
            ],
            _large=[
                Set(_.dims).to((4, 5, 6)),
            ],
        ),
        Sweep(
            _Hehe=[
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
            ],
            _Hehe2=[Set(_.encoder).to(HeheEncoder(depth=1, ch_mult=(3, 4)))],
            _Huhu=[
                SetType(_.encoder)(
                    As(HuhuEncoder).then(
                        lambda e: [
                            Set(e.depth).to(1),
                            Set(e.layers).to(2),
                        ]
                    )
                )
            ],
        ),
    ]


@c.main()
def main(config: Config):
    print("Main")
    print(config)


if __name__ == "__main__":
    main()
