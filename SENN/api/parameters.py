from enum import auto, Enum


class RegLambda(Enum):
    ZERO = 0
    E4 = 1e-4
    E3 = 1e-3
    E2 = 1e-2
    E1 = 1e-1
    ONE = 1


class HType(Enum):
    CNN = auto()
    INPUT = auto()


class NConcepts(Enum):
    FIVE = 5
    TWENTY = 20
