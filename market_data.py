"""Here we hold market inputs; contract and numerical choices live separately."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MarketData:
    spot: float
    interest_rate: float
    volatility: float | None = None
    foreign_interest_rate: float | None = None

    def __post_init__(self):
        supplied_values = [value for value in self.__dict__.values() if value is not None]
        if not all(isfinite(value) for value in supplied_values) or self.spot <= 0:
            raise ValueError("Market inputs must be finite and spot positive")
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("Volatility must be nonnegative")
