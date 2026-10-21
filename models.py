"""Here we define the price dynamics. The simulator supplies timesteps and shocks."""
from dataclasses import dataclass
from math import isfinite
import numpy as np


# GEOMETRIC BROWNIAN MOTION
@dataclass(frozen=True)
class GeometricBrownianMotionModel:
    spot_drift_rate: float
    volatility: float

    def __post_init__(self):
        if self.volatility is None or not all(isfinite(value) for value in
                                             (self.spot_drift_rate, self.volatility)):
            raise ValueError("Supply finite drift and volatility")
        if self.volatility < 0:
            raise ValueError("Volatility must be nonnegative")

    def spot_drift(self, current_spots):
        return self.spot_drift_rate * current_spots

    def spot_diffusion(self, current_spots):
        return self.volatility * current_spots


# STOCHASTIC VOLATILITY — THE SDE PRINTED IN EXERCISE 1
@dataclass(frozen=True)
class StochasticVolatilityModel:
    initial_variance: float
    mean_reversion_speed: float
    long_term_variance: float
    volatility_of_variance: float
    brownian_correlation: float
    spot_drift_rate: float = 0.0

    def __post_init__(self):
        if not all(isfinite(value) for value in self.__dict__.values()):
            raise ValueError("Model parameters must be finite")
        if min(self.initial_variance, self.mean_reversion_speed, self.long_term_variance) <= 0:
            raise ValueError("Initial variance, kappa and theta must be positive")
        if self.volatility_of_variance < 0 or not 0 < self.brownian_correlation <= 1:
            raise ValueError("Exercise 1 requires xi >= 0 and 0 < rho <= 1")
        if 2 * self.mean_reversion_speed * self.long_term_variance <= self.volatility_of_variance**2:
            raise ValueError("Exercise 1 requires 2*kappa*theta > xi**2")
        # Here we enforce the stated inequality; it is not a positivity proof
        # for this diffusion containing S. Euler handles negative proposals.

    def spot_drift(self, current_spots):
        return self.spot_drift_rate * current_spots

    def spot_diffusion(self, current_spots, nonnegative_variances):
        return np.sqrt(nonnegative_variances) * current_spots

    def variance_drift(self, current_variances):
        return self.mean_reversion_speed * (self.long_term_variance - current_variances)

    def variance_diffusion(self, current_spots, nonnegative_variances):
        # Here we retain the printed S factor; this is not standard Heston.
        return self.volatility_of_variance * np.sqrt(nonnegative_variances) * current_spots


# EXERCISE 4 — dS/S = KAPPA * (MU - LOG S) dt + SIGMA dW
@dataclass(frozen=True)
class GasPriceModel:
    mean_reversion_speed: float
    log_price_target: float
    volatility: float

    def __post_init__(self):
        if not all(isfinite(value) for value in self.__dict__.values()):
            raise ValueError("Gas-price parameters must be finite")
        if self.mean_reversion_speed <= 0 or self.volatility < 0:
            raise ValueError("Gas kappa must be positive and sigma nonnegative")

    @property
    def log_price_mean(self):
        # Here Ito's correction makes m = mu - sigma^2/(2*kappa).
        # Both kappa and sigma are quoted in DAILY units in the assignment.
        return self.log_price_target - self.volatility**2 / (2 * self.mean_reversion_speed)
