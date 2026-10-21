"""Here we define option payoffs and gas-storage operating constraints."""
from dataclasses import dataclass
from math import isfinite
import numpy as np


@dataclass(frozen=True)
class AmericanPut:
    strike: float
    maturity_years: float

    def __post_init__(self):
        if not all(isfinite(value) and value > 0
                   for value in (self.strike, self.maturity_years)):
            raise ValueError("Strike and maturity must be finite and positive")

    def payoff(self, spot):
        return np.maximum(self.strike - np.asarray(spot), 0.0)


# EXERCISE 4 — STORAGE RIGHTS AND PHYSICAL CONSTRAINTS
@dataclass(frozen=True)
class GasStorage:
    start_date: str = "2005-07-07"
    end_date: str = "2006-06-30"
    capacity: float = 250000.0
    initial_inventory: float = 100000.0
    final_inventory: float = 100000.0
    injection_limit: float = 2500.0
    withdrawal_limit: float = 7500.0

    def __post_init__(self):
        quantities = (self.capacity, self.initial_inventory, self.final_inventory,
                      self.injection_limit, self.withdrawal_limit)
        if not all(isfinite(value) for value in quantities):
            raise ValueError("Storage quantities must be finite")
        if min(self.capacity, self.injection_limit, self.withdrawal_limit) <= 0:
            raise ValueError("Capacity and daily flow limits must be positive")
        if not (0 <= self.initial_inventory <= self.capacity
                and 0 <= self.final_inventory <= self.capacity):
            raise ValueError("Initial and final inventory must be within capacity")
        if len(self.operating_dates()) == 0:
            raise ValueError("The end date must not precede the start date")
        lower, upper = self.reachable_inventory_bounds(len(self.operating_dates()))
        if not lower <= self.initial_inventory <= upper:
            raise ValueError("The terminal inventory cannot be reached from the start")

    def operating_dates(self):
        # Here both dates are operating days. Gas left after the last trade must
        # equal final_inventory; there is no additional liquidation payment.
        start = np.datetime64(self.start_date, "D")
        end = np.datetime64(self.end_date, "D")
        if np.isnat(start) or np.isnat(end):
            raise ValueError("Supply valid operating dates")
        return np.arange(start, end + np.timedelta64(1, "D"))

    def action_bounds(self, inventory):
        inventory = np.asarray(inventory)
        return (np.maximum(-inventory, -self.withdrawal_limit),
                np.minimum(self.capacity - inventory, self.injection_limit))

    def reachable_inventory_bounds(self, remaining_decisions):
        # Here we enforce the final inventory directly. With n trades left,
        # we can inject at most n*injection_limit or withdraw n*withdrawal_limit.
        return (max(0.0, self.final_inventory - remaining_decisions * self.injection_limit),
                min(self.capacity, self.final_inventory + remaining_decisions * self.withdrawal_limit))

    def cashflow(self, spot, inventory_change):
        # Here positive Delta v buys gas; negative Delta v sells gas.
        return -np.asarray(spot) * np.asarray(inventory_change)
