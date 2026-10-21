"""EXERCISE 4.1 — EXACT DAILY SIMULATION OF THE INSTRUCTED GAS-PRICE MODEL.

Here X=log(S) follows dX=kappa*(m-X)dt+sigma*dW, with
m=mu-sigma^2/(2*kappa). Daily dt=1; no Euler or Milstein approximation is needed.
"""
import numpy as np


def gas_price_moments(initial_spot, model, elapsed_days):
    elapsed_days = np.asarray(elapsed_days, dtype=float)
    if not np.isfinite(initial_spot) or initial_spot <= 0 or np.any(elapsed_days < 0):
        raise ValueError("Supply a positive initial spot and nonnegative elapsed days")
    decay = np.exp(-model.mean_reversion_speed * elapsed_days)
    log_mean = model.log_price_mean + (np.log(initial_spot) - model.log_price_mean) * decay
    log_variance = model.volatility**2 * (-np.expm1(
        -2 * model.mean_reversion_speed * elapsed_days)) / (2 * model.mean_reversion_speed)
    return {"log_mean": log_mean, "log_variance": log_variance,
            "spot_mean": np.exp(log_mean + 0.5 * log_variance)}


def simulate_gas_prices(initial_spot, model, operating_dates, number_of_paths,
                        seed, antithetic=True):
    # 1. KEEP THE OBSERVED FIRST SPOT AND THE DAILY OPERATING GRID EXPLICIT
    dates = np.asarray(operating_dates, dtype="datetime64[D]")
    if dates.ndim != 1 or len(dates) == 0 or np.any(np.isnat(dates)):
        raise ValueError("Supply a nonempty daily operating grid")
    if len(dates) > 1 and np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("Gas storage uses consecutive daily decisions")
    if not np.isfinite(initial_spot) or initial_spot <= 0:
        raise ValueError("The initial spot must be finite and positive")
    if not isinstance(number_of_paths, int) or number_of_paths < 2:
        raise ValueError("Use at least two price paths")
    if antithetic and number_of_paths % 2:
        raise ValueError("Antithetic sampling requires an even number of paths")

    # 2. DRAW INDEPENDENT GAUSSIAN SHOCKS, PAIRED AS Z AND -Z
    generator = np.random.default_rng(seed)
    if antithetic:
        independent_shocks = generator.standard_normal((number_of_paths // 2, len(dates) - 1))
        shocks = np.empty((number_of_paths, len(dates) - 1))
        shocks[0::2] = independent_shocks
        shocks[1::2] = -independent_shocks
    else:
        shocks = generator.standard_normal((number_of_paths, len(dates) - 1))
    # Here adjacent rows form a pair. Every even prefix retains complete pairs.

    # 3. APPLY THE EXACT CONDITIONAL LOG-PRICE TRANSITION
    decay = np.exp(-model.mean_reversion_speed)
    shock_scale = model.volatility * np.sqrt(
        -np.expm1(-2 * model.mean_reversion_speed) / (2 * model.mean_reversion_speed))
    log_prices = np.empty((number_of_paths, len(dates)))
    log_prices[:, 0] = np.log(initial_spot)
    for date_index in range(1, len(dates)):
        log_prices[:, date_index] = (model.log_price_mean
            + (log_prices[:, date_index - 1] - model.log_price_mean) * decay
            + shock_scale * shocks[:, date_index - 1])
    # Here rows are paths and columns are decision dates, including the first.
    prices = np.exp(log_prices)
    prices[:, 0] = initial_spot
    return prices
