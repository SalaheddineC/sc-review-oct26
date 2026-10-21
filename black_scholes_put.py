"""EXERCISE 2: BLACK–SCHOLES PUT VALUE AND ANALYTICAL GAMMA.

Here each function values one case using scalar inputs. The valuation layer
supplies these inputs from our product and market-data objects.
"""
from math import exp, isfinite, log, sqrt
from scipy.stats import norm


# SHARED INPUT CHECKS
def check_black_scholes_put_inputs(spot, strike, maturity_years, volatility):
    # Here NaN or infinity would make the calculated price meaningless.
    inputs = (spot, strike, maturity_years, volatility)
    if not all(isfinite(value) for value in inputs):
        raise ValueError("Analytical put inputs must be finite")
    # Here positive spot and strike make log(S/K) well defined.
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive")
    # Here time remaining and volatility cannot be negative.
    if maturity_years < 0 or volatility < 0:
        raise ValueError("Maturity and volatility must be nonnegative")


# PART 1: KNOWN EXPECTATION OF THE EUROPEAN CONTROL
def value_european_put_black_scholes_zero_rates(
    spot, strike, maturity_years, volatility
):
    # Here the named zero-rate entry point states the assignment's assumptions.
    return value_european_put_black_scholes(
        spot, strike, maturity_years, volatility,
        interest_rate=0.0, dividend_yield=0.0
    )


# GENERAL EUROPEAN PUT FORMULA — ALSO USED INSIDE THE BAW APPROXIMATION
def value_european_put_black_scholes(
    spot, strike, maturity_years, volatility,
    interest_rate=0.0, dividend_yield=0.0
):
    check_black_scholes_put_inputs(spot, strike, maturity_years, volatility)
    if not (isfinite(interest_rate) and isfinite(dividend_yield)):
        raise ValueError("Interest rate and dividend yield must be finite")

    # Here an expired contract is worth its payoff, with no time value.
    if maturity_years == 0:
        return max(strike - spot, 0.0)
    discounted_strike = strike * exp(-interest_rate * maturity_years)
    discounted_spot = spot * exp(-dividend_yield * maturity_years)
    # Here the terminal spot is deterministic; avoid division by sigma*sqrt(T).
    if volatility == 0:
        return max(discounted_strike - discounted_spot, 0.0)

    # Here volatility is annualised and maturity is measured in years.
    total_volatility = volatility * sqrt(maturity_years)
    cost_of_carry = interest_rate - dividend_yield
    d1 = ((log(spot / strike) + cost_of_carry * maturity_years)
          / total_volatility + 0.5 * total_volatility)
    d2 = d1 - total_volatility

    # Here Phi is the standard normal CDF in the Black–Scholes formula.
    probability_minus_d1 = norm.cdf(-d1)
    probability_minus_d2 = norm.cdf(-d2)

    # Here r = yield = 0 makes both discount factors one in the assignment runs.
    return discounted_strike * probability_minus_d2 - discounted_spot * probability_minus_d1


# PART 3: ANALYTICAL GAMMA REFERENCE
def calculate_european_put_gamma_black_scholes_zero_rates(
    spot, strike, maturity_years, volatility
):
    check_black_scholes_put_inputs(spot, strike, maturity_years, volatility)

    # Here the limiting payoff has zero gamma away from its kink at the strike.
    if maturity_years == 0 or volatility == 0:
        if spot == strike:
            raise ValueError("Gamma is undefined at the intrinsic-payoff kink")
        return 0.0

    total_volatility = volatility * sqrt(maturity_years)
    d1 = log(spot / strike) / total_volatility + 0.5 * total_volatility
    normal_density_d1 = norm.pdf(d1)

    # Here gamma = phi(d1) / (S * sigma * sqrt(T)).
    # The finite-bump control mean will use bumped prices, not this derivative.
    return normal_density_d1 / (spot * total_volatility)
