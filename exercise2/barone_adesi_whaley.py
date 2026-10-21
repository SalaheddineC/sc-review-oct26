"""EXERCISE 2, PART 2: BARONE-ADESI–WHALEY AMERICAN PUT APPROXIMATION.

Here we implement the put equations (21)–(25) in Barone-Adesi–Whaley (1987).
The boundary joins the exercise payoff to European value plus the premium.
Rates, continuous yields and volatility are constant. We support nonnegative
rates and yields; positive-rate cases require positive volatility.
"""
from math import exp, expm1, isfinite, log, sqrt
from scipy.stats import norm
from black_scholes_put import (
    check_black_scholes_put_inputs,
    value_european_put_black_scholes,
    value_european_put_black_scholes_zero_rates,
)


# INPUTS: STATE THE DOMAIN OF THIS IMPLEMENTATION
def check_baw_put_inputs(spot, strike, maturity_years, volatility,
                         interest_rate, dividend_yield):
    # Here the shared checks protect logarithms, square roots and finite prices.
    check_black_scholes_put_inputs(spot, strike, maturity_years, volatility)
    if not all(isfinite(value) and value >= 0
               for value in (interest_rate, dividend_yield)):
        raise ValueError("BAW requires finite, nonnegative rates and yields")
    # Here the positive-rate power solution divides by sigma squared.
    if maturity_years > 0 and interest_rate > 0 and volatility == 0:
        raise ValueError("Positive-rate BAW requires positive volatility")


# ZERO RATES: APPLY THE MATHEMATICAL SPECIALISATION REQUIRED BY EXERCISE 2
def value_american_put_baw_zero_rates(spot, strike, maturity_years, volatility):
    # Here r=q=0 makes GBM a martingale. The put payoff is convex, so conditional
    # Jensen shows that waiting is at least as valuable as exercising early.
    # American and European values coincide: the early-exercise premium is zero.
    # In BAW, no positive exercise boundary remains. The power premium D*S**lambda
    # must stay bounded near S=0; lambda<0 therefore requires D=0.
    # We evaluate this derived zero-rate formula directly, without a tiny rate
    # or an undefined S/B calculation at B=0. This is the assignment benchmark.
    return value_european_put_black_scholes_zero_rates(
        spot, strike, maturity_years, volatility)


# POWER EXPONENT: DETERMINE THE SHAPE OF THE EARLY-EXERCISE PREMIUM
def calculate_baw_put_exponent(maturity_years, volatility,
                               interest_rate, dividend_yield):
    # Here BAW neglects the remaining maturity derivative of the transformed
    # premium. Trying S**lambda gives lambda**2+(N-1)*lambda-M/H=0,
    # with N=2*(r-q)/sigma**2 and H=1-exp(-r*T).
    variance = volatility**2
    carry_coefficient_minus_one = 2 * (interest_rate - dividend_yield) / variance - 1
    if interest_rate == 0:
        # Here M/H is a 0/0 limit, not zero: its limit is 2/(sigma**2*T).
        ratio_m_over_h = 2 / (variance * maturity_years)
    else:
        # Here expm1(x)=exp(x)-1 retains precision for rates close to zero.
        ratio_m_over_h = 2 * interest_rate / (
            variance * -expm1(-interest_rate * maturity_years))

    root_discriminant = sqrt(carry_coefficient_minus_one**2 + 4 * ratio_m_over_h)
    # Here the negative root makes the put premium decline as spot increases.
    # The equivalent fraction avoids subtracting almost equal numbers.
    if carry_coefficient_minus_one < 0:
        put_exponent = -2 * ratio_m_over_h / (
            root_discriminant - carry_coefficient_minus_one)
    else:
        put_exponent = -(carry_coefficient_minus_one + root_discriminant) / 2
    return put_exponent


# EXERCISE BOUNDARY: COMPLETE THE PRICING PROBLEM BY FINDING THE UNKNOWN SPOT B
def calculate_baw_put_boundary_and_premium(
    strike, maturity_years, volatility, interest_rate, dividend_yield=0.0
):
    """Return the exercise boundary B, negative exponent and premium at B."""
    # Here no current spot is needed: the boundary depends on the contract/model.
    check_baw_put_inputs(strike, strike, maturity_years, volatility,
                         interest_rate, dividend_yield)
    if maturity_years == 0 or interest_rate == 0:
        raise ValueError("A positive BAW boundary requires positive time and rate")

    put_exponent = calculate_baw_put_exponent(
        maturity_years, volatility, interest_rate, dividend_yield)
    total_volatility = volatility * sqrt(maturity_years)
    rate_discount = exp(-interest_rate * maturity_years)
    yield_discount = exp(-dividend_yield * maturity_years)
    rate_discount_loss = -expm1(-interest_rate * maturity_years)
    yield_discount_loss = -expm1(-dividend_yield * maturity_years)

    # Here the differential equation gives lambda, but not the boundary or premium.
    # At B, value matching requires: K-B = p_BS(B) + A_B.
    # Smooth pasting requires the same slope: -1 = p'_BS(B) + lambda*A_B/B.
    # Thus A_B = -B/lambda * [1-exp(-q*T)*Phi(-d1(B))].
    # Substituting A_B in value matching gives one equation for the unknown B.
    # Solving it is the key step: below B we exercise; above B we continue.

    # Here bisection searches B/K between 0 and 1. Scaling by K makes the
    # boundary tolerance meaningful for different contractual strike levels.
    lower_boundary_ratio, upper_boundary_ratio = 0.0, 1.0
    maximum_iterations = 100
    boundary_ratio_tolerance = 1e-12
    for boundary_iteration in range(maximum_iterations):
        boundary_ratio = (lower_boundary_ratio + upper_boundary_ratio) / 2
        d1_at_boundary = (log(boundary_ratio)
            + (interest_rate - dividend_yield + 0.5 * volatility**2)
            * maturity_years) / total_volatility
        d2_at_boundary = d1_at_boundary - total_volatility
        boundary_slope_term = yield_discount_loss + yield_discount * norm.cdf(d1_at_boundary)

        # Here this is (K-B-p_BS(B)-A_B)/K, rearranged using Phi(-d)=1-Phi(d).
        # The rearrangement avoids subtracting nearly equal option values.
        boundary_gap_per_strike = (
            rate_discount_loss + rate_discount * norm.cdf(d2_at_boundary)
            - boundary_ratio * boundary_slope_term * (1 - 1 / put_exponent))
        if boundary_gap_per_strike > 0:
            lower_boundary_ratio = boundary_ratio
        else:
            upper_boundary_ratio = boundary_ratio
        if upper_boundary_ratio - lower_boundary_ratio <= boundary_ratio_tolerance:
            break

    exercise_boundary = strike * boundary_ratio
    boundary_premium = -exercise_boundary * boundary_slope_term / put_exponent
    return exercise_boundary, put_exponent, boundary_premium


# AMERICAN PUT: SELECT THE ZERO-RATE FORMULA OR THE GENERAL EXERCISE DECISION
def value_american_put_barone_adesi_whaley(
    spot, strike, maturity_years, volatility,
    interest_rate=0.0, dividend_yield=0.0
):
    check_baw_put_inputs(spot, strike, maturity_years, volatility,
                         interest_rate, dividend_yield)
    if maturity_years == 0:
        return max(strike - spot, 0.0)

    # Here the assignment supplies exact zeros. The proof above determines the
    # formula BEFORE solving a boundary; no numerical zero-rate workaround is used.
    if interest_rate == 0 and dividend_yield == 0:
        return value_american_put_baw_zero_rates(spot, strike, maturity_years, volatility)

    european_value = value_european_put_black_scholes(
        spot, strike, maturity_years, volatility, interest_rate, dividend_yield)
    # Here r=0,q>0 also favours waiting: expected spot falls and convexity applies.
    if interest_rate == 0:
        return european_value

    exercise_boundary, put_exponent, boundary_premium = calculate_baw_put_boundary_and_premium(
        strike, maturity_years, volatility, interest_rate, dividend_yield)
    if spot <= exercise_boundary:
        return strike - spot

    # Here BAW equation (25) gives continuation: European value + exercise premium.
    # At S=B this joins K-B in both value and slope, by the boundary conditions.
    return european_value + boundary_premium * (spot / exercise_boundary)**put_exponent
