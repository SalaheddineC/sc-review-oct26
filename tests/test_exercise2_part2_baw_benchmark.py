"""EXERCISE 2, PART 2: THREE CHECKS OF THE BAW PUT BENCHMARK.

Here the assignment uses the exact zero-rate case. The published positive-rate
examples separately check that our general implementation follows the paper.
"""
import csv
import pytest

from black_scholes_put import value_european_put_black_scholes
from exercise2.barone_adesi_whaley import (
    calculate_baw_put_boundary_and_premium,
    value_american_put_barone_adesi_whaley,
)


# ASSIGNMENT: ZERO RATES REMOVE THE EARLY-EXERCISE PREMIUM
def test_exercise2_part2_baw_zero_rates_match_the_european_value():
    for volatility in (0.05, 0.10, 0.20, 0.30):
        for spot in (36.0, 40.0, 44.0):
            reference_value = value_european_put_black_scholes(
                spot, 40.0, 1 / 12, volatility
            )
            assert value_american_put_barone_adesi_whaley(
                spot, 40.0, 1 / 12, volatility
            ) == reference_value
            # Here the positive-rate formula approaches the exact zero-rate limit.
            near_zero_rate_value = value_american_put_barone_adesi_whaley(
                spot, 40.0, 1 / 12, volatility, interest_rate=1e-12
            )
            assert abs(near_zero_rate_value - reference_value) < 1e-9

    # Here expiry and zero-volatility zero-rate cases have known payoffs.
    assert value_american_put_barone_adesi_whaley(36, 40, 0, 0.2, 0.08) == 4
    assert value_american_put_barone_adesi_whaley(36, 40, 1 / 12, 0) == 4
    assert value_american_put_barone_adesi_whaley(40, 40, 1, 0.2, 0, 0.04) == (
        value_european_put_black_scholes(40, 40, 1, 0.2, 0, 0.04)
    )
    with pytest.raises(ValueError, match="positive volatility"):
        value_american_put_barone_adesi_whaley(40, 40, 1, 0, 0.08)
    with pytest.raises(ValueError, match="nonnegative rates"):
        value_american_put_barone_adesi_whaley(40, 40, 1, 0.2, -0.01)


# PAPER: TABLE IV, FIRST BLOCK, QUADRATIC APPROXIMATION COLUMN (PAGE 315)
def test_exercise2_part2_baw_matches_published_positive_rate_prices(test_data, test_data_directory):
    # Here K=100, r=b=8%, sigma=20%, T=0.25 and therefore yield=0.
    # The paper rounds prices to two decimals; tolerance reflects that precision.
    case = test_data["exercise2"]["baw_published_prices"]
    path = test_data_directory / "exercise2" / case["inputs"]["file"]
    with path.open(newline="", encoding="utf-8") as source:
        published_cases = list(csv.DictReader(source))
    assert len(published_cases) == case["inputs"]["rows"]
    for row in published_cases:
        calculated_price = value_american_put_barone_adesi_whaley(
            float(row["spot"]), float(row["strike"]), float(row["maturity_years"]),
            float(row["volatility"]), float(row["interest_rate"]), float(row["dividend_yield"])
        )
        published_price = float(row[case["expected"]["column"]])
        tolerance = float(row[case["tolerances"]["column"]])
        assert abs(calculated_price - published_price) <= tolerance


# BOUNDARY: VALUE MATCHING AND SMOOTH PASTING WITH NONZERO PREMIUM
def test_exercise2_part2_baw_boundary_joins_value_and_slope():
    strike, maturity, volatility, rate = 100.0, 0.5, 0.2, 0.08
    for yield_rate in (0.0, 0.04, 0.12):
        boundary, exponent, premium = calculate_baw_put_boundary_and_premium(
            strike, maturity, volatility, rate, yield_rate
        )
        assert 0 < boundary < strike
        assert exponent < 0 and premium > 0
        european_at_boundary = value_european_put_black_scholes(
            boundary, strike, maturity, volatility, rate, yield_rate
        )
        assert abs(european_at_boundary + premium - (strike - boundary)) < 1e-8

        # Here the right-hand continuation slope must approach payoff slope -1.
        spot_bump = 1e-5 * strike
        price_above_boundary = value_american_put_barone_adesi_whaley(
            boundary + spot_bump, strike, maturity, volatility, rate, yield_rate
        )
        continuation_slope = (price_above_boundary - (strike - boundary)) / spot_bump
        assert abs(continuation_slope + 1) < 2e-4
