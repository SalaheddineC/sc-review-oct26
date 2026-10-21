"""Checks for the independent integration benchmark, using known model identities."""
import numpy as np
import pytest

from models import GasPriceModel
from products import GasStorage
from exercise4.gas_storage_benchmark import (benchmark_price_grid,
    gaussian_expectation_matrix, solve_storage_benchmark, exact_three_day_value)


def standard_model(volatility):
    return GasPriceModel(.05, np.log(20.0) + volatility**2 / .2, volatility)


def test_quadrature_preserves_constant_and_affine_terminal_cashflows():
    model = standard_model(.0945)
    prices = benchmark_price_grid(20.0, model, 501, 6.0)
    matrix = gaussian_expectation_matrix(prices, model, 64)
    # Here the terminal cashflow is affine in spot. This check includes the
    # boundary tails and catches both Gaussian normalisation and interpolation.
    np.testing.assert_allclose(matrix @ np.ones(len(prices)), 1.0, atol=2e-14)
    mean = np.exp(model.log_price_mean + (np.log(prices) - model.log_price_mean) * np.exp(-.05)
                  + .0945**2 * (1 - np.exp(-.1)) / .2)
    np.testing.assert_allclose(matrix @ prices, mean, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(matrix @ (7500 * prices - 1234), 7500 * mean - 1234,
                               rtol=2e-13, atol=1e-8)


@pytest.mark.parametrize("case_name", ["three_day_high", "three_day_low"])
def test_three_day_nonlinear_decision_matches_exact_truncated_lognormal_moments(case_name, test_data):
    # Here three dates already contain a future maximum: this checks the
    # integration of an action-switching kink, not only a linear payoff.
    case = test_data["exercise4"][case_name]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    product, model = GasStorage(**inputs["product"]), GasPriceModel(**inputs["model"])
    exact = exact_three_day_value(product, model, inputs["spot"])
    assert exact == pytest.approx(expected["value"], rel=tolerances["relative"],
                                  abs=tolerances["exact_value_absolute_eur"])
    result = solve_storage_benchmark(product, model, initial_spot=inputs["spot"],
                                     inventory_step=inputs["inventory_step"], **inputs["benchmark"])
    assert result["reference_value"] == pytest.approx(exact, rel=tolerances["relative"],
                                                      abs=tolerances["benchmark_value_absolute_eur"])
    assert result["initial_action"] == expected["initial_action"]


def test_forced_final_trade_and_constant_price_round_trip(test_data):
    case = test_data["exercise4"]["final_day"]
    inputs = case["inputs"]
    model = GasPriceModel(**inputs["model"])
    for row in case["expected"]["rows"]:
        product = GasStorage(**{**inputs["product"], "initial_inventory": row["initial_inventory"]})
        result = solve_storage_benchmark(product, model, initial_spot=inputs["spot"],
                                         inventory_step=inputs["inventory_step"], **inputs["benchmark"])
        assert result["reference_value"] == pytest.approx(row["cashflow"],
            rel=case["tolerances"]["relative"], abs=case["tolerances"]["benchmark_value_absolute_eur"])
        assert result["initial_action"] == row["trade"]
    # Here a full-year constant spot makes every feasible round-trip worth zero.
    case = test_data["exercise4"]["zero_volatility"]
    inputs = case["inputs"]
    result = solve_storage_benchmark(GasStorage(**inputs["product"]), GasPriceModel(**inputs["model"]),
        initial_spot=inputs["spot"], inventory_step=inputs["inventory_step"], **inputs["benchmark"])
    assert result["reference_value"] == pytest.approx(case["expected"]["value"],
        rel=case["tolerances"]["relative"], abs=case["tolerances"]["benchmark_value_absolute_eur"])


def test_separate_recursion_values_the_supplied_decisions_without_optimising_them():
    product = GasStorage(start_date="2006-06-28")
    inventory = np.arange(0.0,250000.0+2500.0,2500.0)
    # Here a deliberately simple decision rule strongly prefers returning to
    # the required inventory. Starting there, it holds on all three dates and
    # earns exactly zero. The independent optimiser can earn a positive value.
    decisions = {"inventory_grid":inventory,
        "actions":np.array([-7500.0,-5000.0,-2500.0,0.0,2500.0]),
        "initial_spot":20.0,"constant_dates":np.ones(3,dtype=bool),
        "constant_continuation":np.tile(-1e9*np.abs(inventory-100000.0),(3,1))}
    result = solve_storage_benchmark(product,standard_model(.0945),price_points=201,
                                    quadrature_points=64,decisions=decisions)
    assert result["saved_policy_value"] == 0.0
    assert result["policy_initial_action"] == 0.0
    assert result["reference_value"] > 0.0
    assert result["policy_value_gap"] == result["reference_value"]
