"""EXERCISE 2, PART 3 — TWO SMALL CHECKS OF THE GAMMA CALCULATION."""
import numpy as np
from exercise2 import gamma
from black_scholes_put import (
    calculate_european_put_gamma_black_scholes_zero_rates,
    value_european_put_black_scholes_zero_rates,
)
from market_data import MarketData
from models import GeometricBrownianMotionModel
from products import AmericanPut
from valuation import SimulationSettings


# CHECK THE DIFFERENCE FORMULA AND ITS DISTINCT FINITE-BUMP CONTROL MEAN
def test_gamma_central_difference_and_black_scholes_reference(test_data):
    case = test_data["exercise2"]["gamma_central_difference"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    np.testing.assert_allclose(
        gamma.calculate_central_difference_gamma(
            inputs["prices_down"], inputs["prices_centre"], inputs["prices_up"],
            inputs["spot_bump"]),
        expected["gamma"], rtol=tolerances["relative"], atol=tolerances["absolute"],
    )
    case = test_data["exercise2"]["gamma_finite_bump"]
    inputs, tolerances = case["inputs"], case["tolerances"]
    spot, strike = inputs["spot"], inputs["strike"]
    maturity, volatility = inputs["maturity_years"], inputs["volatility"]
    analytical_gamma = calculate_european_put_gamma_black_scholes_zero_rates(
        spot, strike, maturity, volatility
    )
    errors = []
    for bump in inputs["spot_bumps"]:
        prices = [value_european_put_black_scholes_zero_rates(
            bumped_spot, strike, maturity, volatility
        ) for bumped_spot in (spot - bump, spot, spot + bump)]
        finite_difference = float(gamma.calculate_central_difference_gamma(*prices, bump))
        errors.append(abs(finite_difference - analytical_gamma))
    # Here reducing the bump reduces the truncation error in the known smooth price.
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < tolerances["smallest_bump_absolute_error_upper_bound"]


# CHECK SHARED PATHS AND EXACT CONTROL CANCELLATION WHEN BOTH PAYOFFS COINCIDE
def test_gamma_shares_bumped_paths_and_uses_gamma_control_mean(monkeypatch):
    calls = []

    def exact_waiting_valuation(product, market_data, model, simulation_method,
                                basis_function, simulation_settings,
                                training_brownian_increments,
                                evaluation_brownian_increments):
        calls.append((market_data.spot, training_brownian_increments,
                      evaluation_brownian_increments, simulation_settings.seed.state))
        pathwise = {}
        for prefix, increments in (
            ("training", training_brownian_increments),
            ("evaluation", evaluation_brownian_increments),
        ):
            terminal_spots = market_data.spot * np.exp(
                -.5 * model.volatility**2 * product.maturity_years
                + model.volatility * increments.sum(axis=1)
            )
            payoffs = product.payoff(terminal_spots)
            if prefix == "training":
                pathwise["training_american_payoffs"] = payoffs
                pathwise["training_european_payoffs"] = payoffs
            else:
                pathwise["american_evaluation_payoffs"] = payoffs
                pathwise["european_evaluation_payoffs"] = payoffs
        return {
            "pathwise": pathwise, "exercise_at_time_zero": False,
            "training_simulation_diagnostics": {"negative_spot_state_count": 0},
            "evaluation_simulation_diagnostics": {"negative_spot_state_count": 0},
        }

    monkeypatch.setattr(gamma.valuation, "value_american_put_with_european_control",
                        exact_waiting_valuation)
    result = gamma.calculate_american_put_gamma(
        AmericanPut(40., 1 / 12), MarketData(40., 0., .2),
        GeometricBrownianMotionModel(0., .2), None, None,
        SimulationSettings(1000, 2000, np.random.SeedSequence(123), 1., 365), .2,
    )
    assert [call[0] for call in calls] == [39.8, 40., 40.2]
    assert all(call[1] is calls[0][1] and call[2] is calls[0][2] for call in calls)
    assert calls[0][1] is not calls[0][2]
    assert calls[0][1].shape == (1000, 31)
    assert calls[0][2].shape == (2000, 31)
    assert result["control_coefficient"] == 1.
    np.testing.assert_allclose(result["controlled_gamma"],
                               result["european_finite_difference_gamma"], atol=1e-14)
    assert result["controlled_estimator_variance"] < 1e-25
    assert result["plain_estimator_variance"] > 0
    assert result["controlled_gamma"] != result["analytical_gamma"]
