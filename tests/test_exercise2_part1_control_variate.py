"""EXERCISE 2, PART 1 — THREE CHECKS OF THE EUROPEAN CONTROL.

Here small known calculations check coupling and estimator arithmetic.
The separate study runner measures effectiveness at the assignment inputs.
"""
import numpy as np
from black_scholes_put import value_european_put_black_scholes_zero_rates
from basis import spot_quadratic_basis
from control_variates import estimate_control_coefficient, apply_control_variate
from euler_maruyama import generate_brownian_increments, simulate_paths_euler_maruyama
from longstaff_schwartz_lsmc import value_american_option_longstaff_schwartz
from market_data import MarketData
from models import GeometricBrownianMotionModel
from products import AmericanPut
from valuation import (
    SimulationSettings, exact_gbm_european_payoffs,
    make_simulation_and_exercise_grids, value_american_put,
    value_american_put_with_european_control,
)


# SHARED RANDOMNESS — PRESERVE EULER AND CHECK THE EXACT CONTROL TRANSITION
def test_part1_brownian_coupling_preserves_exercise1_euler_paths(test_data):
    product = AmericanPut(40., 1 / 12)
    market_data = MarketData(40., 0., volatility=.2)
    model = GeometricBrownianMotionModel(0., .2)
    settings = SimulationSettings(8, 12, np.random.SeedSequence(10),
                                  exercise_dates_per_year=365)
    times, exercise_indices = make_simulation_and_exercise_grids(
        product.maturity_years, settings)
    assert len(times) == 32  # Here one month has 30 full days and a short final step.
    assert 0 < times[-1] - times[-2] < 1 / 365
    original_paths, original_dates, original_diagnostics = simulate_paths_euler_maruyama(
        market_data, model, times, exercise_indices, 12, np.random.SeedSequence(10))
    increments = generate_brownian_increments(times, 12, np.random.SeedSequence(10))
    coupled_paths, coupled_dates, coupled_diagnostics = simulate_paths_euler_maruyama(
        market_data, model, times, exercise_indices, 12, np.random.SeedSequence(999),
        brownian_increments=increments)
    np.testing.assert_array_equal(original_paths["spot"], coupled_paths["spot"])
    np.testing.assert_array_equal(original_dates, coupled_dates)
    assert (original_diagnostics["negative_spot_state_count"]
            == coupled_diagnostics["negative_spot_state_count"])

    # Here the supplied paths check exp(-sigma²*T/2 + sigma*W_T).
    case = test_data["exercise2"]["control_transition"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    product = AmericanPut(inputs["strike"], inputs["maturity_years"])
    market_data = MarketData(inputs["spot"], inputs["interest_rate"],
                             volatility=inputs["volatility"])
    model = GeometricBrownianMotionModel(inputs["interest_rate"], inputs["volatility"])
    known_increments = np.array(inputs["brownian_increments"])
    expected_terminal_spots = inputs["spot"] * np.exp(
        -.5 * inputs["volatility"]**2 * inputs["maturity_years"]
        + inputs["volatility"] * np.array(expected["terminal_brownian_values"]))
    expected_control_payoffs = np.maximum(inputs["strike"] - expected_terminal_spots, 0.)
    np.testing.assert_allclose(
        exact_gbm_european_payoffs(product, market_data, model, known_increments),
        expected_control_payoffs, rtol=tolerances["relative"], atol=tolerances["absolute"])


# CONTROL ARITHMETIC — ESTIMATE BETA AND CHECK THE EXACT-WAITING LIMIT
def test_part1_coefficient_adjustment_and_exact_european_limit(test_data):
    # Here mean-centred products give 3, and squared control deviations give 2.
    case = test_data["exercise2"]["control_arithmetic"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    coefficient = estimate_control_coefficient(
        inputs["training_target"], inputs["training_control"])
    assert coefficient == expected["coefficient"]
    adjusted = apply_control_variate(inputs["evaluation_target"], inputs["evaluation_control"],
                                     inputs["control_mean"], coefficient)
    np.testing.assert_allclose(adjusted, expected["adjusted_observations"],
                               rtol=tolerances["adjusted_relative"],
                               atol=tolerances["adjusted_absolute"])
    assert estimate_control_coefficient(
        inputs["training_target"], inputs["constant_control"]
    ) == expected["constant_control_coefficient"]

    # Here exact simulation plus waiting gives X = Y path by path at zero rates.
    # Estimated beta is then one, and every adjusted observation equals E[Y].
    case = test_data["exercise2"]["exact_waiting_control"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    product = AmericanPut(inputs["strike"], inputs["maturity_years"])
    market_data = MarketData(inputs["spot"], inputs["interest_rate"],
                             volatility=inputs["volatility"])
    model = GeometricBrownianMotionModel(inputs["interest_rate"], inputs["volatility"])
    increments = np.array(inputs["brownian_increments"])
    european_payoffs = exact_gbm_european_payoffs(product, market_data, model, increments)
    exact_mean = value_european_put_black_scholes_zero_rates(
        inputs["spot"], inputs["strike"], inputs["maturity_years"], inputs["volatility"])
    coefficient = estimate_control_coefficient(european_payoffs, european_payoffs)
    adjusted = apply_control_variate(european_payoffs, european_payoffs, exact_mean, coefficient)
    assert coefficient == expected["coefficient"]
    np.testing.assert_allclose(adjusted, exact_mean, rtol=tolerances["relative"],
                               atol=tolerances["absolute"])
    assert np.var(adjusted, ddof=1) < tolerances["variance_upper_bound"]


# INTEGRATION — REUSE THE PLAIN ESTIMATOR AND REPORT EVALUATION VARIANCE / N
def test_part1_controlled_valuation_keeps_the_plain_lsmc_estimator():
    product = AmericanPut(40., 1 / 12)
    market_data = MarketData(40., 0., volatility=.2)
    model = GeometricBrownianMotionModel(0., .2)
    plain_settings = SimulationSettings(512, 1024, np.random.SeedSequence(30),
                                        exercise_dates_per_year=365)
    paired_settings = SimulationSettings(512, 1024, np.random.SeedSequence(30),
                                         exercise_dates_per_year=365)
    plain = value_american_put(
        product, market_data, model, value_american_option_longstaff_schwartz,
        simulate_paths_euler_maruyama, spot_quadratic_basis, plain_settings)
    paired = value_american_put_with_european_control(
        product, market_data, model, simulate_paths_euler_maruyama,
        spot_quadratic_basis, paired_settings)
    observations = paired["pathwise"]
    american = observations["american_evaluation_payoffs"]
    european = observations["european_evaluation_payoffs"]
    controlled = observations["controlled_evaluation_observations"]
    np.testing.assert_array_equal(american, plain["discounted_payoffs"])
    assert paired["plain_price"] == plain["option_value"]
    assert paired["training_seed_spawn_key"] == (0,)
    assert paired["evaluation_seed_spawn_key"] == (1,)

    expected_coefficient = np.cov(
        observations["training_american_payoffs"],
        observations["training_european_payoffs"], ddof=1)[0, 1] / np.var(
            observations["training_european_payoffs"], ddof=1)
    np.testing.assert_allclose(paired["control_coefficient"], expected_coefficient)
    np.testing.assert_allclose(controlled, american - expected_coefficient * (
        european - paired["european_black_scholes_value"]))
    np.testing.assert_allclose(paired["controlled_price"], np.mean(controlled))
    np.testing.assert_allclose(paired["plain_estimator_variance"], np.var(american, ddof=1) / 1024)
    np.testing.assert_allclose(paired["controlled_estimator_variance"], np.var(controlled, ddof=1) / 1024)
    np.testing.assert_allclose(paired["controlled_standard_error"]**2,
                               paired["controlled_estimator_variance"])
    np.testing.assert_allclose(paired["payoff_correlation"], np.corrcoef(american, european)[0, 1])
