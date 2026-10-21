"""EXERCISE 1 — SIX SMALL CHECKS OF THE NUMERICAL CALCULATION.

Here we check known calculations before interpreting the larger assignment
studies. The eight-path fixture is from Longstaff–Schwartz, Section 1.
"""
import json
import numpy as np
from basis import spot_quadratic_basis, spot_variance_quadratic_basis
from euler_maruyama import simulate_paths_euler_maruyama
from longstaff_schwartz_lsmc import (
    apply_fitted_exercise_rules, value_american_option_longstaff_schwartz)
from market_data import MarketData
from models import GeometricBrownianMotionModel, StochasticVolatilityModel
from products import AmericanPut
from exercise1.run_exercise1 import run_exercise1
from valuation import SimulationSettings, make_simulation_and_exercise_grids


class FixedNormalShocks:
    """Here known shocks replace random draws only inside the hand checks."""
    def __init__(self, shocks):
        self.shocks = iter(shocks)

    def standard_normal(self, path_count):
        shocks = np.asarray(next(self.shocks), dtype=float)
        assert shocks.shape == (path_count,)
        return shocks


# EULER — CHECK THE PRINTED EQUATIONS, OLD STATES AND VARIANCE PROJECTION
def test_exercise1_euler_updates_match_hand_calculations(monkeypatch, test_data):
    case = test_data["exercise1"]["gbm_euler_step"]
    inputs, expected, tolerance = case["inputs"], case["expected"], case["tolerances"]
    market_data = MarketData(**inputs["market"])
    gbm = GeometricBrownianMotionModel(**inputs["model"])
    monkeypatch.setattr(np.random, "default_rng",
                        lambda seed: FixedNormalShocks(inputs["normal_shocks"]))
    times = np.asarray(inputs["times"])
    paths, dates, diagnostics = simulate_paths_euler_maruyama(
        market_data, gbm, times, np.arange(len(times)), inputs["path_count"], inputs["seed"])
    # Here S_next = 2 + .1*2*.25 + .2*2*sqrt(.25)*Z.
    np.testing.assert_allclose(paths["spot"], expected["spot_paths"], **tolerance)
    np.testing.assert_array_equal(dates, times)
    assert diagnostics["negative_spot_state_count"] == expected["negative_spot_state_count"]

    case = test_data["exercise1"]["stochastic_volatility_euler_steps"]
    inputs, expected, tolerance = case["inputs"], case["expected"], case["tolerances"]
    market_data = MarketData(**inputs["market"])
    model = StochasticVolatilityModel(**inputs["model"])
    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedNormalShocks(
        inputs["normal_shocks"]))
    times = np.asarray(inputs["times"])
    paths, dates, diagnostics = simulate_paths_euler_maruyama(
        market_data, model, times, np.arange(len(times)), inputs["path_count"], inputs["seed"])
    # Here dW_variance = [.3, -1.1]; the printed extra S gives diffusion .04.
    # Variance proposals [.0545, -.0015] therefore include one projection to zero.
    np.testing.assert_allclose(paths["variance"], expected["variance_paths"], **tolerance)
    # Here both spot updates use the old variance, even when the new one is zero.
    np.testing.assert_allclose(paths["spot"], expected["spot_paths"], **tolerance)
    assert diagnostics["variance_projection_count"] == expected["variance_projection_count"]


# LONGSTAFF–SCHWARTZ — REPRODUCE SECTION 1'S EIGHT-PATH WORKED EXAMPLE
def test_exercise1_lsmc_reproduces_paper_targets_decisions_and_value(
        monkeypatch, test_data, test_data_directory):
    case = test_data["exercise1"]["longstaff_eight_paths"]
    inputs, expected, tolerance = case["inputs"], case["expected"], case["tolerances"]
    fixture = test_data_directory / "exercise1" / inputs["paths_file"]
    paper_paths = np.loadtxt(fixture, delimiter=",", skiprows=1)[:, 1:]
    product = AmericanPut(**inputs["product"])
    market_data = MarketData(**inputs["market"])
    times = np.asarray(inputs["times"])
    indices = np.arange(len(times))
    supplied_seeds = []

    def simulate_paper_paths(market_data, model, simulation_times,
                             exercise_date_indices, path_count, seed):
        supplied_seeds.append(seed.spawn_key)
        assert path_count == len(paper_paths)
        return {"spot": paper_paths.copy()}, simulation_times[exercise_date_indices], {}

    regressions = []
    ordinary_least_squares = np.linalg.lstsq

    def record_regression(basis_matrix, targets, rcond):
        regressions.append((basis_matrix.copy(), targets.copy()))
        return ordinary_least_squares(basis_matrix, targets, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", record_regression)
    result = value_american_option_longstaff_schwartz(
        product, market_data, GeometricBrownianMotionModel(**inputs["model"]),
        simulate_paper_paths, spot_quadratic_basis,
        SimulationSettings(inputs["training_paths"], inputs["evaluation_paths"],
                           np.random.SeedSequence(inputs["seed"])), times, indices)
    # Here we replay the same published paths to reproduce its arithmetic only.
    # Genuine valuation experiments use separate training and evaluation samples.
    discount = np.exp(-market_data.interest_rate)
    arithmetic_tolerance = {"rtol": tolerance["arithmetic_rtol"],
                            "atol": tolerance["arithmetic_atol"]}
    np.testing.assert_allclose(regressions[0][1], discount * np.asarray(
        expected["date2_target_payoffs_before_one_year_discount"]), **arithmetic_tolerance)
    np.testing.assert_allclose(regressions[1][1], discount * np.asarray(
        expected["date1_target_payoffs_before_one_year_discount"]), **arithmetic_tolerance)
    np.testing.assert_allclose(regressions[0][0][:, 1], np.asarray(
        expected["date2_in_the_money_spots"]) / product.strike, **arithmetic_tolerance)
    # Here coefficients are converted from our S/K basis to the paper's S basis.
    coefficient_scaling = np.array([1., product.strike, product.strike**2])
    np.testing.assert_allclose(result["coefficients_by_date"][2] / coefficient_scaling,
        expected["date2_coefficients_in_spot_basis"],
        rtol=tolerance["rounded_coefficient_rtol"], atol=tolerance["rounded_coefficient_atol"])
    np.testing.assert_allclose(result["coefficients_by_date"][1] / coefficient_scaling,
        expected["date1_coefficients_in_spot_basis"],
        rtol=tolerance["rounded_coefficient_rtol"], atol=tolerance["rounded_coefficient_atol"])
    np.testing.assert_array_equal(result["exercise_date_indices"], expected["exercise_date_indices"])
    exercise_times = times[np.asarray(expected["exercise_date_indices"])]
    expected_payoffs = (np.asarray(expected["undiscounted_exercise_payoffs"])
                        * np.exp(-market_data.interest_rate * exercise_times))
    np.testing.assert_allclose(result["discounted_payoffs"], expected_payoffs,
        rtol=tolerance["arithmetic_rtol"], atol=tolerance["discounted_payoff_atol"])
    assert abs(result["option_value"] - expected["option_value"]) < tolerance["option_value_atol"]
    assert result["exercise_at_time_zero"] is expected["exercise_at_time_zero"]
    assert supplied_seeds == [tuple(key) for key in expected["seed_spawn_keys"]]


# EVALUATION — PAY ONCE, DISCOUNT CORRECTLY AND NEVER REFIT
def test_exercise1_frozen_rules_pay_once_without_future_information(monkeypatch, test_data):
    case = test_data["exercise1"]["fixed_exercise_rules"]
    inputs, expected, tolerance = case["inputs"], case["expected"], case["tolerances"]
    product = AmericanPut(**inputs["product"])
    market_data = MarketData(**inputs["market"])
    times = np.asarray(inputs["times"])
    state_paths = {"spot": np.array(inputs["spot_paths"])}
    coefficients = [None if row is None else np.asarray(row)
                    for row in inputs["coefficients_by_date"]]

    def forbid_refitting(*args, **kwargs):
        raise AssertionError("Evaluation must apply coefficients without fitting again")

    monkeypatch.setattr(np.linalg, "lstsq", forbid_refitting)
    result = apply_fitted_exercise_rules(product, market_data, state_paths,
        times, spot_quadratic_basis, coefficients, False)
    expected_payoffs = np.asarray(expected["discounted_payoffs"])
    np.testing.assert_allclose(result["discounted_payoffs"], expected_payoffs,
                               rtol=tolerance["rtol"], atol=tolerance["atol"])
    np.testing.assert_array_equal(result["exercise_date_indices"], expected["exercise_date_indices"])
    assert abs(result["option_value"] - expected_payoffs.mean()) < tolerance["summary_atol"]
    assert abs(result["standard_error"] - expected_payoffs.std(ddof=1)
               / np.sqrt(len(expected_payoffs))) < tolerance["summary_atol"]
    # Here later prices change after exercise; payments and decisions must not.
    state_paths["spot"][0, 2:] = inputs["later_price_replacement"]
    state_paths["spot"][1, 3] = inputs["later_price_replacement"]
    repeated = apply_fitted_exercise_rules(product, market_data, state_paths,
        times, spot_quadratic_basis, coefficients, False)
    np.testing.assert_array_equal(repeated["discounted_payoffs"], result["discounted_payoffs"])
    immediate = apply_fitted_exercise_rules(product, MarketData(**inputs["immediate_market"]),
        state_paths, times, spot_quadratic_basis, coefficients, True)
    np.testing.assert_array_equal(immediate["discounted_payoffs"], expected["immediate_payoffs"])
    assert immediate["standard_error"] == expected["immediate_standard_error"]


# REGRESSION STATES — SCALED TERMS AND THE CONSTANT-VARIANCE LIMIT
def test_exercise1_basis_terms_and_constant_variance_recover_gbm(monkeypatch):
    product = AmericanPut(strike=10., maturity_years=1.)
    states = {"spot": np.array([5., 5., 5., 10., 10., 15.]),
              "variance": np.array([.04, .08, .12, .04, .08, .04])}
    spot_basis = spot_quadratic_basis(states, product)
    full_basis = spot_variance_quadratic_basis(states, product, variance_scale=.04)
    np.testing.assert_allclose(spot_basis[0], [1., .5, .25])
    np.testing.assert_allclose(full_basis[1], [1., .5, 2., .25, 1., 4.])
    assert np.linalg.matrix_rank(spot_basis) == 3
    assert np.linalg.matrix_rank(full_basis) == 6
    # Here xi=0 and theta=v0 keep variance fixed. Both models receive identical
    # spot shocks: using the same seed alone would not couple their extra draws.
    repeated_shocks = [[.5, -.5, 1.]] * 4
    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedNormalShocks(repeated_shocks))
    market_data = MarketData(spot=1., interest_rate=.03)
    model = StochasticVolatilityModel(.04, 2., .04, 0., .5, spot_drift_rate=.03)
    times = np.array([0., .25, .5])
    sv_paths, sv_dates, sv_diagnostics = simulate_paths_euler_maruyama(
        market_data, model, times, np.arange(3), 3, 10)
    gbm_paths, gbm_dates, gbm_diagnostics = simulate_paths_euler_maruyama(
        market_data, GeometricBrownianMotionModel(.03, .2), times, np.arange(3), 3, 10)
    np.testing.assert_array_equal(sv_paths["variance"], np.full((3, 3), .04))
    np.testing.assert_array_equal(sv_paths["spot"], gbm_paths["spot"])
    np.testing.assert_array_equal(sv_dates, gbm_dates)
    assert sv_diagnostics["variance_projection_count"] == gbm_diagnostics["variance_projection_count"] == 0


# TIMESTEP STUDY — HOLD THE EXERCISE GRID FIXED AND REPRODUCE THE PATHS
def test_exercise1_grids_preserve_exercise_dates_and_seed_reproducibility():
    exercise_grids = []
    for days in (1., 2., 4., 7.):
        settings = SimulationSettings(10, 12, np.random.SeedSequence(10), time_step_days=days)
        simulation_times, exercise_indices = make_simulation_and_exercise_grids(.75, settings)
        assert simulation_times[0] == 0. and simulation_times[-1] == .75
        assert np.all(np.diff(simulation_times) > 0)
        assert np.max(np.diff(simulation_times)) <= days/365 + 1e-12
        assert exercise_indices[0] == 0 and exercise_indices[-1] == len(simulation_times)-1
        exercise_grids.append(simulation_times[exercise_indices])
    for exercise_grid in exercise_grids[1:]:
        np.testing.assert_array_equal(exercise_grid, exercise_grids[0])
    market_data = MarketData(40., .06, volatility=.2)
    model = GeometricBrownianMotionModel(.06, .2)
    first_paths, first_dates, first_diagnostics = simulate_paths_euler_maruyama(
        market_data, model, simulation_times, exercise_indices, 12, np.random.SeedSequence(10))
    repeated_paths, repeated_dates, repeated_diagnostics = simulate_paths_euler_maruyama(
        market_data, model, simulation_times, exercise_indices, 12, np.random.SeedSequence(10))
    np.testing.assert_array_equal(first_paths["spot"], repeated_paths["spot"])
    np.testing.assert_array_equal(first_dates, repeated_dates)
    assert first_diagnostics["negative_spot_state_count"] == repeated_diagnostics["negative_spot_state_count"]


# RUNNER — CHANGING THE WORKER COUNT MUST NOT CHANGE VALUATION RESULTS
def test_exercise1_serial_and_parallel_runs_use_the_same_case_seeds():
    serial_results = run_exercise1("gbm_pricing", 32, 48, seed=10, workers=1)
    parallel_results = run_exercise1("gbm_pricing", 32, 48, seed=10, workers=2)
    for serial, parallel in zip(serial_results, parallel_results):
        # Here elapsed runtime is the only field allowed to differ.
        serial.pop("runtime_seconds")
        parallel.pop("runtime_seconds")
        assert json.dumps(serial, sort_keys=True, default=lambda array: array.tolist()) == (
            json.dumps(parallel, sort_keys=True, default=lambda array: array.tolist()))
