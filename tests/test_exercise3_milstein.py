"""EXERCISE 3 — FOCUSED CHECKS OF THE MILSTEIN CALCULATION.

Here exact transitions, limiting models, a conditional characteristic function
and numerical differentiation check different risks in the implementation.
Inputs, reference definitions and tolerances are in test_data/exercise3/cases.json.
"""
import numpy as np
import pytest
from euler_maruyama import (generate_brownian_increments,
    generate_two_driver_brownian_increments, simulate_paths_euler_maruyama)
from market_data import MarketData
from exercise3.milstein import simulate_paths_milstein, sample_wiktorsson_cross_integrals
from models import GeometricBrownianMotionModel, StochasticVolatilityModel


# GBM — USE EXACT PATHS, NOT ANOTHER COPY OF THE MILSTEIN FORMULA
def test_exercise3_gbm_milstein_improves_coupled_terminal_path_error(test_data):
    case = test_data["exercise3"]["gbm_coupled_convergence"]
    inputs, tolerances = case["inputs"], case["tolerances"]
    path_count = inputs["path_count"]
    finest_steps = inputs["finest_steps"]
    fine_times = np.linspace(0, inputs["maturity"], finest_steps + 1)
    fine_increments = generate_brownian_increments(fine_times, path_count, inputs["increment_seed"])
    market = MarketData(**inputs["market"])
    model = GeometricBrownianMotionModel(**inputs["model"])
    exact_spots = market.spot * np.exp(
        (model.spot_drift_rate - .5 * model.volatility**2) * inputs["maturity"]
        + model.volatility * fine_increments.sum(axis=1))
    errors = {"Euler": [], "Milstein": []}
    for step_count in inputs["step_counts"]:
        increments = fine_increments.reshape(path_count, step_count, finest_steps // step_count).sum(axis=2)
        times = np.linspace(0, inputs["maturity"], step_count + 1)
        for name, simulator in (("Euler", simulate_paths_euler_maruyama),
                                ("Milstein", simulate_paths_milstein)):
            paths, dates, diagnostics = simulator(market, model, times,
                np.array([0, step_count]), path_count, inputs["simulation_seed"], brownian_increments=increments)
            errors[name].append(np.sqrt(np.mean((paths["spot"][:, -1] - exact_spots)**2)))
    assert np.all(np.array(errors["Milstein"]) < np.array(errors["Euler"])
        * tolerances["milstein_to_euler_error_upper_ratio"])
    # Here halving h should approximately halve the Milstein RMS error. Broad
    # ranges allow sampling variation; this is not a fitted-order theorem.
    lower, upper = tolerances["milstein_coarse_to_fine_error_ratio"]
    assert lower < errors["Milstein"][0] / errors["Milstein"][-1] < upper
    lower, upper = tolerances["euler_coarse_to_fine_error_ratio"]
    assert lower < errors["Euler"][0] / errors["Euler"][-1] < upper


# TWO MODELS — ZERO VOLATILITY OF VARIANCE MUST RECOVER SCALAR GBM
def test_exercise3_constant_variance_reduces_to_gbm_and_endpoints_stay_paired(test_data):
    case = test_data["exercise3"]["constant_variance_limit"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    times = np.linspace(0, inputs["maturity"], inputs["step_count"] + 1)
    path_count = inputs["path_count"]
    increments = generate_two_driver_brownian_increments(times, path_count, inputs["increment_seed"])
    market = MarketData(**inputs["market"])
    sv_model = StochasticVolatilityModel(**inputs["stochastic_volatility_model"])
    gbm_model = GeometricBrownianMotionModel(**inputs["gbm_model"])
    for simulator in (simulate_paths_euler_maruyama, simulate_paths_milstein):
        sv_paths, sv_dates, sv_diagnostics = simulator(market, sv_model, times,
            np.arange(len(times)), path_count, inputs["simulation_seed"], brownian_increments=increments)
        gbm_paths, gbm_dates, gbm_diagnostics = simulator(market, gbm_model, times,
            np.arange(len(times)), path_count, inputs["simulation_seed"], brownian_increments=increments[:, :, 0])
        np.testing.assert_allclose(sv_paths["spot"], gbm_paths["spot"],
            rtol=tolerances["spot_relative"], atol=tolerances["spot_absolute"])
        np.testing.assert_array_equal(sv_paths["variance"],
            np.full((path_count, len(times)), expected["constant_variance"]))
        assert sv_diagnostics["variance_projection_count"] == expected["variance_projection_count"]
    # Here changing only area randomness cannot change Euler's supplied paths.
    first_paths, first_dates, first_diagnostics = simulate_paths_euler_maruyama(
        market, sv_model, times, np.arange(len(times)), path_count,
        inputs["alternative_simulation_seeds"][0], increments)
    repeated_paths, repeated_dates, repeated_diagnostics = simulate_paths_euler_maruyama(
        market, sv_model, times, np.arange(len(times)), path_count,
        inputs["alternative_simulation_seeds"][1], increments)
    np.testing.assert_array_equal(first_paths["spot"], repeated_paths["spot"])


# CROSS-INTEGRAL LAW — ENDPOINT CONDITIONING MUST BE CORRECT, NOT JUST THE SUM
def test_exercise3_wiktorsson_matches_conditional_characteristic_function(test_data):
    # Here A=(J12-J21)/2. Conditional on endpoint w, its known characteristic
    # function is x/sinh(x)*exp[-|w|^2/(2h)*(x*coth(x)-1)], x=lambda*h/2
    # (Wiktorsson 2001, Section 1, equation (1.5), with the endpoint phase removed).
    # A plain Gaussian with the same variance fails the nonquadratic shape.
    case = test_data["exercise3"]["wiktorsson_conditional_law"]
    inputs, reference, tolerances = case["inputs"], case["expected"], case["tolerances"]
    path_count = inputs["path_count"]
    time_step = inputs["time_step"]
    for endpoint_pair in inputs["endpoint_pairs"]:
        first_increments = np.full(path_count, endpoint_pair[0] * np.sqrt(time_step))
        second_increments = np.full(path_count, endpoint_pair[1] * np.sqrt(time_step))
        endpoint_square = sum(value**2 for value in endpoint_pair)
        for accuracy in inputs["accuracies"]:
            first_second, second_first, terms = sample_wiktorsson_cross_integrals(
                first_increments, second_increments, time_step,
                np.random.default_rng(inputs["seed_base"] + accuracy), accuracy)
            np.testing.assert_allclose(first_second + second_first,
                first_increments * second_increments,
                rtol=tolerances["sum_relative"], atol=tolerances["sum_absolute"])
            assert terms == reference["terms_per_accuracy"] * accuracy
            normalized_area = (first_second - second_first) / (2 * time_step)
            for frequency in inputs["frequencies"]:
                half_frequency = frequency / 2
                expected = half_frequency / np.sinh(half_frequency) * np.exp(
                    -.5 * endpoint_square * (half_frequency / np.tanh(half_frequency) - 1))
                cosine_samples = np.cos(frequency * normalized_area)
                sine_samples = np.sin(frequency * normalized_area)
                # Six sampling standard errors plus a small finite-series allowance. The
                # exact conditional law is an independent reference for sampler.
                assert abs(cosine_samples.mean() - expected) < (
                    tolerances["sampling_standard_errors"] * cosine_samples.std(ddof=1) / np.sqrt(path_count)
                    + tolerances["cosine_finite_series_allowance"])
                assert abs(sine_samples.mean() - reference["sine_mean"]) < (
                    tolerances["sampling_standard_errors"] * sine_samples.std(ddof=1) / np.sqrt(path_count))


# RHO=1 — CHECK THE VECTOR CORRECTION BY NUMERICALLY DIFFERENTIATING DIFFUSION
def test_exercise3_old_state_updates_and_boundary_convention(monkeypatch, test_data):
    case = test_data["exercise3"]["common_driver_boundary"]
    inputs, expected, tolerances = case["inputs"], case["expected"], case["tolerances"]
    market = MarketData(**inputs["market"])
    model = StochasticVolatilityModel(**inputs["model"])
    times = np.array(inputs["times"])
    increments = np.array(inputs["increments"])
    old_state = np.array([market.spot, model.initial_variance])

    def joint_diffusion(state):
        spot, variance = state
        return np.array([model.spot_diffusion(spot, variance),
                         model.variance_diffusion(spot, variance)])

    jacobian = np.empty((2, 2))
    differentiation_step = inputs["differentiation_step"]
    for column in range(2):
        displacement = np.zeros(2)
        displacement[column] = differentiation_step
        jacobian[:, column] = (joint_diffusion(old_state + displacement)
            - joint_diffusion(old_state - displacement)) / (2 * differentiation_step)
    drift = np.array([model.spot_drift(old_state[0]), model.variance_drift(old_state[1])])
    # Here rho=1 gives one common driver, so the general vector formula is an
    # independent check of all scalar coefficient products and their old states.
    time_step = times[1] - times[0]
    first_increment = increments[0, 0, 0]
    expected_raw = (old_state + drift * time_step + joint_diffusion(old_state) * first_increment
        + .5 * (jacobian @ joint_diffusion(old_state)) * (first_increment**2 - time_step))

    def forbid_area_sampling(*args, **kwargs):
        raise AssertionError("A common driver needs no stochastic-area draws")

    monkeypatch.setattr("exercise3.milstein.sample_wiktorsson_cross_integrals", forbid_area_sampling)
    paths, dates, diagnostics = simulate_paths_milstein(market, model, times,
        np.arange(len(times)), len(increments), inputs["simulation_seed"], brownian_increments=increments)
    np.testing.assert_allclose(paths["spot"][0, 1], expected_raw[0],
        rtol=tolerances["spot_relative"], atol=tolerances["spot_absolute"])
    assert expected_raw[1] < 0
    np.testing.assert_array_equal(paths["variance"], expected["variance_path"])
    assert paths["spot"][0, 2] < paths["spot"][0, 1]
    for name, value in expected["diagnostics"].items():
        assert diagnostics[name] == value


# FAILURES — RETAIN NEGATIVE SPOTS, BUT NEVER RETURN A NON-FINITE PATH
def test_exercise3_negative_spots_and_nonfinite_failures_are_explicit(test_data):
    case = test_data["exercise3"]["negative_spot"]
    inputs, expected = case["inputs"], case["expected"]
    market = MarketData(**inputs["market"])
    times = np.array(inputs["times"])
    increments = np.array(inputs["increments"])
    paths, dates, diagnostics = simulate_paths_milstein(market,
        GeometricBrownianMotionModel(**inputs["model"]), times, np.arange(len(times)),
        len(increments), inputs["simulation_seed"], brownian_increments=increments)
    assert paths["spot"][0, -1] == expected["terminal_spot"]
    for name, value in expected["diagnostics"].items():
        assert diagnostics[name] == value
    case = test_data["exercise3"]["nonfinite_spot"]
    inputs, expected = case["inputs"], case["expected"]
    times = np.array(inputs["times"])
    increments = np.array(inputs["increments"])
    with pytest.raises(FloatingPointError, match=expected["message_pattern"]):
        simulate_paths_milstein(MarketData(**inputs["market"]),
            GeometricBrownianMotionModel(**inputs["model"]), times, np.arange(len(times)),
            len(increments), inputs["simulation_seed"], brownian_increments=increments)


# SAVED RESULTS — DATA INTEGRITY IS SEPARATE FROM SOURCE PROVENANCE

def test_exercise3_saved_results_require_published_integrity(tmp_path, monkeypatch):
    import hashlib
    import json
    import results_integrity
    from exercise3.ui_exercise3 import load_exercise3_results
    document = {"exercise": 3, "gbm_prices": [{"plain_price": 1.0}],
        "code_sha256": {"models.py": "0" * 64}, "config": {}, "environment": {}}
    results = tmp_path / "exercise3_results.json"
    results.write_text(json.dumps(document))
    manifest = {"format_version": 1, "files_sha256": {
        results.name: hashlib.sha256(results.read_bytes()).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(results_integrity, "RESULTS_DIRECTORY", tmp_path)
    # Here saved data are checked against their manifest, independently of
    # the current source files. Changing a saved price must fail the check.
    assert load_exercise3_results(results) == document
    (tmp_path / "models.py").write_text("changed current source")
    assert load_exercise3_results(results) == document
    document["gbm_prices"][0]["plain_price"] = 2.0
    results.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Published result integrity mismatch"):
        load_exercise3_results(results)


def test_exercise3_loader_rejects_incomplete_result_structure(tmp_path):
    import json
    from exercise3.ui_exercise3 import load_exercise3_results
    results = tmp_path / "exercise3_results.json"
    results.write_text(json.dumps({"exercise": 3, "gbm_prices": [{"plain_price": 1.0}]}))
    with pytest.raises(ValueError, match="Incomplete saved result structure"):
        load_exercise3_results(results)
