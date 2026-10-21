"""Here we connect the economic objects to the supplied numerical methods."""
from dataclasses import dataclass
from math import isfinite
import numpy as np
from black_scholes_put import value_european_put_black_scholes_zero_rates
from control_variates import estimate_control_coefficient, apply_control_variate
from euler_maruyama import generate_brownian_increments
from longstaff_schwartz_lsmc import (
    estimate_exercise_rules_longstaff_schwartz, apply_fitted_exercise_rules)
from models import GeometricBrownianMotionModel


@dataclass(frozen=True)
class SimulationSettings:
    training_paths: int
    evaluation_paths: int
    # Here the seed belongs to one valuation; spawning its children advances it.
    seed: np.random.SeedSequence
    time_step_days: float = 1.0
    exercise_dates_per_year: int = 50

    def __post_init__(self):
        if (not isinstance(self.training_paths, int) or not isinstance(self.evaluation_paths, int)
                or min(self.training_paths, self.evaluation_paths) < 2):
            raise ValueError("Training and evaluation counts must be integers >= 2")
        if not all(isfinite(value) and value > 0 for value in
                   (self.time_step_days, self.exercise_dates_per_year)):
            raise ValueError("Timestep and exercise frequency must be finite and positive")
        if not isinstance(self.seed, np.random.SeedSequence):
            raise TypeError("Supply a SeedSequence spawned for this valuation")


# SIMULATION AND EXERCISE GRIDS
def make_simulation_and_exercise_grids(maturity_years, simulation_settings):
    # Here we use calendar days / 365 and retain a short final interval if needed.
    integration_times = np.append(np.arange(
        0, maturity_years, simulation_settings.time_step_days / 365), maturity_years)
    exercise_times = np.append(np.arange(
        0, maturity_years, 1 / simulation_settings.exercise_dates_per_year), maturity_years)
    # Here we insert exercise dates into the simulation grid. Rounding only merges
    # floating-point representations of the same date; we restore exact maturity.
    exercise_times = np.unique(np.round(exercise_times, 12))
    if len(exercise_times) < 2:
        raise ValueError("Maturity is too short for the grid's 12-decimal year precision")
    exercise_times[-1] = maturity_years
    simulation_times = np.unique(np.round(np.concatenate((integration_times, exercise_times)), 12))
    simulation_times[-1] = maturity_years
    exercise_date_indices = np.searchsorted(simulation_times, exercise_times)
    return simulation_times, exercise_date_indices


# AMERICAN PUT VALUATION
def value_american_put(product, market_data, model, valuation_method,
                       simulation_method, basis_function, simulation_settings):
    simulation_times, exercise_date_indices = make_simulation_and_exercise_grids(
        product.maturity_years, simulation_settings)
    return valuation_method(
        product=product, market_data=market_data, model=model,
        simulation_method=simulation_method, basis_function=basis_function,
        simulation_settings=simulation_settings, simulation_times=simulation_times,
        exercise_date_indices=exercise_date_indices)


# EXERCISE 2 — PAIR THE AMERICAN PAYOFF WITH AN EXACT EUROPEAN CONTROL
def value_american_put_with_european_control(
        product, market_data, model, simulation_method, basis_function,
        simulation_settings, training_brownian_increments=None,
        evaluation_brownian_increments=None):
    # 1. USE THE ASSIGNMENT'S ZERO-RATE GBM AND EXPLICIT DATE GRIDS
    if (not isinstance(model, GeometricBrownianMotionModel)
            or market_data.interest_rate != 0 or model.spot_drift_rate != 0
            or market_data.foreign_interest_rate not in (None, 0.0)):
        raise ValueError("Exercise 2 requires GBM with zero rates, yield and drift")
    if market_data.volatility != model.volatility:
        raise ValueError("Market and model volatility must agree")
    simulation_times, exercise_indices = make_simulation_and_exercise_grids(
        product.maturity_years, simulation_settings)
    training_seed, evaluation_seed = simulation_settings.seed.spawn(2)
    if training_brownian_increments is None:
        training_brownian_increments = generate_brownian_increments(
            simulation_times, simulation_settings.training_paths, training_seed)
    if evaluation_brownian_increments is None:
        evaluation_brownian_increments = generate_brownian_increments(
            simulation_times, simulation_settings.evaluation_paths, evaluation_seed)

    # 2. SIMULATE TRAINING PATHS AND ESTIMATE THE EXERCISE RULES
    training_paths, exercise_times, training_diagnostics = simulation_method(
        market_data, model, simulation_times, exercise_indices,
        simulation_settings.training_paths, training_seed,
        brownian_increments=training_brownian_increments)
    exercise_rules = estimate_exercise_rules_longstaff_schwartz(
        product, market_data, training_paths, exercise_times, basis_function)
    training_valuation = apply_fitted_exercise_rules(
        product, market_data, training_paths, exercise_times, basis_function,
        exercise_rules["coefficients_by_date"], exercise_rules["exercise_at_time_zero"])
    training_american_payoffs = training_valuation["discounted_payoffs"]

    # 3. ESTIMATE BETA FROM THE PAIRED TRAINING PAYOFFS
    # Here the exact terminal GBM shares every dW with its simulated American path.
    # The control needs its claimed known expectation: Black–Scholes.
    # An Euler or Milstein European payoff generally has a different expectation.
    # Exact GBM avoids that extra control bias; the American scheme is supplied.
    training_european_payoffs = exact_gbm_european_payoffs(
        product, market_data, model, training_brownian_increments)
    beta = estimate_control_coefficient(
        training_american_payoffs, training_european_payoffs)
    # Here we reuse training observations; there is no additional pilot batch.
    # This minimises training variance. Its benefit is measured on evaluation paths.

    # 4. VALUE THE SAME EXERCISE RULES ON INDEPENDENT EVALUATION PATHS
    evaluation_paths, evaluation_times, evaluation_diagnostics = simulation_method(
        market_data, model, simulation_times, exercise_indices,
        simulation_settings.evaluation_paths, evaluation_seed,
        brownian_increments=evaluation_brownian_increments)
    evaluation_valuation = apply_fitted_exercise_rules(
        product, market_data, evaluation_paths, evaluation_times, basis_function,
        exercise_rules["coefficients_by_date"], exercise_rules["exercise_at_time_zero"])
    american_payoffs = evaluation_valuation["discounted_payoffs"]
    european_payoffs = exact_gbm_european_payoffs(
        product, market_data, model, evaluation_brownian_increments)

    # 5. CENTRE THE CONTROL AT ITS KNOWN EXPECTATION
    known_european_mean = value_european_put_black_scholes_zero_rates(
        market_data.spot, product.strike, product.maturity_years, model.volatility)
    controlled_observations = apply_control_variate(
        american_payoffs, european_payoffs, known_european_mean, beta)
    # Here beta is unchanged from training, so independent evaluation preserves
    # the American estimator's expectation. Simulation and policy errors remain.
    # Here Z is an adjusted statistical observation, not another contract payoff.
    # It can be negative. Clipping it would change the estimator's expectation.

    # 6. REPORT MEANS AND ESTIMATOR VARIANCES ON THE SAME EVALUATION SAMPLE
    plain_variance = float(np.var(american_payoffs, ddof=1))
    controlled_variance = float(np.var(controlled_observations, ddof=1))
    european_variance = float(np.var(european_payoffs, ddof=1))
    number_of_paths = len(american_payoffs)
    correlation = (float(np.corrcoef(american_payoffs, european_payoffs)[0, 1])
                   if plain_variance > 0 and european_variance > 0 else None)
    # Here variance / n is valid for our independent evaluation observations.
    # It conditions on the estimated rules and beta; it excludes training variation.
    return {
        "plain_price": float(np.mean(american_payoffs)),
        "controlled_price": float(np.mean(controlled_observations)),
        "european_black_scholes_value": known_european_mean,
        "simulated_european_price": float(np.mean(european_payoffs)),
        "control_coefficient": beta, "payoff_correlation": correlation,
        "plain_payoff_variance": plain_variance,
        "controlled_observation_variance": controlled_variance,
        "plain_estimator_variance": plain_variance / number_of_paths,
        "controlled_estimator_variance": controlled_variance / number_of_paths,
        "plain_standard_error": float(np.sqrt(plain_variance / number_of_paths)),
        "controlled_standard_error": float(np.sqrt(controlled_variance / number_of_paths)),
        "variance_reduction_factor": (plain_variance / controlled_variance
                                      if controlled_variance > 0 else None),
        "training_control_variance": float(np.var(training_european_payoffs, ddof=1)),
        "training_payoff_covariance": float(np.cov(
            training_american_payoffs, training_european_payoffs, ddof=1)[0, 1]),
        "training_simulation_diagnostics": training_diagnostics,
        "evaluation_simulation_diagnostics": evaluation_diagnostics,
        "regression_diagnostics": exercise_rules["regression_diagnostics"],
        "exercise_times": exercise_times,
        "training_seed_spawn_key": training_seed.spawn_key,
        "evaluation_seed_spawn_key": evaluation_seed.spawn_key,
        "exercise_at_time_zero": exercise_rules["exercise_at_time_zero"],
        "pathwise": {
            "training_american_payoffs": training_american_payoffs,
            "training_european_payoffs": training_european_payoffs,
            "american_evaluation_payoffs": american_payoffs,
            "european_evaluation_payoffs": european_payoffs,
            "controlled_evaluation_observations": controlled_observations}}


# EXACT EUROPEAN CONTROL — ONE TERMINAL PAYOFF PER PATH
def exact_gbm_european_payoffs(product, market_data, model, brownian_increments):
    # Here rows are paths and columns are daily increments. Summing columns
    # gives W_T on each path; no extra random numbers are generated.
    terminal_spots = market_data.spot * np.exp(
        (model.spot_drift_rate - 0.5 * model.volatility**2) * product.maturity_years
        + model.volatility * np.sum(brownian_increments, axis=1))
    return product.payoff(terminal_spots) * np.exp(
        -market_data.interest_rate * product.maturity_years)


# EXERCISE 3 — COMPARE SCHEMES ON THE SAME SUPPLIED BROWNIAN PATHS
def value_american_put_with_supplied_brownian_increments(
        product, market_data, model, simulation_method, basis_function,
        simulation_settings, training_brownian_increments,
        evaluation_brownian_increments):
    simulation_times, exercise_indices = make_simulation_and_exercise_grids(
        product.maturity_years, simulation_settings)
    training_seed, evaluation_seed = simulation_settings.seed.spawn(2)

    # 1. SIMULATE TRAINING STATES AND REUSE THE EXISTING LONGSTAFF–SCHWARTZ FIT
    training_states, exercise_times, training_diagnostics = simulation_method(
        market_data, model, simulation_times, exercise_indices,
        simulation_settings.training_paths, training_seed,
        brownian_increments=training_brownian_increments)
    rules = estimate_exercise_rules_longstaff_schwartz(
        product, market_data, training_states, exercise_times, basis_function)
    del training_states

    # 2. APPLY THIS SCHEME'S RULES TO INDEPENDENT EVALUATION STATES
    evaluation_states, exercise_times, evaluation_diagnostics = simulation_method(
        market_data, model, simulation_times, exercise_indices,
        simulation_settings.evaluation_paths, evaluation_seed,
        brownian_increments=evaluation_brownian_increments)
    result = apply_fitted_exercise_rules(
        product, market_data, evaluation_states, exercise_times, basis_function,
        rules["coefficients_by_date"], rules["exercise_at_time_zero"])

    # 3. RETAIN REALISED PAYMENTS AND ONE JOINT PATH FOR THE EXPLANATORY GRAPH
    # Here each state has shape (paths, exercise dates); row 0 is one scenario.
    result.update({"exercise_times": exercise_times,
        "example_state_path": {name: values[0].copy()
                               for name, values in evaluation_states.items()},
        "training_simulation_diagnostics": training_diagnostics,
        "evaluation_simulation_diagnostics": evaluation_diagnostics})
    return result


# EXERCISE 4 — DAILY STORAGE SIMULATION AND REPEATED OPERATING DECISIONS
@dataclass(frozen=True)
class StorageSimulationSettings:
    training_paths: int
    evaluation_paths: int
    seed: np.random.SeedSequence
    inventory_step: float = 2500.0
    antithetic: bool = True

    def __post_init__(self):
        counts = (self.training_paths, self.evaluation_paths)
        if any(not isinstance(count, int) or count < 2 for count in counts):
            raise ValueError("Storage training and evaluation counts must be integers >= 2")
        if self.antithetic and any(count % 2 for count in counts):
            raise ValueError("Antithetic paths must be generated in complete pairs")
        if not isinstance(self.seed, np.random.SeedSequence):
            raise TypeError("Supply a SeedSequence for this storage valuation")
        if not isfinite(self.inventory_step) or self.inventory_step <= 0:
            raise ValueError("Storage inventory spacing must be finite and positive")


def value_gas_storage(product, market_data, model, simulation_settings, progress=None):
    from exercise4.gas_price_simulation import simulate_gas_prices
    from exercise4.gas_storage_lsmc import estimate_storage_decisions, apply_storage_decisions
    from models import GasPriceModel

    # 1. USE THE ZERO-RATE STORAGE CONTRACT AND ITS DAILY OPERATING DATES
    if (not isinstance(model, GasPriceModel) or market_data.interest_rate != 0
            or market_data.foreign_interest_rate not in (None, 0.0)):
        raise ValueError("Exercise 4 requires GasPriceModel and zero interest rates")
    dates = product.operating_dates()
    training_seed, evaluation_seed = simulation_settings.seed.spawn(2)

    # 2. ESTIMATE CONTINUATION FUNCTIONS AND STORAGE DECISIONS
    if progress is not None:
        progress("Simulate training prices", 0, 1)
    training_prices = simulate_gas_prices(market_data.spot, model, dates,
        simulation_settings.training_paths, training_seed, simulation_settings.antithetic)
    decisions = estimate_storage_decisions(training_prices, product, model,
                                           simulation_settings.inventory_step, progress=progress)

    # 3. APPLY THESE DECISIONS TO INDEPENDENT PRICE PATHS
    if progress is not None:
        progress("Simulate independent evaluation prices", 0, 1)
    evaluation_prices = simulate_gas_prices(market_data.spot, model, dates,
        simulation_settings.evaluation_paths, evaluation_seed, simulation_settings.antithetic)
    evaluation = apply_storage_decisions(evaluation_prices, product, decisions,
                                         simulation_settings.antithetic, progress=progress)
    return {"training_value": decisions["training_value"],
        "value": evaluation["value"], "standard_error": evaluation["standard_error"],
        "decisions": decisions, "evaluation": evaluation,
        "operating_dates": dates,
        "training_spot_paths": training_prices,
        "evaluation_spot_paths": evaluation_prices,
        "training_sample_prices": training_prices[:3].copy(),
        "evaluation_sample_prices": evaluation_prices[:3].copy(),
        "training_seed_spawn_key": training_seed.spawn_key,
        "evaluation_seed_spawn_key": evaluation_seed.spawn_key}
