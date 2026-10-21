"""EXERCISE 2, PART 3: PUT GAMMA AND ITS ESTIMATOR VARIANCE.

Here gamma measures how the put's delta changes with spot. We estimate it
from three American valuations, using the same paths across the spot bumps.
"""
from dataclasses import replace
from time import perf_counter
import numpy as np
from black_scholes_put import (
    calculate_european_put_gamma_black_scholes_zero_rates,
    value_european_put_black_scholes_zero_rates,
)
from control_variates import apply_control_variate, estimate_control_coefficient
from euler_maruyama import generate_brownian_increments
import valuation


# CENTRAL DIFFERENCE — USED FOR BOTH PATHWISE PAYOFFS AND KNOWN PRICES
def calculate_central_difference_gamma(lower_values, middle_values, upper_values, spot_bump):
    if not np.isfinite(spot_bump) or spot_bump <= 0:
        raise ValueError("The spot bump must be finite and positive")
    lower_values = np.asarray(lower_values, dtype=float)
    middle_values = np.asarray(middle_values, dtype=float)
    upper_values = np.asarray(upper_values, dtype=float)
    if not (lower_values.shape == middle_values.shape == upper_values.shape):
        raise ValueError("The three bumped observations must have matching shapes")
    # Here each array entry belongs to the same path at S-h, S and S+h.
    return (upper_values - 2 * middle_values + lower_values) / spot_bump**2


def calculate_american_put_gamma(
    product, market_data, model, simulation_method, basis_function,
    simulation_settings, spot_bump,
):
    started_at = perf_counter()
    if not np.isfinite(spot_bump) or not 0 < spot_bump < market_data.spot:
        raise ValueError("The spot bump must be positive and smaller than spot")

    # 1. SHARE THE BROWNIAN INCREMENTS ACROSS ALL THREE SPOT BUMPS
    simulation_times, exercise_date_indices = valuation.make_simulation_and_exercise_grids(
        product.maturity_years, simulation_settings
    )
    # Here this saved seed state lets each bumped valuation record the same streams.
    # The increments below already contain the randomness; they are not redrawn.
    seed_state_before_spawning = simulation_settings.seed.state
    training_seed, evaluation_seed = simulation_settings.seed.spawn(2)
    training_increments = generate_brownian_increments(
        simulation_times, simulation_settings.training_paths, training_seed
    )
    evaluation_increments = generate_brownian_increments(
        simulation_times, simulation_settings.evaluation_paths, evaluation_seed
    )

    # 2. REFIT THE AMERICAN EXERCISE RULES AT EACH BUMPED SPOT
    bumped_spots = [market_data.spot - spot_bump, market_data.spot,
                    market_data.spot + spot_bump]
    paired_payoffs = []
    european_values = []
    bumped_diagnostics = []
    for bumped_spot in bumped_spots:
        bumped_market_data = replace(market_data, spot=bumped_spot)
        bumped_settings = replace(
            simulation_settings,
            seed=np.random.SeedSequence(**seed_state_before_spawning),
        )
        # Here each fit uses training paths; its evaluation paths use those rules.
        bumped_result = valuation.value_american_put_with_european_control(
            product, bumped_market_data, model, simulation_method, basis_function,
            bumped_settings, training_brownian_increments=training_increments,
            evaluation_brownian_increments=evaluation_increments,
        )
        paired_payoffs.append(bumped_result["pathwise"])
        bumped_diagnostics.append({
            "spot": float(bumped_spot),
            "exercise_at_time_zero": bumped_result["exercise_at_time_zero"],
            "training_negative_spot_states": bumped_result[
                "training_simulation_diagnostics"]["negative_spot_state_count"],
            "evaluation_negative_spot_states": bumped_result[
                "evaluation_simulation_diagnostics"]["negative_spot_state_count"],
        })
        european_values.append(value_european_put_black_scholes_zero_rates(
            bumped_spot, product.strike, product.maturity_years, model.volatility
        ))

    # 3. FORM ONE GAMMA OBSERVATION PER PATH
    gamma_observations = {}
    for payoff_name in (
        "training_american_payoffs", "training_european_payoffs",
        "american_evaluation_payoffs", "european_evaluation_payoffs",
    ):
        # Here the resulting vector has shape (paths,), not (spots, paths).
        gamma_observations[payoff_name] = calculate_central_difference_gamma(
            paired_payoffs[0][payoff_name], paired_payoffs[1][payoff_name],
            paired_payoffs[2][payoff_name], spot_bump,
        )

    # 4. ESTIMATE A GAMMA-SPECIFIC CONTROL COEFFICIENT FROM TRAINING
    gamma_control_beta = estimate_control_coefficient(
        gamma_observations["training_american_payoffs"],
        gamma_observations["training_european_payoffs"],
    )
    # Here E[G_E] is the finite difference of the three Black–Scholes prices.
    # Analytical gamma is a derivative; at a nonzero bump these are different.
    known_control_mean = float(calculate_central_difference_gamma(
        european_values[0], european_values[1], european_values[2], spot_bump
    ))
    plain_observations = gamma_observations["american_evaluation_payoffs"]
    european_observations = gamma_observations["european_evaluation_payoffs"]
    controlled_observations = apply_control_variate(
        plain_observations, european_observations,
        known_control_mean, gamma_control_beta,
    )

    # 5. REPORT GAMMA AND THE VARIANCE OF ITS ESTIMATED MEAN
    # Here variance / evaluation paths measures sampling uncertainty conditional
    # on the fitted rules and coefficient. Full-run repetitions also vary training.
    evaluation_count = simulation_settings.evaluation_paths
    # Here s_G**2 describes individual gamma observations. Dividing by n gives
    # the estimated variance of their average: the requested gamma estimator.
    plain_observation_variance = float(np.var(plain_observations, ddof=1))
    controlled_observation_variance = float(np.var(controlled_observations, ddof=1))
    plain_estimator_variance = plain_observation_variance / evaluation_count
    controlled_estimator_variance = controlled_observation_variance / evaluation_count
    analytical_gamma = calculate_european_put_gamma_black_scholes_zero_rates(
        market_data.spot, product.strike, product.maturity_years, model.volatility
    )
    return {
        "spot": float(market_data.spot), "strike": float(product.strike),
        "moneyness": float(market_data.spot / product.strike),
        "volatility": float(model.volatility), "maturity_years": product.maturity_years,
        "spot_bump": float(spot_bump), "bump_fraction": float(spot_bump / product.strike),
        "plain_gamma": float(np.mean(plain_observations)),
        "controlled_gamma": float(np.mean(controlled_observations)),
        "analytical_gamma": analytical_gamma,
        "european_finite_difference_gamma": known_control_mean,
        "control_coefficient": gamma_control_beta,
        "plain_estimator_variance": plain_estimator_variance,
        "controlled_estimator_variance": controlled_estimator_variance,
        "plain_standard_error": float(np.sqrt(plain_estimator_variance)),
        "controlled_standard_error": float(np.sqrt(controlled_estimator_variance)),
        "training_paths": simulation_settings.training_paths,
        "evaluation_paths": evaluation_count,
        "training_seed_spawn_key": list(training_seed.spawn_key),
        "evaluation_seed_spawn_key": list(evaluation_seed.spawn_key),
        "simulation_intervals": len(simulation_times) - 1,
        "exercise_intervals": len(exercise_date_indices) - 1,
        "policy_treatment": "Refit at each bumped spot; apply to separate evaluation paths",
        "variance_scope": "Conditional on the three fitted policies and training coefficient",
        "gamma_units": "inverse spot-price unit",
        "runtime_seconds": perf_counter() - started_at,
        "bumped_valuation_diagnostics": bumped_diagnostics,
        "example_observations": [
            {"american_gamma": float(plain_observations[index]),
             "european_gamma": float(european_observations[index]),
             "controlled_gamma": float(controlled_observations[index])}
            for index in range(min(5, evaluation_count))
        ],
    }
