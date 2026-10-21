"""EXERCISE 1 REQUIREMENT: SIMULATE USING EULER–MARUYAMA.

Here we use X_next = X + drift(X)*dt + diffusion(X)*sqrt(dt)*Z.
Both coefficients use the current state. GBM is not simulated with an exact
exponential transition. The supplied grids separate simulation from exercise.
"""
import numpy as np
from models import StochasticVolatilityModel


# SHARED GBM RANDOMNESS FOR THE AMERICAN PATH AND EUROPEAN CONTROL
def generate_brownian_increments(simulation_times, path_count, seed):
    time_steps = np.diff(simulation_times)
    if np.any(time_steps <= 0):
        raise ValueError("Simulation dates must increase")
    random_numbers = np.random.default_rng(seed)
    # Here drawing dates first preserves Exercise 1's GBM random-number order.
    # The returned shape is (paths, intervals), so column j is dW on interval j.
    normal_shocks = random_numbers.standard_normal((len(time_steps), path_count)).T
    return normal_shocks * np.sqrt(time_steps)


# SHARED TWO-DRIVER RANDOMNESS FOR THE EULER–MILSTEIN COMPARISON
def generate_two_driver_brownian_increments(simulation_times, path_count, seed):
    time_steps = np.diff(simulation_times)
    if np.any(time_steps <= 0):
        raise ValueError("Simulation dates must increase")
    random_numbers = np.random.default_rng(seed)
    # Here the two drivers are INDEPENDENT. Each simulator applies model rho.
    # Shape is (paths, intervals, 2); drawing date, driver, path preserves the
    # original Euler ordering: all spot shocks, then all second-driver shocks.
    normal_shocks = random_numbers.standard_normal((len(time_steps), 2, path_count))
    return normal_shocks.transpose(2, 0, 1) * np.sqrt(time_steps)[None, :, None]


def simulate_paths_euler_maruyama(market_data, model, simulation_times,
                                  exercise_date_indices, path_count, seed,
                                  brownian_increments=None):
    # 1. PREPARE THE RANDOM GENERATOR AND DATES
    random_numbers = np.random.default_rng(seed)
    exercise_times = simulation_times[exercise_date_indices]
    has_stochastic_variance = isinstance(model, StochasticVolatilityModel)
    if brownian_increments is not None:
        # Here supplied increments couple Euler and Milstein without redrawing.
        # For stochastic variance, the final axis contains independent drivers.
        brownian_increments = np.asarray(brownian_increments, dtype=float)
        expected_shape = (path_count, len(simulation_times) - 1)
        if has_stochastic_variance:
            expected_shape += (2,)
        if brownian_increments.shape != expected_shape:
            raise ValueError(f"Brownian increments must have shape {expected_shape}")
        if not np.all(np.isfinite(brownian_increments)):
            raise ValueError("Brownian increments must be finite")

    # 2. SET THE INITIAL STATES AND ALLOCATE THE SAVED PATHS
    current_spots = np.full(path_count, float(market_data.spot))
    # Here rows are paths, columns are exercise dates; [:, 0] is every initial spot.
    spot_paths = np.empty((path_count, len(exercise_times)))
    spot_paths[:, 0] = current_spots
    if has_stochastic_variance:
        current_variances = np.full(path_count, float(model.initial_variance))
        variance_paths = np.empty_like(spot_paths)
        variance_paths[:, 0] = current_variances
    variance_projection_count = 0
    zero_variance_start_count = 0
    negative_spot_state_count = 0
    next_exercise_column = 1

    # 3. MOVE THROUGH THE SIMULATION DATES
    for next_date_index in range(1, len(simulation_times)):
        current_date_index = next_date_index - 1
        time_step = simulation_times[next_date_index] - simulation_times[current_date_index]

        # 4. DRAW BROWNIAN INCREMENTS
        if brownian_increments is None:
            spot_normal_shocks = random_numbers.standard_normal(path_count)
            spot_brownian_increments = np.sqrt(time_step) * spot_normal_shocks
        elif has_stochastic_variance:
            spot_brownian_increments = brownian_increments[:, current_date_index, 0]
        else:
            spot_brownian_increments = brownian_increments[:, current_date_index]
        if has_stochastic_variance:
            correlation = model.brownian_correlation
            if brownian_increments is None:
                # Here we retain the original draw order AND arithmetic order;
                # existing Exercise 1 results therefore remain reproducible.
                independent_normal_shocks = random_numbers.standard_normal(path_count)
                variance_brownian_increments = np.sqrt(time_step) * (
                    correlation * spot_normal_shocks
                    + np.sqrt(1 - correlation**2) * independent_normal_shocks)
            else:
                independent_increments = brownian_increments[:, current_date_index, 1]
                variance_brownian_increments = (correlation * spot_brownian_increments
                    + np.sqrt(1 - correlation**2) * independent_increments)

        # 5. APPLY EULER–MARUYAMA USING THE CURRENT STATES
        if has_stochastic_variance:
            zero_variance_start_count += int(np.count_nonzero(current_variances == 0))
            proposed_variances = (current_variances
                + model.variance_drift(current_variances) * time_step
                + model.variance_diffusion(current_spots, current_variances)
                * variance_brownian_increments)
            # Here we catch numerical failure before projection can hide it.
            if not np.all(np.isfinite(proposed_variances)):
                raise FloatingPointError("Euler produced a non-finite variance")
            variance_projection_count += int(np.count_nonzero(proposed_variances < 0))
            # Here we project negative proposals to zero so sqrt(v) stays defined.
            # This is projected Euler, not full truncation; it can introduce bias.
            next_variances = np.maximum(proposed_variances, 0.0)
            spot_diffusion = model.spot_diffusion(current_spots, current_variances)
        else:
            spot_diffusion = model.spot_diffusion(current_spots)
        next_spots = (current_spots + model.spot_drift(current_spots) * time_step
                      + spot_diffusion * spot_brownian_increments)
        if not np.all(np.isfinite(next_spots)):
            raise FloatingPointError("Euler produced a non-finite spot")
        # Here we retain negative Euler spots and count them, without clipping.
        negative_spot_state_count += int(np.count_nonzero(next_spots < 0))
        current_spots = next_spots
        if has_stochastic_variance:
            current_variances = next_variances

        # 6. SAVE STATES ON EXERCISE DATES
        if (next_exercise_column < len(exercise_date_indices)
                and next_date_index == exercise_date_indices[next_exercise_column]):
            spot_paths[:, next_exercise_column] = current_spots
            if has_stochastic_variance:
                variance_paths[:, next_exercise_column] = current_variances
            next_exercise_column += 1

    # 7. RETURN THE PATHS, DATES AND SIMULATION DIAGNOSTICS
    state_paths = {"spot": spot_paths}
    if has_stochastic_variance:
        state_paths["variance"] = variance_paths
    # Here counts refer to path-step observations, not distinct paths.
    diagnostics = {"simulation_times": simulation_times,
                   "variance_projection_count": variance_projection_count,
                   "negative_spot_state_count": negative_spot_state_count,
                   "zero_variance_start_count": zero_variance_start_count,
                   "path_step_count": path_count * (len(simulation_times) - 1)}
    return state_paths, exercise_times, diagnostics
