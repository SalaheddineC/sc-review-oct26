"""EXERCISE 3 — MILSTEIN FOR GBM AND THE PRINTED SPOT–VARIANCE MODEL.

Here we answer Exercise 3 by replacing the Euler step while retaining the
products, models and Longstaff-Schwartz valuation from Exercises 1 and 2.
The runner compares both simulators on shared Brownian increments.

Milstein (1975, Sections 2 and 5) supplies the correction. The report's joint
Milstein derivation explains the GBM and printed stochastic-variance updates.
The printed variance diffusion contains spot: xi*S*sqrt(v). We keep that S.
For this joint model, scalar corrections alone would omit cross integrals.
Wiktorsson (2001, equations 2.1-2.2, algorithm p. 481, Theorem 4.1) supplies
their sampler; the joint Milstein derivation explains the boundary convention.
"""
import numpy as np
from models import GeometricBrownianMotionModel, StochasticVolatilityModel


# CROSS INTEGRALS — WIKTORSSON'S TWO-INDEPENDENT-DRIVER CONSTRUCTION
def sample_wiktorsson_cross_integrals(first_increments, second_increments,
                                      time_step, random_numbers,
                                      area_accuracy_multiplier=1):
    """Return J12, J21 and the term count, conditional on endpoint increments.

    Here first_increments and second_increments are independent dB1 and dB2,
    one value per path: shape (paths,). They are not yet the correlated shocks.
    Jij = integral (B_i(u)-B_i(t)) dB_j(u): first i, then integration against j.
    The area convention is A = (J12-J21)/2, so J12 = dB1*dB2/2 + A.
    The truncated Fourier series retains endpoint dependence; only its tail is
    approximated by a conditional Gaussian. The whole area is not Gaussian.
    """
    # Here matching finite arrays provide one paired increment entry per path.
    first_increments = np.asarray(first_increments, dtype=float)
    second_increments = np.asarray(second_increments, dtype=float)
    if (first_increments.ndim != 1 or first_increments.shape != second_increments.shape
            or not np.all(np.isfinite(first_increments))
            or not np.all(np.isfinite(second_increments))):
        raise ValueError("Supply two finite endpoint arrays with shape (paths,)")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("The cross-integral timestep must be finite and positive")
    if not np.isfinite(area_accuracy_multiplier) or area_accuracy_multiplier < 1:
        raise ValueError("Area accuracy multiplier must be finite and at least one")

    # 1. SELECT ACCURACY FROM THE PUBLISHED TWO-DRIVER ERROR BOUND
    # Here E[(J12-J12_approx)^2] <= 5*h^2/(6*pi^2*n^2). We choose <= h^3
    # with time in years. The report's cross-integral accuracy check tests this by
    # doubling the terms. More terms require more random draws and computation.
    minimum_series_terms = int(np.ceil(np.sqrt(5 / (6 * np.pi**2 * time_step))))
    fourier_term_count = int(np.ceil(minimum_series_terms * area_accuracy_multiplier))
    path_count = len(first_increments)
    first_driver_endpoint_shift = np.sqrt(2 / time_step) * first_increments
    second_driver_endpoint_shift = np.sqrt(2 / time_step) * second_increments
    finite_area_series = np.zeros(path_count)
    included_inverse_square_sum = 0.0

    # 2. SIMULATE THE FINITE SERIES, CONDITIONAL ON THE GIVEN ENDPOINTS
    # Here x and y name the paper's independent normal Fourier coefficients.
    # Each row has one draw per path; the endpoint shifts retain conditioning.
    for term_index in range(1, fourier_term_count + 1):
        (first_driver_fourier_x, second_driver_fourier_x,
         first_driver_fourier_y, second_driver_fourier_y) = random_numbers.standard_normal((4, path_count))
        finite_area_series += (
            first_driver_fourier_x * (second_driver_fourier_y + second_driver_endpoint_shift)
            - second_driver_fourier_x * (first_driver_fourier_y + first_driver_endpoint_shift)
        ) / term_index
        included_inverse_square_sum += 1 / term_index**2

    # 3. ADD THE CONDITIONAL GAUSSIAN TAIL AND RECONSTRUCT BOTH INTEGRALS
    # Here a_n is the remaining sum of 1/k^2. The variance below belongs to
    # the unscaled series tail; h/(2*pi) then scales the complete area.
    # The endpoints fix J12+J21. The sampled area supplies their difference.
    remaining_inverse_square_sum = max(np.pi**2 / 6 - included_inverse_square_sum, 0.0)
    unscaled_tail_variance = 2 * remaining_inverse_square_sum * (
        1 + (first_increments**2 + second_increments**2) / time_step)
    gaussian_area_tail = np.sqrt(unscaled_tail_variance) * random_numbers.standard_normal(path_count)
    stochastic_area = time_step / (2 * np.pi) * (finite_area_series + gaussian_area_tail)
    first_then_second_integral = 0.5 * first_increments * second_increments + stochastic_area
    second_then_first_integral = first_increments * second_increments - first_then_second_integral
    return first_then_second_integral, second_then_first_integral, fourier_term_count


# JOINT SIMULATION — BOTH STATE UPDATES USE THE SAME OLD SPOT AND VARIANCE
def simulate_paths_milstein(market_data, model, simulation_times,
                            exercise_date_indices, path_count, seed,
                            brownian_increments=None, area_seed=None,
                            area_accuracy_multiplier=1):
    """Return states at exercise dates, those dates, and numerical diagnostics.

    Here the signature matches Euler so valuation can switch the simulator.
    Supplied increments have shape (paths, simulation intervals) for GBM,
    or (paths, simulation intervals, 2) for independent SV drivers B1 and B2.
    Returned spot/variance arrays have shape (paths, exercise dates).
    """
    # 1. PREPARE THE DATES AND TWO SEPARATE RANDOM STREAMS
    # Here the grids stay separate: a smaller integration step must not silently
    # add exercise opportunities. Checks prevent invalid dates and array shapes.
    simulation_times = np.asarray(simulation_times, dtype=float)
    exercise_date_indices = np.asarray(exercise_date_indices, dtype=int)
    if (simulation_times.ndim != 1 or len(simulation_times) < 2
            or not np.all(np.isfinite(simulation_times))
            or np.any(np.diff(simulation_times) <= 0)):
        raise ValueError("Simulation dates must be finite and strictly increasing")
    if (exercise_date_indices.ndim != 1 or len(exercise_date_indices) < 2
            or exercise_date_indices[0] != 0
            or exercise_date_indices[-1] >= len(simulation_times)
            or np.any(np.diff(exercise_date_indices) <= 0)):
        raise ValueError("Exercise indices must increase from the initial date")
    if not isinstance(path_count, (int, np.integer)) or path_count < 1:
        raise ValueError("Path count must be a positive integer")
    if not isinstance(model, (GeometricBrownianMotionModel, StochasticVolatilityModel)):
        raise TypeError("Milstein supports the assignment's GBM and printed SV models")
    if not np.isfinite(area_accuracy_multiplier) or area_accuracy_multiplier < 1:
        raise ValueError("Area accuracy multiplier must be finite and at least one")
    endpoint_random_numbers = np.random.default_rng(seed)
    if area_seed is None:
        # Here the area gets its own reproducible child key without advancing the
        # supplied SeedSequence. The fixed child label 314159 only selects a
        # stream; it is not a model parameter. Area draws leave shared shocks intact.
        brownian_seed_sequence = (seed if isinstance(seed, np.random.SeedSequence)
                                  else np.random.SeedSequence(seed))
        area_seed = np.random.SeedSequence(brownian_seed_sequence.entropy,
            spawn_key=brownian_seed_sequence.spawn_key + (314159,),
            pool_size=brownian_seed_sequence.pool_size)
    area_random_numbers = np.random.default_rng(area_seed)
    has_stochastic_variance = isinstance(model, StochasticVolatilityModel)
    simulation_interval_count = len(simulation_times) - 1
    if brownian_increments is not None:
        brownian_increments = np.asarray(brownian_increments, dtype=float)
        expected_shape = ((path_count, simulation_interval_count, 2) if has_stochastic_variance
                          else (path_count, simulation_interval_count))
        if brownian_increments.shape != expected_shape:
            raise ValueError(f"Brownian increments must have shape {expected_shape}")
        if not np.all(np.isfinite(brownian_increments)):
            raise ValueError("Brownian increments must be finite")

    # 2. ALLOCATE THE STATES SAVED AT EXERCISE DATES
    exercise_times = simulation_times[exercise_date_indices]
    current_spots = np.full(path_count, float(market_data.spot))
    # Here column 0 contains the known initial state; later columns feed LSMC.
    spot_paths = np.empty((path_count, len(exercise_times)))
    spot_paths[:, 0] = current_spots
    if has_stochastic_variance:
        current_variances = np.full(path_count, float(model.initial_variance))
        variance_paths = np.empty_like(spot_paths)
        variance_paths[:, 0] = current_variances
    variance_projection_count = 0
    zero_variance_start_count = 0
    negative_spot_state_count = 0
    maximum_fourier_terms = 0
    additional_area_normal_draw_count = 0
    next_saved_exercise_column = 1

    for next_date_index in range(1, len(simulation_times)):
        current_date_index = next_date_index - 1
        time_step = simulation_times[next_date_index] - simulation_times[current_date_index]

        # 3. READ THE INDEPENDENT ENDPOINT INCREMENTS, THEN APPLY CORRELATION
        # Here [:, current_date_index, 0] selects every path's first driver
        # for this interval. Last-axis index 1 selects the independent second driver.
        if brownian_increments is None:
            spot_brownian_increments = np.sqrt(time_step) * endpoint_random_numbers.standard_normal(path_count)
        elif has_stochastic_variance:
            spot_brownian_increments = brownian_increments[:, current_date_index, 0]
        else:
            spot_brownian_increments = brownian_increments[:, current_date_index]
        if has_stochastic_variance:
            if brownian_increments is None:
                second_increments = np.sqrt(time_step) * endpoint_random_numbers.standard_normal(path_count)
            else:
                second_increments = brownian_increments[:, current_date_index, 1]
            spot_variance_correlation = model.brownian_correlation
            independent_driver_weight = np.sqrt(1 - spot_variance_correlation**2)
            # Here dW_v = rho*dW_S + sqrt(1-rho^2)*dB2 imposes the correlation
            # required by the assignment. Each increment already includes sqrt(h).
            variance_brownian_increments = (
                spot_variance_correlation * spot_brownian_increments
                + independent_driver_weight * second_increments)
            volatility_of_variance = model.volatility_of_variance

            # 4. SIMULATE CROSS INTEGRALS WHEN TWO RANDOM DIRECTIONS MATTER
            # Here I_SS = ((dW_S)^2-h)/2 is known from the endpoint.
            # I_Sv means spot Brownian movement integrated against variance's Brownian
            # motion; I_vS reverses the order. The joint update needs both.
            spot_spot_integral = 0.5 * (spot_brownian_increments**2 - time_step)
            if spot_variance_correlation == 1 or volatility_of_variance == 0:
                # Here rho=1 leaves one driver. When xi=0, both cross terms
                # in the updates vanish, so drawing an area would do no work.
                spot_then_variance_integral = spot_variance_correlation * spot_spot_integral
                variance_then_spot_integral = spot_variance_correlation * spot_spot_integral
            else:
                (first_then_second_integral, second_then_first_integral,
                 fourier_term_count) = sample_wiktorsson_cross_integrals(
                    spot_brownian_increments, second_increments, time_step,
                    area_random_numbers, area_accuracy_multiplier)
                spot_then_variance_integral = (
                    spot_variance_correlation * spot_spot_integral
                    + independent_driver_weight * first_then_second_integral)
                variance_then_spot_integral = (
                    spot_variance_correlation * spot_spot_integral
                    + independent_driver_weight * second_then_first_integral)
                maximum_fourier_terms = max(maximum_fourier_terms, fourier_term_count)
                additional_area_normal_draw_count += path_count * (4 * fourier_term_count + 1)

            # 5. ADVANCE BOTH STATES FROM THE OLD PAIR, THEN PROJECT VARIANCE
            # Here the report's diffusion expansion gives the added terms:
            # spot: S*v*I_SS + (xi*S^2/2)*I_vS;
            # variance: (xi^2*S^2/4)*((dW_v)^2-h) + xi*S*v*I_Sv.
            # Both equations use current S and v: updating spot first inside
            # the variance equation would implement a different scheme.
            zero_variance_start_count += int(np.count_nonzero(current_variances == 0))
            with np.errstate(over="ignore", invalid="ignore"):
                spot_diffusion_coefficient = current_spots * np.sqrt(current_variances)
                next_spots = (current_spots + model.spot_drift(current_spots) * time_step
                    + spot_diffusion_coefficient * spot_brownian_increments
                    + current_spots * current_variances * spot_spot_integral
                    + 0.5 * volatility_of_variance * current_spots**2 * variance_then_spot_integral)
                unprojected_next_variances = (current_variances + model.variance_drift(current_variances) * time_step
                    + volatility_of_variance * spot_diffusion_coefficient * variance_brownian_increments
                    + 0.25 * volatility_of_variance**2 * current_spots**2 * (variance_brownian_increments**2 - time_step)
                    + volatility_of_variance * current_spots * current_variances * spot_then_variance_integral)
            if not np.all(np.isfinite(unprojected_next_variances)):
                raise FloatingPointError(f"Milstein produced non-finite variance at simulation step {next_date_index}")
            # Here we retain Euler's projection so the next sqrt(v) is defined.
            # At v=0 we use the finite limits of the simplified correction
            # products above. This is our boundary convention, not a proof of
            # positivity or strong order one. We count encounters for the report.
            variance_projection_count += int(np.count_nonzero(unprojected_next_variances < 0))
            next_variances = np.maximum(unprojected_next_variances, 0.0)
        else:
            # Here GBM adds 0.5*sigma^2*S*((dW)^2-h) to the Euler step.
            # It follows from b(S)*b'(S)=sigma^2*S. This is the scalar
            # correction whose pathwise convergence we test against exact GBM.
            with np.errstate(over="ignore", invalid="ignore"):
                next_spots = (current_spots + model.spot_drift(current_spots) * time_step
                    + model.spot_diffusion(current_spots) * spot_brownian_increments
                    + 0.5 * model.volatility**2 * current_spots * (spot_brownian_increments**2 - time_step))

        # 6. REPORT NUMERICAL PROBLEMS WITHOUT CLIPPING OR DISCARDING SPOTS
        # Here clipping spot would change payoffs. We retain finite proposals,
        # count negative states and fail explicitly if the update is non-finite.
        if not np.all(np.isfinite(next_spots)):
            raise FloatingPointError(f"Milstein produced non-finite spot at simulation step {next_date_index}")
        negative_spot_state_count += int(np.count_nonzero(next_spots < 0))
        current_spots = next_spots
        if has_stochastic_variance:
            current_variances = next_variances

        # 7. SAVE THE JOINT STATES FOR THE EXISTING LSMC VALUATION
        # Here only exercise-date states enter the valuation. In the joint model
        # we keep both S and v because continuation depends on both variables.
        if (next_saved_exercise_column < len(exercise_date_indices)
                and next_date_index == exercise_date_indices[next_saved_exercise_column]):
            spot_paths[:, next_saved_exercise_column] = current_spots
            if has_stochastic_variance:
                variance_paths[:, next_saved_exercise_column] = current_variances
            next_saved_exercise_column += 1

    state_paths = {"spot": spot_paths}
    if has_stochastic_variance:
        state_paths["variance"] = variance_paths
    # Here diagnostics support the boundary and cost discussion. A more accurate
    # path step alone does not establish better American prices; the runner also
    # compares valuations and learned exercise behaviour. Count all path-steps.
    diagnostics = {"simulation_times": simulation_times,
        "path_step_count": path_count * simulation_interval_count,
        "variance_projection_count": variance_projection_count,
        "zero_variance_start_count": zero_variance_start_count,
        "negative_spot_state_count": negative_spot_state_count,
        "nonfinite_state_count": 0,
        "area_terms_max": maximum_fourier_terms,
        "area_normal_draw_count": additional_area_normal_draw_count,
        "area_accuracy_multiplier": float(area_accuracy_multiplier)}
    return state_paths, exercise_times, diagnostics
