"""Here we estimate continuation by least squares and compare it with exercise.

The regression target is a realised later payment discounted to the current date.
The fitted value estimates its conditional expectation given the current state.
Basis functions and simulation are supplied; the exercise algorithm is shared.
"""
import numpy as np


# LONGSTAFF–SCHWARTZ AMERICAN OPTION VALUATION
def value_american_option_longstaff_schwartz(
        product, market_data, model, simulation_method, basis_function,
        simulation_settings, simulation_times, exercise_date_indices):
    # 1. SIMULATE PATHS USING THE SUPPLIED SIMULATION FUNCTION
    # Here the case seed is already a SeedSequence: spawn directly from it.
    training_seed, evaluation_seed = simulation_settings.seed.spawn(2)
    training_paths, training_exercise_times, training_simulation_diagnostics = simulation_method(
        market_data=market_data, model=model, simulation_times=simulation_times,
        exercise_date_indices=exercise_date_indices,
        path_count=simulation_settings.training_paths, seed=training_seed)

    exercise_rules = estimate_exercise_rules_longstaff_schwartz(
        product, market_data, training_paths, training_exercise_times, basis_function)
    coefficients_by_date = exercise_rules["coefficients_by_date"]
    exercise_at_time_zero = exercise_rules["exercise_at_time_zero"]
    initial_continuation = exercise_rules["initial_continuation"]
    regression_diagnostics = exercise_rules["regression_diagnostics"]

    # 8. APPLY THE FITTED RULES TO EVALUATION PATHS AND CALCULATE VALUE AND ERROR
    evaluation_paths, evaluation_exercise_times, evaluation_simulation_diagnostics = simulation_method(
        market_data=market_data, model=model, simulation_times=simulation_times,
        exercise_date_indices=exercise_date_indices,
        path_count=simulation_settings.evaluation_paths, seed=evaluation_seed)
    result = apply_fitted_exercise_rules(
        product, market_data, evaluation_paths, evaluation_exercise_times,
        basis_function, coefficients_by_date, exercise_at_time_zero)
    result.update({"coefficients_by_date": coefficients_by_date,
        "exercise_times": evaluation_exercise_times,
        "exercise_at_time_zero": exercise_at_time_zero,
        "initial_continuation": initial_continuation,
        "regression_diagnostics": regression_diagnostics,
        "training_simulation_diagnostics": training_simulation_diagnostics,
        "evaluation_simulation_diagnostics": evaluation_simulation_diagnostics})
    return result


# STEPS 2–7 — ESTIMATE EXERCISE RULES FROM THE TRAINING PATHS
def estimate_exercise_rules_longstaff_schwartz(
        product, market_data, training_paths, exercise_times, basis_function):
    # 2. CALCULATE THE PAYOFF AT MATURITY
    maturity_date_index = len(exercise_times) - 1
    # Here each state array has shape (paths, dates). [:, maturity_date_index]
    # selects the maturity spot on every path.
    selected_cash_flows = product.payoff(training_paths["spot"][:, maturity_date_index])
    coefficients_by_date = [None] * len(exercise_times)
    regression_diagnostics = []

    # 3. WORK BACKWARDS THROUGH EXERCISE DATES
    last_date_before_maturity = maturity_date_index - 1
    time_zero_index = 0
    backward_step = -1
    # Here range excludes zero: all initial states are identical, so no fit there.
    for exercise_index in range(last_date_before_maturity, time_zero_index, backward_step):
        time_to_next_date = (exercise_times[exercise_index + 1]
                             - exercise_times[exercise_index])

        # 4. SELECT IN-THE-MONEY PATHS
        # Here we follow Longstaff–Schwartz: fit where exercise is relevant.
        # All paths still contribute to the final valuation.
        immediate_payoffs = product.payoff(training_paths["spot"][:, exercise_index])
        in_the_money = immediate_payoffs > 0

        # 5. REGRESS DISCOUNTED FUTURE CASH FLOWS ON THE CURRENT-STATE BASIS
        # Here selected cash flows are already valued at the next date. Discount
        # one interval so the target and immediate exercise use the same date.
        discounted_future_cash_flows = selected_cash_flows * np.exp(
            -market_data.interest_rate * time_to_next_date)
        selected_cash_flows = discounted_future_cash_flows.copy()
        if not np.any(in_the_money):
            continue
        # Here current_state contains one vector per state: spot, and variance
        # when present. The Boolean mask selects only the ITM rows at this date.
        current_state = {name: paths[in_the_money, exercise_index]
                         for name, paths in training_paths.items()}
        basis_matrix = basis_function(current_state, product)
        least_squares_result = np.linalg.lstsq(
            basis_matrix, discounted_future_cash_flows[in_the_money], rcond=None)
        coefficients = least_squares_result[0]
        coefficients_by_date[exercise_index] = coefficients
        regression_diagnostics.append({"date_index": exercise_index,
            "itm_paths": int(np.count_nonzero(in_the_money)),
            "basis_columns": basis_matrix.shape[1],
            "rank": int(least_squares_result[2])})

        # 6. COMPARE IMMEDIATE EXERCISE WITH ESTIMATED CONTINUATION
        # Here @ sums basis values times fitted coefficients for each path.
        estimated_continuation = basis_matrix @ coefficients
        exercise_now = immediate_payoffs[in_the_money] > estimated_continuation

        # 7. KEEP THE CASH FLOW CORRESPONDING TO THE CHOSEN DECISION
        # Here np.where selects the actual exercise payment or discounted later
        # payment. We never replace a realised payment with a regression estimate.
        selected_cash_flows[in_the_money] = np.where(
            exercise_now, immediate_payoffs[in_the_money],
            discounted_future_cash_flows[in_the_money])

    # Here we compare the common initial payoff with continuation estimated from
    # training cash flows; this also fixes the time-zero decision before evaluation.
    initial_continuation = float(np.mean(selected_cash_flows) * np.exp(
        -market_data.interest_rate * exercise_times[1]))
    exercise_at_time_zero = bool(product.payoff(market_data.spot) > initial_continuation)

    return {"coefficients_by_date": coefficients_by_date,
            "exercise_at_time_zero": exercise_at_time_zero,
            "initial_continuation": initial_continuation,
            "regression_diagnostics": regression_diagnostics}


# STEP 8 — APPLY THE STORED COEFFICIENTS; DO NOT REFIT THEM
def apply_fitted_exercise_rules(product, market_data, state_paths, exercise_times,
                                basis_function, coefficients_by_date, exercise_at_time_zero):
    number_of_paths, number_of_dates = state_paths["spot"].shape
    maturity_date_index = number_of_dates - 1
    # Here a path stays active until its single exercise payment is recorded.
    active_paths = np.ones(number_of_paths, dtype=bool)
    discounted_payoffs = np.zeros(number_of_paths)
    exercise_date_indices = np.full(number_of_paths, maturity_date_index, dtype=int)
    if exercise_at_time_zero:
        discounted_payoffs[:] = product.payoff(market_data.spot)
        exercise_date_indices[:] = 0
        active_paths[:] = False

    # Here range starts after time zero and excludes maturity, handled below.
    for exercise_index in range(1, maturity_date_index):
        coefficients = coefficients_by_date[exercise_index]
        if coefficients is None or not np.any(active_paths):
            continue
        immediate_payoffs = product.payoff(state_paths["spot"][:, exercise_index])
        eligible_paths = active_paths & (immediate_payoffs > 0)
        if not np.any(eligible_paths):
            continue
        current_state = {name: paths[eligible_paths, exercise_index]
                         for name, paths in state_paths.items()}
        estimated_continuation = basis_function(current_state, product) @ coefficients
        exercise_now = immediate_payoffs[eligible_paths] > estimated_continuation
        # Here we map the smaller eligible-path array back to original path rows.
        eligible_path_indices = np.flatnonzero(eligible_paths)
        exercising_path_indices = eligible_path_indices[exercise_now]
        discounted_payoffs[exercising_path_indices] = (
            immediate_payoffs[exercising_path_indices]
            * np.exp(-market_data.interest_rate * exercise_times[exercise_index]))
        exercise_date_indices[exercising_path_indices] = exercise_index
        active_paths[exercising_path_indices] = False

    # Here only remaining paths receive the maturity payoff, discounted to zero.
    discounted_payoffs[active_paths] = (
        product.payoff(state_paths["spot"][active_paths, maturity_date_index])
        * np.exp(-market_data.interest_rate * exercise_times[maturity_date_index]))
    # Here the SE measures evaluation sampling with the fitted rules held fixed.
    # It excludes training variability, basis error and time discretisation bias.
    return {"option_value": float(np.mean(discounted_payoffs)),
            "standard_error": float(np.std(discounted_payoffs, ddof=1) / np.sqrt(number_of_paths)),
            "discounted_payoffs": discounted_payoffs,
            "exercise_date_indices": exercise_date_indices}
