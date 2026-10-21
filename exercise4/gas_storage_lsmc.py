"""EXERCISE 4.2 — STORAGE CONTINUATION, FEASIBLE ACTIONS AND REALISED CASHFLOWS.

Here inventory is measured immediately BEFORE each daily trade. An injection
Delta v is positive, v_next=v+Delta v, and the immediate cashflow is -S*Delta v.
We regress once per resulting inventory, as in Boogert–de Jong equation (16).
All price paths enter each regression: storage has no in-the-money filter.
"""
import numpy as np
from basis import gas_storage_bspline_basis, gas_storage_spline_knots
from exercise4.gas_price_simulation import gas_price_moments


def storage_inventory_grid(product, inventory_step):
    quantities = np.array([product.capacity, product.initial_inventory,
                           product.final_inventory, product.injection_limit,
                           product.withdrawal_limit])
    if not np.isfinite(inventory_step) or inventory_step <= 0:
        raise ValueError("Inventory spacing must be finite and positive")
    if not np.allclose(quantities / inventory_step, np.round(quantities / inventory_step),
                       rtol=0, atol=1e-10):
        raise ValueError("This implementation requires capacity, endpoints and flow limits on the inventory grid")
    return np.arange(round(product.capacity / inventory_step) + 1) * inventory_step


def storage_grid_actions(product, inventory_step):
    # Here every grid-aligned volume is considered, not only full withdrawal,
    # holding and full injection. The paper warns against that restriction.
    return np.arange(-round(product.withdrawal_limit / inventory_step),
                     round(product.injection_limit / inventory_step) + 1) * inventory_step


def feasible_storage_transitions(product, inventory_grid, actions, date_index, number_of_dates):
    step = inventory_grid[1] - inventory_grid[0]
    next_inventories = inventory_grid[:, None] + actions[None, :]
    lower_now, upper_now = product.reachable_inventory_bounds(number_of_dates - date_index)
    lower_next, upper_next = product.reachable_inventory_bounds(number_of_dates - date_index - 1)
    feasible = ((inventory_grid[:, None] >= lower_now)
                & (inventory_grid[:, None] <= upper_now)
                & (next_inventories >= lower_next) & (next_inventories <= upper_next))
    indices = np.rint(next_inventories / step).astype(int)
    return np.where(feasible, indices, -1)


def preferred_storage_actions(actions):
    # Here exact ties prefer the smallest trade, then withdrawal over injection.
    # This makes a flat-price case hold when possible without changing value.
    return np.lexsort((actions, np.abs(actions)))


def validate_storage_paths(spot_paths, product):
    spot_paths = np.asarray(spot_paths, dtype=float)
    if (spot_paths.ndim != 2 or spot_paths.shape[1] != len(product.operating_dates())
            or spot_paths.shape[0] == 0 or not np.all(np.isfinite(spot_paths))
            or np.any(spot_paths <= 0)):
        raise ValueError("Storage prices must be positive finite paths x operating dates")
    return spot_paths


def estimate_storage_decisions(spot_paths, product, model, inventory_step=2500.0,
                               snapshot_dates=None, progress=None):
    # 1. INITIALISE TERMINAL VALUES AND THE REACHABLE INVENTORY STATES
    spot_paths = validate_storage_paths(spot_paths, product)
    if not np.all(spot_paths[:, 0] == spot_paths[0, 0]):
        raise ValueError("Training paths must share their observed initial price")
    number_of_paths, number_of_dates = spot_paths.shape
    inventory_grid = storage_inventory_grid(product, inventory_step)
    number_of_inventories = len(inventory_grid)
    actions = storage_grid_actions(product, inventory_step)
    action_order = preferred_storage_actions(actions)
    final_index = round(product.final_inventory / inventory_step)
    initial_index = round(product.initial_inventory / inventory_step)
    future_cashflows = np.full((number_of_paths, number_of_inventories), np.nan)
    future_cashflows[:, final_index] = 0.0
    # Here NaN marks an unreachable state. It is never a regression target.
    coefficients = np.full((number_of_dates, number_of_inventories, 6), np.nan)
    knots = np.full((number_of_dates, 10), np.nan)
    constant_dates = np.zeros(number_of_dates, dtype=bool)
    constant_continuation = np.full((number_of_dates, number_of_inventories), np.nan)
    diagnostics, snapshots = [], []
    moments = gas_price_moments(spot_paths[0, 0], model, np.arange(number_of_dates))
    if snapshot_dates is None:
        snapshot_dates = {1, number_of_dates // 2, number_of_dates - 2}
    else:
        snapshot_dates = set(snapshot_dates)

    # 2. MOVE BACKWARDS THROUGH THE OPERATING DATES
    for date_index in range(number_of_dates - 1, -1, -1):
        spots = spot_paths[:, date_index]
        transitions = feasible_storage_transitions(
            product, inventory_grid, actions, date_index, number_of_dates)
        next_indices = np.unique(transitions[transitions >= 0])
        next_slice = slice(next_indices[0], next_indices[-1] + 1)
        regression_targets = future_cashflows[:, next_slice]
        if not np.all(np.isfinite(regression_targets)):
            raise RuntimeError("An unreachable inventory entered the regression")

        # 3. FIT CONTINUATION AT EACH RESULTING INVENTORY USING THE SAME b6 MATRIX
        # Here Y has shape (paths, reachable next inventories), B has (paths, 6).
        # One SVD solves all ordinary least-squares regressions simultaneously.
        constant = np.all(spots == spots[0]) or date_index == number_of_dates - 1
        constant_dates[date_index] = constant
        design = None
        if constant:
            means = np.mean(regression_targets, axis=0)
            constant_continuation[date_index, next_indices] = means
            coefficients[date_index, next_indices] = means[:, None]
            fitted_continuation = np.broadcast_to(means, regression_targets.shape)
            rank, condition, tail_paths = 1, 1.0, 0
        else:
            knots[date_index] = gas_storage_spline_knots(
                moments["log_mean"][date_index], moments["log_variance"][date_index])
            design = gas_storage_bspline_basis(spots, knots[date_index])
            left, singular_values, right = np.linalg.svd(design, full_matrices=False)
            cutoff = np.finfo(float).eps * max(design.shape) * singular_values[0]
            retained = singular_values > cutoff
            rank = int(np.sum(retained))
            inverse = np.zeros_like(singular_values)
            inverse[retained] = 1.0 / singular_values[retained]
            fitted_coefficients = (right.T * inverse) @ (left.T @ regression_targets)
            coefficients[date_index, next_indices] = fitted_coefficients.T
            fitted_continuation = design @ fitted_coefficients
            condition = (float(singular_values[0] / singular_values[-1])
                         if rank == 6 else None)
            tail_paths = int(np.count_nonzero((spots < knots[date_index, 3])
                                             | (spots > knots[date_index, 6])))
        diagnostics.append({"date_index": date_index, "rank": rank,
            "constant_state": bool(constant), "condition_number": condition,
            "tail_paths": tail_paths, "regressed_inventories": len(next_indices),
            "method": ("Known terminal continuation: zero" if date_index == number_of_dates - 1
                       else "Sample mean at a common spot" if constant else "OLS b6 by SVD")})
        continuation = np.full_like(future_cashflows, np.nan)
        continuation[:, next_slice] = fitted_continuation

        # Here we save selected observations and fitted values for the regression plot.
        if date_index in snapshot_dates and design is not None:
            desired_inventory = (min(final_index + 1, number_of_inventories - 1)
                                 if date_index == number_of_dates - 2 else initial_index)
            snapshot_inventory = int(next_indices[np.argmin(np.abs(next_indices - desired_inventory))])
            observations = np.linspace(0, number_of_paths - 1, min(160, number_of_paths), dtype=int)
            snapshots.append({"date_index": date_index,
                "inventory": float(inventory_grid[snapshot_inventory]),
                "spots": spots[observations].copy(),
                "realised_future_cashflows": future_cashflows[observations, snapshot_inventory].copy(),
                "fitted_continuations": continuation[observations, snapshot_inventory].copy(),
                "basis": design[observations].copy(),
                "coefficients": coefficients[date_index, snapshot_inventory].copy(),
                "knots": knots[date_index].copy()})

        # 4. COMPARE EVERY FEASIBLE ACTION USING IMMEDIATE CASHFLOW + CONTINUATION
        estimated_values = np.full_like(future_cashflows, -np.inf)
        current_cashflows = np.full_like(future_cashflows, np.nan)
        for action_index in action_order:
            current_indices = np.flatnonzero(transitions[:, action_index] >= 0)
            if len(current_indices) == 0:
                continue
            resulting_indices = transitions[current_indices, action_index]
            # Here reachable inventories form an interval. Slices expose that
            # interval without making a large paths x inventories copy.
            current_slice = slice(current_indices[0], current_indices[-1] + 1)
            resulting_slice = slice(resulting_indices[0], resulting_indices[-1] + 1)
            immediate_cashflows = product.cashflow(spots, actions[action_index])[:, None]
            candidate_values = immediate_cashflows + continuation[:, resulting_slice]
            improves = candidate_values > estimated_values[:, current_slice] + 1e-10
            np.copyto(estimated_values[:, current_slice], candidate_values, where=improves)

            # 5. CARRY BACK REALISED FUTURE CASHFLOWS, NOT THE FITTED CONTINUATION
            realised_values = immediate_cashflows + future_cashflows[:, resulting_slice]
            np.copyto(current_cashflows[:, current_slice], realised_values, where=improves)
        future_cashflows = current_cashflows
        if progress is not None and (date_index % 10 == 0 or date_index == number_of_dates - 1):
            progress("Estimate decisions backwards", number_of_dates - date_index,
                     number_of_dates, date_index)

    # 6. RETURN THE INITIAL VALUE AND THE FUNCTIONS NEEDED FOR NEW PRICE PATHS
    training_cashflows = future_cashflows[:, initial_index].copy()
    return {"coefficients": coefficients, "knots": knots,
        "constant_dates": constant_dates, "constant_continuation": constant_continuation,
        "initial_continuation": constant_continuation[0].copy(),
        "inventory_grid": inventory_grid, "actions": actions,
        "training_cashflows": training_cashflows,
        "training_value": float(np.mean(training_cashflows)),
        "initial_spot": float(spot_paths[0, 0]),
        "regression_diagnostics": sorted(diagnostics, key=lambda row: row["date_index"]),
        "regression_snapshot": sorted(snapshots, key=lambda row: row["date_index"])}


def storage_continuation_values(spots, date_index, decisions):
    spots = np.atleast_1d(np.asarray(spots, dtype=float))
    if spots.ndim != 1 or not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("Continuation requires a finite positive spot vector")
    if decisions["constant_dates"][date_index]:
        return np.broadcast_to(decisions["constant_continuation"][date_index],
                               (len(spots), len(decisions["inventory_grid"])))
    design = gas_storage_bspline_basis(spots, decisions["knots"][date_index])
    return design @ decisions["coefficients"][date_index].T


def choose_storage_actions(total_values, actions):
    order = preferred_storage_actions(actions)
    # Here ordering makes an exact tie reproducible, with holding first.
    reordered = total_values[..., order]
    best = np.max(reordered, axis=-1, keepdims=True)
    return order[np.argmax(reordered >= best - 1e-10, axis=-1)]


def compare_storage_actions(spot, date_index, inventory, product, decisions,
                            include_infeasible=False):
    grid, actions = decisions["inventory_grid"], decisions["actions"]
    step = grid[1] - grid[0]
    inventory_index = int(round(inventory / step))
    if (not 0 <= inventory_index < len(grid)
            or not np.isclose(inventory, grid[inventory_index], atol=1e-8, rtol=0)):
        raise ValueError("The inventory must be a node of the fitted grid")
    number_of_dates = len(product.operating_dates())
    if not 0 <= date_index < number_of_dates:
        raise ValueError("The selected operating date is outside the contract")
    transitions = feasible_storage_transitions(product, grid, actions, date_index, number_of_dates)
    next_indices = transitions[inventory_index]
    feasible = next_indices >= 0
    if not np.any(feasible):
        raise ValueError("The selected inventory cannot reach the terminal requirement")
    continuation = np.full(len(actions), np.nan)
    continuation[feasible] = storage_continuation_values([spot], date_index, decisions)[0, next_indices[feasible]]
    cashflows = product.cashflow(spot, actions)
    totals = np.where(feasible, cashflows + continuation, -np.inf)
    chosen = int(choose_storage_actions(totals, actions))
    reasons = []
    lower, upper = product.reachable_inventory_bounds(number_of_dates - date_index - 1)
    for next_inventory, valid in zip(inventory + actions, feasible):
        if valid:
            reasons.append("")
        elif next_inventory < 0:
            reasons.append("Not enough gas to withdraw")
        elif next_inventory > product.capacity:
            reasons.append("Not enough space to inject")
        elif not lower <= next_inventory <= upper:
            reasons.append("Cannot return the required final inventory in the remaining days")
        else:
            reasons.append("Current inventory cannot reach the final requirement")
    kept = np.arange(len(actions)) if include_infeasible else np.flatnonzero(feasible)
    return {"actions": actions[kept], "next_inventories": inventory + actions[kept],
        "immediate_cashflows": cashflows[kept], "continuations": continuation[kept],
        "total_values": totals[kept], "feasible": feasible[kept],
        "reasons": [reasons[index] for index in kept],
        "chosen_index": int(np.flatnonzero(kept == chosen)[0])}


def storage_cashflow_statistics(cashflows, antithetic=True):
    cashflows = np.asarray(cashflows, dtype=float)
    if cashflows.ndim != 1 or len(cashflows) < 2 or not np.all(np.isfinite(cashflows)):
        raise ValueError("Supply at least two finite path cashflows")
    if antithetic and len(cashflows) % 2:
        raise ValueError("Antithetic cashflows must contain complete adjacent pairs")
    # Here pairs, not individual paths, are independent statistical observations.
    independent_observations = cashflows.reshape(-1, 2).mean(axis=1) if antithetic else cashflows
    standard_error = (float(np.std(independent_observations, ddof=1)
                            / np.sqrt(len(independent_observations)))
                      if len(independent_observations) > 1 else None)
    return {"value": float(np.mean(cashflows)), "standard_error": standard_error,
        "path_cashflow_standard_deviation": float(np.std(cashflows, ddof=1)),
        "independent_observations": len(independent_observations)}


def apply_storage_decisions(spot_paths, product, decisions, antithetic=True, record_paths=3,
                            progress=None):
    # 7. APPLY THE PREVIOUSLY ESTIMATED DECISIONS WITHOUT REFITTING
    spot_paths = validate_storage_paths(spot_paths, product)
    if not np.allclose(spot_paths[:, 0], decisions["initial_spot"], rtol=0, atol=1e-12):
        raise ValueError("A different initial spot requires fitting new storage decisions")
    number_of_paths, number_of_dates = spot_paths.shape
    grid, actions = decisions["inventory_grid"], decisions["actions"]
    step = grid[1] - grid[0]
    inventories = np.full(number_of_paths, round(product.initial_inventory / step), dtype=int)
    cashflows = np.zeros(number_of_paths)
    recorded = min(max(int(record_paths), 0), number_of_paths)
    inventory_paths = np.empty((recorded, number_of_dates + 1))
    inventory_paths[:, 0] = product.initial_inventory
    action_paths = np.empty((recorded, number_of_dates))
    daily_cashflows = np.empty((recorded, number_of_dates))
    minimum_inventory, maximum_inventory = product.initial_inventory, product.initial_inventory
    flow_violations = 0
    rows = np.arange(number_of_paths)
    for date_index in range(number_of_dates):
        spots = spot_paths[:, date_index]
        transitions = feasible_storage_transitions(product, grid, actions, date_index, number_of_dates)
        next_indices = transitions[inventories]
        feasible = next_indices >= 0
        if not np.all(np.any(feasible, axis=1)):
            raise RuntimeError("A replay path has no feasible storage action")
        continuation = storage_continuation_values(spots, date_index, decisions)
        totals = np.where(feasible, product.cashflow(spots[:, None], actions[None, :])
                          + continuation[rows[:, None], np.maximum(next_indices, 0)], -np.inf)
        chosen = choose_storage_actions(totals, actions)
        changes = actions[chosen]
        flow_violations += int(np.count_nonzero((changes < -product.withdrawal_limit)
                                               | (changes > product.injection_limit)))
        daily_cash = product.cashflow(spots, changes)
        cashflows += daily_cash
        inventories = next_indices[rows, chosen]
        minimum_inventory = min(minimum_inventory, float(np.min(grid[inventories])))
        maximum_inventory = max(maximum_inventory, float(np.max(grid[inventories])))
        inventory_paths[:, date_index + 1] = grid[inventories[:recorded]]
        action_paths[:, date_index] = changes[:recorded]
        daily_cashflows[:, date_index] = daily_cash[:recorded]
        if progress is not None and (date_index % 10 == 0 or date_index == number_of_dates - 1):
            progress("Evaluate new paths forwards", date_index + 1,
                     number_of_dates, date_index)
    terminal_violations = int(np.count_nonzero(grid[inventories] != product.final_inventory))
    if terminal_violations:
        raise RuntimeError("The fitted decisions did not return the required final inventory")
    return {**storage_cashflow_statistics(cashflows, antithetic), "cashflows": cashflows,
        "inventory_paths": inventory_paths, "action_paths": action_paths,
        "daily_cashflows": daily_cashflows,
        "constraint_diagnostics": {"minimum_inventory": minimum_inventory,
            "maximum_inventory": maximum_inventory, "terminal_violations": terminal_violations,
            "capacity_violations": int(minimum_inventory < 0 or maximum_inventory > product.capacity),
            "flow_violations": flow_violations}}


def storage_value_surface(date_index, spots, product, decisions):
    spots = np.asarray(spots, dtype=float)
    if date_index == 0 and not np.allclose(spots, decisions["initial_spot"], rtol=0, atol=1e-12):
        raise ValueError("The initial-date continuation is estimated only at the observed initial spot")
    grid, actions = decisions["inventory_grid"], decisions["actions"]
    transitions = feasible_storage_transitions(product, grid, actions, date_index,
                                               len(product.operating_dates()))
    continuation = storage_continuation_values(spots, date_index, decisions)
    values = np.full((len(spots), len(grid)), -np.inf)
    for action_index, action in enumerate(actions):
        current = np.flatnonzero(transitions[:, action_index] >= 0)
        following = transitions[current, action_index]
        values[:, current] = np.maximum(values[:, current],
            product.cashflow(spots, action)[:, None] + continuation[:, following])
    values[~np.isfinite(values)] = np.nan
    return {"spots": spots, "inventory_grid": grid, "values": values}


def replay_storage_path(spot_path, product, decisions):
    spot_path = validate_storage_paths(np.asarray(spot_path)[None, :], product)[0]
    grid, actions = decisions["inventory_grid"], decisions["actions"]
    number_of_dates = len(spot_path)
    shape = (number_of_dates, len(grid), len(actions))
    feasible = np.zeros(shape, dtype=bool)
    continuations, totals = np.full(shape, np.nan), np.full(shape, np.nan)
    chosen = np.full((number_of_dates, len(grid)), -1, dtype=int)
    immediate = product.cashflow(spot_path[:, None], actions[None, :])
    for date_index, spot in enumerate(spot_path):
        transitions = feasible_storage_transitions(product, grid, actions, date_index, number_of_dates)
        feasible[date_index] = transitions >= 0
        continuation = storage_continuation_values([spot], date_index, decisions)[0]
        continuations[date_index] = np.where(transitions >= 0,
            continuation[np.maximum(transitions, 0)], np.nan)
        totals[date_index] = immediate[date_index] + continuations[date_index]
        reachable = np.any(transitions >= 0, axis=1)
        chosen[date_index, reachable] = choose_storage_actions(
            np.where(transitions[reachable] >= 0, totals[date_index, reachable], -np.inf), actions)
    path_result = apply_storage_decisions(
        np.repeat(spot_path[None, :], 2, axis=0), product, decisions, antithetic=False, record_paths=1)
    return {"dates": product.operating_dates().astype(str), "spots": spot_path,
        "inventory_grid": grid, "actions": actions, "feasible": feasible,
        "continuation_values": continuations, "immediate_cashflows": immediate,
        "total_values": totals, "chosen_action_indices": chosen,
        "recommended_inventory": path_result["inventory_paths"][0],
        "recommended_action": path_result["action_paths"][0],
        "recommended_cashflows": path_result["daily_cashflows"][0]}
