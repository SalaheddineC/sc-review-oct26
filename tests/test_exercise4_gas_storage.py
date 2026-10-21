"""EXERCISE 4 — CHECKS DERIVED FROM THE INSTRUCTED MODEL AND CONTRACT.

Here we check the price model against its analytical moments and the storage
decisions against the contract limits in Exhibit 1.
"""
import numpy as np
import pytest

from basis import gas_storage_bspline_basis
from exercise4.gas_price_simulation import simulate_gas_prices
from exercise4.gas_storage_lsmc import (apply_storage_decisions, compare_storage_actions,
    estimate_storage_decisions, replay_storage_path, storage_continuation_values)
from models import GasPriceModel
from products import GasStorage


def standard_model(volatility=.0945):
    # Here the stationary expected price is 20 EUR/MWh, as in our main run.
    return GasPriceModel(.05, np.log(20.) + volatility**2 / (4 * .05), volatility)


# 4.1 — THE EXACT LOG-PRICE LAW, INCLUDING ITS DAILY UNITS AND ITO CORRECTION
def test_gas_price_marginals_and_antithetic_pairs():
    model = standard_model()
    dates = np.arange(np.datetime64("2005-07-07"), np.datetime64("2005-09-06"))
    prices = simulate_gas_prices(20., model, dates, 20000, 4101)
    assert prices.shape == (20000, len(dates))
    np.testing.assert_allclose(prices[:, 0], 20., rtol=1e-15)
    for day in (1, 14, 60):
        # Here we derive the analytical law from the stated SDE, independently
        # of the simulator's moment helper. Adjacent paths share opposite Z.
        log_mean = np.log(20.) - .0945**2 / (4 * .05) * (1 - np.exp(-.05 * day))
        log_variance = .0945**2 / (.1) * (1 - np.exp(-.1 * day))
        log_prices = np.log(prices[:, day])
        np.testing.assert_allclose((log_prices[0::2] + log_prices[1::2]) / 2,
                                   log_mean, rtol=0, atol=5e-14)
        independent_squared_errors = (log_prices[0::2] - log_mean)**2
        # Here 10,000 independent pairs, not 20,000 independent observations,
        # determine the sampling allowance for this known normal variance.
        variance_standard_error = log_variance * np.sqrt(2 / 10000)
        assert abs(independent_squared_errors.mean() - log_variance) < 6 * variance_standard_error


# 4.2 — ON THE LAST DAY, THE CONTRACT DETERMINES THE ONLY POSSIBLE TRADE
def test_final_day_forces_inventory_and_signed_cashflow(test_data):
    case = test_data["exercise4"]["final_day"]
    inputs, expected = case["inputs"], case["expected"]
    for row in expected["rows"]:
        inventory = row["initial_inventory"]
        product = GasStorage(**{**inputs["product"], "initial_inventory": inventory})
        prices = np.full((inputs["paths"], 1), inputs["spot"])
        decisions = estimate_storage_decisions(prices, product, GasPriceModel(**inputs["model"]),
                                                 inventory_step=inputs["inventory_step"])
        comparison = compare_storage_actions(inputs["spot"], 0, inventory, product, decisions)
        np.testing.assert_array_equal(comparison["actions"], [row["trade"]])
        np.testing.assert_array_equal(comparison["continuations"], [expected["continuation"]])
        result = apply_storage_decisions(prices, product, decisions, record_paths=inputs["paths"])
        np.testing.assert_array_equal(result["inventory_paths"][:, -1], expected["final_inventory"])
        np.testing.assert_array_equal(result["cashflows"], row["cashflow"])
    # Here 95,000 MWh cannot be refilled to 100,000 in a single operating day.
    with pytest.raises(ValueError, match=expected["unreachable_inventory_error"]):
        GasStorage(**{**inputs["product"], "initial_inventory": inputs["unreachable_initial_inventory"]})


# 4.2 — THE PRECEDING CONTINUATION HAS AN EXACT CONDITIONAL EXPECTATION
def test_penultimate_continuation_and_decision_match_analytic_reference():
    product = GasStorage(start_date="2006-06-28")
    model = standard_model()
    prices = simulate_gas_prices(20., model, product.operating_dates(), 40000, 4103)
    decisions = estimate_storage_decisions(prices, product, model)
    query_spots = np.quantile(prices[:, 1], [.1, .5, .9])

    def next_spot_mean(spots):
        # Here the next price is lognormal: E[S_next|s] = exp(mean + variance/2).
        decay = np.exp(-.05)
        long_term_log_mean = np.log(20.) - .0945**2 / (4 * .05)
        return np.exp(decay * np.log(spots) + (1 - decay) * long_term_log_mean
                      + .0945**2 / (4 * .05) * (1 - decay**2))

    next_inventories = np.arange(97500., 107500. + 2500., 2500.)
    inventory_difference = next_inventories - 100000.
    exact_continuation = next_spot_mean(query_spots)[:, None] * inventory_difference
    fitted_continuation = storage_continuation_values(query_spots, 1, decisions)[
        :, (next_inventories / 2500).astype(int)]

    # Here we separate sampling from the b6 approximation. OLS weights map the
    # terminal-price noise into a fitted conditional mean; adjacent contributions
    # are combined because the simulation uses antithetic pairs.
    design = gas_storage_bspline_basis(prices[:, 1], decisions["knots"][1])
    query_design = gas_storage_bspline_basis(query_spots, decisions["knots"][1])
    regression_weights = query_design @ np.linalg.pinv(design)
    price_noise = prices[:, 2] - next_spot_mean(prices[:, 1])
    pair_contributions = (regression_weights * price_noise).reshape(3, -1, 2).sum(axis=2)
    sampling_error = pair_contributions.std(axis=1, ddof=1) * np.sqrt(20000)
    projected_exact_mean = regression_weights @ next_spot_mean(prices[:, 1])
    approximation_error = np.abs(projected_exact_mean - next_spot_mean(query_spots))
    assert np.max(approximation_error) < .001  # EUR/MWh, for these model-derived states.
    allowance = (6 * sampling_error + approximation_error)[:, None] * np.abs(inventory_difference)
    assert np.all(np.abs(fitted_continuation - exact_continuation) <= allowance + 1e-8)

    for spot in query_spots:
        comparison = compare_storage_actions(spot, 1, 100000., product, decisions)
        exact_values = (-spot * comparison["actions"]
            + (comparison["next_inventories"] - 100000.) * next_spot_mean(spot))
        assert comparison["chosen_index"] == np.argmax(exact_values)


# 4.2–4.3 — THE BACKWARD CASHFLOWS MUST BE REALISABLE BY FORWARD DECISIONS
def test_full_contract_replay_constraints_cashflows_and_no_future_information():
    product, model = GasStorage(), standard_model()
    dates = product.operating_dates()
    assert len(dates) == 359
    training_prices = simulate_gas_prices(20., model, dates, 400, 4104)
    decisions = estimate_storage_decisions(training_prices, product, model)
    training_replay = apply_storage_decisions(training_prices, product, decisions)
    # Here replacing realised targets with fitted continuation would break this
    # path-by-path agreement, even if the two averages happened to look close.
    np.testing.assert_allclose(training_replay["cashflows"], decisions["training_cashflows"],
                               rtol=0, atol=1e-7)

    prices = simulate_gas_prices(20., model, dates, 200, 4105)
    result = apply_storage_decisions(prices, product, decisions, record_paths=200)
    inventory, actions = result["inventory_paths"], result["action_paths"]
    assert np.all((inventory >= 0) & (inventory <= 250000.))
    assert np.all((actions >= -7500.) & (actions <= 2500.))
    np.testing.assert_array_equal(np.diff(inventory, axis=1), actions)
    np.testing.assert_array_equal(inventory[:, -1], 100000.)
    np.testing.assert_allclose(result["cashflows"], (-prices * actions).sum(axis=1),
                               rtol=0, atol=1e-7)
    pair_cashflows = result["cashflows"].reshape(-1, 2).mean(axis=1)
    assert result["standard_error"] == pytest.approx(pair_cashflows.std(ddof=1) / np.sqrt(100))

    # Here changing unseen future prices cannot change the earlier decisions.
    altered_prices = prices.copy()
    altered_prices[:, 180:] *= 2
    altered = apply_storage_decisions(altered_prices, product, decisions, record_paths=200)
    np.testing.assert_array_equal(altered["action_paths"][:, :180], actions[:, :180])

    # Here alternative trades must still obey the operating constraints.
    # Alternating the smallest and largest feasible trades must
    # still finish at the contractual inventory, with the displayed cashflows.
    replay = replay_storage_path(prices[0], product, decisions)
    np.testing.assert_array_equal(replay["recommended_inventory"], inventory[0])
    current_inventory = 100000.
    for date_index in range(len(dates)):
        inventory_index = int(current_inventory / 2500.)
        feasible_actions = np.flatnonzero(replay["feasible"][date_index, inventory_index])
        action_index = feasible_actions[0 if date_index % 2 else -1]
        change = replay["actions"][action_index]
        assert -min(current_inventory, 7500.) <= change <= min(250000. - current_inventory, 2500.)
        assert replay["immediate_cashflows"][date_index, action_index] == -prices[0, date_index] * change
        current_inventory += change
    assert current_inventory == 100000.


# LIMITING CASE — CONSTANT PRICES GIVE ZERO CASHFLOW FOR EVERY ROUND TRIP
def test_zero_volatility_has_zero_value_and_no_degenerate_spline_fit(test_data):
    case = test_data["exercise4"]["zero_volatility"]
    inputs, expected = case["inputs"], case["expected"]
    product, model = GasStorage(**inputs["product"]), GasPriceModel(**inputs["model"])
    prices = simulate_gas_prices(inputs["spot"], model, product.operating_dates(),
                                 inputs["paths"], inputs["seed"])
    decisions = estimate_storage_decisions(prices, product, model, inventory_step=inputs["inventory_step"])
    result = apply_storage_decisions(prices, product, decisions, record_paths=inputs["paths"])
    assert np.all(decisions["constant_dates"]) == expected["all_dates_constant"]
    np.testing.assert_allclose(decisions["training_cashflows"], expected["value"],
                               rtol=0, atol=case["tolerances"]["cashflow_absolute_eur"])
    np.testing.assert_allclose(result["cashflows"], expected["value"],
                               rtol=0, atol=case["tolerances"]["cashflow_absolute_eur"])
    np.testing.assert_array_equal(result["inventory_paths"][:, -1], expected["final_inventory"])


# SAVED OPERATING PATH — READABLE TEST INPUTS AND A REPRODUCIBLE REPLAY
def test_saved_operating_path_matches_cashflows_constraints_and_decisions(test_data, test_data_directory):
    case = test_data["exercise4"]["saved_operating_path"]
    inputs, expected = case["inputs"], case["expected"]
    product = GasStorage(**inputs["product"])
    path = np.genfromtxt(test_data_directory / "exercise4" / inputs["file"],
                        delimiter=",", names=True, dtype=None, encoding="utf-8")
    assert len(path) == expected["operating_dates"]
    np.testing.assert_array_equal(path["operating_date"], product.operating_dates().astype(str))
    before, after, trades = path["inventory_before_mwh"], path["inventory_after_mwh"], path["trade_mwh"]
    assert before[0] == expected["initial_inventory"]
    assert after[-1] == expected["final_inventory"]
    np.testing.assert_array_equal(before[1:], after[:-1])
    np.testing.assert_array_equal(after - before, trades)
    assert np.all((before >= 0) & (before <= product.capacity))
    assert np.all((after >= 0) & (after <= product.capacity))
    assert np.all((trades >= -product.withdrawal_limit) & (trades <= product.injection_limit))
    np.testing.assert_array_equal(trades / inputs["inventory_step"],
                                   np.round(trades / inputs["inventory_step"]))
    allowance = case["tolerances"]["cashflow_absolute_eur"]
    np.testing.assert_allclose(path["daily_cashflow_eur"], -path["spot_eur_per_mwh"] * trades,
                               rtol=0, atol=allowance)
    np.testing.assert_allclose(path["cumulative_cashflow_eur"], np.cumsum(path["daily_cashflow_eur"]),
                               rtol=0, atol=allowance)
    assert path["cumulative_cashflow_eur"][-1] == pytest.approx(
        expected["total_cashflow"], rel=0, abs=allowance)

    # Here the CSV exposes a saved implementation result, not an independent benchmark.
    prefix, index = inputs["case"] + "_", inputs["path_index"]
    with np.load(test_data_directory.parent / inputs["arrays_file"], allow_pickle=False) as saved:
        for column, field in (("spot_eur_per_mwh", "spot_paths"), ("trade_mwh", "action_paths"),
                              ("daily_cashflow_eur", "daily_cashflows")):
            np.testing.assert_array_equal(path[column], saved[prefix + field][index])
        np.testing.assert_array_equal(before, saved[prefix + "inventory_paths"][index, :-1])
        np.testing.assert_array_equal(after, saved[prefix + "inventory_paths"][index, 1:])
        fields = ("coefficients", "knots", "inventory_grid", "actions", "initial_continuation",
                  "constant_dates", "constant_continuation")
        decisions = {field: saved[prefix + field] for field in fields}
    decisions["initial_spot"] = inputs["spot"]
    replay = replay_storage_path(path["spot_eur_per_mwh"], product, decisions)
    np.testing.assert_array_equal(replay["recommended_action"], trades)
    np.testing.assert_array_equal(replay["recommended_inventory"], np.r_[before[0], after])
    np.testing.assert_allclose(replay["recommended_cashflows"], path["daily_cashflow_eur"],
                               rtol=0, atol=allowance)

