"""EXERCISE 4.3 — AN INDEPENDENT FULL-YEAR STORAGE BENCHMARK.

Here we solve the same daily Bellman recursion by Gaussian quadrature on a
price grid, without simulations or regressions. The known lognormal transition
supplies tomorrow's distribution. This is a numerical reference, not an exact
price or a proved upper bound. Price-grid, quadrature and domain refinements
are reported separately.

Run from the project root: python -m exercise4.gas_storage_benchmark
"""
import argparse
from results_integrity import load_result_json, verified_result_bytes
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.special import roots_hermitenorm, ndtr
from threadpoolctl import threadpool_limits

from products import GasStorage
from models import GasPriceModel
from exercise4.gas_storage_lsmc import (storage_inventory_grid,
    storage_grid_actions, feasible_storage_transitions, preferred_storage_actions,
    storage_continuation_values)

PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
NUMERICAL_FILES = ("products.py", "models.py", "basis.py",
    "exercise4/gas_storage_lsmc.py", "exercise4/gas_storage_benchmark.py")


def benchmark_price_grid(initial_spot, model, price_points, domain_standard_deviations):
    if (not isinstance(price_points, int) or price_points < 5 or price_points % 2 == 0
            or not np.isfinite(initial_spot) or initial_spot <= 0
            or not np.isfinite(domain_standard_deviations) or domain_standard_deviations <= 0):
        raise ValueError("Use a positive spot and domain width, and an odd grid size of at least five")
    # Here a symmetric log grid includes the observed initial spot EXACTLY.
    # Its edges extend beyond both initial and stationary log-price means.
    center = np.log(initial_spot)
    stationary_deviation = model.volatility / np.sqrt(2 * model.mean_reversion_speed)
    radius = max(domain_standard_deviations * stationary_deviation
                 + abs(center - model.log_price_mean), .01)
    prices = np.exp(np.linspace(center - radius, center + radius, price_points))
    prices[price_points // 2] = initial_spot
    return prices


def gaussian_expectation_matrix(prices, model, quadrature_points):
    if not isinstance(quadrature_points, int) or quadrature_points < 2:
        raise ValueError("Use at least two Gauss-Hermite quadrature points")
    # 1. INTEGRATE THE EXACT CONDITIONAL LOG-PRICE LAW
    # Here roots_hermitenorm integrates exp(-z*z/2); dividing the weights by
    # sqrt(2*pi) converts that integral to an expectation for Z ~ N(0,1).
    nodes, weights = roots_hermitenorm(quadrature_points)
    weights /= np.sqrt(2 * np.pi)
    decay = np.exp(-model.mean_reversion_speed)
    variance = model.volatility**2 * (-np.expm1(-2 * model.mean_reversion_speed)) / (
        2 * model.mean_reversion_speed)
    next_prices = np.exp(model.log_price_mean
        + (np.log(prices[:, None]) - model.log_price_mean) * decay
        + np.sqrt(variance) * nodes[None, :])

    # 2. INTERPOLATE VALUES LINEARLY IN SPOT ON THE LOG-SPACED GRID
    # Here affine cashflows in S, including the last forced trade, are represented
    # exactly. Outside the grid we extend the nearest segment linearly. Those
    # tail weights may be negative: domain refinement is necessary, and this
    # approximation is NOT a proof of an upper/lower bound on the true value.
    lower = np.clip(np.searchsorted(prices, next_prices) - 1, 0, len(prices) - 2)
    fraction = (next_prices - prices[lower]) / (prices[lower + 1] - prices[lower])
    rows = np.broadcast_to(np.arange(len(prices))[:, None], next_prices.shape)
    matrix = csr_matrix((np.concatenate(((weights * (1 - fraction)).ravel(),
                                        (weights * fraction).ravel())),
        (np.tile(rows.ravel(), 2), np.concatenate((lower.ravel(), (lower + 1).ravel())))),
        shape=(len(prices), len(prices)))
    # Here time-homogeneous model parameters let every date reuse this matrix.
    return matrix


def solve_storage_benchmark(product, model, initial_spot=20.0, inventory_step=2500.0,
                            price_points=3201, quadrature_points=256,
                            domain_standard_deviations=6.0, decisions=None):
    started = perf_counter()
    prices = benchmark_price_grid(initial_spot, model, price_points, domain_standard_deviations)
    expectation = gaussian_expectation_matrix(prices, model, quadrature_points)
    inventories = storage_inventory_grid(product, inventory_step)
    actions = storage_grid_actions(product, inventory_step)
    order = preferred_storage_actions(actions)
    number_of_dates = len(product.operating_dates())
    initial_index = round(product.initial_inventory / inventory_step)
    initial_price_index = price_points // 2
    if decisions is not None:
        if (not np.array_equal(decisions["inventory_grid"], inventories)
                or not np.array_equal(decisions["actions"], actions)
                or decisions["initial_spot"] != initial_spot):
            raise ValueError("The saved decisions must use the same inventory grid, actions and initial spot")

    # 3. SET ZERO TERMINAL CONTINUATION AT THE REQUIRED FINAL INVENTORY
    # Here unreachable entries are unused zeros. Feasible transitions alone
    # select continuation values; no unreachable state can enter maximisation.
    future_values = np.zeros((price_points, len(inventories)))
    policy_future_values = np.zeros_like(future_values) if decisions is not None else None
    initial_action = None
    policy_initial_action = None
    with threadpool_limits(limits=1):
        for day in range(number_of_dates - 1, -1, -1):
            destinations = feasible_storage_transitions(product, inventories, actions, day, number_of_dates)
            continuation = expectation @ future_values if day < number_of_dates - 1 else future_values
            values = np.full_like(future_values, -np.inf)
            if decisions is not None:
                policy_continuation = (expectation @ policy_future_values
                    if day < number_of_dates - 1 else policy_future_values)
                fitted_continuation = storage_continuation_values(prices, day, decisions)
                fitted_best = np.full_like(values, -np.inf)
                policy_values = np.zeros_like(values)

            # 4. MAXIMISE TODAY'S CASHFLOW PLUS INTEGRATED TOMORROW'S VALUE
            for action_index in order:
                current = np.flatnonzero(destinations[:, action_index] >= 0)
                following = destinations[current, action_index]
                cashflow = product.cashflow(prices, actions[action_index])[:, None]
                candidate = cashflow + continuation[:, following]
                improves = candidate > values[:, current] + 1e-10
                if day == 0 and initial_index in current:
                    column = int(np.flatnonzero(current == initial_index)[0])
                    if improves[initial_price_index, column]:
                        initial_action = float(actions[action_index])
                values[:, current] = np.where(improves, candidate, values[:, current])

                # 5. ALSO VALUE THE EXISTING LSMC DECISIONS WITHOUT REFITTING
                # Here the fitted continuation selects the trade. Its actual
                # future value comes from a SECOND quadrature recursion, so it
                # does not confuse estimated continuation with realised value.
                if decisions is not None:
                    fitted_score = cashflow + fitted_continuation[:, following]
                    selected = fitted_score > fitted_best[:, current] + 1e-10
                    fitted_best[:, current] = np.where(selected, fitted_score, fitted_best[:, current])
                    policy_values[:, current] = np.where(selected,
                        cashflow + policy_continuation[:, following], policy_values[:, current])
                    if day == 0 and initial_index in current:
                        if selected[initial_price_index, column]:
                            policy_initial_action = float(actions[action_index])
            future_values = np.where(np.isfinite(values), values, 0.0)
            if decisions is not None:
                policy_future_values = policy_values

    reference_value = float(future_values[initial_price_index, initial_index])
    result = {"price_points": price_points, "quadrature_points": quadrature_points,
        "domain_standard_deviations": domain_standard_deviations,
        "minimum_spot": float(prices[0]), "maximum_spot": float(prices[-1]),
        "log_price_spacing": float(np.log(prices[-1] / prices[0]) / (price_points - 1)),
        "reference_value": reference_value, "initial_action": initial_action,
        "runtime_seconds": perf_counter() - started}
    if decisions is not None:
        policy_value = float(policy_future_values[initial_price_index, initial_index])
        result.update({"saved_policy_value": policy_value,
            "policy_value_gap": reference_value - policy_value,
            "policy_value_gap_percent": 100 * (reference_value - policy_value) / reference_value,
            "policy_initial_action": policy_initial_action})
    return result


def exact_three_day_value(product, model, initial_spot):
    """Here the final forced trade makes a three-date benchmark analytical."""
    if len(product.operating_dates()) != 3 or model.volatility <= 0:
        raise ValueError("This analytic check needs three dates and positive volatility")
    decay = np.exp(-model.mean_reversion_speed)
    variance = model.volatility**2 * (1 - decay**2) / (2 * model.mean_reversion_speed)
    log_mean = model.log_price_mean + (np.log(initial_spot) - model.log_price_mean) * decay
    coefficient = np.exp((1 - decay) * model.log_price_mean + variance / 2)
    threshold_log = np.log(coefficient) / (1 - decay)
    def moment(power, below_threshold=False):
        result = np.exp(power * log_mean + power**2 * variance / 2)
        if below_threshold:
            result *= ndtr((threshold_log - log_mean - power * variance) / np.sqrt(variance))
        return result
    # Here g(s)=E[S_next|s]=coefficient*s**decay. The max chooses the largest
    # feasible trade when g(s)>s and the smallest otherwise. Truncated lognormal
    # moments integrate this kink exactly, independently of grid/quadrature.
    expected_spot = moment(1)
    expected_next_spot = coefficient * moment(decay)
    positive_spread = coefficient * moment(decay, True) - moment(1, True)
    values = []
    for change in storage_grid_actions(product, 2500.0):
        next_inventory = product.initial_inventory + change
        lower, upper = product.reachable_inventory_bounds(2)
        if not lower <= next_inventory <= upper:
            continue
        final_lower, final_upper = product.reachable_inventory_bounds(1)
        minimum_change = max(-product.withdrawal_limit, final_lower - next_inventory)
        maximum_change = min(product.injection_limit, final_upper - next_inventory)
        continuation = ((next_inventory - product.final_inventory) * expected_next_spot
            + minimum_change * (expected_next_spot - expected_spot)
            + (maximum_change - minimum_change) * positive_spread)
        values.append(-initial_spot * change + continuation)
    return float(max(values))


def load_storage_benchmark(results_path=None, baseline_path=None):
    results_path = Path(results_path or PROJECT_DIRECTORY / "results/exercise4_benchmark.json")
    document = load_result_json(results_path, exercise=4, required=("code_sha256", "cases",
        "baseline_results_file", "baseline_results_sha256", "baseline_arrays_sha256"))
    baseline_path = Path(baseline_path or results_path.with_name(document["baseline_results_file"]))
    verified_result_bytes(baseline_path, document["baseline_results_sha256"])
    baseline = load_result_json(baseline_path, exercise=4, required=("arrays_file", "arrays_sha256"))
    verified_result_bytes(baseline_path.with_name(baseline["arrays_file"]), document["baseline_arrays_sha256"])
    if document["baseline_arrays_sha256"] != baseline["arrays_sha256"]:
        raise ValueError("The benchmark comparison refers to different baseline arrays")
    return document


def run_benchmark_study(results_path):
    results_path = Path(results_path)
    original = json.loads(results_path.read_text(encoding="utf-8"))
    arrays_path = results_path.with_name(original["arrays_file"])
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != original["arrays_sha256"]:
        raise ValueError("The saved LSMC decisions do not match the recorded arrays hash")
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    source_hashes = {name: hashlib.sha256((PROJECT_DIRECTORY / name).read_bytes()).hexdigest()
                     for name in NUMERICAL_FILES}
    config = original["config"]
    product = GasStorage(**config["product"])
    initial_spot = config["initial_spot"]
    step = config["inventory_step"]
    fields = ("coefficients", "knots", "inventory_grid", "actions", "initial_continuation",
              "constant_dates", "constant_continuation")
    document = {"exercise": 4, "method": "Daily Bellman recursion with Gauss-Hermite quadrature",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": source_hashes,
        "baseline_results_file": results_path.name,
        "baseline_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "baseline_arrays_sha256": original["arrays_sha256"],
        "product": asdict(product), "initial_spot": initial_spot, "inventory_step": step,
        "interest_rate": config["interest_rate"], "number_of_decision_dates": len(product.operating_dates()),
        "interpolation": "Linear in spot on a log-spaced grid, with linear endpoint extrapolation",
        "scope": "Same daily dates, five actions and 101 inventory states as the assignment implementation",
        "limitations": "Numerical reference, not an exact value or a rigorous bound; refinements are empirical stability checks. This does not validate the market model or remove inventory discretisation.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": []}
    for baseline in original["main_cases"]:
        case = baseline["case"]
        model = GasPriceModel(**baseline["model"])
        decisions = {name: arrays[f"{case}_{name}"] for name in fields}
        decisions["initial_spot"] = initial_spot
        # Here each comparison changes only the named numerical approximation.
        settings = [("price_grid_coarse", 801, 128, 6.0),
                    ("price_grid_medium", 1601, 128, 6.0),
                    ("price_grid_fine", 3201, 128, 6.0),
                    ("quadrature_coarse", 3201, 64, 6.0),
                    ("quadrature_fine", 3201, 256, 6.0)]
        narrow = benchmark_price_grid(initial_spot, model, 3201, 6.0)
        wide = benchmark_price_grid(initial_spot, model, 3201, 8.0)
        ratio = np.log(wide[-1] / wide[0]) / np.log(narrow[-1] / narrow[0])
        wide_points = 2 * int(np.ceil(1600 * ratio)) + 1
        settings.append(("domain_wide", wide_points, 256, 8.0))
        refinements = []
        for label, points, quadrature, width in settings:
            result = solve_storage_benchmark(product, model, initial_spot, step,
                points, quadrature, width, decisions)
            result["refinement"] = label
            refinements.append(result)
            print(f"{case}: {label}, {points} prices / {quadrature} quadrature nodes: "
                  f"reference EUR {result['reference_value']:,.2f}; "
                  f"saved decisions EUR {result['saved_policy_value']:,.2f}; "
                  f"{result['runtime_seconds']:.1f}s", flush=True)
        reference = refinements[-1]
        by_label = {row["refinement"]: row for row in refinements}
        stability = {}
        for name, left, right in (("price_grid", "price_grid_medium", "price_grid_fine"),
                                  ("quadrature", "price_grid_fine", "quadrature_fine"),
                                  ("domain", "quadrature_fine", "domain_wide")):
            stability[name] = {quantity: abs(by_label[left][quantity] - by_label[right][quantity])
                for quantity in ("reference_value", "saved_policy_value", "policy_value_gap")}
        short_product = GasStorage(**{**asdict(product), "start_date": str(
            np.datetime64(product.end_date) - np.timedelta64(2, "D"))})
        exact_value = exact_three_day_value(short_product, model, initial_spot)
        short_result = solve_storage_benchmark(short_product, model, initial_spot, step,
            3201, 1024, 8.0)
        zero_model = GasPriceModel(model.mean_reversion_speed, np.log(initial_spot), 0.0)
        zero_result = solve_storage_benchmark(product, zero_model, initial_spot, step, 101, 8, 6.0)
        document["cases"].append({"case": case, "label": baseline["label"], "model": asdict(model),
            "refinements": refinements, "reference": reference, "stability_changes_eur": stability,
            "baseline_evaluation_value": baseline["evaluation_value"],
            "baseline_standard_error": baseline["standard_error"],
            "baseline_error_relative_to_saved_policy": baseline["evaluation_value"] - reference["saved_policy_value"],
            "baseline_error_in_standard_errors": (baseline["evaluation_value"] - reference["saved_policy_value"])
                                                    / baseline["standard_error"],
            "analytic_three_day_check": {"exact_value": exact_value,
                "numerical_value": short_result["reference_value"],
                "absolute_error": abs(short_result["reference_value"] - exact_value),
                "quadrature_points": 1024, "price_points": 3201},
            "constant_price_full_year_check": {"exact_value": 0.0,
                "numerical_value": zero_result["reference_value"]}})
    current_hashes = {name: hashlib.sha256((PROJECT_DIRECTORY / name).read_bytes()).hexdigest()
                      for name in NUMERICAL_FILES}
    if source_hashes != current_hashes:
        raise ValueError("Benchmark source changed during the study; rerun with fixed source")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_DIRECTORY / "results/exercise4_results.json")
    parser.add_argument("--output", type=Path,
                        help="JSON destination; default: a new runs/<run-id>/ folder")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-storage-benchmark"
    args.output = args.output or (PROJECT_DIRECTORY / "runs" / run_id / "exercise4_benchmark.json")
    document = run_benchmark_study(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Saved {args.output}", flush=True)
