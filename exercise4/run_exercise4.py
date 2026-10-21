"""EXERCISE 4 — DAILY GAS PRICES, CONTINUATION VALUES AND STORAGE VALUE.

Here the three numbered sections answer 4.1, 4.2 and 4.3 of the assignment.
The contract and price parameters come from the assignment and Boogert–de Jong.
Run: python -m exercise4.run_exercise4 --study all --workers 2
"""
import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.special import ndtri
from threadpoolctl import threadpool_limits

from market_data import MarketData
from products import GasStorage
from models import GasPriceModel
from exercise4.gas_price_simulation import gas_price_moments
from exercise4.gas_storage_lsmc import (
    storage_inventory_grid, storage_grid_actions, feasible_storage_transitions,
    compare_storage_actions)
from valuation import StorageSimulationSettings, value_gas_storage

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
PATH_COUNTS = (50, 500, 5000, 10000, 20000)
NUMERICAL_FILES = ("market_data.py", "products.py", "models.py", "basis.py",
    "valuation.py", "exercise4/gas_price_simulation.py", "exercise4/gas_storage_lsmc.py",
    "exercise4/run_exercise4.py", "requirements.txt")


def exercise4_source_hashes():
    return {name: hashlib.sha256((PROJECT_DIRECTORY / name).read_bytes()).hexdigest()
            for name in NUMERICAL_FILES}


def pair_statistics(cashflows):
    # Here adjacent paths are antithetic. Independent observations are PAIR means.
    observations = np.asarray(cashflows, dtype=float).reshape(-1, 2).mean(axis=1)
    return (float(observations.mean()),
            float(observations.std(ddof=1) / np.sqrt(len(observations))))


# 4.1 — DECLARE THE PRICE INPUTS AND CHECK THE EXACT DAILY LAW
def gas_model(initial_spot=20.0, mean_reversion_speed=0.05, volatility=0.0945):
    if (not np.all(np.isfinite((initial_spot,mean_reversion_speed,volatility)))
            or min(initial_spot,mean_reversion_speed) <= 0 or volatility < 0):
        raise ValueError("Use positive finite spot and mean reversion, and nonnegative volatility")
    # Here mu is a log-price target, not an arithmetic mean spot. This choice
    # makes stationary E[S] equal to initial_spot, our illustrative EUR 20 scale.
    log_price_target = np.log(initial_spot) + volatility**2 / (4 * mean_reversion_speed)
    return GasPriceModel(mean_reversion_speed, log_price_target, volatility)


def price_moment_checks(spot_paths, model):
    # Here each date checks a known model identity, not an invented price path.
    selected_dates = sorted(set(min(day, spot_paths.shape[1]-1)
                               for day in (1, 90, 180, 358)))
    rows = []
    for day in selected_dates:
        logs = np.log(spot_paths[:, day])
        moments = gas_price_moments(float(spot_paths[0, 0]), model, day)
        expected_mean, expected_variance = map(float,
            (moments["log_mean"], moments["log_variance"]))
        pair_second_moments = ((logs - expected_mean)**2).reshape(-1, 2).mean(axis=1)
        variance_se = float(pair_second_moments.std(ddof=1) /
                            np.sqrt(len(pair_second_moments)))
        observed_variance = float(pair_second_moments.mean())
        rows.append({"elapsed_days": day,
            "analytic_log_mean": expected_mean, "simulated_log_mean": float(logs.mean()),
            "analytic_log_variance": expected_variance,
            "simulated_log_variance": observed_variance,
            "variance_standard_error": variance_se,
            "variance_z_score": ((observed_variance-expected_variance) / variance_se
                                 if variance_se else 0.0)})
    return rows


# 4.2 — USE THE ACTUAL MODEL TO CHECK THE LAST TWO DECISIONS
def last_two_day_comparison(product, model, decisions, initial_spot):
    date_index = len(product.operating_dates()) - 2
    moments = gas_price_moments(initial_spot, model, date_index)
    probabilities = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    spots = np.exp(moments["log_mean"] + np.sqrt(moments["log_variance"]) * ndtri(probabilities))
    rows = []
    for spot, probability in zip(spots, probabilities):
        comparison = compare_storage_actions(float(spot), date_index,
            product.initial_inventory, product, decisions)
        actions = np.asarray(comparison["actions"], dtype=float)
        inventories = np.asarray(comparison["next_inventories"], dtype=float)
        fitted = np.asarray(comparison["continuations"], dtype=float)
        totals = np.asarray(comparison["total_values"], dtype=float)
        feasible = np.isfinite(totals)
        # Here tomorrow's trade is forced, so its expected cashflow is known.
        next_mean = float(np.exp(model.log_price_mean
            + (np.log(spot)-model.log_price_mean)*np.exp(-model.mean_reversion_speed)
            + model.volatility**2 * (-np.expm1(-2*model.mean_reversion_speed))
              / (4*model.mean_reversion_speed)))
        exact_continuation = (inventories-product.final_inventory) * next_mean
        exact_totals = -spot*actions + exact_continuation
        exact_totals[~feasible] = -np.inf
        chosen = int(comparison["chosen_index"])
        exact_best = float(exact_totals.max())
        rows.append({"spot_quantile": float(probability), "spot": float(spot),
            "inventory": product.initial_inventory,
            "chosen_action": float(actions[chosen]),
            "fitted_continuation": float(fitted[chosen]),
            "analytic_continuation": float(exact_continuation[chosen]),
            "fitted_action_value": float(totals[chosen]),
            "analytic_value_of_chosen_action": float(exact_totals[chosen]),
            "analytic_optimal_value": exact_best,
            "decision_value_loss": max(0.0, exact_best-float(exact_totals[chosen])),
            "maximum_continuation_error": float(np.max(
                np.abs(fitted[feasible]-exact_continuation[feasible])))})
    return rows


def fixed_storage_schedule(product, expected_prices, inventory_step=2500.0):
    # Here volumes are chosen once using model E[S_t]. This is an EXPECTED
    # fixed-schedule value, not locked-in forward profit: forward trading is absent.
    grid = storage_inventory_grid(product, inventory_step)
    actions = storage_grid_actions(product, inventory_step)
    number_of_dates = len(expected_prices)
    next_values = np.full(len(grid), -np.inf)
    next_values[int(round(product.final_inventory/inventory_step))] = 0.0
    selected = np.zeros((number_of_dates, len(grid)), dtype=int)
    for day in range(number_of_dates-1, -1, -1):
        destinations = feasible_storage_transitions(product, grid, actions, day, number_of_dates)
        candidates = -expected_prices[day]*actions[None, :] + next_values[np.maximum(destinations, 0)]
        candidates[destinations < 0] = -np.inf
        choices = np.argmax(candidates, axis=1)
        selected[day] = choices
        next_values = candidates[np.arange(len(grid)), choices]
    initial_index = int(round(product.initial_inventory/inventory_step))
    value = float(next_values[initial_index])
    schedule = np.empty(number_of_dates)
    index = initial_index
    for day in range(number_of_dates):
        schedule[day] = actions[selected[day, index]]
        index += int(round(schedule[day]/inventory_step))
    if not np.isclose(grid[index], product.final_inventory):
        raise ValueError("The fixed schedule failed the final-inventory constraint")
    return value, schedule


# 4.3 — ESTIMATE, APPLY AND REPEAT THE SAME STORAGE ALGORITHM
def run_storage_repetition(job, progress=None):
    case, repetition, counts, config = job
    product = GasStorage(**config["product"])
    volatility = config["volatility_cases"][case]
    model = gas_model(config["initial_spot"], config["mean_reversion_speed"], volatility)
    market = MarketData(config["initial_spot"], 0.0)
    moments = gas_price_moments(market.spot, model, np.arange(len(product.operating_dates())))
    fixed_value, schedule = fixed_storage_schedule(product, moments["spot_mean"], config["inventory_step"])
    rows, main, arrays, checks = [], None, {}, {}
    for count in counts:
        # Here each count restarts the same branch. Even prefixes contain the
        # same training pairs; a separate child stream supplies common evaluation.
        seed = np.random.SeedSequence(config["seed"],
            spawn_key=(4, 0 if case == "high" else 1, repetition))
        settings = StorageSimulationSettings(count, config["evaluation_paths"], seed,
                                            inventory_step=config["inventory_step"], antithetic=True)
        started = perf_counter()
        # Here one BLAS thread avoids repeatedly coordinating many threads for
        # six-column regressions. Independent study repetitions can still run
        # in separate processes. This does not change the numerical method.
        def report_progress(stage, completed, total, date_index=None):
            if progress is not None:
                progress({"case": case, "stage": stage, "completed": completed,
                          "total": total, "date_index": date_index,
                          "training_paths": count, "repetition": repetition + 1})
        with threadpool_limits(limits=1, user_api="blas"):
            result = value_gas_storage(product, market, model, settings,
                                      progress=report_progress if progress is not None else None)
        elapsed = perf_counter() - started
        evaluation = result["evaluation"]
        rows.append({"case": case, "repetition": repetition+1,
            "volatility": volatility, "training_paths": count,
            "evaluation_paths": settings.evaluation_paths,
            "training_value": float(result["training_value"]),
            "evaluation_value": float(result["value"]), "value": float(result["value"]),
            "standard_error": float(result["standard_error"]),
            "constraint_diagnostics": evaluation["constraint_diagnostics"],
            "runtime_seconds": elapsed})
        print(f"{case}, run {repetition+1}, {count:,} estimation paths: "
              f"EUR {result['value']:,.0f}; {elapsed:.2f}s", flush=True)
        if repetition == 0 and count == config["training_paths"]:
            fixed_cashflows = -result["evaluation_spot_paths"] @ schedule
            gain, gain_se = pair_statistics(evaluation["cashflows"] - fixed_cashflows)
            main = dict(rows[-1], label=("High" if case=="high" else "Low")+" volatility",
                model=asdict(model), fixed_schedule_value=fixed_value,
                simulated_fixed_schedule_value=float(fixed_cashflows.mean()),
                gain_over_fixed=gain, gain_standard_error=gain_se,
                constraint_diagnostics=evaluation["constraint_diagnostics"],
                regression_diagnostics=result["decisions"]["regression_diagnostics"],
                cashflow_quantiles=dict(zip(("minimum","p05","median","p95","maximum"),
                    map(float,np.quantile(evaluation["cashflows"],[0,.05,.5,.95,1])))))
            decisions = result["decisions"]
            for name in ("coefficients","knots","inventory_grid","actions","initial_continuation",
                         "constant_dates","constant_continuation"):
                arrays[case+"_"+name] = decisions[name]
            arrays.update({case+"_spot_paths": result["evaluation_spot_paths"][:3],
                case+"_training_cashflows": decisions["training_cashflows"],
                case+"_evaluation_cashflows": evaluation["cashflows"],
                case+"_inventory_paths": evaluation["inventory_paths"],
                case+"_action_paths": evaluation["action_paths"],
                case+"_daily_cashflows": evaluation["daily_cashflows"],
                case+"_fixed_schedule": schedule, case+"_mean_prices": moments["spot_mean"]})
            checks = {"simulation": price_moment_checks(result["training_spot_paths"], model),
                "two_day": last_two_day_comparison(product,model,decisions,market.spot),
                "regression_examples": decisions["regression_snapshot"]}
    return {"rows": rows, "main": main, "arrays": arrays, "checks": checks}


def run_exercise4(study="all", training_paths=10000, evaluation_paths=10000,
                  path_counts=PATH_COUNTS, repetitions=10, seed=20261011, workers=1,
                  initial_spot=20.0, mean_reversion_speed=0.05, volatility=0.0945,
                  progress=None):
    if study not in ("main", "all"):
        raise ValueError("Study must be main or all")
    counts = tuple(sorted(set((*path_counts, training_paths)))) if study=="all" else (training_paths,)
    if any(not isinstance(n, int) or n < 4 or n%2 for n in (*counts,evaluation_paths)):
        raise ValueError("Use even path counts of at least four to retain independent pairs")
    if repetitions < 1 or workers < 1:
        raise ValueError("Repetitions and workers must be positive")
    if progress is not None and workers != 1:
        raise ValueError("Interactive progress uses one worker; study repetitions may run in parallel without it")
    source_hashes = exercise4_source_hashes()
    config = {"product": asdict(GasStorage()), "initial_spot": initial_spot,
        "mean_reversion_speed": mean_reversion_speed,
        "volatility_cases": {"high": volatility, "low": 0.0315},
        "training_paths": training_paths, "evaluation_paths": evaluation_paths,
        "path_counts": list(counts), "repetitions": repetitions if study=="all" else 1,
        "seed": seed, "antithetic": True, "inventory_step": 2500.0,
        "daily_time_step": 1.0, "interest_rate": 0.0,
        "price_level_choice": "Initial and stationary arithmetic mean spot are equal.",
        "basis": "Six cubic B-splines; outer probability limits .0001/.9999, interior thirds; linear tails.",
        "terminal_condition": "Exact reachability to 100000 MWh after the final trade.",
        "evaluation_design": "New independent pairs; common evaluation sample across M within each repetition.",
        "study": study, "workers": workers, "blas_threads": 1}
    jobs = [("high", repetition, counts, config) for repetition in range(config["repetitions"])]
    jobs += [("low", repetition, (training_paths,), config)
             for repetition in range(config["repetitions"])]
    document = {"exercise": 4, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": source_hashes, "config": config, "main_cases": [],
        "convergence": [], "validation": {}, "regression_examples": {},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__}}
    arrays = {}
    def keep_result(result):
        document["convergence"].extend(result["rows"])
        if result["main"] is not None:
            case = result["main"]["case"]
            document["main_cases"].append(result["main"])
            document["validation"][case] = {name: value for name,value in result["checks"].items()
                                           if name != "regression_examples"}
            document["regression_examples"][case] = result["checks"]["regression_examples"]
            arrays.update(result["arrays"])
    started = perf_counter()
    if workers == 1:
        for job in jobs:
            keep_result(run_storage_repetition(job, progress=progress))
    else:
        with get_context("spawn").Pool(processes=min(workers,len(jobs))) as pool:
            for result in pool.imap(run_storage_repetition, jobs):
                keep_result(result)
    if source_hashes != exercise4_source_hashes():
        raise ValueError("Numerical source changed during this study; do not label mixed-source results as one run")
    document["total_wall_seconds"] = perf_counter()-started
    return document, arrays


def json_values(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_exercise4_results(document, arrays, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output_path.with_name(output_path.stem.replace("_results","")+"_arrays.npz")
    temporary_arrays = arrays_path.with_suffix(".npz.tmp")
    with temporary_arrays.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary_arrays.replace(arrays_path)
    document["arrays_file"] = arrays_path.name
    document["arrays_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    temporary_json = output_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(document,default=json_values,indent=2,allow_nan=False))
    temporary_json.replace(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("main","all"), default="all")
    parser.add_argument("--training-paths", type=int, default=10000)
    parser.add_argument("--evaluation-paths", type=int, default=10000)
    parser.add_argument("--path-counts", type=int, nargs="+", default=list(PATH_COUNTS))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="JSON destination; companion NPZ stays beside it; default: runs/<run-id>/")
    args = parser.parse_args()
    # Here the JSON and arrays form one new run; published evidence is preserved.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-exercise4"
    args.output = args.output or (PROJECT_DIRECTORY / "runs" / run_id / "exercise4_results.json")
    document, arrays = run_exercise4(args.study,args.training_paths,args.evaluation_paths,
        tuple(args.path_counts),args.repetitions,args.seed,args.workers)
    save_exercise4_results(document,arrays,args.output)
    print(f"Saved {args.output}",flush=True)

