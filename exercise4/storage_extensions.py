"""OPTIONAL EXERCISE 4 EXTENSIONS — FINER TRADES AND SCRAMBLED SOBOL.

Here we test whether finer trading volumes improve storage value and whether
scrambled Sobol sampling estimates the value of fixed decisions more precisely. The bridge changes Gaussian input ordering, not the
exact daily model, information available to decisions, or monitoring frequency.

Run from the project root: python -m exercise4.storage_extensions --workers 4
Small mathematical checks only: python -m exercise4.storage_extensions --check
"""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.special import ndtri
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from results_integrity import load_result_json, verified_result_bytes
from models import GasPriceModel
from products import GasStorage
from exercise4.gas_price_simulation import simulate_gas_prices
from exercise4.gas_storage_lsmc import estimate_storage_decisions, apply_storage_decisions
from exercise4.gas_storage_benchmark import solve_storage_benchmark, load_storage_benchmark

PROJECT = Path(__file__).resolve().parents[1]
METHODS = ("antithetic_mc", "sobol_chronological", "sobol_bridge")
PATH_COUNTS = (8192, 16384)
EXTENSION_SEED = 20261012
SOURCES = ("models.py", "products.py", "basis.py", "exercise4/gas_price_simulation.py",
           "exercise4/gas_storage_lsmc.py", "exercise4/gas_storage_benchmark.py",
           "exercise4/storage_extensions.py", "requirements.txt")


def source_hashes():
    return {name: sha256((PROJECT / name).read_bytes()).hexdigest() for name in SOURCES}


def load_baseline():
    # Here the existing loader verifies published data and matching array identities.
    from exercise4.ui_exercise4 import load_exercise4_results
    return load_exercise4_results()


def saved_decisions(document, arrays, case):
    fields = ("coefficients", "knots", "inventory_grid", "actions", "initial_continuation",
              "constant_dates", "constant_continuation")
    result = {name: arrays[f"{case}_{name}"] for name in fields}
    result["initial_spot"] = document["config"]["initial_spot"]
    return result


def gaussian_log_paths(normals, initial_spot, model, bridge=False):
    """Rows are paths; columns are independent normal coordinates, one per day.

    For l<t<r, q(h)=sigma^2*(1-exp(-2*kappa*h))/(2*kappa).
    Given X_l and X_r, the conditional mean at t is
    m+a*(X_l-m)+b*q(t-l)/q(r-l)*(X_r-m-a*b*(X_l-m)),
    and its variance is q(t-l)*q(r-t)/q(r-l).
    Here a=exp(-kappa*(t-l)), b=exp(-kappa*(r-t)), m=E[X_infinity].
    These Gaussian conditioning identities preserve the full joint daily law.
    """
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or not np.all(np.isfinite(normals)) or initial_spot <= 0:
        raise ValueError("Supply finite paths x daily-transition normals and a positive spot")
    paths, days = normals.shape
    logs = np.empty((paths, days + 1))
    logs[:, 0] = np.log(initial_spot)
    kappa, sigma, mean = model.mean_reversion_speed, model.volatility, model.log_price_mean
    if kappa <= 0:
        raise ValueError("This extension uses the assignment's positive mean reversion")
    variance = lambda h: sigma**2 * (-np.expm1(-2 * kappa * h)) / (2 * kappa)
    if not bridge or sigma == 0:
        decay, scale = np.exp(-kappa), np.sqrt(variance(1))
        for day in range(1, days + 1):
            logs[:, day] = mean + decay * (logs[:, day - 1] - mean) + scale * normals[:, day - 1]
        return logs
    if days == 0:
        return logs
    # 1. USE THE FIRST COORDINATE FOR THE TERMINAL LOG PRICE
    logs[:, days] = mean + np.exp(-kappa * days) * (logs[:, 0] - mean) + np.sqrt(variance(days)) * normals[:, 0]
    intervals, coordinate = deque([(0, days)]), 1
    # 2. FILL MIDPOINTS IN BREADTH-FIRST ORDER, CONDITIONING ON BOTH ENDPOINTS
    while intervals:
        left, right = intervals.popleft()
        if right - left <= 1:
            continue
        middle = (left + right) // 2
        h1, h2 = middle - left, right - middle
        a, b = np.exp(-kappa * h1), np.exp(-kappa * h2)
        q1, q2, total = variance(h1), variance(h2), variance(h1 + h2)
        expected = mean + a * (logs[:, left] - mean)
        expected += b * q1 / total * (logs[:, right] - mean - a * b * (logs[:, left] - mean))
        logs[:, middle] = expected + np.sqrt(q1 * q2 / total) * normals[:, coordinate]
        coordinate += 1
        intervals.extend(((left, middle), (middle, right)))
    if coordinate != days:
        raise RuntimeError("Every bridge coordinate must fill exactly one daily date")
    return logs


def sampling_prices(method, count, repetition, product, model, initial_spot):
    if method not in METHODS or count < 2 or count & (count - 1):
        raise ValueError("Use a named method and a power-of-two path budget")
    # Here chronological and bridge Sobol share each scramble for a paired
    # comparison. Replications are independent. Budgets within a replication
    # are nested prefixes; they are not independent experiments.
    family = 0 if method == "antithetic_mc" else 1
    seed = np.random.SeedSequence(EXTENSION_SEED, spawn_key=(4, family, repetition))
    if method == "antithetic_mc":
        return simulate_gas_prices(initial_spot, model, product.operating_dates(), count, seed, True)
    dimensions = len(product.operating_dates()) - 1
    generator = qmc.Sobol(dimensions, scramble=True, bits=52, rng=np.random.default_rng(seed))
    uniforms = generator.random_base2(int(np.log2(count)))
    # Here endpoint clipping only prevents an infinite inverse-normal value.
    # We retain the first Sobol point, with no thinning or antithetic doubling.
    normals = ndtri(np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps))
    return np.exp(gaussian_log_paths(normals, initial_spot, model, method == "sobol_bridge"))


def make_replay(document, arrays, case, prices, decisions, evaluation, step, method):
    # Here we reuse the existing presentation formatter, after numerical work.
    # Only three actual paths and their action comparisons are stored for the UI.
    from exercise4.ui_exercise4 import build_storage_replay_data
    temporary = deepcopy(document)
    temporary["config"]["inventory_step"] = step
    row = next(row for row in temporary["main_cases"] if row["case"] == case)
    row["evaluation_value"] = float(evaluation["cashflows"].mean())
    row["standard_error"] = evaluation["standard_error"] if method == "antithetic_mc" else None
    saved = dict(arrays)
    for name in ("coefficients", "knots", "inventory_grid", "initial_continuation", "constant_dates", "constant_continuation"):
        saved[f"{case}_{name}"] = decisions[name]
    saved[f"{case}_spot_paths"] = prices[:3]
    for name in ("inventory_paths", "action_paths", "daily_cashflows"):
        saved[f"{case}_{name}"] = evaluation[name]
    replay = build_storage_replay_data(temporary, saved, case)
    replay["method"] = method
    replay["inventory_step"] = step
    replay["evaluation_paths"] = len(prices)
    replay["path_note"] = ("Three saved paths from the first scrambled Sobol bridge repetition. These are not antithetic pairs. All start at EUR 20/MWh."
        if method == "sobol_bridge" else "Three saved evaluation paths, shared with the assignment baseline. Paths 1 and 2 are an antithetic pair; path 3 starts another pair.")
    return replay


def finer_grid_case(case):
    document, arrays = load_baseline()
    benchmark = load_storage_benchmark()
    config = document["config"]
    row = next(row for row in document["main_cases"] if row["case"] == case)
    reference = next(row for row in benchmark["cases"] if row["case"] == case)
    product, model = GasStorage(**config["product"]), GasPriceModel(**row["model"])
    seed = np.random.SeedSequence(config["seed"], spawn_key=(4, 0 if case == "high" else 1, 0))
    training_seed, evaluation_seed = seed.spawn(2)
    with threadpool_limits(limits=1):
        training = simulate_gas_prices(config["initial_spot"], model, product.operating_dates(), 10000, training_seed)
        evaluation_prices = simulate_gas_prices(config["initial_spot"], model, product.operating_dates(), 10000, evaluation_seed)
        if not np.array_equal(evaluation_prices[:3], arrays[f"{case}_spot_paths"]):
            raise RuntimeError("The common evaluation prices do not reproduce the saved baseline")
        baseline = saved_decisions(document, arrays, case)
        started = perf_counter()
        coarse = estimate_storage_decisions(training, product, model, 2500.0, snapshot_dates=())
        coarse_fit_seconds = perf_counter() - started
        if not np.allclose(coarse["coefficients"], baseline["coefficients"], rtol=1e-11, atol=1e-7, equal_nan=True):
            raise RuntimeError("Original training prices do not reproduce the saved coefficient vectors")
        started = perf_counter()
        fine = estimate_storage_decisions(training, product, model, 1250.0, snapshot_dates=())
        fine_fit_seconds = perf_counter() - started
        evaluated = apply_storage_decisions(evaluation_prices, product, fine)
        difference = evaluated["cashflows"] - arrays[f"{case}_evaluation_cashflows"]
        pairs = difference.reshape(-1, 2).mean(axis=1)
        refinements = []
        # Here each fine-grid reference matches a saved coarse-grid integration
        # setting. The final wide-domain setting is identical to the baseline.
        for label in ("price_grid_medium", "quadrature_fine", "domain_wide"):
            old = next(x for x in reference["refinements"] if x["refinement"] == label)
            fresh = solve_storage_benchmark(product, model, config["initial_spot"], 1250.0,
                old["price_points"], old["quadrature_points"], old["domain_standard_deviations"], fine)
            fresh.update({"refinement": label, "coarse_reference_value": old["reference_value"],
                          "optimized_grid_uplift": fresh["reference_value"] - old["reference_value"]})
            refinements.append(fresh)
        replay = make_replay(document, arrays, case, evaluation_prices, fine, evaluated, 1250.0, "antithetic_mc")
    return {"case": case, "grids_mwh": [2500, 1250], "inventory_levels": [101, 201], "action_counts": [5, 9],
        "training_paths": 10000, "evaluation_paths": 10000, "coarse_fit_seconds": coarse_fit_seconds,
        "fine_fit_seconds": fine_fit_seconds, "training_coefficient_reproduction": True,
        "common_evaluation_prices_reproduction": True, "fine_training_value": fine["training_value"],
        "fine_evaluation_value": evaluated["value"], "fine_evaluation_standard_error": evaluated["standard_error"],
        "paired_value_change": float(pairs.mean()), "paired_change_standard_error": float(pairs.std(ddof=1) / np.sqrt(len(pairs))),
        "fine_policy_value_change": refinements[-1]["saved_policy_value"] - reference["reference"]["saved_policy_value"],
        "constraint_diagnostics": evaluated["constraint_diagnostics"], "refinements": refinements, "replay": replay}


def sampling_repetition(job):
    method, repetition = job
    document, arrays = load_baseline()
    row = next(row for row in document["main_cases"] if row["case"] == "high")
    product, model = GasStorage(**document["config"]["product"]), GasPriceModel(**row["model"])
    decisions = saved_decisions(document, arrays, "high")
    results, replay = [], None
    with threadpool_limits(limits=1):
        for count in PATH_COUNTS:
            started = perf_counter()
            prices = sampling_prices(method, count, repetition, product, model, document["config"]["initial_spot"])
            evaluated = apply_storage_decisions(prices, product, decisions, method == "antithetic_mc")
            elapsed = perf_counter() - started
            # Never use the ordinary per-path SE for dependent Sobol points.
            results.append({"method": method, "repetition": repetition + 1, "paths": count,
                "value": float(evaluated["cashflows"].mean()), "runtime_seconds": elapsed,
                "constraint_diagnostics": evaluated["constraint_diagnostics"]})
            if method == "sobol_bridge" and repetition == 0 and count == PATH_COUNTS[-1]:
                replay = make_replay(document, arrays, "high", prices, decisions, evaluated, 2500.0, method)
    return results, replay


def mathematical_checks():
    # Here an exact covariance factor test checks the ENTIRE joint law, without
    # accepting a sampler merely because noisy marginal moments look plausible.
    model = GasPriceModel(.05, np.log(20.0), .0945)
    initial = np.exp(model.log_price_mean)
    errors = {}
    for days in (1, 7, 358):
        mean = gaussian_log_paths(np.zeros((1, days)), initial, model, True)[0, 1:]
        factor = (gaussian_log_paths(np.eye(days), initial, model, True)[:, 1:] - mean).T
        t = np.arange(1, days + 1)
        exact = model.volatility**2 / (2 * model.mean_reversion_speed) * (
            np.exp(-model.mean_reversion_speed * np.abs(t[:, None] - t[None, :]))
            - np.exp(-model.mean_reversion_speed * (t[:, None] + t[None, :])))
        error = float(np.max(np.abs(factor @ factor.T - exact)))
        if error > 1e-12:
            raise AssertionError("Bridge covariance differs from the exact Gaussian process")
        errors[str(days)] = error
    product = GasStorage(end_date="2005-07-14")
    for method in METHODS:
        small = sampling_prices(method, 8, 0, product, model, 20.0)
        large = sampling_prices(method, 16, 0, product, model, 20.0)
        if not np.array_equal(small, large[:8]):
            raise AssertionError("Budget prefixes are not reproducible")
    return {"bridge_maximum_covariance_error": errors, "nested_budget_prefixes": True}


def run_extensions(workers=4, repetitions=16):
    if repetitions < 2 or workers < 1:
        raise ValueError("Use at least two independent repetitions and one worker")
    started = perf_counter()
    hashes = source_hashes()
    checks = mathematical_checks()
    baseline, _ = load_baseline()
    benchmark = load_storage_benchmark()
    reference = next(row for row in benchmark["cases"] if row["case"] == "high")
    result = {"exercise": 4, "scope": "Optional extensions; assignment baseline unchanged",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": hashes,
        "baseline_sha256": sha256((PROJECT / "results/exercise4_results.json").read_bytes()).hexdigest(),
        "baseline_arrays_sha256": baseline["arrays_sha256"], "seed": EXTENSION_SEED,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__, "workers": workers,
        "repetitions": repetitions, "paths": list(PATH_COUNTS), "checks": checks,
        "policy_reference": reference["reference"]["saved_policy_value"],
        "reference_stability": reference["stability_changes_eur"],
        "uncertainty": "SD across independent replications; SE of their mean = SD/sqrt(R). No Sobol per-path SE. Budgets are nested and the two Sobol constructions share scramble seeds.",
        "timing": "One BLAS thread per worker; measured generation plus evaluation, excluding data loading, fitting and UI export. Worker wall times include concurrent contention.",
        "references": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html",
                       "https://docs.scipy.org/doc/scipy/reference/stats.qmc.html",
                       "https://www2.maths.ox.ac.uk/~gilesm/mc/module_6/QMC.pdf"]}
    # Keep long grid fits separate from the matched sampling experiment so their
    # workload does not distort the three samplers' measured evaluation cost.
    with get_context("spawn").Pool(min(workers, 2)) as pool:
        result["grid"] = []
        for row in pool.imap(finer_grid_case, ("high", "low")):
            result["grid"].append(row)
            print(f"Finer grid {row['case']}: optimized uplift EUR {row['refinements'][-1]['optimized_grid_uplift']:.6f}; paired estimate change EUR {row['paired_value_change']:.2f}", flush=True)
    jobs = [(method, repetition) for repetition in range(repetitions) for method in METHODS]
    observations = []
    with get_context("spawn").Pool(workers) as pool:
        for rows, replay in pool.imap(sampling_repetition, jobs):
            observations.extend(rows)
            if replay is not None:
                result["sobol_bridge_replay"] = replay
            print(f"Sampling {rows[0]['method']} repetition {rows[0]['repetition']} complete", flush=True)
    result["observations"], result["sampling"] = observations, []
    for method in METHODS:
        for count in PATH_COUNTS:
            rows = [row for row in observations if row["method"] == method and row["paths"] == count]
            values = np.array([row["value"] for row in rows])
            errors = values - result["policy_reference"]
            result["sampling"].append({"method": method, "paths": count, "mean_value": float(values.mean()),
                "mean_error": float(errors.mean()), "replication_sd": float(values.std(ddof=1)),
                "mean_standard_error": float(values.std(ddof=1) / np.sqrt(repetitions)),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in rows]))})
    if hashes != source_hashes():
        raise RuntimeError("Numerical extension source changed during the run")
    result["total_runtime_seconds"] = perf_counter() - started
    return result


def load_storage_extensions(results_path=None, baseline_path=None):
    results_path = Path(results_path or PROJECT / "results/exercise4_extensions.json")
    document = load_result_json(results_path, exercise=4, required=("code_sha256", "grid",
        "sampling", "observations", "sobol_bridge_replay", "baseline_sha256", "baseline_arrays_sha256"))
    baseline_path = Path(baseline_path or PROJECT / "results/exercise4_results.json")
    baseline = load_result_json(baseline_path, exercise=4, required=("arrays_file", "arrays_sha256"))
    verified_result_bytes(baseline_path, document["baseline_sha256"])
    verified_result_bytes(baseline_path.with_name(baseline["arrays_file"]), document["baseline_arrays_sha256"])
    if document["baseline_arrays_sha256"] != baseline["arrays_sha256"]:
        raise ValueError("Extensions refer to different baseline arrays")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        help="JSON destination; default: a new runs/<run-id>/ folder")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(mathematical_checks(), indent=2))
    else:
        output = run_extensions(args.workers, args.repetitions)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-storage-extensions"
        path = args.output or (PROJECT / "runs" / run_id / "exercise4_extensions.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(output, indent=2, allow_nan=False))
        temporary.replace(path)
        print(f"Saved {path}", flush=True)
