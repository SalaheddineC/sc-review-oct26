"""EXERCISE 3 — MILSTEIN, PRICING ERRORS AND CONVERGENCE.

Here we change the SDE stepping method and retain the economic objects, basis,
Longstaff–Schwartz exercise rules and exact European control from Exercises 1–2.
Every pricing error is reference minus estimate, following the assignment.
"""
import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from exercise2.barone_adesi_whaley import value_american_put_barone_adesi_whaley
from basis import spot_quadratic_basis
from euler_maruyama import (generate_brownian_increments,
    generate_two_driver_brownian_increments, simulate_paths_euler_maruyama)
from exercise2.gamma import calculate_american_put_gamma
from market_data import MarketData
from exercise3.milstein import simulate_paths_milstein
from models import GeometricBrownianMotionModel
from products import AmericanPut
from exercise1.run_exercise1 import make_stochastic_volatility_cases
from exercise2.run_exercise2 import NUMERICAL_FILES, read_published_at_the_money_case
from valuation import (SimulationSettings, make_simulation_and_exercise_grids,
    value_american_put_with_european_control,
    value_american_put_with_supplied_brownian_increments)

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
SCHEMES = {"Euler": simulate_paths_euler_maruyama, "Milstein": simulate_paths_milstein}
EVALUATION_COUNTS = (10000, 25000, 50000, 75000, 100000)
VOLATILITIES = (0.05, 0.10, 0.20, 0.30)


def exercise3_source_hashes():
    names = tuple(NUMERICAL_FILES) + ("exercise1/run_exercise1.py", "exercise3/milstein.py", "exercise3/run_exercise3.py")
    return {name: hashlib.sha256((PROJECT_DIRECTORY / name).read_bytes()).hexdigest()
            for name in names}


def stochastic_volatility_baseline():
    # Here we use the same stochastic-volatility baseline as Exercise 1.
    return next(case for case in make_stochastic_volatility_cases()
        if case.study == "sv_sensitivities/mean_reversion_speed"
        and case.model.mean_reversion_speed == 2.0)


def simulation_settings(training, evaluation, seed, branch, step_days=1.0):
    return SimulationSettings(int(training), int(evaluation),
        np.random.SeedSequence(seed, spawn_key=tuple(branch)), step_days, 365)


def mean_statistics(observations):
    count = len(observations)
    variance = float(np.var(observations, ddof=1) / count)
    return float(np.mean(observations)), variance, float(np.sqrt(variance))


# PART 2 — GBM PRICES WITH AND WITHOUT THE EUROPEAN CONTROL

def run_gbm_pricing(training, counts, seed):
    paper_case = read_published_at_the_money_case()
    product = AmericanPut(paper_case["strike"], 1 / 12)
    rows = []
    for volatility_index, volatility in enumerate(VOLATILITIES):
        market = MarketData(paper_case["spot"], 0.0, volatility, 0.0)
        model = GeometricBrownianMotionModel(0.0, volatility)
        branch = (3, 0, volatility_index)
        settings = simulation_settings(training, max(counts), seed, branch)
        times, exercise_indices = make_simulation_and_exercise_grids(product.maturity_years, settings)
        training_seed, evaluation_seed = np.random.SeedSequence(seed, spawn_key=branch).spawn(2)
        training_increments = generate_brownian_increments(times, training, training_seed)
        evaluation_increments = generate_brownian_increments(times, max(counts), evaluation_seed)
        reference = value_american_put_barone_adesi_whaley(
            market.spot, product.strike, product.maturity_years, volatility, 0.0, 0.0)
        paired_scheme_payments = {}
        for scheme_name, simulator in SCHEMES.items():
            started = perf_counter()
            result = value_american_put_with_european_control(product, market, model,
                simulator, spot_quadratic_basis,
                simulation_settings(training, max(counts), seed, branch),
                training_increments, evaluation_increments)
            elapsed = perf_counter() - started
            pathwise = result["pathwise"]
            paired_scheme_payments[scheme_name] = pathwise["american_evaluation_payoffs"]
            for count in counts:
                # Here the same fitted policy and beta serve nested evaluation
                # prefixes. More paths do not change the training calculation.
                american = pathwise["american_evaluation_payoffs"][:count]
                european = pathwise["european_evaluation_payoffs"][:count]
                controlled = pathwise["controlled_evaluation_observations"][:count]
                price, variance, standard_error = mean_statistics(american)
                controlled_price, controlled_variance, controlled_standard_error = mean_statistics(controlled)
                correlation = (float(np.corrcoef(american, european)[0, 1])
                    if np.var(american) > 0 and np.var(european) > 0 else None)
                rows.append({"scheme": scheme_name, "volatility": volatility,
                    "training_paths": training, "evaluation_paths": count,
                    "plain_price": price, "controlled_price": controlled_price,
                    "plain_estimator_variance": variance,
                    "controlled_estimator_variance": controlled_variance,
                    "plain_standard_error": standard_error,
                    "controlled_standard_error": controlled_standard_error,
                    "baw_reference": reference, "plain_error": reference - price,
                    "controlled_error": reference - controlled_price,
                    "payoff_correlation": correlation,
                    "control_coefficient": result["control_coefficient"],
                    "runtime_seconds": elapsed,
                    "runtime_scope": "One training fit and the full maximum evaluation sample; prefixes share this run.",
                    "diagnostics_scope": "Full training and maximum evaluation samples, not the selected prefix.",
                    "diagnostic_evaluation_paths": max(counts),
                    "seed_spawn_key": branch,
                    "diagnostics": {"training": result["training_simulation_diagnostics"],
                                    "evaluation": result["evaluation_simulation_diagnostics"]}})
            print(f"GBM {scheme_name}, volatility {volatility:.0%}: completed in {elapsed:.2f}s", flush=True)
        difference = paired_scheme_payments["Milstein"] - paired_scheme_payments["Euler"]
        for row in rows[-2 * len(counts):]:
            mean, difference_variance, standard_error = mean_statistics(difference[:row["evaluation_paths"]])
            row["milstein_minus_euler_plain_price"] = mean
            row["paired_scheme_difference_standard_error"] = standard_error
    return rows


# PART 3 — THE SAME JOINT SPOT–VARIANCE SCENARIOS FOR BOTH SCHEMES

def run_sv_pricing(training, counts, seed, step_days=1.0, area_accuracy_multiplier=1,
                   branch=(3, 1), schemes=None, supplied_increments=None):
    case = stochastic_volatility_baseline()
    settings = simulation_settings(training, max(counts), seed, branch, step_days)
    times, exercise_indices = make_simulation_and_exercise_grids(case.product.maturity_years, settings)
    training_seed, evaluation_seed = np.random.SeedSequence(seed, spawn_key=branch).spawn(2)
    if supplied_increments is None:
        training_increments = generate_two_driver_brownian_increments(times, training, training_seed)
        evaluation_increments = generate_two_driver_brownian_increments(times, max(counts), evaluation_seed)
    else:
        training_increments, evaluation_increments = supplied_increments
    # Here 8% BAW is deliberately a constant-volatility proxy, not an SV truth.
    baw = value_american_put_barone_adesi_whaley(case.market_data.spot, case.product.strike,
        case.product.maturity_years, np.sqrt(case.model.initial_variance), 0.0, 0.0)
    rows, examples, payments = [], {}, {}
    for scheme_name in (schemes or tuple(SCHEMES)):
        simulator = SCHEMES[scheme_name]
        if scheme_name == "Milstein":
            simulator = partial(simulator, area_accuracy_multiplier=area_accuracy_multiplier)
        started = perf_counter()
        result = value_american_put_with_supplied_brownian_increments(
            case.product, case.market_data, case.model, simulator, case.basis_function,
            simulation_settings(training, max(counts), seed, branch, step_days),
            training_increments, evaluation_increments)
        elapsed = perf_counter() - started
        payments[scheme_name] = result["discounted_payoffs"]
        for count in counts:
            price, variance, standard_error = mean_statistics(payments[scheme_name][:count])
            rows.append({"scheme": scheme_name, "training_paths": training,
                "evaluation_paths": count, "plain_price": price,
                "plain_estimator_variance": variance, "plain_standard_error": standard_error,
                "baw_reference": baw, "plain_error": baw - price,
                "time_step_days": step_days, "area_accuracy_multiplier": area_accuracy_multiplier,
                "early_exercise_fraction": float(np.mean(result["exercise_date_indices"][:count]
                    < len(result["exercise_times"]) - 1)),
                "runtime_seconds": elapsed,
                "runtime_scope": "Full maximum evaluation sample plus training; shared by nested prefixes.",
                "diagnostics_scope": "Full training and maximum evaluation samples, not the selected prefix.",
                "diagnostic_evaluation_paths": max(counts),
                "seed_spawn_key": branch,
                "diagnostics": {"training": result["training_simulation_diagnostics"],
                                "evaluation": result["evaluation_simulation_diagnostics"]}})
        examples["times"] = result["exercise_times"]
        examples["spot_" + scheme_name.lower()] = result["example_state_path"]["spot"]
        examples["variance_" + scheme_name.lower()] = result["example_state_path"]["variance"]
        print(f"SV {scheme_name}, step {step_days:g} day, area multiplier {area_accuracy_multiplier}: {elapsed:.2f}s", flush=True)
    if len(payments) == 2:
        difference = payments["Milstein"] - payments["Euler"]
        for row in rows:
            mean, variance, se = mean_statistics(difference[:row["evaluation_paths"]])
            row.update(milstein_minus_euler_plain_price=mean,
                       paired_scheme_difference_standard_error=se)
    return rows, examples


# PART 4 — STRONG GBM ERROR, SEPARATE FROM REGRESSION AND EXERCISE

def run_gbm_convergence(seed, path_count=10000):
    maturity = 1 / 12
    interval_counts = (8, 16, 32, 64, 128, 256)
    finest_times = np.linspace(0.0, maturity, interval_counts[-1] + 1)
    increments = generate_brownian_increments(finest_times, path_count,
        np.random.SeedSequence(seed, spawn_key=(3, 2)))
    rows, orders = [], []
    for volatility in VOLATILITIES:
        model = GeometricBrownianMotionModel(0.0, volatility)
        market = MarketData(40.0, 0.0, volatility, 0.0)
        exact_terminal = market.spot * np.exp(-0.5 * volatility**2 * maturity
                                             + volatility * increments.sum(axis=1))
        for intervals in interval_counts:
            times = np.linspace(0.0, maturity, intervals + 1)
            # Here fine increments are summed, so all resolutions share W_T and
            # the underlying Brownian scenario. Only the numerical step changes.
            coarse = increments.reshape(path_count, intervals,
                                        interval_counts[-1] // intervals).sum(axis=2)
            for scheme_name, simulator in SCHEMES.items():
                started = perf_counter()
                paths, dates, diagnostics = simulator(market, model, times,
                    np.array([0, intervals]), path_count, seed,
                    brownian_increments=coarse)
                terminal_error = paths["spot"][:, -1] - exact_terminal
                rows.append({"scheme": scheme_name, "volatility": volatility,
                    "time_step_days": maturity * 365 / intervals,
                    "terminal_spot_rmse": float(np.sqrt(np.mean(terminal_error**2))),
                    "runtime_seconds": perf_counter() - started})
        for scheme_name in SCHEMES:
            selected = [row for row in rows if row["scheme"] == scheme_name
                        and row["volatility"] == volatility]
            slope = np.polyfit(np.log([row["time_step_days"] for row in selected]),
                               np.log([row["terminal_spot_rmse"] for row in selected]), 1)[0]
            orders.append({"scheme": scheme_name, "volatility": volatility,
                           "observed_order": float(slope)})
    return {"rows": rows, "orders": orders, "paths": path_count,
            "settings": {"maturity_years": maturity, "interval_counts": interval_counts,
                "seed_spawn_key": (3, 2), "coupling": "Sum increments from the finest grid.",
                "order_fit": "OLS slope of log terminal RMSE against log timestep over all six grids."},
            "meaning": "Terminal spot RMSE against exact GBM on coupled Brownian paths; not American pricing error."}


# A REFINED SAME-MODEL REFERENCE — MATURITY PAYMENTS, NOT ANOTHER LSMC FIT

def run_sv_reference(seed, path_count=100000, subdivisions=(4, 8), batch_size=5000):
    case = stochastic_volatility_baseline()
    finest_intervals = 365 * max(subdivisions)
    fine_times = np.linspace(0.0, 1.0, finest_intervals + 1)
    payoffs = {level: [] for level in subdivisions}
    totals = {level: {"variance_projection_count": 0, "negative_spot_state_count": 0,
                      "path_step_count": 0, "zero_variance_start_count": 0} for level in subdivisions}
    started = perf_counter()
    batches = int(np.ceil(path_count / batch_size))
    seeds = np.random.SeedSequence(seed, spawn_key=(3, 3)).spawn(batches)
    for batch_index, batch_seed in enumerate(seeds):
        count = min(batch_size, path_count - batch_index * batch_size)
        increments = generate_two_driver_brownian_increments(fine_times, count, batch_seed)
        for level in subdivisions:
            intervals = 365 * level
            coarse = increments.reshape(count, intervals, max(subdivisions) // level, 2).sum(axis=2)
            paths, dates, diagnostics = simulate_paths_euler_maruyama(
                case.market_data, case.model, np.linspace(0.0, 1.0, intervals + 1),
                np.array([0, intervals]), count, batch_seed, brownian_increments=coarse)
            payoffs[level].append(case.product.payoff(paths["spot"][:, -1]))
            for key in totals[level]:
                totals[level][key] += int(diagnostics.get(key, count * intervals if key == "path_step_count" else 0))
        if (batch_index + 1) % 5 == 0:
            print(f"SV refined reference: {min((batch_index + 1) * batch_size, path_count):,} paths", flush=True)
    rows, previous = [], None
    for level in subdivisions:
        observations = np.concatenate(payoffs[level])
        price, variance, se = mean_statistics(observations)
        change, change_se = None, None
        if previous is not None:
            change, difference_variance, change_se = mean_statistics(observations - previous)
        rows.append({"time_step_days": 1 / level, "price": price, "standard_error": se,
            "change_from_coarser": change, "paired_change_standard_error": change_se,
            "diagnostics": totals[level]})
        previous = observations
    return {"rows": rows, "reference_price": rows[-1]["price"],
        "reference_standard_error": rows[-1]["standard_error"], "paths": path_count,
        "runtime_seconds": perf_counter() - started,
        "batch_size": batch_size, "seed_spawn_key": (3, 3),
        "meaning": "Projected Euler terminal-payoff numerical reference; sampling uncertainty and residual refinement/projection error remain."}


# FOCUSED SV CHECK — REFINEMENT AND DOUBLED CROSS-INTEGRAL ACCURACY

def run_sv_stability(seed, training=10000, evaluation=10000):
    records = []
    # Here a daily increment is the sum of its two half-day increments. This
    # preserves the Brownian scenario when we refine the integration grid.
    case = stochastic_volatility_baseline()
    half_day_times, exercise_indices = make_simulation_and_exercise_grids(
        case.product.maturity_years, simulation_settings(training, evaluation, seed, (3, 4), 0.5))
    training_seed, evaluation_seed = np.random.SeedSequence(seed, spawn_key=(3, 4)).spawn(2)
    fine_training = generate_two_driver_brownian_increments(half_day_times, training, training_seed)
    fine_evaluation = generate_two_driver_brownian_increments(half_day_times, evaluation, evaluation_seed)
    for step, multiplier, schemes in [(1.0, 1, None), (1.0, 2, ("Milstein",)),
                                      (0.5, 1, None)]:
        if step == 1.0:
            increments = (fine_training.reshape(training, 365, 2, 2).sum(axis=2),
                          fine_evaluation.reshape(evaluation, 365, 2, 2).sum(axis=2))
        else:
            increments = (fine_training, fine_evaluation)
        rows, example = run_sv_pricing(training, (evaluation,), seed, step, multiplier,
            branch=(3, 4), schemes=schemes, supplied_increments=increments)
        for row in rows:
            records.append({"scheme": row["scheme"], "time_step_days": step,
                "area_accuracy_multiplier": multiplier, "price": row["plain_price"],
                "standard_error": row["plain_standard_error"],
                "training_paths": training, "evaluation_paths": evaluation,
                "diagnostics": row["diagnostics"], "runtime_seconds": row["runtime_seconds"]})
    return {"rows": records,
        "settings": {"training_paths": training, "evaluation_paths": evaluation,
            "maturity_years": case.product.maturity_years, "exercise_dates_per_year": 365,
            "seed_spawn_key": (3, 4), "valuation": "Fresh LSMC fit for every scheme/setting.",
            "coupling": "Brownian endpoints coupled across schemes, timesteps and area accuracy. Area tails are not a nested strong coupling."},
        "meaning": "A representative complete-fit stability check; doubled sampler accuracy is not a proof of strong pathwise order."}


# LIMITED CARRYOVER — THE EXISTING GBM GAMMA PROFILE WITH EACH SCHEME

def value_gamma_case(scheme_name, volatility, moneyness, training, evaluation, seed, case_index):
    paper = read_published_at_the_money_case()
    product = AmericanPut(paper["strike"], 1 / 12)
    market = MarketData(product.strike * moneyness, 0.0, volatility, 0.0)
    model = GeometricBrownianMotionModel(0.0, volatility)
    result = calculate_american_put_gamma(product, market, model, SCHEMES[scheme_name],
        spot_quadratic_basis, simulation_settings(training, evaluation, seed, (3, 5, case_index)),
        0.005 * product.strike)
    return {"scheme": scheme_name, **result}


def run_gamma_comparison(training, evaluation, seed, workers=1):
    jobs = []
    for case_index, (volatility, moneyness) in enumerate(
            (v, m) for v in VOLATILITIES for m in np.linspace(0.9, 1.1, 21)):
        jobs.extend((scheme, volatility, float(moneyness), training, evaluation, seed, case_index)
                    for scheme in SCHEMES)
    if workers == 1:
        results = [value_gamma_case(*job) for job in jobs]
    else:
        with get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(value_gamma_case, jobs)
    return {"settings": {"moneyness": np.linspace(0.9, 1.1, 21), "volatilities": VOLATILITIES,
        "training_paths": training, "evaluation_paths": evaluation, "spot_bump": 0.2,
        "bump_fraction_of_strike": 0.005, "maturity_years": 1 / 12, "strike": 40.0,
        "interpretation": "Limited GBM carryover of Exercise 2; policies refit at each bump; conditional variance only, no claim about full-training gamma variance."},
        "rows": results}


# ONE EXERCISE DOCUMENT — CONFIGURATION, EVIDENCE AND PROVENANCE

def run_exercise3(study="all", training_paths=50000, evaluation_counts=EVALUATION_COUNTS,
                  seed=20261004, workers=1, existing_document=None):
    counts = sorted(set(int(value) for value in evaluation_counts))
    if not counts or min(counts) < 2 or training_paths < 2 or workers < 1:
        raise ValueError("Use at least two training/evaluation paths and one worker")
    if study not in ("all", "pricing", "diagnostics", "gamma"):
        raise ValueError("Unknown Exercise 3 study")
    if existing_document is not None:
        if existing_document.get("code_sha256") != exercise3_source_hashes():
            raise ValueError("Cannot merge results produced by different numerical source")
        old = existing_document["config"]
        if any(old[key] != value for key, value in
               [("training_paths", training_paths), ("evaluation_counts", counts), ("seed", seed)]):
            raise ValueError("Cannot merge studies with different experiment settings")
    case = stochastic_volatility_baseline()
    document = dict(existing_document or {})
    document.update(exercise=3, created_at_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=exercise3_source_hashes(),
        config={"training_paths": training_paths, "evaluation_counts": counts,
            "volatilities": VOLATILITIES, "seed": seed, "workers": workers,
            "main_time_step_days": 1.0, "exercise_dates_per_year": 365,
            "gbm_spot": 40.0, "gbm_strike": 40.0, "gbm_maturity": 1 / 12,
            "sv_spot": 1.0, "sv_strike": 1.0, "sv_maturity": 1.0,
            "sv_parameters": asdict(case.model)},
        environment={"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        requirements={"1": "Explain bias and variance of the implemented American price estimator.",
            "2": "GBM Milstein versus Euler and BAW, retaining the European control.",
            "3": "Printed stochastic-volatility Milstein versus Euler, BAW proxy and same-model refinement.",
            "4": "Errors across 10,000–100,000 evaluation paths and explain convergence."},
        assumptions={"inputs": "GBM uses the Exercise 2 inputs; the stochastic-volatility model and contract use the Exercise 1 baseline. Each study simulates new paths.",
            "exercise_grid": "Daily for both models; old Exercise 1 SV results used 50/year and are not matched comparators.",
            "path_counts": "Training fixed; evaluation uses nested prefixes of a maximum sample. Prefix estimates are dependent.",
            "error_sign": "Reference minus estimate. SV BAW gap is a model comparison, not pure numerical error.",
            "uncertainty": "Sample variance/N conditions on fitted rules and beta. Excludes training variability and numerical bias.",
            "control": "Exact GBM European terminal payoff shares American Brownian increments. Training beta fixed during evaluation. No SV Black–Scholes control.",
            "sampler": "Wiktorsson two-driver Fourier series with conditional Gaussian tail. n=ceil(sqrt(5/(6*pi²*h))); chosen integral MSE budget h³, h in years; separate area draws.",
            "boundary": "Finite interior coefficient products extended to zero, then variance proposal projected to zero. No strong-order theorem asserted at the boundary.",
            "reference": "Under a well-defined nonnegative continuous spot solution with zero drift, put waiting is optimal. Refined projected-Euler maturity payments give a numerical, not exact, reference.",
            "gamma": "GBM gamma repeats the Exercise 2 profile at one evaluation path count. The stochastic-volatility study compares option prices."})
    if study in ("all", "pricing"):
        document["gbm_prices"] = run_gbm_pricing(training_paths, counts, seed)
        document["sv_prices"], document["joint_paths"] = run_sv_pricing(training_paths, counts, seed)
    if study in ("all", "diagnostics"):
        document["gbm_convergence"] = run_gbm_convergence(seed)
        document["sv_reference"] = run_sv_reference(seed, max(counts))
        document["sv_stability"] = run_sv_stability(seed)
    if study in ("all", "gamma"):
        document["gamma"] = run_gamma_comparison(training_paths, max(counts), seed, workers)
    if "sv_reference" in document:
        reference = document["sv_reference"]
        for row in document.get("sv_prices", []):
            row.update(same_model_reference=reference["reference_price"],
                same_model_error=reference["reference_price"] - row["plain_price"],
                same_model_reference_standard_error=reference["reference_standard_error"],
                same_model_gap_standard_error=float(np.hypot(row["plain_standard_error"], reference["reference_standard_error"])))
    return document


def save_exercise3_results(document, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value), allow_nan=False) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("all", "pricing", "diagnostics", "gamma"), default="all")
    parser.add_argument("--training-paths", type=int, default=50000)
    parser.add_argument("--evaluation-counts", type=int, nargs="+", default=EVALUATION_COUNTS)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="JSON destination; reuse the same new path for staged studies")
    arguments = parser.parse_args()
    # Here stages merge only when explicitly directed to the same output file.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-exercise3"
    arguments.output = arguments.output or (PROJECT_DIRECTORY / "runs" / run_id
                                             / "exercise3_results.json")
    existing = None
    if arguments.output.exists() and arguments.study != "all":
        existing = json.loads(arguments.output.read_text())
    result = run_exercise3(arguments.study, arguments.training_paths, arguments.evaluation_counts,
                          arguments.seed, arguments.workers, existing)
    save_exercise3_results(result, arguments.output)
    print(f"Saved Exercise 3 {arguments.study} to {arguments.output}", flush=True)
