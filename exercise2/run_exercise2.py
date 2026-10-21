"""EXERCISE 2: CONTROL VARIATE, BAW COMPARISON AND NUMERICAL GAMMA.

Here we reuse Exercise 1's economic objects and American valuation. We generate
new payoff pairs at zero rates; the published Table 1 prices are not reused.
Part 1 measures correlation and estimator variance. Part 2 reuses those prices.
Part 3 uses the coupled bumped valuation in the separate gamma.py file.
"""
import argparse
import csv
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

from exercise2.barone_adesi_whaley import value_american_put_barone_adesi_whaley
from basis import spot_quadratic_basis
from euler_maruyama import simulate_paths_euler_maruyama
from exercise2.gamma import calculate_american_put_gamma
from market_data import MarketData
from models import GeometricBrownianMotionModel
from products import AmericanPut
from valuation import SimulationSettings, value_american_put_with_european_control


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
NUMERICAL_FILES = (
    "market_data.py", "products.py", "models.py", "basis.py",
    "euler_maruyama.py", "longstaff_schwartz_lsmc.py", "valuation.py",
    "black_scholes_put.py", "exercise2/barone_adesi_whaley.py", "control_variates.py",
    "exercise2/gamma.py", "exercise2/run_exercise2.py", "test_data/exercise1/longstaff_table1.csv", "requirements.txt",
)


# RECORD THE EXACT CODE AND INPUT FILES USED BY THIS STUDY
def exercise2_source_hashes():
    return {name: hashlib.sha256((PROJECT_DIRECTORY / name).read_bytes()).hexdigest()
            for name in NUMERICAL_FILES}


# REUSE EXERCISE 1'S PUBLISHED SPOT AND STRIKE
def read_published_at_the_money_case():
    with (PROJECT_DIRECTORY / "test_data/exercise1/longstaff_table1.csv").open() as source:
        for paper_row in csv.DictReader(source):
            if float(paper_row["spot"]) == float(paper_row["strike"]):
                return {name: float(value) for name, value in paper_row.items()}
    raise ValueError("Longstaff-Schwartz Table 1 must contain an at-the-money case")


# PART 1: VALUE ONE AMERICAN PUT WITH ITS PAIRED EUROPEAN CONTROL
def value_control_variate_case(paper_case, volatility, simulation_settings):
    start_time = perf_counter()
    market_data = MarketData(
        spot=paper_case["spot"], interest_rate=0.0, volatility=volatility,
        foreign_interest_rate=0.0)
    product = AmericanPut(strike=paper_case["strike"], maturity_years=1.0 / 12.0)
    model = GeometricBrownianMotionModel(spot_drift_rate=0.0, volatility=volatility)

    # Here the valuation returns every paired payoff. The saved study retains
    # summary statistics and five example rows, keeping the output readable.
    result = value_american_put_with_european_control(
        product=product, market_data=market_data, model=model,
        simulation_method=simulate_paths_euler_maruyama,
        basis_function=spot_quadratic_basis, simulation_settings=simulation_settings)
    pathwise = result["pathwise"]
    example_count = min(5, simulation_settings.evaluation_paths)
    examples = [{
        "path_number": path_index + 1,
        "american_payoff": float(pathwise["american_evaluation_payoffs"][path_index]),
        "european_payoff": float(pathwise["european_evaluation_payoffs"][path_index]),
        "controlled_observation": float(
            pathwise["controlled_evaluation_observations"][path_index]),
    } for path_index in range(example_count)]

    return {
        "study": "part_1_european_control_variate",
        "market_data": asdict(market_data), "product": asdict(product),
        "model": asdict(model), "model_name": type(model).__name__,
        "american_scheme": "Euler-Maruyama", "european_control_scheme": "Exact GBM",
        "training_paths": simulation_settings.training_paths,
        "evaluation_paths": simulation_settings.evaluation_paths,
        "requested_time_step_days": simulation_settings.time_step_days,
        "exercise_dates_per_year": simulation_settings.exercise_dates_per_year,
        "seed_entropy": simulation_settings.seed.entropy,
        "case_seed_spawn_key": simulation_settings.seed.spawn_key,
        "runtime_seconds": perf_counter() - start_time,
        "valuation": {name: value for name, value in result.items() if name != "pathwise"},
        "example_evaluation_payoffs": examples,
    }


# PART 1: CORRELATION AND ESTIMATOR VARIANCE ACROSS THE REQUESTED VOLATILITIES
def run_control_variate_study(training_paths=50000, evaluation_paths=100000,
                              seed=20261004, workers=1):
    if workers < 1:
        raise ValueError("Use at least one worker")
    paper_case = read_published_at_the_money_case()
    # Here Part 2 explicitly requests these four volatilities. Using them in
    # Part 1 as well lets both assignment answers share the same pricing runs.
    volatilities = (0.05, 0.10, 0.20, 0.30)
    case_seeds = np.random.SeedSequence(seed).spawn(len(volatilities))
    jobs = [(paper_case, volatility,
             SimulationSettings(training_paths, evaluation_paths, case_seed,
                                time_step_days=1.0, exercise_dates_per_year=365))
            for volatility, case_seed in zip(volatilities, case_seeds)]
    # Here each case receives its seed before parallel execution. Worker count
    # changes scheduling; it does not change the case's random numbers.
    if workers == 1:
        return [value_control_variate_case(*job) for job in jobs]
    with get_context("spawn").Pool(processes=workers) as pool:
        return pool.starmap(value_control_variate_case, jobs)


# PART 2: COMPARE THE SAME ONE-MONTH ESTIMATES WITH BAW
def compare_prices_with_baw(pricing_results):
    comparisons = []
    for record in pricing_results:
        market_data, product = record["market_data"], record["product"]
        valuation = record["valuation"]
        # Here the general BAW implementation explicitly selects its zero-rate,
        # zero-yield branch. This reference then equals the European BS value.
        reference = value_american_put_barone_adesi_whaley(
            spot=market_data["spot"], strike=product["strike"],
            maturity_years=product["maturity_years"], volatility=market_data["volatility"],
            interest_rate=market_data["interest_rate"], dividend_yield=0.0)
        comparisons.append({
            "volatility": market_data["volatility"], "baw_reference": reference,
            "baw_branch": "Exact zero-rate, zero-yield specialisation",
            "plain_price": valuation["plain_price"],
            "controlled_price": valuation["controlled_price"],
            "plain_error": valuation["plain_price"] - reference,
            "controlled_error": valuation["controlled_price"] - reference,
            "plain_standard_error": valuation["plain_standard_error"],
            "controlled_standard_error": valuation["controlled_standard_error"],
        })
    return comparisons


# PART 3: VALUE ONE SPOT CASE USING THREE COUPLED BUMPED VALUATIONS
def value_gamma_case(paper_case, volatility, moneyness, simulation_settings, spot_bump):
    market_data = MarketData(spot=paper_case["strike"] * moneyness,
        interest_rate=0.0, volatility=volatility, foreign_interest_rate=0.0)
    product = AmericanPut(strike=paper_case["strike"], maturity_years=1.0 / 12.0)
    model = GeometricBrownianMotionModel(spot_drift_rate=0.0, volatility=volatility)
    return calculate_american_put_gamma(
        product=product, market_data=market_data, model=model,
        simulation_method=simulate_paths_euler_maruyama,
        basis_function=spot_quadratic_basis, simulation_settings=simulation_settings,
        spot_bump=spot_bump)


# PART 3: GAMMA PROFILES, A SMALL BUMP CHECK AND INDEPENDENT FULL REPETITIONS
def run_gamma_study(training_paths=50000, evaluation_paths=100000,
                    seed=20261004, moneyness_values=None, bump_fraction=0.005, workers=1):
    if workers < 1 or not np.isfinite(bump_fraction) or bump_fraction <= 0:
        raise ValueError("Use at least one worker and a finite positive bump fraction")
    paper_case = read_published_at_the_money_case()
    # Here 21 equally spaced spots cover S/K=0.90 to 1.10 and include ATM,
    # where the low-volatility gamma peak is narrow. This grid and h=0.5%K are our numerical choices.
    if moneyness_values is None:
        moneyness_values = np.linspace(0.90, 1.10, 21).tolist()
    else:
        moneyness_values = list(moneyness_values)
    if not moneyness_values or any(value <= bump_fraction or not np.isfinite(value)
                                  for value in moneyness_values):
        raise ValueError("Moneyness values must be finite and exceed the bump fraction")
    volatilities = (0.05, 0.10, 0.20, 0.30)
    spot_bump = bump_fraction * paper_case["strike"]
    # Here the (2,) branch keeps gamma randomness separate from pricing, while
    # all three bumps inside one gamma valuation use common random numbers.
    profile_seed, bump_seed, repetition_seed = np.random.SeedSequence(
        seed, spawn_key=(2,)).spawn(3)
    profile_cases = [(volatility, moneyness) for volatility in volatilities
                     for moneyness in moneyness_values]
    profile_seeds = profile_seed.spawn(len(profile_cases))
    jobs = [(paper_case, volatility, moneyness,
             SimulationSettings(training_paths, evaluation_paths, case_seed,
                                time_step_days=1.0, exercise_dates_per_year=365), spot_bump)
            for (volatility, moneyness), case_seed in zip(profile_cases, profile_seeds)]
    profile_count = len(jobs)

    # Here h/2,h,2h share identical random increments at each representative
    # spot. Fresh SeedSequence objects preserve that coupling across calls.
    check_spots = (0.98, 1.0, 1.02)
    for moneyness, case_seed in zip(check_spots, bump_seed.spawn(len(check_spots))):
        for bump_multiplier in (0.5, 1.0, 2.0):
            shared_seed = np.random.SeedSequence(case_seed.entropy,
                                                spawn_key=case_seed.spawn_key)
            settings = SimulationSettings(training_paths, evaluation_paths, shared_seed,
                                          time_step_days=1.0, exercise_dates_per_year=365)
            jobs.append((paper_case, 0.20, moneyness, settings, spot_bump * bump_multiplier))
    bump_count = len(jobs) - profile_count

    # Here ten independently seeded complete ATM runs also change training.
    # Their empirical variance is a diagnostic, separate from conditional SEs.
    repetition_count = 10
    for case_seed in repetition_seed.spawn(repetition_count):
        settings = SimulationSettings(training_paths, evaluation_paths, case_seed,
                                      time_step_days=1.0, exercise_dates_per_year=365)
        jobs.append((paper_case, 0.20, 1.0, settings, spot_bump))
    if workers == 1:
        results = [value_gamma_case(*job) for job in jobs]
    else:
        with get_context("spawn").Pool(processes=workers) as pool:
            results = pool.starmap(value_gamma_case, jobs)
    rows = results[:profile_count]
    bump_checks = results[profile_count:profile_count + bump_count]
    repetitions = results[profile_count + bump_count:]
    return {
        "settings": {
            "strike": paper_case["strike"], "maturity_years": 1.0 / 12.0,
            "rates_and_yields": 0.0, "volatilities": volatilities,
            "moneyness_values": moneyness_values, "bump_fraction_of_strike": bump_fraction,
            "spot_bump": spot_bump, "training_paths": training_paths,
            "evaluation_paths": evaluation_paths, "time_step_days": 1.0,
            "exercise_dates_per_year": 365, "seed": seed, "workers": workers,
            "basis": "[1, S/K, (S/K)^2]", "spot_source": "Fixed strike from test_data/exercise1/longstaff_table1.csv; S/K sensitivity grid is our choice.",
            "coefficient": "Gamma-specific coefficient estimated from existing training gamma observations; no pilot batch.",
            "uncertainty": "Main variance curves condition on three fitted exercise rules and beta. Ten complete repetitions additionally vary training.",
        },
        "rows": rows, "bump_checks": bump_checks,
        "repetitions": {
            "case": "S=K=40, volatility=20%, maturity=1/12 year, main bump",
            "count": repetition_count, "rows": repetitions,
            "plain_empirical_variance": float(np.var([row["plain_gamma"] for row in repetitions], ddof=1)),
            "controlled_empirical_variance": float(np.var([row["controlled_gamma"] for row in repetitions], ddof=1)),
            "mean_plain_conditional_variance": float(np.mean([row["plain_estimator_variance"] for row in repetitions])),
            "mean_controlled_conditional_variance": float(np.mean([row["controlled_estimator_variance"] for row in repetitions])),
            "interpretation": "Empirical variances describe a single complete run at these path counts, not the variance of an average of ten runs; ten repetitions give a noisy diagnostic.",
        },
    }


# SAVE RESULTS, EXPLICIT CHOICES AND PROVENANCE TOGETHER
def save_exercise2_results(pricing_results, output_path, seed, workers):
    document = {
        "exercise": 2, "completed_parts": [1, 2], "gamma_part_3_included": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": exercise2_source_hashes(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "root_seed": seed, "workers": workers,
        "published_input_source": {
            "file": "test_data/exercise1/longstaff_table1.csv", "paper": "Longstaff-Schwartz (2001), Table 1",
            "selected_original_row": read_published_at_the_money_case(),
            "reused_fields": ["spot", "strike"],
            "published_prices_used_as_exercise2_benchmarks": False,
        },
        "assumptions": {
            "rates_and_yields": "All zero, as required by Exercise 2; GBM drift is zero.",
            "volatilities": "5%, 10%, 20%, 30% are specified in Part 2 and reused in Part 1.",
            "maturity": "One month means 1/12 year; Part 1 shares Part 2's one-month cases.",
            "dates": "Days/365, daily simulation and exercise, shorter final interval.",
            "basis": "[1, S/K, (S/K)^2], retained from Exercise 1.",
            "counts": "50,000 training and 100,000 evaluation default to Exercise 1's full runs; CLI overrides are recorded per case.",
            "control": "Exact-GBM European payoff, paired using the American Euler path's Brownian increments. Euler European expectation is not generally Black-Scholes.",
            "coefficient": "Estimate beta separately for each volatility from existing training payoff pairs; no extra pilot batch. Apply it unchanged to independent evaluation paths.",
            "coefficient_limitation": "Training payments use fitted exercise rules, so training beta need not minimise independent evaluation variance. Evaluate its effectiveness on evaluation paths.",
            "uncertainty": "Estimator variance is sample observation variance divided by evaluation path count, conditional on fitted exercise rules and beta. It excludes training variability, Euler bias and policy error.",
            "baw": "A general BAW implementation is available. These required runs use its exact zero-rate, zero-yield branch; they do not validate general BAW accuracy.",
        },
        "units": {"prices_and_payoffs": "Contract currency units; no market currency assigned",
                  "payoff_variance_and_estimator_variance": "Squared contract currency units",
                  "volatility": "Annualised decimal", "time": "Years, days/365"},
        "pricing_results": pricing_results,
        "baw_comparisons": compare_prices_with_baw(pricing_results),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2,
                                     default=lambda value: value.tolist(), allow_nan=False) + "\n")
    return document


# PART 3: SAVE GAMMA RESULTS SEPARATELY FROM THE SHARED PART 1/2 PRICES
def save_exercise2_gamma_results(gamma_results, output_path, seed):
    document = {
        "exercise": 2, "completed_parts": [3],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": exercise2_source_hashes(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "root_seed": seed, "gamma_results": gamma_results,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2,
                                     default=lambda value: value.tolist(), allow_nan=False) + "\n")
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("pricing", "gamma", "all"), default="all")
    parser.add_argument("--training-paths", type=int, default=50000)
    parser.add_argument("--evaluation-paths", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="Price JSON destination; default: a new runs/<run-id>/ folder")
    parser.add_argument("--gamma-output", type=Path,
                        help="Gamma JSON destination; default: beside the price JSON")
    parser.add_argument("--bump-fraction", type=float, default=0.005,
                        help="Main gamma spot bump as a fraction of the fixed strike")
    arguments = parser.parse_args()
    # Here both outputs stay together, outside the published snapshot by default.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-exercise2"
    arguments.output = arguments.output or (PROJECT_DIRECTORY / "runs" / run_id
                                             / "exercise2_control_variates.json")
    arguments.gamma_output = arguments.gamma_output or arguments.output.with_name("exercise2_gamma.json")
    if arguments.study == "all" and arguments.output.resolve() == arguments.gamma_output.resolve():
        parser.error("Price and gamma outputs must use different files")
    if arguments.study in ("pricing", "all"):
        pricing_results = run_control_variate_study(
            arguments.training_paths, arguments.evaluation_paths, arguments.seed, arguments.workers)
        save_exercise2_results(pricing_results, arguments.output, arguments.seed, arguments.workers)
        print(f"Saved Parts 1 and 2: {len(pricing_results)} one-month cases to {arguments.output}")
    if arguments.study in ("gamma", "all"):
        gamma_results = run_gamma_study(
            training_paths=arguments.training_paths, evaluation_paths=arguments.evaluation_paths,
            seed=arguments.seed, bump_fraction=arguments.bump_fraction, workers=arguments.workers)
        save_exercise2_gamma_results(gamma_results, arguments.gamma_output, arguments.seed)
        print(f"Saved Part 3: gamma profiles and variance to {arguments.gamma_output}")
