"""Here we run one explicitly named part of Exercise 1 at a time.

Market data -> product -> model -> numerical methods -> valuation.
Table 1 supplies published scenarios, not observed market quotes. Other grids
below are our sensitivity choices. Each study records its inputs, price
estimates and sampling uncertainty.
"""
import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import partial
from itertools import product as combinations
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Callable
import numpy as np
from market_data import MarketData
from products import AmericanPut
from models import GeometricBrownianMotionModel, StochasticVolatilityModel
from basis import spot_quadratic_basis, spot_variance_quadratic_basis
from euler_maruyama import simulate_paths_euler_maruyama
from longstaff_schwartz_lsmc import value_american_option_longstaff_schwartz
from valuation import SimulationSettings, value_american_put


@dataclass(frozen=True)
class Exercise1Case:
    study: str
    market_data: MarketData
    product: AmericanPut
    model: GeometricBrownianMotionModel | StochasticVolatilityModel
    basis_function: Callable
    time_step_days: float = 1.0
    paper_reference: dict | None = None


# EXERCISE 1 — COMPARE WITH LONGSTAFF–SCHWARTZ TABLE 1
def make_table1_cases():
    with (Path(__file__).resolve().parent.parent / "test_data/exercise1/longstaff_table1.csv").open() as source:
        paper_rows = list(csv.DictReader(source))
    cases = []
    for paper_row in paper_rows:
        reference = {name: float(value) for name, value in paper_row.items()}
        market_data = MarketData(spot=reference["spot"], interest_rate=reference["rate"],
                                 volatility=reference["volatility"])
        product = AmericanPut(strike=reference["strike"], maturity_years=reference["maturity"])
        model = GeometricBrownianMotionModel(
            spot_drift_rate=market_data.interest_rate, volatility=market_data.volatility)
        cases.append(Exercise1Case("table1", market_data, product, model,
                                    spot_quadratic_basis, paper_reference=reference))
    return cases


# EXERCISE 1 — VARY STRIKE, VOLATILITY AND MATURITY
def make_gbm_pricing_cases():
    cases = []
    for strike, volatility, maturity in combinations((36., 40., 44.), (.2, .4), (1., 2.)):
        market_data = MarketData(spot=40., interest_rate=.06, volatility=volatility)
        product = AmericanPut(strike=strike, maturity_years=maturity)
        model = GeometricBrownianMotionModel(
            spot_drift_rate=market_data.interest_rate, volatility=market_data.volatility)
        cases.append(Exercise1Case("gbm_pricing", market_data, product, model, spot_quadratic_basis))
    return cases


# EXERCISE 1 — COMPARE ONE-DAY TO ONE-WEEK EULER STEPS
def make_gbm_timestep_cases():
    # Here every published case is repeated at each step; exercise dates stay fixed.
    return [replace(case, study="gbm_timestep", time_step_days=days)
            for case in make_table1_cases() for days in (1., 2., 4., 7.)]


# EXERCISE 1 — REPEAT THE EVALUATIONS UNDER PRINTED STOCHASTIC VOLATILITY
def make_stochastic_volatility_cases():
    # Here S0=1, v0=0.08**2 and positive correlation follow the assignment.
    # K, T, kappa=2, theta=v0, xi=.04 and rho=.5 are our working scenarios.
    # We choose r=mu=0 for this non-dividend baseline; this is not Table 1's rate.
    market_data = MarketData(spot=1., interest_rate=0.)
    model = StochasticVolatilityModel(initial_variance=.08**2, mean_reversion_speed=2.,
        long_term_variance=.08**2, volatility_of_variance=.04, brownian_correlation=.5)
    basis_function = partial(spot_variance_quadratic_basis, variance_scale=model.initial_variance)
    baseline = Exercise1Case("sv_sensitivities", market_data,
                            AmericanPut(strike=1., maturity_years=1.), model, basis_function)
    cases = []
    # Here theta controls the long-term variance; v0 stays at the specified value.
    for strike, maturity, variance_level in combinations((.9, 1., 1.1), (1., 2.), (.04**2, .08**2, .16**2)):
        pricing_case = replace(baseline, study="sv_pricing",
            product=AmericanPut(strike=strike, maturity_years=maturity),
            model=replace(model, long_term_variance=variance_level))
        cases.append(pricing_case)
        cases.extend(replace(pricing_case, study="sv_timestep", time_step_days=days)
                     for days in (1., 2., 4., 7.))
    # Here we change one model parameter at a time; all choices satisfy 2*kappa*theta > xi**2.
    parameter_values = {"long_term_variance": (.04**2, .08**2, .16**2),
        "mean_reversion_speed": (1., 2., 4.), "volatility_of_variance": (.02, .04, .06),
        "brownian_correlation": (.2, .5, .8)}
    for parameter_name, values in parameter_values.items():
        cases.extend(replace(baseline, study="sv_sensitivities/" + parameter_name,
                             model=replace(model, **{parameter_name: value})) for value in values)
    return cases


# VALUE ONE COMPLETE CASE — ALSO THE MULTIPROCESSING WORKER
def value_exercise1_case(case, simulation_settings):
    start_time = perf_counter()
    result = value_american_put(
        product=case.product, market_data=case.market_data, model=case.model,
        valuation_method=value_american_option_longstaff_schwartz,
        simulation_method=simulate_paths_euler_maruyama,
        basis_function=case.basis_function, simulation_settings=simulation_settings)
    # Here the runner keeps the small study record. LSMC itself also returns
    # pathwise payoffs and exercise dates for later controls, Greeks and plots.
    record = {"study": case.study, "market_data": asdict(case.market_data),
        "product": asdict(case.product), "model": asdict(case.model),
        "model_name": type(case.model).__name__, "scheme": "Euler-Maruyama",
        "training_paths": simulation_settings.training_paths,
        "evaluation_paths": simulation_settings.evaluation_paths,
        "requested_time_step_days": simulation_settings.time_step_days,
        "exercise_dates_per_year": simulation_settings.exercise_dates_per_year,
        "seed_entropy": simulation_settings.seed.entropy,
        "case_seed_spawn_key": simulation_settings.seed.spawn_key,
        "runtime_seconds": perf_counter() - start_time,
        "valuation": {name: value for name, value in result.items()
                      if name not in ("discounted_payoffs", "exercise_date_indices")}}
    if case.paper_reference is not None:
        reference = case.paper_reference
        record["table1_comparison"] = {**reference,
            "difference_from_paper_lsm": result["option_value"] - reference["american_lsm"],
            "difference_from_paper_fd": result["option_value"] - reference["american_fd"],
            "estimated_early_exercise_premium": result["option_value"] - reference["european"]}
    return record


# RUN THE REQUESTED STUDY WITH REPRODUCIBLE CASE SEEDS
def run_exercise1(study, training_paths=25000, evaluation_paths=50000, seed=20261004, workers=1):
    if workers < 1:
        raise ValueError("Use at least one worker")
    if study == "table1":
        cases = make_table1_cases()
    elif study == "gbm_pricing":
        cases = make_gbm_pricing_cases()
    elif study == "gbm_timestep":
        cases = make_gbm_timestep_cases()
    elif study in ("sv_pricing", "sv_timestep", "sv_sensitivities"):
        cases = [case for case in make_stochastic_volatility_cases() if case.study.startswith(study)]
    else:
        raise ValueError("Unknown Exercise 1 study")
    # Here we create the root afresh for each run and assign seeds before workers.
    # Parallelism changes scheduling, not the seed attached to a valuation case.
    case_seeds = np.random.SeedSequence(seed).spawn(len(cases))
    jobs = [(case, SimulationSettings(training_paths, evaluation_paths, case_seed,
                                      time_step_days=case.time_step_days))
            for case, case_seed in zip(cases, case_seeds)]
    if workers == 1:
        return [value_exercise1_case(case, settings) for case, settings in jobs]
    with get_context("spawn").Pool(processes=workers) as pool:
        return pool.starmap(value_exercise1_case, jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("table1", "gbm_pricing", "gbm_timestep",
                                         "sv_pricing", "sv_timestep", "sv_sensitivities"))
    parser.add_argument("--training-paths", type=int, default=25000)
    parser.add_argument("--evaluation-paths", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        help="JSON destination; default: a new runs/<run-id>/ folder")
    arguments = parser.parse_args()
    # Here a fresh run folder protects the published results used by the UI.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-exercise1"
    output_path = arguments.output or (Path(__file__).resolve().parent.parent
                                       / "runs" / run_id / f"{arguments.study}.json")
    study_results = run_exercise1(arguments.study, arguments.training_paths,
        arguments.evaluation_paths, arguments.seed, arguments.workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Here NumPy arrays become ordinary lists in the saved JSON file.
    output_path.write_text(json.dumps(study_results, indent=2,
                                      default=lambda array: array.tolist(), allow_nan=False) + "\n")
    print(f"Saved {len(study_results)} valuations to {output_path}")
