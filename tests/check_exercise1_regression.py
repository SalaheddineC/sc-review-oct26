"""EXERCISE 1 — MANUAL SENSITIVITY AND SPARSE-REGRESSION STUDY.

Here we use the Exercise 1 valuation. Shared spot/variance shocks
make differences easier to measure; each model still gets its own fitted rules.
The paired standard error conditions on those trained rules. This uses large
path budgets and is intentionally separate from the fast pytest suite.
"""
import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import sys

# Here this supporting check uses the same numerical modules as the study runner.
PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

import numpy as np
from euler_maruyama import simulate_paths_euler_maruyama
from longstaff_schwartz_lsmc import (
    value_american_option_longstaff_schwartz, apply_fitted_exercise_rules)
from exercise1.run_exercise1 import make_gbm_pricing_cases, make_stochastic_volatility_cases
from valuation import SimulationSettings, value_american_put


# SENSITIVITIES — USE THE SAME RANDOM DRAWS FOR EACH MODEL SCENARIO
def check_parameter_sensitivities():
    cases = [case for case in make_stochastic_volatility_cases()
             if case.study.startswith("sv_sensitivities")]
    baseline_case = cases[1]  # Here the second theta value defines the sensitivity baseline.
    results = []
    baseline_payoffs = None
    for case in [baseline_case] + cases:
        # Here recreating the case seed couples both training and evaluation
        # shocks. Training and evaluation still use different child streams.
        settings = SimulationSettings(50000, 100000,
                                       np.random.SeedSequence(20261004).spawn(1)[0])
        result = value_american_put(
            product=case.product, market_data=case.market_data, model=case.model,
            valuation_method=value_american_option_longstaff_schwartz,
            simulation_method=simulate_paths_euler_maruyama,
            basis_function=case.basis_function, simulation_settings=settings)
        if baseline_payoffs is None:
            baseline_payoffs = result["discounted_payoffs"]
            continue
        difference = result["discounted_payoffs"] - baseline_payoffs
        parameter_name = case.study.split("/")[-1]
        results.append({"parameter": parameter_name, "parameter_value": asdict(case.model)[parameter_name],
            "option_value": result["option_value"], "standard_error": result["standard_error"],
            "difference_from_baseline": float(np.mean(difference)),
            "paired_standard_error": float(np.std(difference, ddof=1) / np.sqrt(len(difference)))})
    return {"training_paths": 50000, "evaluation_paths": 100000, "seed": 20261004,
            "baseline_model": asdict(baseline_case.model), "comparisons": results}


# SPARSE REGRESSIONS — REPLAY THE SAVED RULES ON THEIR ORIGINAL EVALUATION PATHS
def check_sparse_regressions(study, cases):
    saved_results = json.loads((PROJECT_DIRECTORY / "results" / f"{study}.json").read_text())
    comparisons = []
    for case, saved in zip(cases, saved_results):
        valuation = saved["valuation"]
        sparse_dates = [row["date_index"] for row in valuation["regression_diagnostics"]
                        if row["rank"] < row["basis_columns"]]
        if not sparse_dates:
            continue
        # Here child 1 is the evaluation stream used by LSMC. This reproduces
        # the saved run, without changing the fitted coefficients or seeds.
        evaluation_seed = np.random.SeedSequence(saved["seed_entropy"],
            spawn_key=tuple(saved["case_seed_spawn_key"]) + (1,))
        simulation_times = np.asarray(valuation["evaluation_simulation_diagnostics"]["simulation_times"])
        exercise_times = np.asarray(valuation["exercise_times"])
        exercise_indices = np.searchsorted(simulation_times, exercise_times)
        paths, dates, diagnostics = simulate_paths_euler_maruyama(
            case.market_data, case.model, simulation_times, exercise_indices,
            saved["evaluation_paths"], evaluation_seed)
        original = apply_fitted_exercise_rules(case.product, case.market_data, paths, dates,
            case.basis_function, valuation["coefficients_by_date"], valuation["exercise_at_time_zero"])
        np.testing.assert_allclose(original["option_value"], valuation["option_value"], rtol=0, atol=1e-12)
        coefficients_without_sparse_fits = list(valuation["coefficients_by_date"])
        for date_index in sparse_dates:
            coefficients_without_sparse_fits[date_index] = None
        # Here this diagnostic simply continues on sparse dates; all other
        # stored rules, including the initial decision, remain unchanged.
        alternative = apply_fitted_exercise_rules(case.product, case.market_data, paths, dates,
            case.basis_function, coefficients_without_sparse_fits, valuation["exercise_at_time_zero"])
        difference = alternative["discounted_payoffs"] - original["discounted_payoffs"]
        european_payoffs = case.product.payoff(paths["spot"][:, -1]) * np.exp(
            -case.market_data.interest_rate * dates[-1])
        holding_difference = european_payoffs - original["discounted_payoffs"]
        comparisons.append({"study": study, "product": asdict(case.product), "model": asdict(case.model),
            "sparse_dates": sparse_dates, "original_value": original["option_value"],
            "original_standard_error": original["standard_error"],
            "changed_exercise_paths": int(np.count_nonzero(
                alternative["exercise_date_indices"] != original["exercise_date_indices"])),
            "value_change_if_sparse_dates_continue": float(np.mean(difference)),
            "paired_standard_error": float(np.std(difference, ddof=1) / np.sqrt(len(difference))),
            "hold_to_maturity_minus_original": float(np.mean(holding_difference)),
            "holding_difference_standard_error": float(np.std(holding_difference, ddof=1)
                                                         / np.sqrt(len(holding_difference)))})
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="JSON destination; default: a new runs/<run-id>/ folder")
    arguments = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-exercise1-diagnostics"
    destination = arguments.output or (PROJECT_DIRECTORY / "runs" / run_id / "regression_diagnostics.json")
    sv_cases = make_stochastic_volatility_cases()
    sparse_checks = check_sparse_regressions("gbm_pricing", make_gbm_pricing_cases())
    for study in ("sv_pricing", "sv_timestep"):
        sparse_checks.extend(check_sparse_regressions(study,
            [case for case in sv_cases if case.study == study]))
    output = {"paired_sensitivities": check_parameter_sensitivities(),
              "sparse_regression_diagnostic": sparse_checks}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Saved {destination}")
