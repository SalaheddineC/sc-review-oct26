TEST DATA — EXXONMOBIL QUANTITATIVE MODELLING ASSIGNMENT

This folder supplies the numerical test data for all four exercises. The files
can be inspected without running a valuation. To execute the checks, run this
command from the project folder after installing requirements.txt:

    python -m pytest -q

The tests read cases.json through tests/conftest.py and load the CSVs directly.
Each case records its purpose, source, reference type, units, inputs, expected
answer and tolerance. Where an answer depends on a random draw, the file gives
the seed and analytical reference formula; the test evaluates that formula
independently. Inputs used only for integration and error-handling checks also
remain beside those tests.

1 — AMERICAN OPTION VALUATION

exercise1/longstaff_table1.csv
    Twenty published scenarios from Longstaff and Schwartz (2001), Table 1.
    Used by the Exercise 1 runner for its published-price comparison; Exercises
    2 and 3 reuse its at-the-money spot and strike with their required inputs.
    Prices share the units of spot and strike; maturity is in years and rates
    and volatility are decimal annual values. american_fd is the published
    finite-difference value, american_lsm the published least-squares value,
    and lsm_se its reported standard error. The other price columns give the
    European value, early-exercise premiums and their difference.
    These are rounded published comparison results, not exact simulation
    targets. The CSV also records the paper's path count and exercise frequency.

exercise1/longstaff_eight_paths.csv
    The paper's eight paths at dates 0, 1, 2 and 3. Together with cases.json,
    checks regression coefficients, stopping decisions, cashflows and value.
    This reproduces the worked example on its supplied paths; it is not an
    independent evaluation sample.

exercise1/cases.json
    Four cases: two prescribed Euler updates, the published eight-path example
    and prescribed exercise rules. Sources and tolerances distinguish rounded
    published coefficients from arithmetic checks.

2 — CONTROL VARIATES AND GAMMA

exercise2/barone_adesi_whaley_table_iv.csv
    Five published American-put prices from Barone-Adesi and Whaley (1987),
    Table IV, with the corresponding inputs and absolute price tolerance.
    These positive-rate cases validate the approximation separately from the
    assignment's zero-rate studies. The source is also identified in cases.json.

exercise2/cases.json
    Six cases: prescribed control transitions, control adjustment arithmetic,
    exact cancellation when waiting to maturity, the published price cases,
    centred gamma differences and the analytical European finite-bump check.

3 — EULER AND MILSTEIN

exercise3/cases.json
    Six cases: coupled convergence against exact GBM, constant-variance recovery
    of GBM, Wiktorsson's conditional cross-integral law, a common-driver boundary
    check, a negative-spot step and explicit rejection of non-finite output.
    Supplied increments, seeds, model inputs and acceptance bounds are included.
    Analytical formulae supply the references; simulator output does not define
    its own expected answer.

4 — GAS STORAGE

exercise4/cases.json
    Five cases: forced final-day trades, zero value at constant prices, exact
    three-day values at both volatilities and one saved full-year replay.
    The short contracts use analytical truncated-lognormal moments to check
    the independent integration benchmark. Contract and model inputs, grid
    settings, expected values and tolerances are explicit.

exercise4/operating_path.csv
    All 359 operating dates of the first saved high-volatility evaluation path,
    from 7 July 2005 to 30 June 2006. Columns give the date, spot in EUR/MWh,
    inventory before and after trading in MWh, signed trade in MWh, daily
    cashflow in EUR and cumulative cashflow in EUR. Positive trades inject gas;
    negative trades withdraw it. With zero interest and no trading costs,
    daily cashflow = -spot * trade.
    Tests check inventory limits, daily trading limits, terminal inventory,
    cashflow arithmetic and reproduction using the saved decision coefficients.
    This is a saved implementation replay, not an independent price benchmark
    or an estimate of expected profit from a single path.

TRAINING DATA, EVALUATION DATA AND SAVED RESULTS

The assignment uses simulated data under its specified models, not a historical
market-data feed. Training paths estimate continuation values and decision rules;
independent evaluation paths price the resulting rules. These larger datasets
are generated in memory by the supplied code rather than stored as large CSVs.

results/ contains the numerical outputs used by the report and application.
The JSON files record model and contract inputs, path counts, seeds and numerical
settings. Exercise 4's companion NPZ stores fitted coefficients, training and
evaluation cashflows and selected price/inventory paths for replay. It does not
store every simulated price path. The CSV above makes one of those stored paths
readable without opening an NPZ archive.

Use the recorded settings to reproduce a saved study; the runners' default
budgets need not equal every saved run. For example, the saved Exercise 1 Table 1
study uses 50,000 training and 100,000 evaluation paths:

    python -m exercise1.run_exercise1 table1 --training-paths 50000 --evaluation-paths 100000 --seed 20261004

Each runner accepts --help. New results go under runs/ by default. The application
continues to display the supplied results/. Historical source paths and hashes
inside saved result metadata describe the original run; the Longstaff-Schwartz
CSVs are now centralised in test_data/exercise1/.

The additional Sobol/bridge checks are available through:

    python -m exercise4.storage_extensions --check

Published references, analytical identities and saved replay checks serve
different purposes and are labelled accordingly. Small arithmetic tolerances
allow floating-point rounding; published-price tolerances allow quoted rounding;
statistical tests use the stated sampling bounds. No full valuation study is
needed to run the test suite or explore the saved results in the application.
