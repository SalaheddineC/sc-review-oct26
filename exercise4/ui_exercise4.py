"""EXERCISE 4 — SAVED RESULTS, FIGURES AND THE STORAGE DECISION REPLAY.

Here the presentation reads the same numerical observations as the report.
The browser receives complete action tables from gas_storage_lsmc.py. It does
not implement another valuation algorithm or fit another set of decisions.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from results_integrity import load_result_json, verified_result_bytes
from products import GasStorage
from exercise4.gas_storage_lsmc import (
    feasible_storage_transitions, storage_continuation_values, storage_grid_actions,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
COLOURS = {"high": "#176d9a", "low": "#c78026"}


# 1. LOAD THE PUBLISHED CALCULATION AND ITS MATCHING ARRAYS
def load_exercise4_results(results_path=None):
    results_path = Path(results_path or PROJECT_DIRECTORY / "results/exercise4_results.json")
    document = load_result_json(results_path, exercise=4, required=("code_sha256", "config", "arrays_file", "arrays_sha256"))
    if document.get("exercise") != 4 or not document.get("main_cases"):
        raise ValueError("Expected completed Exercise 4 storage results")
    arrays_path = results_path.with_name(document.get("arrays_file", results_path.stem.replace("results", "arrays") + ".npz"))
    verified_result_bytes(arrays_path, document["arrays_sha256"])
    with np.load(arrays_path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    fields = ("coefficients", "knots", "inventory_grid", "initial_continuation",
              "constant_dates", "constant_continuation", "spot_paths", "inventory_paths",
              "action_paths", "daily_cashflows")
    for case in document["main_cases"]:
        if any(f"{case['case']}_{field}" not in arrays for field in fields):
            raise ValueError("Incomplete Exercise 4 saved decisions or operating paths")
    dates = len(GasStorage(**document["config"]["product"]).operating_dates())
    for row in document["main_cases"]:
        case = row["case"]
        paths = arrays[f"{case}_spot_paths"]
        if paths.ndim != 2 or paths.shape[1] != dates or any(
            arrays[f"{case}_{field}"].shape != (len(paths), dates + extra)
            for field, extra in (("inventory_paths", 1), ("action_paths", 0), ("daily_cashflows", 0))):
            raise ValueError("Exercise 4 saved operating paths have incompatible dimensions")
    return document, arrays


def saved_storage_decisions(document, arrays, case="high"):
    product = GasStorage(**document["config"]["product"])
    fields = ("coefficients", "knots", "inventory_grid", "initial_continuation",
              "constant_dates", "constant_continuation")
    decisions = {field: arrays[f"{case}_{field}"] for field in fields}
    decisions["actions"] = storage_grid_actions(product, document["config"]["inventory_step"])
    decisions["initial_spot"] = document["config"]["initial_spot"]
    return product, decisions


def clean_json_values(value):
    # Here unreachable numerical states become JSON null rather than NaN.
    if isinstance(value, np.ndarray):
        return clean_json_values(value.tolist())
    if isinstance(value, dict):
        return {key: clean_json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_values(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# 2. REPLAY SAVED PRICES AND TRADES; EVALUATE ONLY THEIR SAVED CONTINUATIONS
def build_storage_replay_data(document, arrays, case="high"):
    product, decisions = saved_storage_decisions(document, arrays, case)
    dates = product.operating_dates().astype(str).tolist()
    grid, actions = decisions["inventory_grid"], decisions["actions"]
    paths = []
    for path_index, prices in enumerate(arrays[f"{case}_spot_paths"]):
        inventories = arrays[f"{case}_inventory_paths"][path_index]
        trades = arrays[f"{case}_action_paths"][path_index]
        cashflows = arrays[f"{case}_daily_cashflows"][path_index]
        options = []
        for date_index, spot in enumerate(prices):
            inventory_index = int(round(inventories[date_index] / document["config"]["inventory_step"]))
            transitions = feasible_storage_transitions(product, grid, actions, date_index, len(dates))[inventory_index]
            continuation = storage_continuation_values([spot], date_index, decisions)[0]
            rows = []
            for action_index, next_index in enumerate(transitions):
                cash = float(product.cashflow(spot, actions[action_index]))
                if next_index < 0:
                    resulting = inventories[date_index] + actions[action_index]
                    reason = ("Not enough gas" if resulting < 0 else "Storage is full" if resulting > product.capacity
                              else "Cannot meet final inventory")
                    rows.append({"cash": cash, "continuation": None, "total": None, "reason": reason})
                else:
                    value = float(continuation[next_index])
                    rows.append({"cash": cash, "continuation": value, "total": cash + value, "reason": ""})
            options.append(rows)
        paths.append({"label": f"Sample price path {path_index + 1}", "prices": prices,
            "inventory": inventories, "trades": trades, "cashflows": cashflows,
            "cumulative": np.concatenate(([0.0], np.cumsum(cashflows))), "options": options})
    row = next(row for row in document["main_cases"] if row["case"] == case)
    return clean_json_values({"case": case, "label": row["label"], "volatility": row["volatility"],
        "dates": dates, "actions": actions, "capacity": product.capacity,
        "initial_inventory": product.initial_inventory, "final_inventory": product.final_inventory,
        "paths": paths, "mean_value": row["evaluation_value"], "standard_error": row["standard_error"]})


def storage_replay_html(document, arrays, case="high"):
    template = (PROJECT_DIRECTORY / "assets/gas_storage_replay.html").read_text(encoding="utf-8")
    # Here both volatility cases use three saved evaluation paths. We do not
    # simulate prices, refit decisions, or rerun a valuation to open the replay.
    data = {"selected_case": case, "cases": [build_storage_replay_data(document, arrays, row["case"])
        for row in document["main_cases"]], "training_paths": document["config"]["training_paths"],
        "evaluation_paths": document["config"]["evaluation_paths"], "seed": document["config"]["seed"]}
    for row in data["cases"]:
        row["variant"] = "baseline"
        row["base_case"] = row["case"]
        row["study_note"] = "Assignment baseline: 2,500 MWh grid, five daily volume choices, and 10,000 antithetic evaluation paths, independent of training."
    extension_path = PROJECT_DIRECTORY / "results/exercise4_extensions.json"
    if extension_path.exists():
        from exercise4.storage_extensions import load_storage_extensions
        extensions = load_storage_extensions()
        for result in extensions["grid"]:
            replay = dict(result["replay"])
            replay.update(variant="finer", base_case=result["case"], case=result["case"] + "_finer")
            uplift = result["refinements"][-1]["optimized_grid_uplift"]
            replay["study_note"] = (f"Optional finer grid: 1,250 MWh spacing gives nine volume choices, including partial injection. "
                f"The independently optimized value changes by EUR {uplift:,.2f}. "
                f"Fitting took {result['fine_fit_seconds']/result['coarse_fit_seconds']:.1f} times as long in this check. "
                "The same price paths are used as the baseline; greater detail need not add value.")
            data["cases"].append(replay)
        replay = dict(extensions["sobol_bridge_replay"])
        count = max(extensions["paths"])
        comparison = {r["method"]: r for r in extensions["sampling"] if r["paths"] == count}
        mc, chronological, bridge = (comparison[name] for name in ("antithetic_mc", "sobol_chronological", "sobol_bridge"))
        replay.update(variant="bridge", base_case="high", case="high_bridge")
        replay["study_note"] = (f"Optional Sobol + mean-reverting bridge: unchanged fitted coefficients, {count:,} paths per run. "
            f"Across {extensions['repetitions']} independent repetitions, pricing RMSE is EUR {bridge['rmse']:,.0f}, "
            f"versus EUR {mc['rmse']:,.0f} for antithetic Monte Carlo and EUR {chronological['rmse']:,.0f} for chronological Sobol. "
            "The reference integrates these same decisions. Playback shows one saved scramble, not a new valuation.")
        data["cases"].append(replay)
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    return template.replace("/*__STORAGE_DATA__*/", payload.replace("</", "<\\/"))


def render_storage_replay(document, arrays):
    # Here the overview and Exercise 4 display the same saved operating year.
    import streamlit as st
    import streamlit.components.v1 as components

    @st.cache_data(show_spinner=False, max_entries=2)
    def cached_replay(saved_document, saved_arrays, template_version, extension_version):
        return storage_replay_html(saved_document, saved_arrays)

    template_version = sha256((PROJECT_DIRECTORY / "assets/gas_storage_replay.html").read_bytes()).hexdigest()
    extension_path = PROJECT_DIRECTORY / "results/exercise4_extensions.json"
    extension_version = sha256(extension_path.read_bytes()).hexdigest() if extension_path.exists() else ""
    replay_html = cached_replay(document, arrays, template_version, extension_version)
    components.html(replay_html, height=1270, scrolling=True)
    return replay_html


# 3. ANSWER THE THREE ASSIGNMENT PARTS WITH READABLE FIGURES
def style_storage_figure(figure, title, x_title=None, y_title=None, height=420):
    figure.update_layout(template="plotly_white", title={"text": title, "font": {"size": 15}}, height=height,
        paper_bgcolor="white", plot_bgcolor="white", font={"family": "Arial, sans-serif", "color": "#172b45", "size": 12},
        margin={"l": 75, "r": 25, "t": 60, "b": 100},
        legend={"orientation": "h", "y": -0.3, "font": {"size": 11}}, hovermode="x unified")
    figure.update_xaxes(gridcolor="#e5ebf1", tickfont={"size": 11}, title_font={"size": 12}, automargin=True)
    figure.update_yaxes(gridcolor="#e5ebf1", tickfont={"size": 11}, title_font={"size": 12}, automargin=True)
    if x_title is not None:
        figure.update_xaxes(title_text=x_title)
    if y_title is not None:
        figure.update_yaxes(title_text=y_title)
    figure.update_annotations(font={"size": 12})
    return figure


def storage_price_figure(document, arrays, case="high"):
    dates = GasStorage(**document["config"]["product"]).operating_dates().astype(str).tolist()
    figure = go.Figure()
    for index, path in enumerate(arrays[f"{case}_spot_paths"]):
        label = ["Path 1 · original shocks", "Path 2 · opposite shocks to Path 1", "Path 3 · new shocks"][index]
        figure.add_trace(go.Scatter(x=dates, y=path, mode="lines", name=label,
            line={"width": 1.5, "color": ["#176d9a", "#b58231", "#7661a9"][index % 3]}))
    figure.add_trace(go.Scatter(x=dates, y=arrays[f"{case}_mean_prices"], name="Exact expected spot",
        line={"color": "#172b45", "dash": "dot", "width": 2.5}))
    style_storage_figure(figure, "4.1 · Three saved daily gas-price paths", "Operating date", "Spot · €/MWh")
    figure.update_xaxes(dtick="M2", tickformat="%b %Y")
    return figure


def storage_convergence_table(document):
    observations = pd.DataFrame(document.get("convergence", []))
    if observations.empty:
        return observations
    table = observations.groupby(["case", "training_paths"], as_index=False).agg(
        repetitions=("evaluation_value", "count"),
        mean_training_value=("training_value", "mean"),
        mean_evaluation_value=("evaluation_value", "mean"),
        run_standard_deviation=("evaluation_value", "std"),
        mean_evaluation_standard_error=("standard_error", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"))
    return table


def storage_convergence_figure(document):
    table = storage_convergence_table(document)
    figure = go.Figure()
    if table.empty:
        return style_storage_figure(figure, "4.3 · No convergence study in this calculation")
    for case, rows in table.groupby("case", sort=False):
        rows = rows.sort_values("training_paths")
        if len(rows) < 2:
            continue  # Here a single path count is not a convergence curve.
        figure.add_trace(go.Scatter(x=rows.training_paths, y=rows.mean_evaluation_value,
            error_y={"type": "data", "array": rows.run_standard_deviation.fillna(0)},
            name=f"{case.title()} volatility · independent evaluation", mode="lines+markers",
            line={"color": COLOURS.get(case, "#176d9a"), "width": 3}, marker={"size": 8}))
        figure.add_trace(go.Scatter(x=rows.training_paths, y=rows.mean_training_value,
            name=f"{case.title()} volatility · training sample", mode="lines+markers",
            line={"color": COLOURS.get(case, "#176d9a"), "width": 1.5, "dash": "dot"}))
    style_storage_figure(figure, "4.3 · High volatility: value against training-path count", "Training paths · logarithmic spacing", "Mean storage value · €")
    counts = sorted(table[table["case"] == "high"]["training_paths"].unique())
    figure.update_xaxes(type="log", tickmode="array", tickvals=counts, ticktext=[f"{count:,}" for count in counts])
    return figure


def storage_distribution_figure(document, arrays):
    figure = make_subplots(rows=1, cols=len(document["main_cases"]),
        subplot_titles=[row["label"] for row in document["main_cases"]])
    for index, row in enumerate(document["main_cases"], 1):
        case = row["case"]
        figure.add_trace(go.Histogram(x=arrays[f"{case}_evaluation_cashflows"], nbinsx=45,
            name=row["label"], marker_color=COLOURS.get(case, "#176d9a"), opacity=.86), row=1, col=index)
        figure.add_vline(x=row["evaluation_value"], line_dash="dash", line_color="#172b45", row=1, col=index,
            annotation_text="Mean", annotation_position="top right")
        figure.update_xaxes(title_text="Path operating cashflow · €", row=1, col=index)
    style_storage_figure(figure, "4.3 · Cashflow distributions, following Exhibit 3", y_title="Number of paths")
    figure.update_layout(showlegend=False, bargap=.06, hovermode="closest")
    return figure


# 4. PRESENT THE COMPLETED STUDIES AS ONE EXECUTIVE PAGE
def render_exercise4_page():
    import streamlit as st
    from exercise1.ui_exercise1 import summary_table_html
    from exercise4.gas_storage_benchmark import load_storage_benchmark

    st.caption("EXERCISE 4 / PHYSICAL GAS STORAGE")
    st.title("The value of operating flexibility")
    st.write("Buy, sell or wait: we value the right to adapt daily gas trades to uncertain prices, "
             "while meeting capacity, flow and final-inventory requirements. This page replays completed results; "
             "no new valuation is needed.")
    try:
        document, arrays = load_exercise4_results()
        benchmark = load_storage_benchmark()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"The saved storage evidence cannot be verified: {error}")
        return

    config = document["config"]
    columns = st.columns(len(document["main_cases"]))
    for column, row in zip(columns, document["main_cases"]):
        column.metric(row["label"] + " · expected operating cashflow", f"€{row['evaluation_value']/1e6:.2f}m")
        column.caption(f"10,000 new evaluation paths · approximate 95% sampling interval ±€{1.96*row['standard_error']:,.0f}.")
    st.markdown(summary_table_html(pd.DataFrame([
        ["4.1 · Simulate daily prices", "Exact daily log-price transition; high and low volatility; antithetic shocks.",
         "Checked against exact log-price moments; finite-sample deviations recorded."],
        ["4.2 · Choose feasible trades", "Compare today's cashflow plus continuation, fitted with the six B-spline basis.",
         "No capacity, daily-flow or final-inventory violations on the evaluation paths."],
        ["4.3 · Value and verify", "10,000 training paths; new evaluation paths; independent numerical integration.",
         "Decision value loss is about 0.04% in both cases against the refined reference."],
    ], columns=["Assignment question", "What we implemented", "What we found"])), unsafe_allow_html=True)

    st.subheader("Watch a completed operating year")
    st.write("We built this replay to connect the saved numerical decisions to physical storage operations. "
             "The three linked views show today's decision, the gas in storage, and inventory through the year. "
             "Choose a saved price path, then play or move the date. Every displayed trade comes from the saved calculation.")
    replay_html = render_storage_replay(document, arrays)
    st.caption("What to take away: the best trade balances cash received or paid today with the value of gas left for later. "
               "Each illustrated year starts and ends with 100,000 MWh; one path's cashflow is not the contract's average value.")

    st.subheader("How close are our decisions to an independent reference?")
    st.write("No full-year benchmark was supplied for this exact setup. We solve the same daily decision problem by "
             "numerical integration, without continuation regressions. We also integrate the cashflows of our saved decisions "
             "unchanged, so the comparison measures decision quality without Monte Carlo evaluation noise.")
    comparison = pd.DataFrame([{
        "Scenario": row["label"], "Numerical reference": f"€{row['reference']['reference_value']:,.0f}",
        "Our saved decisions, integrated": f"€{row['reference']['saved_policy_value']:,.0f}",
        "Value loss": f"€{row['reference']['policy_value_gap']:,.0f} ({row['reference']['policy_value_gap_percent']:.3f}%)",
    } for row in benchmark["cases"]])
    st.markdown(summary_table_html(comparison), unsafe_allow_html=True)
    st.write("The decisions retain about 99.96% of the reference value in both volatility cases. "
             "Separate refinements of price spacing, integration points and the price domain support this comparison. "
             "This is a numerical result under the same model and inventory grid, not a calibrated market quotation.")

    with st.expander("Explore the saved results · prices, cashflows and path counts"):
        view = st.radio("Evidence to display", ["4.1 · Price paths", "4.3 · Cashflow distribution", "4.3 · Path-count study"],
                        horizontal=True, key="ex4_evidence")
        if view == "4.1 · Price paths":
            case = st.selectbox("Volatility case", [row["case"] for row in document["main_cases"]],
                format_func=lambda selected: next(row["label"] for row in document["main_cases"] if row["case"] == selected),
                key="ex4_price_case")
            st.write("Three saved daily gas-price paths from the new evaluation sample, all starting at €20/MWh. "
                     "Paths 1 and 2 use opposite Gaussian shocks and form an antithetic pair. Path 3 starts another pair. "
                     "The dotted curve is the model's exact expected price.")
            st.plotly_chart(storage_price_figure(document, arrays, case), use_container_width=True, theme=None)
            st.caption("Different price histories create different buying and selling opportunities. "
                       "These three illustrations explain the dynamics; the separate analytic moment checks establish simulation consistency.")
        elif view == "4.3 · Cashflow distribution":
            st.write("Each histogram contains the realised operating cashflows of 10,000 new price paths, "
                     "following the estimated decisions. The dashed line is their average; both cases share the same contract and price scale.")
            st.plotly_chart(storage_distribution_figure(document, arrays), use_container_width=True, theme=None)
            st.caption("Greater volatility increases both opportunities to trade and variation in the realised annual cashflow. "
                       "These are path outcomes, not the sampling distribution of the estimated contract value.")
        else:
            st.write("The high-volatility study repeats each training-path count ten times. Solid lines show average value "
                     "on new paths; dotted lines show the corresponding training-sample values. Bars show one standard deviation across repetitions.")
            st.plotly_chart(storage_convergence_figure(document), use_container_width=True, theme=None)
            table = storage_convergence_table(document)
            rows = table[table["case"] == "high"].sort_values("training_paths")
            if len(rows) >= 2:
                earlier, latest = rows.iloc[-2], rows.iloc[-1]
                change = latest.mean_evaluation_value - earlier.mean_evaluation_value
                st.caption(f"From {int(earlier.training_paths):,} to {int(latest.training_paths):,} training paths, "
                           f"the repeated mean changes by €{change:+,.0f} ({change/earlier.mean_evaluation_value:+.2%}). "
                           "This supports observed stability; the separate numerical reference assesses decision quality.")

    with st.expander("Inputs, checks and business limitations"):
        st.write("7 July–30 June · 250,000 MWh capacity · 100,000 MWh initially and finally · "
                 "daily actions −7,500, −5,000, −2,500, 0 or +2,500 MWh. "
                 "All price paths enter the storage regressions; there is no in-the-money filter.")
        st.write(f"{config['training_paths']:,} training and {config['evaluation_paths']:,} new evaluation paths · "
                 f"seed {config['seed']} · mean reversion {config['mean_reversion_speed']:g} per day · "
                 "daily volatilities 9.45% and 3.15% · zero rates. "
                 "Antithetic pairs are used in both stages; sampling intervals use independent pair averages.")
        st.write("We use €20/MWh as an illustrative initial and stationary mean price, near the midpoint of the paper's "
                 "quoted forward-price extremes. This is not a historical observation or a current market calibration. "
                 "The constant price target isolates the value of adapting to uncertainty.")
        rows = []
        for case in benchmark["cases"]:
            for name, changes in case["stability_changes_eur"].items():
                rows.append([case["label"], name.replace("_", " "), f"€{changes['reference_value']:,.2f}",
                             f"€{changes['policy_value_gap']:,.2f}"])
        st.write("The following are absolute changes under separate benchmark refinements. Smaller changes support "
                 "numerical stability; they do not provide a rigorous error bound.")
        st.markdown(summary_table_html(pd.DataFrame(rows, columns=["Scenario", "Refinement", "Reference change", "Value-loss change"])), unsafe_allow_html=True)
        st.write("These values are expected operating cashflows before any unspecified storage rent. "
                 "We assume fixed flow limits and trading at model spot, without bid–offer costs, fuel losses or outages. "
                 "A commercial valuation would need the actual contract, forward curve, seasonality and operating constraints. "
                 "More paths alone cannot remove market-model or fixed-basis error.")
    st.download_button("Download the saved operating replay · HTML", replay_html,
                       "exercise4_interactive.html", "text/html", key="ex4_replay_download")


