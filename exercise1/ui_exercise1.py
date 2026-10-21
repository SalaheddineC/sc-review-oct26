"""Here we turn saved Exercise 1 results into tables and Plotly figures.

Simulation and regression remain in the numerical modules.
"""
import json
from html import escape
from pathlib import Path

from results_integrity import load_result_json, verified_result_bytes

import pandas as pd
import plotly.graph_objects as go


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
COLORS = ["#2463c5", "#0c8873", "#da8532"]
PARAMETER_NAMES = {
    "long_term_variance": "Long-run variance θ",
    "mean_reversion_speed": "Mean-reversion speed κ",
    "volatility_of_variance": "Volatility of variance ξ",
    "brownian_correlation": "Spot–variance correlation ρ",
}


# LOAD THE VALIDATED RESULTS
def load_saved_results():
    # Here we check the saved file hashes before displaying their results.
    results_directory = PROJECT_DIRECTORY / "results"
    summary = load_result_json(results_directory / "validation_summary.json",
        required=("core_source_sha256", "result_file_sha256", "monotonicity"))
    studies = {}
    for name, expected_hash in summary["result_file_sha256"].items():
        contents = verified_result_bytes(results_directory / name, expected_hash)
        studies[Path(name).stem] = json.loads(contents)
    for name in ("table1", "gbm_pricing", "gbm_timestep", "sv_pricing", "sv_timestep", "sv_sensitivities"):
        records = studies[name]
        if not isinstance(records, list) or not records or any(
            not {"market_data", "product", "model", "valuation"}.issubset(row) for row in records):
            raise ValueError(f"Incomplete Exercise 1 study: {name}")
    return studies, summary


# ONE ROW PER VALUATION CASE
def valuation_table(records):
    # Here we select the economic inputs and reported outputs from each saved run.
    rows = []
    for record in records:
        market_data = record["market_data"]
        product = record["product"]
        valuation = record["valuation"]
        rows.append({
            "spot": market_data["spot"], "strike": product["strike"],
            "maturity_years": product["maturity_years"],
            "volatility": market_data["volatility"],
            "long_term_variance": record["model"].get("long_term_variance"),
            "step_days": record["requested_time_step_days"],
            "price": valuation["option_value"],
            "standard_error": valuation["standard_error"],
            "two_standard_errors": 2 * valuation["standard_error"],
        })
    return pd.DataFrame(rows).dropna(axis="columns", how="all")


# EXERCISE 1 — TABLE 1 COMPARISONS
def table1_comparison_table(records):
    table = valuation_table(records)
    table["paper_lsm"] = [record["table1_comparison"]["american_lsm"] for record in records]
    table["paper_fd"] = [record["table1_comparison"]["american_fd"] for record in records]
    table["paper_lsm_se"] = [record["table1_comparison"]["lsm_se"] for record in records]
    table["difference_from_lsm"] = table["price"] - table["paper_lsm"]
    table["difference_from_fd"] = table["price"] - table["paper_fd"]
    # Here independent sampling errors add in variance, not directly in SE.
    table["two_combined_errors"] = 2 * (table["standard_error"] ** 2
                                            + table["paper_lsm_se"] ** 2) ** 0.5
    table["early_exercise_premium"] = [
        record["table1_comparison"]["estimated_early_exercise_premium"] for record in records]
    return table


# SUMMARISE PRICE AGREEMENT IN PLAIN UNITS AND PERCENTAGES
def benchmark_summary(table, reference_column="paper_lsm"):
    differences = (table["price"] - table[reference_column]).abs()
    return {"cases": len(table), "mean_absolute_gap": differences.mean(),
            "largest_absolute_gap": differences.max(),
            "mean_relative_gap": (differences / table[reference_column]).mean()}


# READABLE SUMMARY TABLES FOR THE APPLICATION
def summary_table_html(table, caution_rows=()):
    # Here we escape every label/value before adding presentation markup.
    # Amber marks a qualification; the table never substitutes a pass/fail score.
    headers = "".join(f'<th scope="col">{escape(str(column))}</th>' for column in table.columns)
    rows = []
    for row_index, values in enumerate(table.itertuples(index=False, name=None)):
        cells = []
        for value in values:
            display = f"{value:.4f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{escape(display)}</td>")
        row_class = ' class="qualified-result"' if row_index in caution_rows else ""
        rows.append(f"<tr{row_class}>{''.join(cells)}</tr>")
    return ('<div class="summary-table-wrap"><table class="summary-table">'
            f"<thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


# CONSISTENT FORMATTING — ONE SELECTED COMPARISON PER GRAPH
def format_figure(figure, x_label, y_label="American put value (price units)"):
    figure.update_layout(template="plotly_white", height=415,
        font=dict(family="Arial, sans-serif", size=14, color="#26374b"),
        margin=dict(l=40, r=20, t=65, b=50), xaxis_title=x_label, yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, font=dict(size=13)),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified")
    figure.update_xaxes(showgrid=False, automargin=True, title_standoff=16)
    figure.update_yaxes(gridcolor="#e9edf2", zeroline=False, automargin=True, title_standoff=18)
    return figure


# EXERCISE 1 — OUR PRICES BESIDE THE AUTHORS' TABLE 1 PRICES
def table1_figure(selected_cases, show_uncertainty=False):
    table = selected_cases.sort_values("spot")
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=table["spot"], y=table["paper_lsm"],
        name="Authors’ Table 1 prices", mode="lines+markers",
        line=dict(color="#68778b", width=3, dash="dash"), marker=dict(size=12, symbol="circle-open"),
        hovertemplate="Table 1 price: %{y:.6f}<extra></extra>"))
    figure.add_trace(go.Scatter(x=table["spot"], y=table["price"],
        name="Our Longstaff–Schwartz prices", mode="lines+markers",
        line=dict(color=COLORS[0], width=2), marker=dict(size=8, symbol="diamond"),
        error_y=dict(type="data", array=table["two_standard_errors"], visible=show_uncertainty),
        customdata=table[["difference_from_lsm", "standard_error"]].to_numpy(),
        hovertemplate="Our price: %{y:.6f}<br>Difference: %{customdata[0]:+.6f}<br>Sampling SE: %{customdata[1]:.6f}<extra></extra>"))
    figure.update_xaxes(tickvals=table["spot"])
    figure.update_yaxes(rangemode="tozero")
    return format_figure(figure, "Initial spot price")


# EXERCISE 1 — THE ECONOMIC EFFECT OF ONE CHOSEN INPUT
def pricing_figure(selected_cases, variable, variable_label):
    table = selected_cases.sort_values(variable)
    figure = go.Figure(go.Scatter(x=table[variable], y=table["price"],
        name="Our American put values", mode="lines+markers",
        line=dict(color=COLORS[1], width=3), marker=dict(size=9),
        error_y=dict(type="data", array=table["two_standard_errors"]),
        hovertemplate="Input: %{x}<br>Put value: %{y:.6f}<extra></extra>"))
    figure.update_xaxes(tickvals=table[variable])
    figure.update_yaxes(rangemode="tozero")
    return format_figure(figure, variable_label)


# EXERCISE 1 — COMPARE TIMESTEPS WHILE HOLDING THE CONTRACT FIXED
def timestep_figure(selected_cases, stochastic_volatility=False):
    table = selected_cases.sort_values("step_days")
    value_column = "price" if stochastic_volatility else "difference_from_fd"
    figure = go.Figure(go.Scatter(x=table["step_days"], y=table[value_column],
        name="Our values" if stochastic_volatility else "Difference from Table 1 reference",
        mode="lines+markers", line=dict(color=COLORS[2], width=3), marker=dict(size=9),
        error_y=dict(type="data", array=table["two_standard_errors"]),
        hovertemplate="Maximum step: %{x} days<br>Value: %{y:.6f}<extra></extra>"))
    reference = table.loc[table["step_days"] == 1, "price"].iloc[0] if stochastic_volatility else 0
    figure.add_hline(y=reference, line_dash="dot", line_color="#68778b")
    figure.update_xaxes(tickvals=[1, 2, 4, 7])
    y_label = "American put value (price units)" if stochastic_volatility else "Our value − Table 1 numerical reference"
    return format_figure(figure, "Maximum Euler–Maruyama step (calendar days)", y_label)


# EXERCISE 1 — EXPLAIN ONE STOCHASTIC-VOLATILITY PARAMETER EFFECT
def sensitivity_figure(selected_comparisons, parameter):
    table = selected_comparisons.sort_values("parameter_value")
    figure = go.Figure(go.Scatter(x=table["parameter_value"], y=table["difference_from_baseline"],
        name="Change from baseline put value", mode="lines+markers",
        line=dict(color="#7851b2", width=3), marker=dict(size=9),
        error_y=dict(type="data", array=2 * table["paired_standard_error"]),
        customdata=table[["option_value"]].to_numpy(),
        hovertemplate="Parameter: %{x}<br>Value change: %{y:+.6f}<br>Put value: %{customdata[0]:.6f}<extra></extra>"))
    figure.add_hline(y=0, line_dash="dot", line_color="#68778b")
    figure.update_xaxes(tickvals=table["parameter_value"])
    figure.update_yaxes(tickformat=".5f")
    return format_figure(figure, PARAMETER_NAMES[parameter], "Change in put value from baseline (price units)")
