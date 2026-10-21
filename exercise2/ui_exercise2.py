"""EXERCISE 2: TABLES AND PLOTS FROM SAVED PRICES AND GAMMA CALCULATIONS.

Here the saved results supply the application tables and figures.
Opening a page never starts a simulation.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from results_integrity import load_result_json

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent


PLAIN_COLOUR = "#1464a5"
CONTROL_COLOUR = "#bd4b00"


# EXECUTIVE RESULTS — DERIVE EVERY HEADLINE FROM THE SAVED EXPERIMENTS
def build_exercise2_executive_metrics(document, gamma_document=None):
    valuations = [row["valuation"] for row in document["pricing_results"]]
    comparisons = document["baw_comparisons"]
    variance_factors = [row["variance_reduction_factor"] for row in valuations
                        if row["variance_reduction_factor"] is not None]
    metrics = {
        "price_cases": len(valuations),
        "reduced_price_variance_cases": sum(
            row["controlled_estimator_variance"] < row["plain_estimator_variance"]
            for row in valuations),
        "minimum_price_variance_factor": min(variance_factors) if variance_factors else None,
        "maximum_price_variance_factor": max(variance_factors) if variance_factors else None,
        "mean_plain_absolute_price_error": sum(abs(row["plain_error"]) for row in comparisons) / len(comparisons),
        "mean_controlled_absolute_price_error": sum(abs(row["controlled_error"]) for row in comparisons) / len(comparisons),
        "mean_controlled_absolute_relative_price_error": sum(
            abs(row["controlled_error"]) / row["baw_reference"] for row in comparisons
        ) / len(comparisons),
        "controlled_errors_outside_conditional_interval": sum(
            abs(row["controlled_error"]) > 1.96 * row["controlled_standard_error"]
            for row in comparisons),
    }
    if gamma_document:
        gamma_summary, bump_checks = build_exercise2_gamma_tables(gamma_document)
        reductions = [1 - row["Mean controlled estimator variance"] / row["Mean plain estimator variance"]
                      for row in gamma_summary.to_dict("records") if row["Mean plain estimator variance"] > 0]
        rows = gamma_document["gamma_results"]["rows"]
        metrics.update({
            "gamma_cases": len(rows),
            "gamma_spot_cases": len({row["spot"] for row in rows}),
            "gamma_volatility_cases": len({row["volatility"] for row in rows}),
            "minimum_mean_gamma_variance_reduction": min(reductions) if reductions else None,
            "maximum_mean_gamma_variance_reduction": max(reductions) if reductions else None,
            "negative_controlled_gamma_cases": sum(row["controlled_gamma"] < 0 for row in rows),
        })
    return metrics


# LOAD THE PUBLISHED DATA AND RETAIN THEIR ORIGINAL SOURCE IDENTIFIERS
def load_exercise2_results(results_path):
    document = load_result_json(results_path, exercise=2, required=("code_sha256", "environment", "root_seed"))
    if document.get("exercise") != 2 or not document.get("pricing_results"):
        raise ValueError("Expected saved Exercise 2 pricing results")
    return document


# LOAD PART 3 UNDER THE SAME DATA-INTEGRITY CHECKS
def load_exercise2_gamma_results(results_path):
    document = load_result_json(results_path, exercise=2, required=("code_sha256", "environment", "root_seed"))
    if document.get("exercise") != 2 or not document.get("gamma_results", {}).get("rows"):
        raise ValueError("Expected saved Exercise 2 gamma results")
    return document


# REQUIRED TABLES: NAME THE ESTIMATORS AND THEIR BENCHMARK EXPLICITLY
def build_exercise2_tables(document):
    control_rows = []
    for record in document["pricing_results"]:
        valuation = record["valuation"]
        control_rows.append({
            "Volatility": record["market_data"]["volatility"],
            "American/European payoff correlation": valuation["payoff_correlation"],
            "Estimated beta": valuation["control_coefficient"],
            "Plain estimator variance": valuation["plain_estimator_variance"],
            "Controlled estimator variance": valuation["controlled_estimator_variance"],
            "Variance-reduction factor": valuation["variance_reduction_factor"],
        })
    benchmark_rows = [{
        "Volatility": row["volatility"], "BAW zero-rate reference": row["baw_reference"],
        "Plain estimate": row["plain_price"], "Controlled estimate": row["controlled_price"],
        "Plain error": row["plain_error"], "Controlled error": row["controlled_error"],
        "Plain SE": row["plain_standard_error"], "Controlled SE": row["controlled_standard_error"],
    } for row in document["baw_comparisons"]]
    return pd.DataFrame(control_rows), pd.DataFrame(benchmark_rows)


# CHART STYLE — KEEP LABELS READABLE IN LIGHT AND DARK APPLICATION THEMES
def style_exercise2_figure(figure, height=430):
    figure.update_layout(
        template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        height=height, font={"family": "Arial, sans-serif", "size": 12, "color": "#14243a"},
        title={"font": {"size": 15, "color": "#14243a"}, "x": 0.03},
        margin={"l": 65, "r": 20, "t": 55, "b": 100},
        legend={"orientation": "h", "yanchor": "top", "y": -0.2,
                "font": {"color": "#14243a", "size": 11}, "bgcolor": "#ffffff"},
        hovermode="x unified", hoverlabel={"bgcolor": "#ffffff", "font_color": "#14243a"})
    figure.update_xaxes(color="#14243a", gridcolor="#e4eaf1", zerolinecolor="#c5cfdb",
                       title_font={"color": "#14243a", "size": 12},
                       tickfont={"color": "#14243a", "size": 11}, automargin=True)
    figure.update_yaxes(color="#14243a", gridcolor="#e4eaf1", zerolinecolor="#c5cfdb",
                       title_font={"color": "#14243a", "size": 12},
                       tickfont={"color": "#14243a", "size": 11}, automargin=True)
    figure.update_annotations(font={"color": "#14243a", "size": 12})
    return figure


# REQUIRED PLOTS — COMPARE THE SAME EVALUATION SAMPLES AT EACH VOLATILITY
def build_exercise2_figures(document):
    records = sorted(document["pricing_results"], key=lambda row: row["market_data"]["volatility"])
    # Here volatility labels are categories: each computed scenario has equal space.
    volatility_labels = [f"{row['market_data']['volatility']:.0%}" for row in records]
    correlation = go.Figure(go.Scatter(
        x=volatility_labels, y=[row["valuation"]["payoff_correlation"] for row in records],
        mode="markers", name="American / European payments",
        line={"color": CONTROL_COLOUR, "width": 3}, marker={"size": 9},
        hovertemplate="Volatility %{x}<br>Payoff correlation %{y:.4f}<extra></extra>"))
    correlation.update_layout(title="American / European payment correlation", yaxis_title="Payment correlation", showlegend=False)
    correlation.update_yaxes(range=[-1.02, 1.02])
    variance = go.Figure()
    for key, label, colour in (
        ("plain_estimator_variance", "Plain LSMC", PLAIN_COLOUR),
        ("controlled_estimator_variance", "With European control", CONTROL_COLOUR),
    ):
        variance.add_trace(go.Bar(
            x=volatility_labels, y=[row["valuation"][key] for row in records],
            name=label, marker_color=colour,
            hovertemplate="Volatility %{x}<br>Estimator variance %{y:.6g} currency²<extra>%{fullData.name}</extra>"))
    variance.update_layout(title="Variance of the put price estimate", yaxis_title="Estimator variance (currency²)",
                           barmode="group", bargap=0.35, bargroupgap=0.08)
    variance.update_yaxes(rangemode="tozero", tickformat=".1e")
    for figure in (correlation, variance):
        style_exercise2_figure(figure)
        figure.update_xaxes(title="Volatility scenario", type="category", categoryorder="array",
                            categoryarray=volatility_labels)
    return {"payoff_correlation": correlation, "estimator_variance": variance,
            "one_month_baw_errors": build_exercise2_price_comparison(document, view="errors")}


# PART 2 — VIEW PRICES OR THEIR SIGNED DISTANCE FROM THE SAME BAW BENCHMARK
def build_exercise2_price_comparison(document, view="prices", relative=False, show_uncertainty=True):
    if view not in ("prices", "errors"):
        raise ValueError("Choose prices or errors for the BAW comparison")
    rows = sorted(document["baw_comparisons"], key=lambda row: row["volatility"])
    volatilities = [f"{row['volatility']:.0%}" for row in rows]
    figure = go.Figure()
    # Here relative errors are percentages of BAW; relative prices use BAW = 100%.
    scale = [100 / row["baw_reference"] if relative else 1 for row in rows]
    if view == "prices":
        figure.add_trace(go.Scatter(x=volatilities,
            y=[row["baw_reference"] * factor for row, factor in zip(rows, scale)],
            mode="lines+markers", name="BAW / Black–Scholes",
            line={"color": "#14243a", "width": 3, "dash": "dot"}, marker={"size": 9}))
    for estimator, name, colour in (
        ("plain", "Plain LSMC", PLAIN_COLOUR),
        ("controlled", "With European control", CONTROL_COLOUR),
    ):
        field = f"{estimator}_price" if view == "prices" else f"{estimator}_error"
        figure.add_trace(go.Scatter(x=volatilities,
            y=[row[field] * factor for row, factor in zip(rows, scale)],
            error_y={"type": "data", "array": [1.96 * row[f"{estimator}_standard_error"] * factor
                       for row, factor in zip(rows, scale)], "visible": show_uncertainty},
            mode="lines+markers", name=name,
            line={"color": colour, "width": 3 if estimator == "plain" else 2, "dash": "solid" if estimator == "plain" else "dash"},
            marker={"size": 10 if estimator == "plain" else 7, "symbol": "circle-open" if estimator == "plain" else "diamond"}))
    if view == "errors":
        figure.add_hline(y=0, line_color="#667085", line_dash="dash")
    ylabel = ("Price / BAW (%)" if relative else "Put price (currency units)") if view == "prices" else (
        "Signed error / BAW (%)" if relative else "Estimate − BAW (currency units)")
    figure.update_layout(title="One-month put prices against BAW" if view == "prices" else "Price gap against BAW",
                         xaxis_title="Volatility scenario", yaxis_title=ylabel)
    figure.update_xaxes(type="category", categoryorder="array", categoryarray=volatilities)
    return style_exercise2_figure(figure)


# PART 3 — KEEP A SINGLE VOLATILITY SLICE LARGE ENOUGH TO EXPLORE
def build_exercise2_gamma_slice(gamma_document, volatility, view="gamma", show_uncertainty=True):
    if view not in ("gamma", "variance", "errors"):
        raise ValueError("Choose gamma, variance or errors for a spot profile")
    rows = sorted((row for row in gamma_document["gamma_results"]["rows"]
                   if abs(row["volatility"] - volatility) < 1e-12), key=lambda row: row["spot"])
    if not rows:
        raise ValueError("No saved gamma cases for the selected volatility")
    figure = go.Figure()
    if view == "gamma":
        for field, label, colour, dash in (
            ("analytical_gamma", "Analytical Black–Scholes gamma", "#14243a", "dot"),
            ("european_finite_difference_gamma", "European finite-bump reference", "#7b8494", "dot"),
        ):
            figure.add_trace(go.Scatter(x=[row["spot"] for row in rows], y=[row[field] for row in rows],
                name=label, mode="lines", line={"color": colour, "dash": dash, "width": 2.5},
                visible="legendonly" if field == "european_finite_difference_gamma" else True))
    for estimator, label, colour in (
        ("plain", "Plain American estimator", PLAIN_COLOUR),
        ("controlled", "Controlled American estimator", CONTROL_COLOUR),
    ):
        field = f"{estimator}_estimator_variance" if view == "variance" else f"{estimator}_gamma"
        values = [row[field] - row["analytical_gamma"] if view == "errors" else row[field] for row in rows]
        figure.add_trace(go.Scatter(x=[row["spot"] for row in rows], y=values,
            name=label, mode="lines+markers",
            marker={"size": 10 if estimator == "plain" else 6,
                    "symbol": "circle-open" if estimator == "plain" else "diamond",
                    "color": colour, "line": {"color": colour, "width": 1.5}},
            line={"color": colour, "width": 3.5 if estimator == "plain" else 2,
                  "dash": "solid" if estimator == "plain" else "dash"},
            error_y={"type": "data", "array": [1.96 * row[f"{estimator}_standard_error"] for row in rows],
                     "visible": show_uncertainty and view != "variance"}))
    titles = {"gamma": "American put gamma", "variance": "Variance of estimated American put gamma", "errors": "American gamma − analytical gamma"}
    if view == "errors":
        figure.add_hline(y=0, line_color="#667085", line_dash="dash")
    figure.update_layout(title=f"{titles[view]} · {volatility:.0%} volatility",
                         xaxis_title="Initial spot (currency units)",
                         yaxis_title="Variance (1 / currency unit²)" if view == "variance" else "Gamma (1 / currency unit)")
    if view == "variance":
        figure.update_yaxes(rangemode="tozero", tickformat=".1e")
    return style_exercise2_figure(figure, height=540)


# PART 3 — A ROTATABLE VIEW OF THE COMPUTED SPOT / VOLATILITY GRID
GAMMA_SURFACE_LABELS = {
    "controlled_gamma": "American put gamma with European control", "plain_gamma": "American put gamma from plain LSMC",
    "analytical_gamma": "Analytical Black–Scholes gamma",
    "controlled_estimator_variance": "Controlled gamma-estimator variance",
    "plain_estimator_variance": "Plain gamma-estimator variance",
    "controlled_error": "Controlled gamma − analytical gamma",
    "plain_error": "Plain gamma − analytical gamma",
}


def build_exercise2_gamma_surface(gamma_document, metric="controlled_gamma"):
    if metric not in GAMMA_SURFACE_LABELS:
        raise ValueError("Unknown gamma surface metric")
    rows = gamma_document["gamma_results"]["rows"]
    spots = sorted({row["spot"] for row in rows})
    volatilities = sorted({row["volatility"] for row in rows})
    observations = {(row["volatility"], row["spot"]): row for row in rows}
    values = []
    for volatility in volatilities:
        values_at_volatility = []
        for spot in spots:
            row = observations.get((volatility, spot))
            value = None if row is None else (row[metric.replace("_error", "_gamma")] - row["analytical_gamma"]
                     if metric.endswith("_error") else row[metric])
            values_at_volatility.append(value)
        values.append(values_at_volatility)
    is_variance = "variance" in metric
    quantity_label = "Gamma variance" if is_variance else ("Gamma error" if metric.endswith("_error") else "Gamma")
    units = "1 / currency unit²" if is_variance else "1 / currency unit"
    figure = go.Figure(go.Surface(x=spots, y=[100 * value for value in volatilities], z=values,
        colorscale="RdBu_r" if metric.endswith("_error") else "Viridis", connectgaps=False,
        colorbar={"title": {"text": quantity_label, "font": {"size": 11}},
                  "tickfont": {"color": "#14243a", "size": 10}, "thickness": 12, "len": 0.65},
        hovertemplate="Spot %{x:.2f}<br>Volatility %{y:.0f}%<br>Value %{z:.5g}<extra></extra>"))
    style_exercise2_figure(figure, height=490)
    axis_style = {"color": "#14243a", "gridcolor": "#d7e0eb", "showbackground": True,
                  "backgroundcolor": "#f6f8fb", "tickfont": {"size": 10}}
    figure.update_layout(title=GAMMA_SURFACE_LABELS[metric], margin={"l": 15, "r": 50, "t": 50, "b": 45},
        scene={"xaxis": dict(axis_style, title={"text": "Initial spot", "font": {"size": 11}}),
               "yaxis": dict(axis_style, title={"text": "Volatility (%)", "font": {"size": 11}},
                            tickvals=[100 * value for value in volatilities]),
               "zaxis": dict(axis_style, title={"text": quantity_label, "font": {"size": 11}}), "bgcolor": "#ffffff",
               "camera": {"eye": {"x": 1.6, "y": -1.65, "z": 1.05}}},
        uirevision="gamma-surface-camera", showlegend=False)
    if is_variance:
        figure.update_layout(scene_zaxis_tickformat=".1e")
        figure.update_traces(colorbar_tickformat=".1e")
    figure.add_annotation(x=0, y=-0.04, xref="paper", yref="paper", showarrow=False,
        xanchor="left", text=f"{quantity_label} units: {units}. Surface joins saved grid points.",
        font={"size": 11, "color": "#43536a"})
    return figure


# EXPLAIN EACH PRICE CHART USING THE RESULTS THAT ARE ACTUALLY DISPLAYED
def explain_exercise2_results(document):
    payments = [record["valuation"] for record in document["pricing_results"]]
    comparisons = document["baw_comparisons"]
    case_count = len(payments)
    reduced_count = sum(row["controlled_estimator_variance"] < row["plain_estimator_variance"] for row in payments)
    factors = [row["variance_reduction_factor"] for row in payments if row["variance_reduction_factor"] is not None]
    correlations = [row["payoff_correlation"] for row in payments if row["payoff_correlation"] is not None]
    variance_conclusion = f"The control lowers price-estimator variance in {reduced_count} of {case_count} volatility cases."
    if factors and min(factors) > 1:
        variance_conclusion += f" The reduction is about {sum(factors)/len(factors):.0f}× on average, at the same evaluation path count."
    correlation_conclusion = (f"The observed correlations range from {min(correlations):.2f} to {max(correlations):.2f}. "
                             "The variance comparison shows how much precision this shared movement actually adds."
                             if correlations else "Correlation is undefined because the observed payments have zero sample variance.")
    relative_gaps = [abs(row["controlled_error"]) / row["baw_reference"] for row in comparisons]
    below_count = sum(row["controlled_error"] < 0 for row in comparisons)
    improved_count = sum(abs(row["controlled_error"]) < abs(row["plain_error"]) for row in comparisons)
    price_conclusion = f"The largest absolute controlled-price gap is {max(relative_gaps):.2%} of BAW."
    if below_count == case_count:
        price_conclusion += " All controlled estimates remain below the benchmark."
    error_conclusion = (f"The control gives a smaller absolute pricing error in {improved_count} of {case_count} cases. "
                        "Improved sampling precision can coexist with Euler and fitted-exercise errors.")
    return {
        "variance": {
            "description": "We compare the sampling variance of the estimated American put price, before and after the European-put control. Lower variance means a more precise Monte Carlo price; both estimates use the same paths.",
            "conclusion": variance_conclusion,
        },
        "correlation": {
            "description": "At each volatility, we correlate the American put payment selected by LSMC with the European put payment at maturity, across paired paths with the same Brownian shocks.",
            "conclusion": correlation_conclusion,
        },
        "prices": {
            "description": "The dotted curve is the BAW benchmark for the American put. Blue is plain LSMC; orange is the same American valuation after the European-put control. At zero rates and yields, BAW equals Black–Scholes.",
            "conclusion": price_conclusion,
        },
        "errors": {
            "description": "Each point is (estimated American put price − BAW) / BAW, expressed as a percentage. Zero means agreement; negative values mean underpricing relative to the benchmark.",
            "conclusion": error_conclusion,
        },
    }


# PART 3: EXECUTIVE GAMMA COMPARISON AND THE TWO REQUESTED PROFILES
def build_exercise2_gamma_tables(gamma_document):
    rows = pd.DataFrame(gamma_document["gamma_results"]["rows"])
    summary = []
    for volatility, cases in rows.groupby("volatility", sort=True):
        plain_variance = cases["plain_estimator_variance"].mean()
        controlled_variance = cases["controlled_estimator_variance"].mean()
        summary.append({
            "Volatility": volatility, "Spot cases": len(cases),
            "Plain mean absolute gamma error": (cases["plain_gamma"] - cases["analytical_gamma"]).abs().mean(),
            "Controlled mean absolute gamma error": (cases["controlled_gamma"] - cases["analytical_gamma"]).abs().mean(),
            "Mean plain estimator variance": plain_variance,
            "Mean controlled estimator variance": controlled_variance,
            "Ratio of mean variances": plain_variance / controlled_variance if controlled_variance > 0 else None,
        })
    bump_checks = pd.DataFrame(gamma_document["gamma_results"].get("bump_checks", []))
    return pd.DataFrame(summary), bump_checks


# EXPLAIN THE SELECTED GAMMA SLICE OR SURFACE, INCLUDING NEW SESSION RUNS
def explain_exercise2_gamma_results(gamma_document, volatility=None, metric="controlled_gamma", surface=False):
    rows = gamma_document["gamma_results"]["rows"]
    if volatility is not None:
        rows = [row for row in rows if abs(row["volatility"] - volatility) < 1e-12]
    plain_variance = sum(row["plain_estimator_variance"] for row in rows) / len(rows)
    controlled_variance = sum(row["controlled_estimator_variance"] for row in rows) / len(rows)
    estimator = "plain" if metric.startswith("plain") else "controlled"
    mean_gap = sum(abs(row[f"{estimator}_gamma"] - row["analytical_gamma"]) for row in rows) / len(rows)
    if plain_variance > 0:
        reduction = 100 * (1 - controlled_variance / plain_variance)
        variance_conclusion = f"The control {'reduces' if reduction >= 0 else 'increases'} mean gamma-estimator variance by {abs(reduction):.1f}% across the displayed points."
    else:
        variance_conclusion = "The plain observations have zero sample variance here; that alone does not establish accurate gamma."
    axes = ("Surface height shows this quantity as initial spot and volatility change, with strike fixed. "
            if surface else "Initial spot changes along the horizontal axis, with strike and volatility fixed. ")
    if metric == "analytical_gamma":
        peak = max(rows, key=lambda row: row["analytical_gamma"])
        return {"description": axes + "This is the analytical Black–Scholes gamma of the zero-rate American put, which equals its European counterpart.",
                "conclusion": f"On this grid, the reference peaks at spot {peak['spot']:g} and volatility {peak['volatility']:.0%}. Use this smooth reference to assess the numerical estimates."}
    if "variance" in metric:
        description = axes + "We measure the sampling variance of the estimated American put gamma. Lower values mean more precise gamma estimates, conditional on the fitted exercise rules and control coefficient."
        conclusion = variance_conclusion + " This excludes training variability and finite-bump error."
    elif metric.endswith("_error"):
        description = axes + "The plotted value is numerical American put gamma minus analytical Black–Scholes gamma. Zero means agreement; values below zero underestimate the reference."
        conclusion = f"Mean absolute {estimator}-gamma error is {mean_gap:.3g} per currency unit across these points. " + variance_conclusion
    else:
        description = axes + "Gamma measures the curvature of the American put price as spot changes. We calculate it from three coupled valuations at S−h, S and S+h."
        conclusion = f"Mean absolute {estimator}-gamma gap to the analytical reference is {mean_gap:.3g} per currency unit. " + variance_conclusion
    return {"description": description, "conclusion": conclusion}


# OFFLINE EXPLORER — THE SAME SAVED OBSERVATIONS, WITH SELECTABLE VIEWS

# LOCAL INTERFACE — ONE PAGE OF SAVED RESULTS, WITH USEFUL DISPLAY CONTROLS
def render_exercise2_page():
    import logging
    import streamlit as st

    result_directory = PROJECT_DIRECTORY / "results"
    try:
        saved_prices = load_exercise2_results(result_directory / "exercise2_control_variates.json")
        saved_gamma = load_exercise2_gamma_results(result_directory / "exercise2_gamma.json")
    except (OSError, ValueError, KeyError) as error:
        st.error("The saved Exercise 2 study could not be loaded. Restore the complete results/ folder from the supplied ZIP. New numerical runs are saved separately under runs/.")
        logging.getLogger(__name__).error("Exercise 2 saved results: %s", error)
        return

    def show_table(frame):
        # Here the saved values render as HTML without an Arrow conversion.
        st.markdown('<div class="summary-table-wrap">' + frame.to_html(
            index=False, classes="summary-table", border=0, float_format=lambda x: f"{x:.6g}") + '</div>',
            unsafe_allow_html=True)

    def show_plot(figure, key):
        # The explanation above the chart supplies its title, avoiding clipped text.
        figure.update_layout(title_text=None, margin_t=20)
        st.plotly_chart(figure, use_container_width=True, theme=None, key=key,
            config={"displaylogo": False, "scrollZoom": False,
                    "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2}})

    st.caption('EXERCISE 2 / MONTE CARLO PRECISION')
    st.title('Control variates, BAW & gamma')
    st.write('The European control improves American-put price precision. The BAW comparison checks the remaining pricing gap; gamma shows where the improvement is limited.')
    st.caption('Saved results · one-month American put · initial spot = strike = 40 · zero rates and yields · daily Euler simulation and exercise. Controls select saved observations; they do not run simulations.')

    # SUMMARY — ALL THREE ASSIGNMENT QUESTIONS REMAIN VISIBLE ON ONE PAGE.
    metrics = build_exercise2_executive_metrics(saved_prices, saved_gamma)
    left, middle, right = st.columns(3)
    left.metric('1 · Price variance', f"{metrics['minimum_price_variance_factor']:.1f}–{metrics['maximum_price_variance_factor']:.1f}× lower")
    middle.metric('2 · Mean price gap to BAW', f"{metrics['mean_controlled_absolute_relative_price_error']:.2%}",
        help='Mean absolute controlled-price error as a percentage of BAW across all four saved volatilities.')
    right.metric('3 · Gamma variance reduction',
        f"{100*metrics['minimum_mean_gamma_variance_reduction']:.1f}–{100*metrics['maximum_mean_gamma_variance_reduction']:.1f}%",
        help='Reduction in mean conditional variance across the spot grid, for each saved volatility.')
    st.caption('Headlines cover the complete assignment grid. The conclusions below follow the displayed selections.')

    volatilities = sorted({row['market_data']['volatility'] for row in saved_prices['pricing_results']})
    selected = st.multiselect('Volatility scenarios for the two price charts', volatilities,
        default=volatilities, format_func=lambda value: f'{value:.0%}', key='exercise2_summary_volatilities')
    if not selected:
        st.info('Choose at least one saved volatility to display the price comparisons.')
    else:
        # Here selecting scenarios filters both the observations and their interpretation.
        displayed_prices = dict(saved_prices)
        displayed_prices['pricing_results'] = [row for row in saved_prices['pricing_results']
            if row['market_data']['volatility'] in selected]
        displayed_prices['baw_comparisons'] = [row for row in saved_prices['baw_comparisons']
            if row['volatility'] in selected]
        figures = build_exercise2_figures(displayed_prices)
        notes = explain_exercise2_results(displayed_prices)
        left, right = st.columns(2)
        with left:
            st.subheader('1 · Does the control improve precision?')
            measure = st.selectbox('Price precision measure', ['Estimator variance', 'Payment correlation'],
                key='exercise2_summary_precision')
            quantity = 'variance' if measure == 'Estimator variance' else 'correlation'
            st.write(notes[quantity]['description'])
            key = 'estimator_variance' if quantity == 'variance' else 'payoff_correlation'
            show_plot(figures[key], 'exercise2_summary_precision_chart')
            st.markdown('**Conclusion.** ' + notes[quantity]['conclusion'])
        with right:
            st.subheader('2 · Do prices agree with BAW?')
            measure = st.selectbox('Benchmark comparison', ['Put prices', 'Price gap (%)'],
                key='exercise2_summary_benchmark')
            quantity = 'prices' if measure == 'Put prices' else 'errors'
            st.write(notes[quantity]['description'])
            figure = build_exercise2_price_comparison(displayed_prices, view=quantity,
                relative=quantity == 'errors', show_uncertainty=False)
            show_plot(figure, 'exercise2_summary_benchmark_chart')
            st.markdown('**Conclusion.** ' + notes[quantity]['conclusion'])
        st.caption('The four volatility scenarios are equally spaced for comparison. Their labels retain the actual 5%, 10%, 20% and 30% inputs; connecting lines do not add simulated cases.')

    # GAMMA — SWITCH THE SAME SAVED GRID BETWEEN A SURFACE AND A SPOT PROFILE.
    st.subheader('3 · Does the same improvement extend to gamma?')
    controls = st.columns(3)
    dimensions = controls[0].selectbox('Gamma view', ['3D surface', 'Spot profile'],
        key='exercise2_summary_gamma_view')
    measure = controls[1].selectbox('Gamma measure', ['Gamma', 'Estimator variance'],
        key='exercise2_summary_gamma_measure')
    view = 'gamma' if measure == 'Gamma' else 'variance'
    gamma_volatilities = sorted({row['volatility'] for row in saved_gamma['gamma_results']['rows']})
    volatility = None
    if dimensions == '3D surface':
        estimators = ['With European control', 'Plain LSMC', 'Analytical reference'] if view == 'gamma' else ['With European control', 'Plain LSMC']
        estimator = controls[2].selectbox('Gamma estimator', estimators, key='exercise2_summary_gamma_estimator')
        metric = {'gamma': {'With European control': 'controlled_gamma', 'Plain LSMC': 'plain_gamma',
                           'Analytical reference': 'analytical_gamma'},
                  'variance': {'With European control': 'controlled_estimator_variance',
                               'Plain LSMC': 'plain_estimator_variance'}}[view][estimator]
        figure = build_exercise2_gamma_surface(saved_gamma, metric)
    else:
        volatility = controls[2].selectbox('Gamma volatility', gamma_volatilities,
            index=min(range(len(gamma_volatilities)), key=lambda index: abs(gamma_volatilities[index] - .2)),
            format_func=lambda value: f'{value:.0%}', key='exercise2_summary_gamma_volatility')
        metric = 'controlled_gamma' if view == 'gamma' else 'controlled_estimator_variance'
        figure = build_exercise2_gamma_slice(saved_gamma, volatility, view, show_uncertainty=False)
        figure.update_layout(height=430)
    notes = explain_exercise2_gamma_results(saved_gamma, volatility, metric, dimensions == '3D surface')
    st.write(notes['description'])
    st.caption('Saved gamma grid · strike 40 · one month · spot 36–44 · volatility 5%, 10%, 20%, 30% · spot bump 0.20. Gamma units: 1/currency; variance units: 1/currency².')
    show_plot(figure, 'exercise2_summary_gamma_chart')
    st.markdown('**Conclusion.** ' + notes['conclusion'])
    st.caption('Gamma remains noisy: lower conditional variance alone does not establish an accurate hedge. Drag the surface to rotate; select Spot profile to compare both estimators directly with the analytical reference.')

    with st.expander('Saved inputs and values'):
        record = saved_prices['pricing_results'][0]
        st.write(f"Each case uses {record['training_paths']:,} training paths and {record['evaluation_paths']:,} new evaluation paths. The maturity, strike and rates stay fixed; only the saved volatility and spot grid can be selected here.")
        st.caption('Sampling variance conditions on the estimated exercise decisions and control coefficient. It excludes their training variability and numerical approximation error.')
        controls, benchmarks = build_exercise2_tables(saved_prices)
        show_table(controls)
        show_table(benchmarks)
        st.download_button('Download saved price comparisons · CSV', benchmarks.to_csv(index=False),
            file_name='exercise2_price_comparisons.csv', mime='text/csv', key='exercise2_summary_csv')
    report = PROJECT_DIRECTORY / 'reports/ExxonMobil_Quantitative_Modelling_Assignment.pdf'
    if report.exists():
        st.download_button('Read the complete report · PDF', report.read_bytes(),
            file_name=report.name, mime='application/pdf', key='exercise2_report_download')


