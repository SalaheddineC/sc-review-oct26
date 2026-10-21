"""Interactive views of the four exercises and their saved numerical results.

Launch: python -m streamlit run app.py
The interface explores saved results. Numerical studies remain available through the exercise runners.
"""
import pandas as pd
import streamlit as st

from exercise1.ui_exercise1 import (
    PARAMETER_NAMES, load_saved_results, valuation_table, table1_comparison_table,
    benchmark_summary, table1_figure, pricing_figure, timestep_figure, sensitivity_figure,
)


def show_chart(figure, name):
    compact_chart(figure)
    figure.update_xaxes(color="#172b45", gridcolor="#e5eaf0")
    figure.update_yaxes(color="#172b45", gridcolor="#e5eaf0")
    st.plotly_chart(figure, use_container_width=True, theme=None, key=name,
        config={"displaylogo": False, "scrollZoom": False,
                "toImageButtonOptions": {"format": "png", "filename": name, "scale": 2}})


def select_pricing_comparison(table, key, stochastic_volatility=False):
    """Vary one economic input and hold the other inputs fixed."""
    variables = {"Strike": "strike", "Maturity (years)": "maturity_years"}
    variance_label = "Long-run variance θ" if stochastic_volatility else "Volatility"
    variables[variance_label] = "long_term_variance" if stochastic_volatility else "volatility"
    selected_label = st.selectbox("Which input do we vary?", list(variables), key=f"{key}_vary")
    variable = variables[selected_label]
    selected = table.copy()
    control_columns = st.columns(2)
    fixed_inputs = [(label, column) for label, column in variables.items() if column != variable]
    for control, (label, column) in zip(control_columns, fixed_inputs):
        values = sorted(table[column].unique().tolist())
        default = 0 if column in ("maturity_years", "volatility") else len(values) // 2
        display_value = (lambda value: f"{value:.0%}") if column == "volatility" else (lambda value: f"{value:g}")
        chosen = control.selectbox(f"Fixed {label.lower()}", values, index=default,
                                   format_func=display_value, key=f"{key}_{column}")
        selected = selected[selected[column] == chosen]
    return selected, variable, selected_label


# Application layout and navigation.

st.set_page_config(page_title="ExxonMobil - Quantitative Modelling Assignment", page_icon="📈", layout="wide")
st.markdown("""<style>
.stApp {background: #ffffff; color: #172b45; color-scheme: light;}
[data-testid="stHeader"] {background: #ffffff;}
[data-testid="stSidebar"] {background: #f2f5f9; color: #172b45; border-right: 1px solid #dfe6ef;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stExpander"] {color: #172b45;}
[data-testid="stMarkdownContainer"] :is(p,h1,h2,h3,h4,li),
[data-testid="stCaptionContainer"] p, [data-testid="stWidgetLabel"] p {color: #172b45;}
[data-testid="stExpander"] summary {color: #172b45; background: #ffffff;}
[data-baseweb="select"] > div, [data-baseweb="input"], input {
    background-color: #ffffff !important; color: #172b45 !important;}
[role="listbox"], [role="option"] {background: #ffffff !important; color: #172b45 !important;}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    background: #ffffff; color: #172b45; border: 1px solid #ccd6e3;}
[data-testid="stButton"] button[kind="primary"] {background: #cf2434; color: #ffffff;}
[data-testid="stButton"] button[kind="primary"] p {color: #ffffff !important;}
[data-testid="stAlert"] {color: #172b45;}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label {color: #172b45;}
input:disabled {color: #53657d !important;}
.katex {color: #172b45;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"], [data-testid="stMetric"], .project-hero, .project-result {
    font-family: Arial, sans-serif;}

.block-container {max-width: 1440px; padding-top: 4.25rem; padding-bottom: 3rem;}
h1 {font-size: 1.8rem !important; letter-spacing: -.025em; color: #18324f;}
h2 {font-size: 1.3rem !important; color: #18324f;}
h3 {font-size: 1.1rem !important; color: #18324f;}
[data-baseweb="tab-panel"] {background: white; border: 1px solid #e3e9f1; border-radius: 12px; padding: 24px;}
[data-testid="stMetric"] {background: #f5f8fe; border: 1px solid #dce5f4; border-top: 4px solid #2463c5; border-radius: 9px; padding: 12px 16px;}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: #193d68;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
[data-baseweb="tab-list"] {gap: 7px; padding: 8px 0 14px;}
button[data-baseweb="tab"] {height: auto; min-height: 46px; padding: 10px 13px; border-radius: 7px; font-size: .90rem; font-weight: 600;}
button[data-baseweb="tab"]:nth-child(1) {background: #eaf2ff; color: #2155a3;}
button[data-baseweb="tab"]:nth-child(2) {background: #e6f5ee; color: #117354;}
button[data-baseweb="tab"]:nth-child(3) {background: #e9f6f8; color: #117387;}
button[data-baseweb="tab"]:nth-child(4) {background: #fff2df; color: #99570b;}
button[data-baseweb="tab"]:nth-child(5) {background: #f1eafa; color: #71439c;}
button[data-baseweb="tab"][aria-selected="true"] {box-shadow: inset 0 -3px currentColor;}
.summary-table-wrap {overflow-x: auto; border: 1px solid #dce4ee; border-radius: 10px; margin: .6rem 0 1rem; box-shadow: 0 2px 8px #18324f08;}
.summary-table {width: 100%; border-collapse: collapse; margin: 0; color: #26394b; font-size: .91rem; line-height: 1.48;}
.summary-table th {background: #18324f; color: white; text-align: left; padding: 13px 16px;}
.summary-table td {padding: 13px 16px; border-bottom: 1px solid #e5ebf2; vertical-align: top;}
.summary-table tr:nth-child(even) td {background: #f4f7fb;}
.summary-table tr:last-child td {border-bottom: 0;}
.summary-table td:first-child {font-weight: 600; color: #18324f;}
.summary-table td:last-child {font-weight: 600;}
.summary-table .qualified-result td:last-child {background: #fff4e2; color: #80530f;}
.project-brand {font: 700 14px/1.5 Arial, sans-serif; color: #d6312c; margin: 0 0 8px;}
.project-hero {background: linear-gradient(115deg, #101f37, #193b58); border-radius: 4px; padding: 16px 24px; margin-bottom: 12px; border-top: 3px solid #d6312c;}
.project-hero .eyebrow {color: #bfcfdf; font-size: 10px; letter-spacing: .13em; text-transform: uppercase; margin-bottom: 7px;}
.project-hero h1 {color: #ffffff !important; font-size: 1.65rem !important; font-weight: 400; line-height: 1.2; margin: 0 0 9px; padding: 0; max-width: 900px;}
.project-hero p {color: #e3ebf4 !important; font-size: 13px; line-height: 1.5; margin: 0; max-width: 880px;}
.project-results {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    border-top: 2px solid #d6312c; border-bottom: 1px solid #dce5ef;
    background: #f5f8fc; margin: 0 0 10px;}
.project-result {padding: 8px 14px; min-width: 0;}
.project-result + .project-result {border-left: 1px solid #dce5ef;}
.project-result .headline {display: flex; align-items: baseline; justify-content: space-between; gap: 10px;}
.project-result .label {font-size: 10px; font-weight: 600; color: #50627a; line-height: 1.35;}
.project-result .number {font-size: 18px; font-weight: 700; letter-spacing: -.025em; line-height: 1.2; color: #153853; white-space: nowrap;}
.project-result p {font-size: 11px; line-height: 1.35; color: #50627a !important; margin: 3px 0 0;}
.project-section {color: #d6312c; font-size: 10px; font-weight: 700; letter-spacing: .13em; margin-top: 4px;}
[data-testid="stSidebar"] .sidebar-brand {font-size: 16px; font-weight: 700; line-height: 1.5; color: #d6312c; margin: 9px 0 6px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding: 7px 0;}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {font-size: 13px; line-height: 1.5;}
@media (max-width: 760px) {
    [data-baseweb="tab-panel"] {padding: 14px;}
    .summary-table {font-size: .82rem;}
    .summary-table th, .summary-table td {padding: 10px;}
    h1 {font-size: 1.75rem !important;}
    .project-hero {padding: 16px 18px;}
    .project-hero h1 {font-size: 1.5rem !important;}
    .project-results {grid-template-columns: 1fr;}
    .project-result {padding: 7px 12px;}
    .project-result + .project-result {border-left: 0; border-top: 1px solid #dce5ef;}
}
</style>""", unsafe_allow_html=True)

def compact_chart(figure, height=350):
    """Apply consistent axes, typography and legend placement."""
    figure.update_layout(height=height, title=None, paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#172b45"},
        margin={"t": 65, "l": 55, "r": 18, "b": 65},
        legend={"orientation": "h", "x": 0, "y": 1.04, "yanchor": "bottom", "font": {"size": 11}})
    figure.update_xaxes(title_font_size=12, tickfont_size=11, automargin=True, title_standoff=12,
                        color="#172b45", gridcolor="#e5eaf0")
    figure.update_yaxes(title_font_size=12, tickfont_size=11, automargin=True, title_standoff=12,
                        color="#172b45", gridcolor="#e5eaf0")
    figure.update_annotations(font_size=12)
    return figure


with st.sidebar:
    st.markdown('<div class="sidebar-brand">ExxonMobil<br>Quantitative Modelling Assignment</div>', unsafe_allow_html=True)
    st.caption("Quantitative Risk Modelling")
    st.divider()
    selected_exercise = st.radio("Explore the project", (
        "Project overview", "Exercise 1 · American valuation", "Exercise 2 · Control variate & gamma",
        "Exercise 3 · Euler & Milstein", "Exercise 4 · Gas storage"), index=0, key="project_navigation")
    descriptions = {
        "Project overview": "One implementation, from option valuation to daily storage decisions.",
        "Exercise 1 · American valuation": "Simulate prices, estimate continuation and test exercise decisions.",
        "Exercise 2 · Control variate & gamma": "Reduce sampling noise and compare prices and gamma with known references.",
        "Exercise 3 · Euler & Milstein": "Compare path accuracy, valuation results and computational cost.",
        "Exercise 4 · Gas storage": "Replay physical trades, inspect constraints and compare with an independent numerical reference.",
    }
    st.caption(descriptions[selected_exercise])
    st.divider()
    st.caption("The controls explore completed calculations. The report contains the mathematics, assumptions and detailed evidence.")

if selected_exercise == "Project overview":
    from pathlib import Path
    from exercise2.ui_exercise2 import load_exercise2_results
    from exercise3.ui_exercise3 import load_exercise3_results
    from exercise4.ui_exercise4 import load_exercise4_results, render_storage_replay
    from exercise4.gas_storage_benchmark import load_storage_benchmark

    st.markdown('<div class="project-brand">ExxonMobil - Quantitative Modelling Assignment</div>', unsafe_allow_html=True)
    st.markdown("""<div class="project-hero"><div class="eyebrow">Quantitative Risk Modelling</div>
        <h1>Explore the gas-storage tool we built</h1>
        <p>We connect simulated prices, regression and operating constraints to daily gas trades.
        The option results establish the methods. Choose a saved run below to follow the decisions.</p>
        </div>""", unsafe_allow_html=True)

    project = Path(__file__).resolve().parent
    overview_studies, overview_validation = load_saved_results()
    summary = benchmark_summary(table1_comparison_table(overview_studies["table1"]))
    second_results = load_exercise2_results(project / "results/exercise2_control_variates.json")
    factors = [row["valuation"]["variance_reduction_factor"] for row in second_results["pricing_results"]]
    third_results = load_exercise3_results(project / "results/exercise3_results.json")
    order = next(row["observed_order"] for row in third_results["gbm_convergence"]["orders"]
                 if row["scheme"] == "Milstein" and row["volatility"] == .2)
    headlines = [
        ("01 / AMERICAN OPTION VALUATION", f"{summary['mean_relative_gap']:.2%}",
         "Mean absolute relative gap across 20 published price cases."),
        ("02 / CONTROL VARIATES", f"{sum(factors)/len(factors):.1f}×",
         "Mean reduction in price-estimator variance across four volatilities."),
        ("03 / EULER & MILSTEIN", f"{order:.2f}",
         "Milstein strong order at 20% volatility; Euler is about 0.5."),
    ]
    results_strip = ''.join(
        f'<div class="project-result"><div class="headline"><span class="label">{label}</span>'
        f'<span class="number">{value}</span></div><p>{explanation}</p></div>'
        for label, value, explanation in headlines)
    st.markdown(f'<div class="project-results" role="region" aria-label="Exercise results">'
                f'{results_strip}</div>', unsafe_allow_html=True)

    try:
        storage_results, storage_arrays = load_exercise4_results()
        render_storage_replay(storage_results, storage_arrays)
        benchmark = load_storage_benchmark()
        high = next(row for row in storage_results["main_cases"] if row["case"] == "high")
        loss = max(row["reference"]["policy_value_gap_percent"] for row in benchmark["cases"])
        st.caption(f"What this shows: daily decisions balance cashflow now against gas retained for later. "
                   f"The high-volatility contract value is €{high['evaluation_value']/1e6:.2f}m. "
                   f"Our decisions lose about {loss:.2f}% against the independent numerical reference in both cases. "
                   "These are model results; a single replayed path is an illustration, not the average value.")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"The saved storage evidence cannot be verified: {error}")

    st.divider()
    st.subheader("One project, with clear responsibilities")
    st.write("Market data and assumptions feed the product and model. Simulation generates paths; valuation uses those paths "
             "to estimate continuation and choose feasible actions. Each exercise has its own study and presentation, while "
             "sharing the same core components. We can change a simulation method without rewriting the product or the exercise logic.")
    st.caption("Open the exercise pages in the sidebar for the questions, implemented methods and saved comparisons. "
               "The storage replay reads saved calculations, so exploring it does not launch a new simulation.")
    st.stop()

if selected_exercise == "Exercise 4 · Gas storage":
    from exercise4.ui_exercise4 import render_exercise4_page
    render_exercise4_page()
    st.stop()
if selected_exercise == "Exercise 3 · Euler & Milstein":
    from exercise3.ui_exercise3 import render_exercise3_page
    render_exercise3_page()
    st.stop()
if selected_exercise == "Exercise 2 · Control variate & gamma":
    from exercise2.ui_exercise2 import render_exercise2_page
    render_exercise2_page()
    st.stop()

# Exercise 1: published benchmarks and economic sensitivity comparisons.
studies, validation = load_saved_results()
benchmark_cases = table1_comparison_table(studies["table1"])
benchmark_results = benchmark_summary(benchmark_cases)
gbm_prices = valuation_table(studies["gbm_pricing"])
gbm_steps = table1_comparison_table(studies["gbm_timestep"])
paired_run = studies["regression_diagnostics"]["paired_sensitivities"]
paired = pd.DataFrame(paired_run["comparisons"])
st.caption("EXERCISE 1 / AMERICAN OPTION VALUATION")
st.title("American puts · results and comparisons")
st.write("Euler–Maruyama simulation and Longstaff–Schwartz exercise decisions. "
         "The six modules below cover the published benchmark, pricing inputs, daily-to-weekly steps and the stochastic-volatility repeats.")
first, second, third = st.columns(3)
first.metric("Mean gap from published prices", f"{benchmark_results['mean_relative_gap']:.2%}")
second.metric("GBM economic comparisons", "24 / 24", help="All matched comparisons have the expected direction in the saved study.")
third.metric("Stochastic-volatility pricing comparisons", "45 / 45", help="Economic consistency checks, not an independent price benchmark.")
st.caption("Controls select completed calculations. Bars show ±2 evaluation standard errors; uncertainty in fitted exercise decisions and numerical bias are additional.")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Published price benchmark")
    st.write("Compare our American put prices with the article’s Table 1 while spot varies. K = 40, rate = 6%, zero yield; daily Euler steps.")
    a, b = st.columns(2)
    vol = a.selectbox("Volatility", [.2, .4], format_func=lambda v: f"{v:.0%}", key="benchmark_volatility")
    maturity = b.selectbox("Maturity (years)", [1., 2.], key="benchmark_maturity")
    selected = benchmark_cases[(benchmark_cases.volatility == vol) & (benchmark_cases.maturity_years == maturity)]
    figure = table1_figure(selected, True)
    for trace, name in zip(figure.data, ["Published prices", "Our prices"]): trace.name = name
    show_chart(figure, "exercise1_table1")
    result = benchmark_summary(selected)
    st.caption(f"Conclusion: for these five spots, the mean absolute relative gap is {result['mean_relative_gap']:.2%}; "
               f"the largest absolute gap is {result['largest_absolute_gap']:.4f} price units. The curves agree closely.")
with right:
    st.subheader("Strike, volatility and maturity")
    st.write("Change one input and hold the other two fixed. Initial spot = 40, rate = 6%, zero yield; daily Euler steps.")
    selected, variable, label = select_pricing_comparison(gbm_prices, "gbm_pricing")
    show_chart(pricing_figure(selected, variable, label), "exercise1_gbm_pricing")
    explanations = {"strike": "A higher strike raises the put value in these cases.", "volatility": "Higher volatility raises the put value in these cases.", "maturity_years": "A longer maturity raises the put value by adding exercise opportunities."}
    st.caption("Conclusion: " + explanations[variable])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Daily to weekly Euler steps")
    st.write("Price difference from the article’s finite-difference reference. Exercise dates stay fixed; the selected number is the maximum simulation step.")
    a, b, c = st.columns(3)
    vol = a.selectbox("Volatility", [.2, .4], format_func=lambda v: f"{v:.0%}", key="step_volatility")
    spot = b.selectbox("Initial spot", [36., 38., 40., 42., 44.], index=2, key="step_spot")
    maturity = c.selectbox("Maturity (years)", [1., 2.], key="step_maturity")
    selected = gbm_steps[(gbm_steps.volatility == vol) & (gbm_steps.spot == spot) & (gbm_steps.maturity_years == maturity)]
    figure = timestep_figure(selected)
    figure.update_yaxes(title="Price difference from reference")
    figure.update_xaxes(title="Maximum Euler step (days)")
    figure.update_layout(showlegend=False)
    show_chart(figure, "exercise1_gbm_timestep")
    st.caption("Conclusion: gaps remain small across the tested steps. Sampling and exercise approximation obscure a consistent improvement with finer steps; one closer point does not establish convergence.")
with right:
    st.subheader("Stochastic-volatility parameter effects")
    st.write("Change one model parameter and compare the put value with the saved baseline, using common random draws. S = K = 1, maturity = 1 year, zero rates.")
    parameter = st.selectbox("Parameter to vary", list(PARAMETER_NAMES), format_func=PARAMETER_NAMES.get, key="sv_parameter")
    selected = paired[paired.parameter == parameter]
    figure = sensitivity_figure(selected, parameter)
    figure.update_layout(showlegend=False)
    figure.update_yaxes(title="Put value change from baseline")
    show_chart(figure, "exercise1_sv_parameter")
    interpretations = {
        "long_term_variance": "Higher long-run variance clearly increases the put value here.",
        "mean_reversion_speed": "Faster reversion produces a small increase for this baseline; the direction is model- and starting-state dependent.",
        "volatility_of_variance": "Higher volatility of variance produces a small decrease here; it does not simply act like increasing constant volatility.",
        "brownian_correlation": "Across the tested positive correlations, changes are too small relative to sampling uncertainty to establish a direction."}
    st.caption("Conclusion: " + interpretations[parameter])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Repeat pricing with stochastic volatility")
    st.write("Repeat the strike, maturity and variance-level comparisons when variance changes randomly. Initial spot = 1; initial variance = 0.0064; zero rates.")
    sv_prices = valuation_table(studies["sv_pricing"])
    selected, variable, label = select_pricing_comparison(sv_prices, "sv_pricing", stochastic_volatility=True)
    show_chart(pricing_figure(selected, variable, label), "exercise1_sv_pricing")
    st.caption("Conclusion: all matched comparisons increase with strike, maturity and long-run variance in this study. These check economic consistency; no independent price reference is supplied for this model.")
with right:
    st.subheader("Repeat timestep checks with stochastic volatility")
    st.write("Compare the same contract across four Euler steps. The dotted line is its one-day estimate, used as a comparison rather than an exact price.")
    a, b, c = st.columns(3)
    strike = a.selectbox("Strike", [.9, 1., 1.1], index=1, key="sv_step_strike")
    maturity = b.selectbox("Maturity (years)", [1., 2.], key="sv_step_maturity")
    variance = c.selectbox("Long-run variance", [.0016, .0064, .0256], index=1, key="sv_step_variance")
    table = valuation_table(studies["sv_timestep"])
    selected = table[(table.strike == strike) & (table.maturity_years == maturity) & (table.long_term_variance == variance)]
    figure = timestep_figure(selected, stochastic_volatility=True)
    figure.update_xaxes(title="Maximum Euler step (days)")
    figure.update_layout(showlegend=False)
    show_chart(figure, "exercise1_sv_timestep")
    st.caption("Conclusion: across all 18 contracts, 17 daily-to-weekly changes lie within two evaluation sampling errors. Smooth convergence is not resolved at this sampling budget.")
st.caption("The report contains the mathematical basis, input choices and limitations. Selecting these graphs never launches a simulation.")
