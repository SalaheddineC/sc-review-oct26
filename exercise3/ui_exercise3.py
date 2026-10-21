"""EXERCISE 3: EULER–MILSTEIN RESULTS, TABLES AND INTERACTIVE PRESENTATION.

Here the application reads the saved numerical observations.
Numerical methods remain in the simulator, valuation layer and exercise runner.
"""
import html
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from results_integrity import load_result_json

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent


SCHEME_COLOURS = {"Euler": "#1464a5", "Milstein": "#b64600"}
TRACE_STYLES = {
    ("Euler", "plain"): ("#1464a5", "solid", "circle-open"),
    ("Euler", "controlled"): ("#1464a5", "dash", "diamond-open"),
    ("Milstein", "plain"): ("#b64600", "solid", "square-open"),
    ("Milstein", "controlled"): ("#b64600", "dash", "diamond"),
}


# LOAD THE PUBLISHED EXPERIMENT AND VERIFY ITS DATA INTEGRITY

def load_exercise3_results(results_path):
    document = load_result_json(results_path, exercise=3, required=("code_sha256", "config", "environment"))
    if document.get("exercise") != 3 or not document.get("gbm_prices"):
        raise ValueError("Expected saved Exercise 3 pricing results")
    return document


# READABLE CHARTS — COLOUR IDENTIFIES THE SCHEME, DASHES IDENTIFY THE CONTROL

def style_exercise3_figure(figure, title, x_title=None, y_title=None, height=450):
    figure.update_layout(template="plotly_white",
        title={"text": title, "font": {"size": 16}, "x": 0.02}, height=height,
        paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#172b45"},
        margin={"l": 66, "r": 25, "t": 62, "b": 118},
        legend={"orientation": "h", "y": -0.24, "yanchor": "top",
                "font": {"size": 11}, "entrywidth": 0.5, "entrywidthmode": "fraction"},
        hovermode="x unified", hoverlabel={"bgcolor": "white", "font_color": "#172b45"})
    figure.update_xaxes(title_text=x_title, title_font_size=12, tickfont_size=11,
                        automargin=True, color="#172b45", gridcolor="#e4eaf1")
    figure.update_yaxes(title_text=y_title, title_font_size=12, tickfont_size=11,
                        automargin=True, color="#172b45", gridcolor="#e4eaf1")
    figure.update_annotations(font_size=11)
    return figure


def add_estimator_trace(figure, rows, x_key, value_key, scheme, estimator, error_key=None):
    colour, dash, symbol = TRACE_STYLES[(scheme, estimator)]
    label = f"{scheme} · {'European-controlled' if estimator == 'controlled' else 'plain'}"
    figure.add_trace(go.Scatter(x=[row[x_key] for row in rows],
        y=[row[value_key] for row in rows], name=label, mode="lines+markers",
        line={"color": colour, "width": 2.7, "dash": dash}, marker={"symbol": symbol, "size": 9},
        error_y={"type": "data", "array": [1.96*row[error_key] for row in rows]} if error_key else None))


# SUMMARY — HEADLINES USE THE MAXIMUM COMPLETED PATH BUDGET

def build_exercise3_summary(document):
    prices = document["gbm_prices"]
    largest_count = max(row["evaluation_paths"] for row in prices)
    full_rows = [row for row in prices if row["evaluation_paths"] == largest_count]
    summaries = []
    for scheme in ("Euler", "Milstein"):
        selected = [row for row in full_rows if row["scheme"] == scheme]
        if selected:
            summaries.append({"Scheme": scheme, "Evaluation paths": largest_count,
                "Mean absolute plain error / BAW": sum(abs(row["plain_error"] / row["baw_reference"]) for row in selected)/len(selected),
                "Mean absolute controlled error / BAW": sum(abs(row["controlled_error"] / row["baw_reference"]) for row in selected)/len(selected),
                "Mean price-variance reduction factor": sum(row["plain_estimator_variance"]/row["controlled_estimator_variance"] for row in selected)/len(selected)})
    return pd.DataFrame(summaries)


def build_exercise3_result_cards(document):
    summary=build_exercise3_summary(document)
    factor=float(summary["Mean price-variance reduction factor"].mean())
    rows=document["sv_prices"]
    count=max(row["evaluation_paths"] for row in rows)
    latest={row["scheme"]:row for row in rows if row["evaluation_paths"]==count}
    euler,milstein=latest["Euler"],latest["Milstein"]
    cost=milstein["runtime_seconds"]/euler["runtime_seconds"]
    farther=abs(milstein["same_model_error"])>=abs(euler["same_model_error"])
    precision="roughly half" if round(factor)==4 else f"about {1/math.sqrt(factor):.0%}"
    return [
        {"label":"PRICE PRECISION", "headline":f"About {factor:.0f}× lower variance",
         "text":f"The European control reduces GBM price-estimator variance. Standard error is {precision} of the plain estimator's; the gain in precision does not imply the same gain in pricing accuracy."},
        {"label":"GBM CONVERGENCE", "headline":"Better paths; pricing gains less clear",
         "text":"Milstein gives the expected improvement in the original terminal-path study. The daily American pricing comparisons show a less clear benefit."},
        {"label":"STOCHASTIC VOLATILITY", "headline":"Similar prices, more computation",
         "text":f"Milstein costs about {cost:.0f}× as much in this run. Its gap to the same-model reference is {'larger, so no pricing improvement is demonstrated' if farther else 'smaller in this sample; this is an observed result, not a convergence guarantee'}."},
    ]


def result_cards_html(document):
    cards=build_exercise3_result_cards(document)
    blocks="".join(f'<div class="card"><span>{html.escape(card["label"])}</span><strong>{html.escape(card["headline"])}</strong><p>{html.escape(card["text"])}</p></div>' for card in cards)
    return '<div class="ex3-cards">'+blocks+'</div>'


# PAIRED COMPARISONS — USE THE DIFFERENCE ALREADY CALCULATED ON SHARED PATHS

def build_exercise3_paired_figure(document, model="gbm", volatility=0.2):
    rows=document["gbm_prices"] if model=="gbm" else document["sv_prices"]
    selected=sorted((row for row in rows if row["scheme"]=="Milstein" and
        (model=="sv" or abs(row["volatility"]-volatility)<1e-12)),key=lambda row:row["evaluation_paths"])
    figure=go.Figure(go.Scatter(x=[row["evaluation_paths"] for row in selected],
        y=[row["milstein_minus_euler_plain_price"] for row in selected],
        error_y={"type":"data","array":[1.96*row["paired_scheme_difference_standard_error"] for row in selected]},
        mode="lines+markers",name="Milstein − Euler · plain American price",
        line={"color":"#6b4394","width":3},marker={"symbol":"diamond","size":9}))
    figure.add_hline(y=0,line_color="#7f8d9d",line_dash="dot")
    counts=sorted({row["evaluation_paths"] for row in selected})
    figure.update_xaxes(tickmode="array",tickvals=counts,
        ticktext=[f"{count:,}" for count in counts])
    return style_exercise3_figure(figure,"Does changing the scheme change the American price?",
        "Independent evaluation paths","Milstein minus Euler price (spot units)")


def explain_exercise3_paired(document, model="gbm", volatility=0.2):
    rows=document["gbm_prices"] if model=="gbm" else document["sv_prices"]
    selected=[row for row in rows if row["scheme"]=="Milstein" and
        (model=="sv" or abs(row["volatility"]-volatility)<1e-12)]
    row=max(selected,key=lambda item:item["evaluation_paths"])
    difference=row["milstein_minus_euler_plain_price"]
    half_interval=1.96*row["paired_scheme_difference_standard_error"]
    resolved=abs(difference)>half_interval
    direction="lower" if difference<0 else "higher"
    interpretation=(f"Milstein gives a {direction} plain price in this sample; the paired interval excludes zero" if resolved
        else "The paired interval includes zero, so this sample does not resolve a change in the plain price")
    return {"description":"We subtract the Euler payment from the Milstein payment on each shared Brownian scenario, then average those differences. Bars use ±1.96 times the standard error of that paired mean, retaining the covariance gained by sharing paths.",
        "conclusion":f"{interpretation}. At {row['evaluation_paths']:,} paths, the difference is about {difference:+.2g} ± {half_interval:.2g} spot units. A price change is not automatically a pricing improvement; training variability is excluded."}


# GBM — SAME AMERICAN PUT, FOUR ESTIMATORS, ONE EXACT ZERO-RATE BENCHMARK

def build_exercise3_price_figure(document, model="gbm", volatility=0.2,
                                view="errors", reference="baw", show_uncertainty=True):
    rows = document["gbm_prices"] if model == "gbm" else document.get("sv_prices", [])
    if model == "gbm":
        rows = [row for row in rows if abs(row["volatility"]-volatility)<1e-12]
    figure = go.Figure()
    for scheme in ("Euler", "Milstein"):
        selected = sorted((row for row in rows if row["scheme"] == scheme), key=lambda row: row["evaluation_paths"])
        for estimator in (("plain", "controlled") if model == "gbm" else ("plain",)):
            if view == "variance":
                key = estimator + "_estimator_variance"
            elif view == "prices":
                key = estimator + "_price"
            else:
                key = "same_model_error" if model == "sv" and reference == "same_model" else estimator + "_error"
            error_key = estimator + "_standard_error" if show_uncertainty and view != "variance" else None
            if model=="sv" and reference=="same_model" and view=="errors" and show_uncertainty:
                error_key="same_model_gap_standard_error"
            add_estimator_trace(figure, selected, "evaluation_paths", key, scheme, estimator, error_key)
    if rows and view == "prices":
        reference_key = "same_model_reference" if reference == "same_model" else "baw_reference"
        label = "Refined same-model terminal estimate" if reference == "same_model" else ("BAW = exact Black–Scholes" if model == "gbm" else "BAW · different-model 8% proxy")
        figure.add_hline(y=rows[0][reference_key], line_dash="dot", line_color="#172b45", annotation_text=label)
    if view == "errors":
        figure.add_hline(y=0, line_color="#7f8d9d", line_dash="dot")
    if view == "variance":
        figure.update_yaxes(type="log")
    counts=sorted({row["evaluation_paths"] for row in rows})
    figure.update_xaxes(tickmode="array",tickvals=counts,
        ticktext=[f"{count:,}" for count in counts])
    title = {"prices": "American put prices", "errors": "Reference minus American put estimate", "variance": "Variance of the estimated American put price"}[view]
    unit = "spot-price units² · log scale" if view == "variance" else "spot-price units"
    return style_exercise3_figure(figure, title, "Independent evaluation paths", unit)


def build_exercise3_correlation_figure(document):
    rows=document["gbm_prices"]
    largest_count=max(row["evaluation_paths"] for row in rows)
    figure=go.Figure()
    for scheme in ("Euler","Milstein"):
        selected=sorted((row for row in rows if row["scheme"]==scheme and row["evaluation_paths"]==largest_count),key=lambda row:row["volatility"])
        add_estimator_trace(figure,selected,"volatility","payoff_correlation",scheme,"plain")
    style_exercise3_figure(figure,"American / European paired-payment correlation","Lognormal volatility","Correlation across paired payments")
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(range=[0,1.02])
    return figure


def explain_exercise3_correlation(document):
    rows=document["gbm_prices"]
    largest_count=max(row["evaluation_paths"] for row in rows)
    values=[row["payoff_correlation"] for row in rows if row["evaluation_paths"]==largest_count and row["payoff_correlation"] is not None]
    return {"description":"At each volatility, correlation is measured across paired American-policy and exact-GBM European payments. It is not the correlation between average prices at different volatilities.",
        "conclusion":f"At {largest_count:,} evaluation paths, payment correlation ranges from {min(values):.3f} to {max(values):.3f}. This explains the control's opportunity to reduce sampling variance; its realised effect is measured separately."}


def explain_exercise3_prices(document, model="gbm", volatility=None, view="errors", reference="baw"):
    rows = document["gbm_prices"] if model == "gbm" else document.get("sv_prices", [])
    if volatility is not None and model == "gbm":
        rows = [row for row in rows if abs(row["volatility"]-volatility)<1e-12]
    largest_count = max(row["evaluation_paths"] for row in rows)
    selected = [row for row in rows if row["evaluation_paths"] == largest_count]
    descriptions = {
        "prices": "Each curve values the same American put. More evaluation paths average more realised payments under the same fitted exercise rules.",
        "errors": "A positive value means the estimate is below the reference. Bars show ±1.96 conditional evaluation standard errors; the fitted rules and control coefficient are held fixed.",
        "variance": "This is the variance of the estimated mean American put price, not the dispersion of individual option payments. Lower values mean greater conditional sampling precision.",
    }
    if model=="sv" and reference=="same_model" and view=="errors":
        descriptions[view]="A positive value means the LSMC estimate is below the refined terminal reference. Bars combine independent evaluation and reference sampling uncertainty (±1.96 SE); they exclude training variability and numerical bias."
    if model == "gbm":
        if view == "variance":
            factors = [row["plain_estimator_variance"]/row["controlled_estimator_variance"] for row in selected]
            conclusion = f"At {largest_count:,} evaluation paths, the control reduces price-estimator variance by factors of {min(factors):.2f}–{max(factors):.2f} in the displayed cases. This does not remove simulation or exercise-rule error."
        else:
            errors = [abs(row["controlled_error"])/row["baw_reference"] for row in selected]
            outside=sorted({row["volatility"] for row in selected if abs(row["controlled_error"])>1.96*row["controlled_standard_error"]})
            discrepancy=(" At "+", ".join(f"{value:.0%}" for value in outside)+" volatility, at least one controlled benchmark gap exceeds its approximate 1.96 evaluation-SE interval. Better simulation and lower sampling variance have not eliminated that discrepancy; training variability is excluded and its cause is not isolated."
                if outside else " These displayed controlled gaps lie within approximate evaluation-only intervals; this does not establish absence of bias.")
            conclusion = f"At {largest_count:,} paths, controlled estimates differ from the zero-rate benchmark by {min(errors):.2%}–{max(errors):.2%}.{discrepancy}"
    else:
        key = "same_model_error" if reference == "same_model" else "plain_error"
        values = "; ".join(f"{row['scheme']}: {row[key]:+.5g}" for row in selected)
        limit = "The refined reference has its own sampling and discretisation uncertainty." if reference == "same_model" else "BAW uses constant 8% volatility; these differences also contain model mismatch."
        conclusion = f"At {largest_count:,} paths, reference-minus-estimate is {values}. {limit}"
    return {"description": descriptions[view], "conclusion": conclusion}


# CONVERGENCE — EXACT GBM TERMINAL SPOTS ISOLATE THE SIMULATION ERROR

def build_exercise3_convergence_figure(document, volatility=0.2, view="error"):
    rows = [row for row in document.get("gbm_convergence", {}).get("rows", []) if abs(row["volatility"]-volatility)<1e-12]
    figure = go.Figure()
    key = "terminal_spot_rmse" if view == "error" else "runtime_seconds"
    for scheme in ("Euler", "Milstein"):
        selected = sorted((row for row in rows if row["scheme"] == scheme), key=lambda row: row["time_step_days"])
        add_estimator_trace(figure, selected, "time_step_days", key, scheme, "plain")
    style_exercise3_figure(figure, "Terminal-spot error against exact GBM" if view == "error" else "Cost of the terminal-spot diagnostic",
        "Maximum simulation step (days)", "Root mean square spot error" if view == "error" else "Measured runtime (seconds)")
    figure.update_xaxes(type="log")
    figure.update_yaxes(type="log")
    return figure


def explain_exercise3_convergence(document, volatility):
    rows = [row for row in document.get("gbm_convergence", {}).get("orders", []) if abs(row["volatility"]-volatility)<1e-12]
    orders = "; ".join(f"{row['scheme']}: {row['observed_order']:.2f}" for row in rows)
    return {"description": "We compare simulated terminal spots with exact GBM using the same Brownian path. The step varies independently of the daily-exercise pricing study; no exercise decision enters this diagnostic.",
        "conclusion": f"Observed log–log slopes are {orders}. This tests strong terminal convergence. It does not establish the same convergence order for American prices or guarantee that every Milstein price is closer to BAW."}


# GAMMA — REUSE THE EXERCISE 2 DEFINITION AND ITS CONDITIONAL UNCERTAINTY

def build_exercise3_gamma_figure(document, volatility=0.2, view="gamma", show_uncertainty=False):
    rows = [row for row in document.get("gamma", {}).get("rows", []) if abs(row["volatility"]-volatility)<1e-12]
    figure = go.Figure()
    for scheme in ("Euler", "Milstein"):
        selected = sorted((row for row in rows if row["scheme"] == scheme), key=lambda row: row["moneyness"])
        for estimator in ("plain", "controlled"):
            key = estimator + ("_gamma" if view=="gamma" else "_estimator_variance")
            add_estimator_trace(figure, selected, "moneyness", key, scheme, estimator,
                estimator+"_standard_error" if view=="gamma" and show_uncertainty else None)
    if view=="gamma":
        selected = sorted((row for row in rows if row["scheme"]=="Euler"), key=lambda row: row["moneyness"])
        figure.add_trace(go.Scatter(x=[row["moneyness"] for row in selected], y=[row["analytical_gamma"] for row in selected],
            name="Analytical Black–Scholes gamma", mode="lines", line={"color":"#172b45","dash":"dot","width":3}))
    else:
        figure.update_yaxes(type="log")
    return style_exercise3_figure(figure, "American put gamma across moneyness" if view=="gamma" else "Variance of the estimated American put gamma",
        "Initial spot / strike", "Gamma (inverse spot-price unit)" if view=="gamma" else "Estimator variance (inverse spot-price unit²)")


def explain_exercise3_gamma(document, volatility=None, scheme=None):
    rows = document.get("gamma", {}).get("rows", [])
    rows = [row for row in rows if (volatility is None or abs(row["volatility"]-volatility)<1e-12) and (scheme is None or row["scheme"]==scheme)]
    if not rows:
        return {"description":"No gamma study is contained in this run.","conclusion":"Run the gamma study before making a gamma comparison."}
    plain = sum(row["plain_estimator_variance"] for row in rows)
    controlled = sum(row["controlled_estimator_variance"] for row in rows)
    reduction = 1-controlled/plain if plain else 0
    effect=f"reduces mean conditional gamma variance by {reduction:.1%}" if reduction>=0 else f"increases mean conditional gamma variance by {-reduction:.1%}"
    return {"description": "We refit at each bumped spot, share Brownian increments across the three bumps and evaluate those fitted rules on independent paths. Gamma is the second price difference divided by the square of the bump; its variance is pathwise gamma sample variance divided by evaluation paths.",
        "conclusion": f"Across the displayed cases, the control {effect}. The overall saved study shows little gamma variance reduction and no consistent improvement in the gamma profile from Milstein. These variances exclude repeating the policy training."}


# OFFLINE HTML — ONE EMBEDDED PLOTLY COPY, GENUINE SAVED FIGURES AND USEFUL SELECTORS


# LOCAL UI — ONE SAVED RESULTS PAGE, WITH EXPLANATION BEFORE EACH GRAPH

def render_exercise3_page():
    import logging
    import streamlit as st
    try:
        document=load_exercise3_results(PROJECT_DIRECTORY/"results"/"exercise3_results.json")
    except (OSError,ValueError,KeyError) as error:
        st.warning("The saved Exercise 3 study could not be loaded. Restore the complete results/ folder from the supplied ZIP.")
        logging.getLogger(__name__).error("Exercise 3 saved results: %s",error)
        return

    def show_chart(figure,notes,key):
        # Here the question and quantities precede the graph; its takeaway follows.
        st.write(notes["description"])
        st.plotly_chart(figure,use_container_width=True,theme=None,key=key,
            config={"displaylogo":False,"scrollZoom":False,"toImageButtonOptions":{"format":"png","scale":2}})
        st.markdown("**What we learn.** "+notes["conclusion"])

    st.caption("EXERCISE 3 / SAVED RESULTS")
    st.title("Euler & Milstein")
    st.write("Does a more accurate path simulation improve American option pricing? We retain the same Longstaff–Schwartz method and compare the completed Euler and Milstein calculations.")
    st.markdown('''<style>.ex3-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:8px 0 24px}.ex3-cards .card{background:white;color:#172b45;padding:16px;border:1px solid #dbe3ed;border-top:3px solid #c92739;border-radius:9px}.ex3-cards .card span{font-size:11px;font-weight:bold}.ex3-cards .card strong{display:block;font-size:19px;line-height:1.3;margin:8px 0}.ex3-cards .card p{font-size:13px;line-height:1.5;margin-bottom:0}@media(max-width:760px){.ex3-cards{grid-template-columns:1fr}}</style>'''+result_cards_html(document),unsafe_allow_html=True)
    st.caption("The selectors below explore saved calculations immediately. Blue denotes Euler; orange denotes Milstein. Plain and European-controlled estimates use solid and dashed lines respectively.")

    st.subheader("1 · What improves: precision or accuracy?")
    st.write("More paths and the European control reduce sampling variance. Milstein targets time-discretisation error. Neither automatically removes errors in the estimated exercise decisions; a single benchmark gap does not isolate bias.")

    st.subheader("2 · GBM prices and the European control")
    volatilities=sorted({row["volatility"] for row in document["gbm_prices"]})
    controls=st.columns(2)
    with controls[0]:
        volatility=st.selectbox("Lognormal volatility",volatilities,
            index=volatilities.index(.2) if .2 in volatilities else 0,
            format_func=lambda value:f"{value:.0%}",key="ex3_gbm_vol")
    with controls[1]:
        measure=st.selectbox("Compare",["American put prices","Price-estimator variance","Gap to BAW","Paired scheme difference","Payment correlation"],key="ex3_gbm_measure")
    st.caption("One-month American put; initial spot = strike = 40; zero rates and yields; daily simulation and exercise. Path counts refer to new paths used to evaluate the same estimated decisions.")
    if measure=="Paired scheme difference":
        figure=build_exercise3_paired_figure(document,volatility=volatility)
        notes=explain_exercise3_paired(document,volatility=volatility)
    elif measure=="Payment correlation":
        figure=build_exercise3_correlation_figure(document)
        notes=explain_exercise3_correlation(document)
        st.caption("This view covers all four volatilities at the largest saved path count.")
    else:
        view={"American put prices":"prices","Price-estimator variance":"variance","Gap to BAW":"errors"}[measure]
        figure=build_exercise3_price_figure(document,volatility=volatility,view=view)
        notes=explain_exercise3_prices(document,volatility=volatility,view=view)
        if view=="prices":
            notes["description"]+=" BAW equals the exact Black–Scholes value here because rates and yields are zero. Bars show approximate 95% evaluation intervals."
    show_chart(figure,notes,"ex3_gbm_prices")

    if document.get("gbm_convergence",{}).get("rows"):
        st.subheader("How much more accurate are the simulated paths?")
        vols=sorted({row["volatility"] for row in document["gbm_convergence"]["rows"]})
        chosen=st.selectbox("Volatility for the path-accuracy comparison",vols,
            index=vols.index(.2) if .2 in vols else 0,
            format_func=lambda value:f"{value:.0%}",key="ex3_convergence_vol")
        show_chart(build_exercise3_convergence_figure(document,chosen),
            explain_exercise3_convergence(document,chosen),"ex3_convergence")

    if document.get("sv_prices"):
        st.subheader("3 · Repeat with stochastic volatility")
        comparison=st.selectbox("Stochastic-volatility comparison",
            ["Prices against the same-model reference","Paired scheme difference","Gap to the constant-volatility BAW proxy"],key="ex3_sv_reference")
        st.caption("Initial spot = strike = 1; maturity = 1 year; initial and long-run variance = 0.08²; mean reversion = 2; variance volatility = 0.04; correlation = 0.5; zero rates; daily exercise.")
        if comparison=="Paired scheme difference":
            figure=build_exercise3_paired_figure(document,model="sv")
            notes=explain_exercise3_paired(document,model="sv")
        else:
            reference="same_model" if comparison.startswith("Prices") else "baw"
            view="prices" if reference=="same_model" else "errors"
            figure=build_exercise3_price_figure(document,model="sv",reference=reference,view=view)
            notes=explain_exercise3_prices(document,model="sv",reference=reference,view=view)
            notes["description"]=("Euler and Milstein simulate both spot and variance, then value the same American put. The horizontal line is a refined terminal-payoff estimate under the same model; its own uncertainty is separate from the displayed pricing intervals." if reference=="same_model" else "We repeat the requested BAW comparison at constant 8% volatility. This proxy uses another model, so its gap includes model differences as well as numerical error.")
        show_chart(figure,notes,"ex3_sv_prices")
        count=max(row["evaluation_paths"] for row in document["sv_prices"])
        selected={row["scheme"]:row for row in document["sv_prices"] if row["evaluation_paths"]==count}
        ratio=selected["Milstein"]["runtime_seconds"]/selected["Euler"]["runtime_seconds"]
        st.caption(f"At {count:,} evaluation paths, Milstein took about {ratio:.0f} times as long as Euler in this experiment. More computation did not demonstrate a better price in the original configuration.")

    if document.get("gamma",{}).get("rows"):
        st.subheader("Carry forward Exercise 2: does gamma improve?")
        vols=sorted({row["volatility"] for row in document["gamma"]["rows"]})
        controls=st.columns(2)
        with controls[0]:
            chosen=st.selectbox("Gamma comparison volatility",vols,
                index=vols.index(.2) if .2 in vols else 0,
                format_func=lambda value:f"{value:.0%}",key="ex3_gamma_vol")
        with controls[1]:
            quantity=st.selectbox("Gamma comparison",["American put gamma","Gamma-estimator variance"],key="ex3_gamma_measure")
        view="gamma" if quantity=="American put gamma" else "variance"
        notes=explain_exercise3_gamma(document,chosen)
        notes["description"]="Gamma measures price curvature when the initial spot changes. We use the same spot bumps and shared Brownian scenarios for both schemes; new paths evaluate the decisions estimated at each bumped spot. "+("The dotted line is the analytical Black–Scholes gamma." if view=="gamma" else "The vertical axis measures variance of the estimated gamma, on a logarithmic scale; lower is more precise.")
        show_chart(build_exercise3_gamma_figure(document,chosen,view),notes,"ex3_gamma")

    st.caption("All figures use the completed assignment study. Sampling intervals condition on the estimated decisions and control coefficient; they exclude variation from estimating those decisions again and numerical bias. Full methods, inputs and validation remain in the written report and command-line code.")
    st.markdown("**Exercise 3 conclusion.** Milstein improves GBM path accuracy. The European control improves price precision. The saved daily pricing and gamma comparisons do not establish a consistent accuracy improvement from Milstein, and joint stochastic-volatility simulation costs more.")


