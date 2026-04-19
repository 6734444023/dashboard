import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Data Loading (only real CSV columns)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "final_csv_icu.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["first_careunit"] = df["first_careunit"].fillna("Unknown")
df["long_title"] = df["long_title"].fillna("Unknown")

CAREUNITS = sorted(df["first_careunit"].unique())
LOS_CATEGORIES = sorted(df["los_category"].unique())

# Top diagnoses for filters (real data)
TOP_DIAGNOSES = df["long_title"].value_counts().head(20).index.tolist()

# ---------------------------------------------------------------------------
# Color Palette & Theme
# ---------------------------------------------------------------------------
BG_DARK = "#f0f4f8"
BG_CARD = "#ffffff"
BG_CARD_INNER = "#f8fafc"
ACCENT_TEAL = "#0891b2"
ACCENT_GOLD = "#b45309"
TEXT_WHITE = "#1e293b"
TEXT_MUTED = "#64748b"
BORDER_COLOR = "#e2e8f0"

CHART_COLORS = ["#0891b2", "#eab308", "#0ea5e9", "#f59e0b", "#06b6d4", "#d97706"]

LIGHT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_WHITE, size=11),
        title=dict(font=dict(color=ACCENT_TEAL, size=14)),
        xaxis=dict(
            gridcolor="rgba(226,232,240,0.8)", zerolinecolor=BORDER_COLOR,
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            gridcolor="rgba(226,232,240,0.8)", zerolinecolor=BORDER_COLOR,
            color=TEXT_MUTED,
        ),
        colorway=CHART_COLORS,
    )
)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ICU Length of Stay (LOS) & Diagnosis Analysis"
server = app.server  # for gunicorn

PANEL_STYLE = {
    "backgroundColor": BG_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_COLOR}",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
    "padding": "16px",
    "height": "100%",
}

KPI_STYLE = {
    "backgroundColor": BG_CARD_INNER,
    "borderRadius": "10px",
    "border": f"1px solid {BORDER_COLOR}",
    "padding": "12px 8px",
    "textAlign": "center",
}

INSIGHT_STYLE = {
    "backgroundColor": "#f0fdfa",
    "borderRadius": "8px",
    "border": f"1px solid {ACCENT_TEAL}",
    "padding": "10px 14px",
    "marginTop": "10px",
    "fontSize": "12px",
    "color": TEXT_MUTED,
}

GRAPH_CONFIG = {"displayModeBar": False}


def make_kpi_card(label, value_id, color=TEXT_WHITE):
    return html.Div(
        [
            html.Div(label, style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "2px"}),
            html.Div(id=value_id, style={"color": color, "fontSize": "24px", "fontWeight": "700"}),
        ],
        style=KPI_STYLE,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"backgroundColor": BG_DARK, "minHeight": "100vh", "padding": "20px 24px", "fontFamily": "'Segoe UI', sans-serif"},
    children=[
        # ===== HEADER =====
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Span("🏥 ", style={"fontSize": "28px"}),
                            html.Span(
                                "ICU Length of Stay (LOS) & Diagnosis Analysis",
                                style={"fontSize": "22px", "fontWeight": "700", "color": TEXT_WHITE},
                            ),
                        ]
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Care Unit", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    dcc.Dropdown(
                                        id="filter-careunit",
                                        options=[{"label": c, "value": c} for c in CAREUNITS],
                                        value=[],
                                        multi=True,
                                        placeholder="All",
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                                md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Label("LOS Category", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    dcc.Dropdown(
                                        id="filter-los-cat",
                                        options=[{"label": c, "value": c} for c in LOS_CATEGORIES],
                                        value=[],
                                        multi=True,
                                        placeholder="All",
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                                md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Log Scale", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    dbc.Switch(
                                        id="toggle-log",
                                        value=False,
                                        style={"marginTop": "4px"},
                                    ),
                                ],
                                md=2,
                                className="d-flex flex-column align-items-center",
                            ),
                        ]
                    ),
                    md=6,
                ),
            ],
            className="mb-3",
            align="center",
        ),
        # ===== ROW 1: Overview KPIs (left) + Box Plot by Care Unit (right) =====
        dbc.Row(
            [
                # --- Left: Overview panel ---
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                [
                                    html.Span("Overview", style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px"}),
                                    html.Br(),
                                    html.Span("Theme: ", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    html.Span("ICU Performance Analysis", style={"color": TEXT_WHITE, "fontSize": "11px"}),
                                    html.Br(),
                                    html.Span("Motivation: ", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    html.Span("Optimizing Bed Management", style={"color": TEXT_WHITE, "fontSize": "11px"}),
                                ],
                                style={"marginBottom": "14px"},
                            ),
                            # KPI Row 1
                            dbc.Row(
                                [
                                    dbc.Col(make_kpi_card("Total Patients:", "kpi-total", ACCENT_TEAL), md=4),
                                    dbc.Col(make_kpi_card("Avg. LOS:", "kpi-avg", TEXT_WHITE), md=4),
                                    dbc.Col(make_kpi_card("Median LOS:", "kpi-median", TEXT_WHITE), md=4),
                                ],
                                className="g-2 mb-2",
                            ),
                            # KPI Row 2
                            dbc.Row(
                                [
                                    dbc.Col(make_kpi_card("Max LOS:", "kpi-max", TEXT_WHITE), md=6),
                                    dbc.Col(make_kpi_card("% Long Stay (>30d):", "kpi-long-pct", ACCENT_GOLD), md=6),
                                ],
                                className="g-2",
                            ),
                        ],
                    ),
                    md=4,
                ),
                # --- Right: Q1 Box Plot by Care Unit ---
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q1 – LOS Distribution by Care Unit",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="box-careunit", config=GRAPH_CONFIG, style={"height": "260px"}),
                            html.Div(
                                id="insight-1",
                                style=INSIGHT_STYLE,
                            ),
                        ],
                    ),
                    md=8,
                ),
            ],
            className="g-3 mb-3",
        ),
        # ===== ROW 2: Top Diagnoses Bar (left) + Histogram (center) + Avg LOS by Unit (right-center) + Summary (right) =====
        dbc.Row(
            [
                # Q2 – Top Diagnoses
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q2 – Top 10 Diagnoses by Admissions",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="bar-top-diag", config=GRAPH_CONFIG, style={"height": "260px"}),
                            html.Div(
                                id="insight-2",
                                style=INSIGHT_STYLE,
                            ),
                        ],
                    ),
                    md=3,
                ),
                # Q3 – Histogram
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q3 – LOS Variation & Outliers",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="hist-los", config=GRAPH_CONFIG, style={"height": "260px"}),
                            html.Div(
                                id="insight-3",
                                style=INSIGHT_STYLE,
                            ),
                        ],
                    ),
                    md=3,
                ),
                # Avg LOS by Care Unit
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Avg LOS by Care Unit",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="bar-avg-los", config=GRAPH_CONFIG, style={"height": "300px"}),
                        ],
                    ),
                    md=3,
                ),
                # Summary & Recommendation
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "SUMMARY & RECOMMENDATION",
                                style={"color": ACCENT_GOLD, "fontWeight": "700", "fontSize": "14px", "marginBottom": "12px"},
                            ),
                            html.Div(
                                [
                                    html.Div("Key Takeaways:", style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "12px", "marginBottom": "6px"}),
                                    html.Div(id="summary-takeaways"),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            html.Div(
                                [
                                    html.Div("LOS Category Breakdown:", style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "12px", "marginBottom": "6px"}),
                                    html.Div(id="summary-breakdown"),
                                ],
                            ),
                        ],
                    ),
                    md=3,
                ),
            ],
            className="g-3 mb-3",
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("kpi-total", "children"),
    Output("kpi-avg", "children"),
    Output("kpi-median", "children"),
    Output("kpi-max", "children"),
    Output("kpi-long-pct", "children"),
    Output("box-careunit", "figure"),
    Output("insight-1", "children"),
    Output("bar-top-diag", "figure"),
    Output("insight-2", "children"),
    Output("hist-los", "figure"),
    Output("insight-3", "children"),
    Output("bar-avg-los", "figure"),
    Output("summary-takeaways", "children"),
    Output("summary-breakdown", "children"),
    Input("filter-careunit", "value"),
    Input("filter-los-cat", "value"),
    Input("toggle-log", "value"),
)
def update_dashboard(careunits, los_cats, use_log):
    dff = df.copy()
    if careunits:
        dff = dff[dff["first_careunit"].isin(careunits)]
    if los_cats:
        dff = dff[dff["los_category"].isin(los_cats)]

    total = len(dff)
    unique_patients = dff["subject_id"].nunique()
    avg_los = dff["los"].mean() if total else 0
    median_los = dff["los"].median() if total else 0
    max_los = dff["los"].max() if total else 0
    long_pct = (dff["los"] > 30).sum() / total * 100 if total else 0

    # KPI values
    kpi_total = f"{unique_patients:,}"
    kpi_avg = f"{avg_los:.1f} Days"
    kpi_median = f"{median_los:.1f} Days"
    kpi_max = f"{max_los:.0f} Days"
    kpi_long = f"{long_pct:.1f}%"

    yaxis_type = "log" if use_log else "linear"

    # ---- Q1: Box plot by care unit (real column: first_careunit) ----
    units = dff["first_careunit"].value_counts().head(8).index.tolist()
    fig_box = go.Figure()
    for i, unit in enumerate(units):
        unit_data = dff[dff["first_careunit"] == unit]["los"]
        fig_box.add_trace(go.Box(
            y=unit_data, name=unit,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
            fillcolor=f"rgba({','.join(str(int(CHART_COLORS[i % len(CHART_COLORS)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.3)",
        ))
    fig_box.update_layout(
        template=LIGHT_TEMPLATE, showlegend=False, margin=dict(l=30, r=10, t=30, b=30),
        yaxis_title="LOS (days)", title="Box and Whisker Plot",
        title_font_size=12, yaxis_type=yaxis_type,
    )

    # Insight 1 (computed from real data)
    avg_by_unit = dff.groupby("first_careunit")["los"].mean().sort_values(ascending=False)
    top2_units = avg_by_unit.head(2)
    insight_1 = f"Key Insight #1: {top2_units.index[0]} (avg {top2_units.iloc[0]:.1f}d) and {top2_units.index[1]} (avg {top2_units.iloc[1]:.1f}d) have the longest average LOS across all care units."

    # ---- Q2: Top 10 diagnoses bar (real column: long_title) ----
    top_diag = dff["long_title"].value_counts().head(10).reset_index()
    top_diag.columns = ["Diagnosis", "Count"]
    top_diag["Diagnosis_short"] = top_diag["Diagnosis"].str[:40]

    fig_diag = go.Figure(go.Bar(
        x=top_diag["Count"],
        y=top_diag["Diagnosis_short"],
        orientation="h",
        marker_color=ACCENT_TEAL,
        text=top_diag["Count"],
        textposition="outside",
        textfont_size=9,
    ))
    fig_diag.update_layout(
        template=LIGHT_TEMPLATE, margin=dict(l=10, r=40, t=30, b=30),
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 9}},
        title="Top 10 Diagnoses", title_font_size=12,
        yaxis_title="",
    )

    top1_diag = top_diag.iloc[0]
    insight_2 = f'Key Insight #2: "{top1_diag["Diagnosis"][:50]}..." is the most common diagnosis with {top1_diag["Count"]:,} admissions.'

    # ---- Q3: Histogram (real column: los) ----
    hist_data = dff[dff["los"] <= 40]["los"]
    fig_hist = go.Figure(go.Histogram(
        x=hist_data, nbinsx=8,
        marker_color=ACCENT_GOLD, marker_line_color=ACCENT_TEAL, marker_line_width=1,
    ))
    fig_hist.update_layout(
        template=LIGHT_TEMPLATE, margin=dict(l=30, r=10, t=30, b=30),
        xaxis_title="LOS in days", yaxis_title="Count",
        title="Histogram", title_font_size=12, bargap=0.05,
        yaxis_type=yaxis_type,
    )

    long_stay_pct = (dff["los"] > 30).sum() / total * 100 if total else 0
    insight_3 = f"Key Insight #3: {long_stay_pct:.1f}% of patients stay over 30 days, representing the long-tail outliers that consume disproportionate ICU resources."

    # ---- Avg LOS by Care Unit bar (real columns: first_careunit, los) ----
    avg_by_cu = (
        dff.groupby("first_careunit")["los"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    avg_by_cu.columns = ["Care Unit", "Avg LOS"]

    fig_avg_los = go.Figure(go.Bar(
        x=avg_by_cu["Avg LOS"],
        y=avg_by_cu["Care Unit"],
        orientation="h",
        marker_color=CHART_COLORS[1],
        text=avg_by_cu["Avg LOS"].round(1).astype(str) + "d",
        textposition="outside",
        textfont_size=9,
    ))
    fig_avg_los.update_layout(
        template=LIGHT_TEMPLATE, margin=dict(l=10, r=40, t=30, b=30),
        xaxis_title="Avg LOS (days)", yaxis_title="",
        title="Avg LOS by Care Unit", title_font_size=12,
    )

    # ---- Summary takeaways (computed from real data) ----
    short_count = (dff["los_category"] == "Short Stay").sum()
    long_count = (dff["los_category"] == "Long Stay").sum()
    short_pct = short_count / total * 100 if total else 0
    long_pct_cat = long_count / total * 100 if total else 0
    busiest = dff["first_careunit"].value_counts().idxmax() if total else "N/A"

    takeaways = html.Ol(
        [
            html.Li(
                f"{busiest} has the most admissions ({dff['first_careunit'].value_counts().iloc[0]:,}).",
                style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "4px"},
            ),
            html.Li(
                f"Average LOS is {avg_los:.1f} days; median is {median_los:.1f} days.",
                style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "4px"},
            ),
            html.Li(
                f"{long_stay_pct:.1f}% of patients stay >30 days.",
                style={"color": TEXT_MUTED, "fontSize": "12px"},
            ),
        ],
        style={"paddingLeft": "18px"},
    )

    breakdown = html.Div(
        [
            html.Div(
                f"Short Stay: {short_count:,} ({short_pct:.1f}%)",
                style={"color": ACCENT_TEAL, "fontSize": "12px", "marginBottom": "4px"},
            ),
            html.Div(
                f"Long Stay: {long_count:,} ({long_pct_cat:.1f}%)",
                style={"color": ACCENT_GOLD, "fontSize": "12px"},
            ),
        ]
    )

    return (
        kpi_total, kpi_avg, kpi_median, kpi_max, kpi_long,
        fig_box, insight_1,
        fig_diag, insight_2,
        fig_hist, insight_3,
        fig_avg_los,
        takeaways,
        breakdown,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
