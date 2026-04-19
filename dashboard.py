import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Data Loading & Feature Engineering
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "final_csv_icu.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["first_careunit"] = df["first_careunit"].fillna("Unknown")
df["long_title"] = df["long_title"].fillna("Unknown")

# Map diagnoses to broad diagnosis groups
DIAGNOSIS_MAP = {
    "Cardiovascular": [
        "coronary", "myocardial", "heart", "cardiac", "aortic", "mitral",
        "valve", "atherosclerotic", "bradycardia", "tachycardia", "atrial",
        "ventricular", "pericarditis", "endocarditis", "cardiomyopathy",
        "angina", "arrhythmia", "hypertensive", "nstemi", "stemi",
    ],
    "Respiratory": [
        "respiratory", "pneumonia", "pneumonitis", "pulmonary", "asthma",
        "bronch", "copd", "ventilat", "tracheostomy", "hemothorax",
        "pneumothorax", "lung", "pleural",
    ],
    "Sepsis/Infection": [
        "sepsis", "septicemia", "infection", "abscess", "cellulitis",
        "meningitis", "endocarditis", "osteomyelitis", "peritonitis",
        "staphylococcus", "streptococ", "escherichia", "pseudomonas",
        "clostridium",
    ],
    "Trauma": [
        "trauma", "fracture", "contusion", "laceration", "wound",
        "hemorrhage", "bleeding", "injury", "burn", "crush",
        "hemothorax", "hematoma",
    ],
    "Post-Op Care": [
        "postoperative", "postprocedural", "complication", "disruption",
        "transplant", "graft", "prosthe", "implant", "surgical",
    ],
}


def classify_diagnosis(title):
    t = title.lower()
    for group, keywords in DIAGNOSIS_MAP.items():
        if any(kw in t for kw in keywords):
            return group
    return "Other"


df["diagnosis_group"] = df["long_title"].apply(classify_diagnosis)

# Simulate a comorbidity index (number of unique diagnoses per patient)
patient_diag_count = (
    df.groupby("subject_id")["long_title"].nunique().reset_index()
)
patient_diag_count.columns = ["subject_id", "comorbidity_index"]
df = df.merge(patient_diag_count, on="subject_id", how="left")

CAREUNITS = sorted(df["first_careunit"].unique())
LOS_CATEGORIES = sorted(df["los_category"].unique())
DIAG_GROUPS = sorted(df["diagnosis_group"].unique())

# ---------------------------------------------------------------------------
# Color Palette & Theme
# ---------------------------------------------------------------------------
BG_DARK = "#0a1628"
BG_CARD = "#0f2744"
BG_CARD_INNER = "#132d4a"
ACCENT_TEAL = "#22d3ee"
ACCENT_GOLD = "#eab308"
TEXT_WHITE = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
BORDER_COLOR = "#1e3a5f"

CHART_COLORS = ["#22d3ee", "#eab308", "#38bdf8", "#f59e0b", "#06b6d4", "#fbbf24"]

DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_WHITE, size=11),
        title=dict(font=dict(color=ACCENT_TEAL, size=14)),
        xaxis=dict(
            gridcolor="rgba(30,58,95,0.5)", zerolinecolor=BORDER_COLOR,
            color=TEXT_MUTED,
        ),
        yaxis=dict(
            gridcolor="rgba(30,58,95,0.5)", zerolinecolor=BORDER_COLOR,
            color=TEXT_MUTED,
        ),
        colorway=CHART_COLORS,
    )
)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "ICU Length of Stay (LOS) & Diagnosis Analysis"
server = app.server  # for gunicorn

PANEL_STYLE = {
    "backgroundColor": BG_CARD,
    "borderRadius": "12px",
    "border": f"1px solid {BORDER_COLOR}",
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
    "backgroundColor": "rgba(34,211,238,0.08)",
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
                                md=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Diagnosis Group", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                                    dcc.Dropdown(
                                        id="filter-diag-group",
                                        options=[{"label": c, "value": c} for c in DIAG_GROUPS],
                                        value=[],
                                        multi=True,
                                        placeholder="All",
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                                md=4,
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
                                md=4,
                            ),
                        ]
                    ),
                    md=6,
                ),
            ],
            className="mb-3",
            align="center",
        ),
        # ===== ROW 1: Overview KPIs (left) + Box Plot Q1 (right) =====
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
                # --- Right: Q1 Box Plot ---
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q1 – Primary Diagnosis Impact on LOS",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="box-diagnosis", config=GRAPH_CONFIG, style={"height": "260px"}),
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
        # ===== ROW 2: Stacked Bar (left) + Histogram (center) + Scatter (right-center) + Summary (right) =====
        dbc.Row(
            [
                # Q2 – MICU vs SICU stacked bar
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q2 – MICU vs SICU Disease Profile",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="stacked-bar", config=GRAPH_CONFIG, style={"height": "260px"}),
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
                # Scatter plot
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "LOS vs Comorbidity Index",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="scatter-comorb", config=GRAPH_CONFIG, style={"height": "300px"}),
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
                                    html.Div("2–3 Key Takeaways:", style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "12px", "marginBottom": "6px"}),
                                    html.Div(id="summary-takeaways"),
                                ],
                                style={"marginBottom": "16px"},
                            ),
                            html.Div(
                                [
                                    html.Div("1 Recommendation:", style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "12px", "marginBottom": "6px"}),
                                    html.Div(
                                        'Implement a "Complex Case" Triage Protocol for high-comorbidity, high-LOS diagnosis patients.',
                                        style={"color": TEXT_MUTED, "fontSize": "12px", "lineHeight": "1.5"},
                                    ),
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
    Output("box-diagnosis", "figure"),
    Output("insight-1", "children"),
    Output("stacked-bar", "figure"),
    Output("insight-2", "children"),
    Output("hist-los", "figure"),
    Output("insight-3", "children"),
    Output("scatter-comorb", "figure"),
    Output("summary-takeaways", "children"),
    Input("filter-careunit", "value"),
    Input("filter-diag-group", "value"),
    Input("filter-los-cat", "value"),
)
def update_dashboard(careunits, diag_groups, los_cats):
    dff = df.copy()
    if careunits:
        dff = dff[dff["first_careunit"].isin(careunits)]
    if diag_groups:
        dff = dff[dff["diagnosis_group"].isin(diag_groups)]
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

    # ---- Q1: Box plot by diagnosis group ----
    top_groups = ["Cardiovascular", "Respiratory", "Sepsis/Infection", "Post-Op Care", "Trauma"]
    box_df = dff[dff["diagnosis_group"].isin(top_groups)]
    fig_box = go.Figure()
    for i, grp in enumerate(top_groups):
        grp_data = box_df[box_df["diagnosis_group"] == grp]["los"]
        fig_box.add_trace(go.Box(
            y=grp_data, name=grp,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
            fillcolor=f"rgba({','.join(str(int(CHART_COLORS[i % len(CHART_COLORS)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.3)",
        ))
    fig_box.update_layout(
        template=DARK_TEMPLATE, showlegend=False, margin=dict(l=30, r=10, t=30, b=30),
        yaxis_title="LOS (days)", title="Box and Whisker Plot",
        title_font_size=12,
    )

    # Insight 1
    avg_by_grp = box_df.groupby("diagnosis_group")["los"].mean().sort_values(ascending=False)
    top2 = avg_by_grp.head(2).index.tolist() if len(avg_by_grp) >= 2 else avg_by_grp.index.tolist()
    insight_1 = f"Key Insight #1: {' and '.join(top2)} are the primary drivers of longer LOS, with significantly higher median values and extreme outliers."

    # ---- Q2: Stacked bar MICU vs SICU ----
    micu_sicu = dff[dff["first_careunit"].isin(["MICU", "SICU"])]
    stacked_data = micu_sicu.groupby(["first_careunit", "diagnosis_group"]).size().reset_index(name="count")
    # Compute percentages
    totals = stacked_data.groupby("first_careunit")["count"].transform("sum")
    stacked_data["pct"] = stacked_data["count"] / totals * 100

    fig_stacked = go.Figure()
    for i, grp in enumerate(top_groups):
        grp_df = stacked_data[stacked_data["diagnosis_group"] == grp]
        fig_stacked.add_trace(go.Bar(
            x=grp_df["first_careunit"], y=grp_df["pct"], name=grp,
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            text=grp_df["pct"].round(0).astype(int).astype(str) + "%",
            textposition="inside", textfont_size=9,
        ))
    fig_stacked.update_layout(
        barmode="stack", template=DARK_TEMPLATE, margin=dict(l=30, r=10, t=30, b=30),
        yaxis_title="Percent of Patients", xaxis_title="Care Unit",
        title="Stacked Bar Chart", title_font_size=12,
        legend=dict(font=dict(size=9), orientation="h", y=-0.25),
    )

    insight_2 = "Key Insight #2: MICU profiles are complex (Respiratory/Sepsis), leading to inherently longer LOS than the predictable Post-Op SICU caseload."

    # ---- Q3: Histogram ----
    hist_data = dff[dff["los"] <= 40]["los"]
    fig_hist = go.Figure(go.Histogram(
        x=hist_data, nbinsx=8,
        marker_color=ACCENT_GOLD, marker_line_color=ACCENT_TEAL, marker_line_width=1,
    ))
    fig_hist.update_layout(
        template=DARK_TEMPLATE, margin=dict(l=30, r=10, t=30, b=30),
        xaxis_title="LOS in days", yaxis_title="Count",
        title="Histogram", title_font_size=12, bargap=0.05,
    )

    long_stay_pct = (dff["los"] > 30).sum() / total * 100 if total else 0
    insight_3 = f"Key Insight #3: A small percentage ({long_stay_pct:.1f}%) of long-stay patients over 30 days significantly impacts resource use. This correlates with higher comorbidity."

    # ---- Scatter: LOS vs Comorbidity Index ----
    scatter_df = dff.sample(n=min(2000, len(dff)), random_state=42) if len(dff) > 2000 else dff
    fig_scatter = go.Figure(go.Scatter(
        x=scatter_df["comorbidity_index"],
        y=scatter_df["los"],
        mode="markers",
        marker=dict(color=ACCENT_TEAL, size=4, opacity=0.4),
    ))
    # Add trendline
    if len(scatter_df) > 10:
        z = np.polyfit(scatter_df["comorbidity_index"], scatter_df["los"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(scatter_df["comorbidity_index"].min(), scatter_df["comorbidity_index"].max(), 50)
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=p(x_line), mode="lines",
            line=dict(color=ACCENT_GOLD, width=2), showlegend=False,
        ))
    fig_scatter.update_layout(
        template=DARK_TEMPLATE, margin=dict(l=30, r=10, t=30, b=30),
        xaxis_title="Comorbidity Index", yaxis_title="LOS (days)",
        title="Scatter Plot", title_font_size=12,
    )

    # ---- Summary takeaways ----
    takeaways = html.Ol(
        [
            html.Li("Diagnosis drives LOS.", style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "4px"}),
            html.Li(
                f"MICU is more complex than SICU.",
                style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "4px"},
            ),
            html.Li(
                "Long-stay patients disproportionately consume resources.",
                style={"color": TEXT_MUTED, "fontSize": "12px"},
            ),
        ],
        style={"paddingLeft": "18px"},
    )

    return (
        kpi_total, kpi_avg, kpi_median, kpi_max, kpi_long,
        fig_box, insight_1,
        fig_stacked, insight_2,
        fig_hist, insight_3,
        fig_scatter,
        takeaways,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
