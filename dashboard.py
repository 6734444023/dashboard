import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "final_csv_icu.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["first_careunit"] = df["first_careunit"].fillna("Unknown")
df["long_title"] = df["long_title"].fillna("Unknown")

CAREUNITS = sorted(df["first_careunit"].unique())
LOS_CATEGORIES = sorted(df["los_category"].unique())

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ICU Dashboard"
server = app.server  # for gunicorn

CARD_STYLE = {
    "textAlign": "center",
    "padding": "16px",
    "borderRadius": "10px",
    "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container(
    fluid=True,
    className="py-3",
    children=[
        # Title
        dbc.Row(
            dbc.Col(
                html.H1(
                    "🏥 ICU Admissions Dashboard",
                    className="text-center mb-3",
                    style={"fontWeight": "700"},
                )
            )
        ),
        # Filters
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Care Unit", className="fw-bold"),
                        dcc.Dropdown(
                            id="filter-careunit",
                            options=[{"label": c, "value": c} for c in CAREUNITS],
                            value=[],
                            multi=True,
                            placeholder="All Care Units",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("LOS Category", className="fw-bold"),
                        dcc.Dropdown(
                            id="filter-los-cat",
                            options=[{"label": c, "value": c} for c in LOS_CATEGORIES],
                            value=[],
                            multi=True,
                            placeholder="All Categories",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label("Length of Stay (days)", className="fw-bold"),
                        dcc.RangeSlider(
                            id="filter-los-range",
                            min=0,
                            max=int(np.ceil(df["los"].quantile(0.99))),
                            step=0.5,
                            value=[0, int(np.ceil(df["los"].quantile(0.99)))],
                            marks={
                                i: str(i)
                                for i in range(
                                    0, int(np.ceil(df["los"].quantile(0.99))) + 1, 5
                                )
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    md=4,
                ),
            ],
            className="mb-4",
        ),
        # KPI Cards
        dbc.Row(id="kpi-cards", className="mb-4 g-3"),
        # Row 1: Admissions by Care Unit + LOS Category Pie
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="bar-careunit"), md=7),
                dbc.Col(dcc.Graph(id="pie-los-cat"), md=5),
            ],
            className="mb-4",
        ),
        # Row 2: LOS Distribution Histogram + Box plot
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="hist-los"), md=6),
                dbc.Col(dcc.Graph(id="box-los"), md=6),
            ],
            className="mb-4",
        ),
        # Row 3: Top Diagnoses
        dbc.Row(
            dbc.Col(dcc.Graph(id="bar-diagnoses"), md=12),
            className="mb-4",
        ),
        # Row 4: Avg LOS by Care Unit + Readmission scatter
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="bar-avg-los"), md=6),
                dbc.Col(dcc.Graph(id="bar-readmit"), md=6),
            ],
            className="mb-4",
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("kpi-cards", "children"),
    Output("bar-careunit", "figure"),
    Output("pie-los-cat", "figure"),
    Output("hist-los", "figure"),
    Output("box-los", "figure"),
    Output("bar-diagnoses", "figure"),
    Output("bar-avg-los", "figure"),
    Output("bar-readmit", "figure"),
    Input("filter-careunit", "value"),
    Input("filter-los-cat", "value"),
    Input("filter-los-range", "value"),
)
def update_dashboard(careunits, los_cats, los_range):
    dff = df.copy()

    if careunits:
        dff = dff[dff["first_careunit"].isin(careunits)]
    if los_cats:
        dff = dff[dff["los_category"].isin(los_cats)]
    if los_range:
        dff = dff[(dff["los"] >= los_range[0]) & (dff["los"] <= los_range[1])]

    # --- KPI cards ---
    total = len(dff)
    unique_patients = dff["subject_id"].nunique()
    avg_los = dff["los"].mean() if total else 0
    median_los = dff["los"].median() if total else 0
    short_pct = (
        (dff["los_category"] == "Short Stay").sum() / total * 100 if total else 0
    )

    kpi_data = [
        ("Total Admissions", f"{total:,}", "primary"),
        ("Unique Patients", f"{unique_patients:,}", "info"),
        ("Avg LOS (days)", f"{avg_los:.2f}", "success"),
        ("Median LOS (days)", f"{median_los:.2f}", "warning"),
        ("Short Stay %", f"{short_pct:.1f}%", "danger"),
    ]

    kpi_cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(label, className="text-muted mb-1"),
                        html.H3(value, className=f"text-{color} fw-bold"),
                    ]
                ),
                style=CARD_STYLE,
            ),
        )
        for label, value, color in kpi_data
    ]

    # --- Bar: Admissions by Care Unit ---
    cu_counts = (
        dff["first_careunit"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Care Unit", "first_careunit": "Care Unit", "count": "Admissions"})
    )
    if "Admissions" not in cu_counts.columns:
        cu_counts.columns = ["Care Unit", "Admissions"]

    fig_bar_cu = px.bar(
        cu_counts,
        x="Care Unit",
        y="Admissions",
        color="Care Unit",
        title="Admissions by Care Unit",
        text_auto=True,
    )
    fig_bar_cu.update_layout(showlegend=False, template="plotly_white")

    # --- Pie: LOS Category ---
    los_cat_counts = (
        dff["los_category"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Category", "los_category": "Category", "count": "Count"})
    )
    if "Count" not in los_cat_counts.columns:
        los_cat_counts.columns = ["Category", "Count"]

    fig_pie = px.pie(
        los_cat_counts,
        names="Category",
        values="Count",
        title="Length of Stay Category",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_layout(template="plotly_white")

    # --- Histogram: LOS ---
    fig_hist = px.histogram(
        dff,
        x="los",
        nbins=60,
        title="Distribution of Length of Stay (days)",
        labels={"los": "Length of Stay (days)"},
        color_discrete_sequence=["#636EFA"],
    )
    fig_hist.update_layout(template="plotly_white", bargap=0.05)

    # --- Box: LOS by Care Unit ---
    fig_box = px.box(
        dff,
        x="first_careunit",
        y="los",
        color="first_careunit",
        title="LOS Distribution by Care Unit",
        labels={"first_careunit": "Care Unit", "los": "LOS (days)"},
    )
    fig_box.update_layout(showlegend=False, template="plotly_white")

    # --- Bar: Top 15 Diagnoses ---
    top_diag = (
        dff["long_title"]
        .value_counts()
        .head(15)
        .reset_index()
        .rename(columns={"index": "Diagnosis", "long_title": "Diagnosis", "count": "Count"})
    )
    if "Count" not in top_diag.columns:
        top_diag.columns = ["Diagnosis", "Count"]

    # Truncate long labels
    top_diag["Diagnosis_short"] = top_diag["Diagnosis"].str[:60]

    fig_diag = px.bar(
        top_diag,
        x="Count",
        y="Diagnosis_short",
        orientation="h",
        title="Top 15 Diagnoses",
        text_auto=True,
        color="Count",
        color_continuous_scale="Tealgrn",
    )
    fig_diag.update_layout(
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},
        height=500,
        yaxis_title="",
    )

    # --- Bar: Avg LOS by Care Unit ---
    avg_by_cu = (
        dff.groupby("first_careunit")["los"]
        .mean()
        .reset_index()
        .rename(columns={"first_careunit": "Care Unit", "los": "Avg LOS (days)"})
        .sort_values("Avg LOS (days)", ascending=False)
    )
    fig_avg_los = px.bar(
        avg_by_cu,
        x="Care Unit",
        y="Avg LOS (days)",
        color="Care Unit",
        title="Average LOS by Care Unit",
        text_auto=".2f",
    )
    fig_avg_los.update_layout(showlegend=False, template="plotly_white")

    # --- Bar: Patients with Multiple Admissions ---
    visits = dff.groupby("subject_id")["stay_id"].nunique().reset_index()
    visits.columns = ["subject_id", "visits"]
    visit_dist = (
        visits["visits"]
        .clip(upper=6)
        .replace(6, "6+")
        .astype(str)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    visit_dist.columns = ["Number of Visits", "Patients"]

    fig_readmit = px.bar(
        visit_dist,
        x="Number of Visits",
        y="Patients",
        title="Patient Visit Frequency",
        text_auto=True,
        color_discrete_sequence=["#EF553B"],
    )
    fig_readmit.update_layout(template="plotly_white")

    return (
        kpi_cards,
        fig_bar_cu,
        fig_pie,
        fig_hist,
        fig_box,
        fig_diag,
        fig_avg_los,
        fig_readmit,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
