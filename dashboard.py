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

CHART_COLORS = [
    "#0891b2", "#eab308", "#0ea5e9", "#f59e0b", "#06b6d4",
    "#d97706", "#10b981", "#ef4444", "#8b5cf6", "#ec4899",
]

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
        # ===== ROW 1: Overview KPIs (left) + Q1 Box Plot LOS by Top Diagnoses (right) =====
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
                # --- Right: Q1 – Primary Diagnosis Impact on LOS ---
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q1 – การวินิจฉัยโรคหลัก (Primary Diagnosis) ส่งผลต่อ LOS อย่างไร?",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="box-diagnosis", config=GRAPH_CONFIG, style={"height": "220px"}),
                            html.Div(id="diag-key-1", style={"marginTop": "4px", "fontSize": "10px", "color": TEXT_MUTED, "lineHeight": "1.5"}),
                            html.Div(id="insight-1", style=INSIGHT_STYLE),
                        ],
                    ),
                    md=8,
                ),
            ],
            className="g-3 mb-3",
        ),
        # ===== ROW 2: Q2 MICU vs SICU (left) + Q3 LOS Variation (center-left) + Avg LOS by Diagnosis (center-right) + Summary (right) =====
        dbc.Row(
            [
                # Q2 – Common Diseases per ICU Type (MICU vs SICU)
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q2 – โรคที่พบบ่อยใน MICU vs SICU",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="bar-micu-sicu", config=GRAPH_CONFIG, style={"height": "300px"}),
                            html.Div(id="diag-key-2", style={"marginTop": "6px", "fontSize": "10px", "color": TEXT_MUTED}),
                            html.Div(id="insight-2", style=INSIGHT_STYLE),
                        ],
                    ),
                    md=3,
                ),
                # Q3 – LOS Variation & Outlier Analysis
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Q3 – ความแตกต่างของ LOS (0–70+ วัน) สะท้อนอะไร?",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="hist-los", config=GRAPH_CONFIG, style={"height": "180px"}),
                            dcc.Graph(id="violin-los", config=GRAPH_CONFIG, style={"height": "180px"}),
                            html.Div(id="insight-3", style=INSIGHT_STYLE),
                        ],
                    ),
                    md=3,
                ),
                # Avg LOS by Top Diagnoses (supporting Q1)
                dbc.Col(
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.Div(
                                "Avg LOS by Top Diagnoses",
                                style={"color": ACCENT_TEAL, "fontWeight": "600", "fontSize": "13px", "marginBottom": "4px"},
                            ),
                            dcc.Graph(id="bar-avg-diag", config=GRAPH_CONFIG, style={"height": "380px"}),
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
                                style={"marginBottom": "10px"},
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
    Output("box-diagnosis", "figure"),
    Output("diag-key-1", "children"),
    Output("insight-1", "children"),
    Output("bar-micu-sicu", "figure"),
    Output("insight-2", "children"),
    Output("hist-los", "figure"),
    Output("violin-los", "figure"),
    Output("insight-3", "children"),
    Output("bar-avg-diag", "figure"),
    Output("diag-key-2", "children"),
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

    # KPI values – always from full dataset
    all_total = len(df)
    all_unique = df["subject_id"].nunique()
    all_avg = df["los"].mean()
    all_median = df["los"].median()
    all_max = df["los"].max()
    all_long_pct = (df["los"] > 30).sum() / all_total * 100

    kpi_total = f"{all_unique:,}"
    kpi_avg = f"{all_avg:.1f} Days"
    kpi_median = f"{all_median:.1f} Days"
    kpi_max = f"{all_max:.0f} Days"
    kpi_long = f"{all_long_pct:.1f}%"

    yaxis_type = "log" if use_log else "linear"

    # ================================================================
    # Q1: Box plot of LOS by Top 10 Diagnoses
    #     → answers "การวินิจฉัยโรคหลักส่งผลต่อ LOS อย่างไร?"
    # ================================================================
    top10_diag = dff["long_title"].value_counts().head(10).index.tolist()
    q1_df = dff[dff["long_title"].isin(top10_diag)]
    q1_code_map = {d: f"D{i+1}" for i, d in enumerate(top10_diag)}
    fig_box = go.Figure()
    for i, diag in enumerate(top10_diag):
        diag_data = q1_df[q1_df["long_title"] == diag]["los"]
        fig_box.add_trace(go.Box(
            y=diag_data, name=q1_code_map[diag],
            marker_color=CHART_COLORS[i % len(CHART_COLORS)],
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
            fillcolor=f"rgba({','.join(str(int(CHART_COLORS[i % len(CHART_COLORS)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.3)",
        ))
    fig_box.update_layout(
        template=LIGHT_TEMPLATE, showlegend=False,
        margin=dict(l=30, r=10, t=30, b=40),
        yaxis_title="LOS (days)", title="LOS Distribution by Primary Diagnosis (Top 10)",
        title_font_size=12, yaxis_type=yaxis_type,
        xaxis_tickangle=0, xaxis_tickfont_size=10,
    )

    # Build colored key for Q1 diagnoses
    q1_key_items = []
    for i, diag in enumerate(top10_diag):
        color = CHART_COLORS[i % len(CHART_COLORS)]
        q1_key_items.append(
            html.Span([
                html.Span("■ ", style={"color": color, "fontSize": "13px"}),
                html.B(f"{q1_code_map[diag]}", style={"color": color}),
                html.Span(f": {diag}  ", style={"color": TEXT_MUTED}),
            ])
        )
    diag_key_1 = html.Div(q1_key_items, style={"display": "flex", "flexWrap": "wrap", "gap": "4px 12px"})

    # Insight Q1: compare diagnosis with highest vs lowest avg LOS
    avg_by_diag = q1_df.groupby("long_title")["los"].agg(["mean", "median"]).sort_values("mean", ascending=False)
    longest_diag = avg_by_diag.index[0][:40]
    shortest_diag = avg_by_diag.index[-1][:40]
    insight_1 = (
        f'Insight Q1: "{longest_diag}" has the highest avg LOS ({avg_by_diag.iloc[0]["mean"]:.1f}d, '
        f'median {avg_by_diag.iloc[0]["median"]:.1f}d), while "{shortest_diag}" has the lowest '
        f'({avg_by_diag.iloc[-1]["mean"]:.1f}d). Diagnosis type significantly influences ICU stay duration.'
    )

    # ================================================================
    # Q2: Grouped bar – Top 5 Diagnoses in MICU vs SICU
    #     → answers "โรคใดพบบ่อยใน MICU เทียบกับ SICU?"
    # ================================================================
    micu_sicu = dff[dff["first_careunit"].isin(["MICU", "SICU"])]
    # Get top 5 diagnoses for each unit
    micu_top5 = micu_sicu[micu_sicu["first_careunit"] == "MICU"]["long_title"].value_counts().head(5)
    sicu_top5 = micu_sicu[micu_sicu["first_careunit"] == "SICU"]["long_title"].value_counts().head(5)
    # Union of top diagnoses
    union_diags = list(dict.fromkeys(micu_top5.index.tolist() + sicu_top5.index.tolist()))

    q2_data = micu_sicu[micu_sicu["long_title"].isin(union_diags)]
    q2_grouped = q2_data.groupby(["first_careunit", "long_title"]).size().reset_index(name="count")
    # Assign short codes D1, D2, … per unique diagnosis
    unique_diags_ordered = list(dict.fromkeys(q2_grouped["long_title"].tolist()))
    diag_code_map = {d: f"D{i+1}" for i, d in enumerate(unique_diags_ordered)}
    q2_grouped["diag_label"] = q2_grouped["long_title"].map(diag_code_map)

    # Butterfly chart: MICU → left (negative), SICU → right (positive)
    pivot_df = q2_grouped.pivot_table(
        index="diag_label", columns="first_careunit",
        values="count", fill_value=0, aggfunc="sum",
    ).reset_index()
    pivot_df.columns.name = None
    code_to_full = {v: k for k, v in diag_code_map.items()}
    pivot_df["full_name"] = pivot_df["diag_label"].map(code_to_full)
    micu_vals = pivot_df["MICU"] if "MICU" in pivot_df.columns else pd.Series([0] * len(pivot_df))
    sicu_vals = pivot_df["SICU"] if "SICU" in pivot_df.columns else pd.Series([0] * len(pivot_df))

    fig_micu_sicu = go.Figure()
    fig_micu_sicu.add_trace(go.Bar(
        y=pivot_df["diag_label"], x=-micu_vals,
        name="MICU", orientation="h",
        marker_color=CHART_COLORS[0],
        text=micu_vals, textposition="inside", textfont_size=9,
        textfont_color="white",
        customdata=pivot_df["full_name"],
        hovertemplate="<b>%{customdata}</b><br>MICU: %{text}<extra></extra>",
    ))
    fig_micu_sicu.add_trace(go.Bar(
        y=pivot_df["diag_label"], x=sicu_vals,
        name="SICU", orientation="h",
        marker_color=CHART_COLORS[1],
        text=sicu_vals, textposition="inside", textfont_size=9,
        textfont_color="white",
        customdata=pivot_df["full_name"],
        hovertemplate="<b>%{customdata}</b><br>SICU: %{text}<extra></extra>",
    ))
    max_val = int(max(micu_vals.max(), sicu_vals.max()))
    tick_step = max(1, max_val // 4)
    tick_vals = list(range(0, max_val + tick_step, tick_step))
    # Mirror for both sides
    all_tick_vals = [-v for v in reversed(tick_vals[1:])] + tick_vals
    all_tick_text = [str(v) for v in reversed(tick_vals[1:])] + [str(v) for v in tick_vals]
    fig_micu_sicu.update_layout(
        barmode="relative", template=LIGHT_TEMPLATE,
        margin=dict(l=5, r=70, t=20, b=20),
        yaxis=dict(tickfont=dict(size=5), automargin=True, categoryorder="array", categoryarray=sorted(pivot_df["diag_label"].tolist())),
        xaxis=dict(
            title=dict(text="Number of Admissions", standoff=20),
            tickvals=all_tick_vals, ticktext=all_tick_text,
            tickfont=dict(size=8),
            range=[-(max_val * 1.25), max_val * 1.25],
            zeroline=True, zerolinewidth=2, zerolinecolor=BORDER_COLOR,
        ),
        title="Top Diagnoses: MICU vs SICU", title_font_size=12,
        legend=dict(font=dict(size=9), orientation="h", y=-0.18, x=0.3),
    )

    # Build abbreviation key table
    diag_key_rows = [
        html.Div(
            [html.B(f"{code}: ", style={"color": TEXT_WHITE}), full],
            style={"marginBottom": "2px"}
        )
        for code, full in sorted(code_to_full.items(), key=lambda x: int(x[0][1:]))
    ]
    diag_key_table = html.Div(
        [html.Div("Diagnosis Key:", style={"fontWeight": "600", "color": ACCENT_TEAL, "marginBottom": "4px", "fontSize": "11px"})] + diag_key_rows,
        style={"lineHeight": "1.4"},
    )

    # Insight Q2
    micu_top1 = micu_top5.index[0][:40] if len(micu_top5) else "N/A"
    sicu_top1 = sicu_top5.index[0][:40] if len(sicu_top5) else "N/A"
    insight_2 = (
        f'Insight Q2: MICU top diagnosis is "{micu_top1}" ({micu_top5.iloc[0]:,}), '
        f'while SICU top is "{sicu_top1}" ({sicu_top5.iloc[0]:,}). '
        f"Disease profiles differ significantly between medical and surgical ICUs."
    )

    # ================================================================
    # Q3: Histogram + Violin – LOS Variation (0 to 70+ days)
    #     → answers "ความแตกต่างของ LOS สะท้อนอะไร?"
    # ================================================================
    # Histogram colored by LOS category
    hist_data_short = dff[(dff["los"] <= 70) & (dff["los_category"] == "Short Stay")]["los"]
    hist_data_long = dff[(dff["los"] <= 70) & (dff["los_category"] == "Long Stay")]["los"]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=hist_data_short, nbinsx=35, name="Short Stay",
        marker_color=ACCENT_TEAL, opacity=0.75,
    ))
    fig_hist.add_trace(go.Histogram(
        x=hist_data_long, nbinsx=35, name="Long Stay",
        marker_color=ACCENT_GOLD, opacity=0.75,
    ))
    fig_hist.update_layout(
        barmode="overlay", template=LIGHT_TEMPLATE,
        margin=dict(l=50, r=10, t=30, b=35),
        xaxis_title="LOS (days)", yaxis_title="",
        title="LOS Distribution by Category", title_font_size=11,
        legend=dict(font=dict(size=8), orientation="h", y=1.0, x=1.0, xanchor="right", bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(range=[0, 72], tickfont=dict(size=8)),
        yaxis=dict(
            tickfont=dict(size=8), automargin=True, type=yaxis_type, rangemode="tozero",
            **({"tickvals": [1, 10, 100, 1000, 10000], "ticktext": ["1", "10", "100", "1k", "10k"]} if use_log else {"tickvals": [2000, 4000, 6000, 8000, 10000]}),
        ),
        bargap=0.03,
    )

    # Violin plot by care unit (shows shape of distribution per unit)
    top_units = dff["first_careunit"].value_counts().head(5).index.tolist()
    violin_df = dff[(dff["first_careunit"].isin(top_units)) & (dff["los"] <= 70)]
    fig_violin = go.Figure()
    for i, unit in enumerate(top_units):
        unit_data = violin_df[violin_df["first_careunit"] == unit]["los"]
        fig_violin.add_trace(go.Violin(
            y=unit_data, name=unit, box_visible=True, meanline_visible=True,
            fillcolor=f"rgba({','.join(str(int(CHART_COLORS[i % len(CHART_COLORS)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.3)",
            line_color=CHART_COLORS[i % len(CHART_COLORS)],
        ))
    fig_violin.update_layout(
        template=LIGHT_TEMPLATE, showlegend=False,
        margin=dict(l=40, r=10, t=30, b=45),
        yaxis_title="LOS (days)", title="LOS Shape by Care Unit (Violin)",
        title_font_size=11, yaxis_type=yaxis_type,
        xaxis=dict(tickfont=dict(size=9), tickangle=0),
        yaxis=dict(
            tickfont=dict(size=8), type=yaxis_type,
            **({"tickvals": [0.1, 1, 10, 70], "ticktext": ["0.1", "1", "10", "70"]} if use_log else {}),
        ),
    )

    # Insight Q3
    pct_under7 = (dff["los"] <= 7).sum() / total * 100 if total else 0
    pct_over30 = (dff["los"] > 30).sum() / total * 100 if total else 0
    pct_over70 = (dff["los"] > 70).sum() / total * 100 if total else 0
    insight_3 = (
        f"Insight Q3: {pct_under7:.1f}% of patients stay ≤7 days (routine cases), "
        f"while {pct_over30:.1f}% stay >30 days and {pct_over70:.1f}% exceed 70 days. "
        f"The wide variation reflects a mix of short acute episodes and complex chronic conditions requiring extended ICU care."
    )

    # ================================================================
    # Supporting: Avg LOS by Top 10 Diagnoses (horizontal bar)
    # ================================================================
    avg_los_by_diag = (
        dff[dff["long_title"].isin(top10_diag)]
        .groupby("long_title")["los"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    avg_los_by_diag.columns = ["Diagnosis", "Avg_LOS"]
    avg_los_by_diag["short"] = avg_los_by_diag["Diagnosis"].str[:30]

    fig_avg_diag = go.Figure()
    for idx, (_, row) in enumerate(avg_los_by_diag.iterrows()):
        fig_avg_diag.add_trace(go.Bar(
            x=[row["Avg_LOS"]],
            y=[row["short"]],
            orientation="h",
            marker_color=CHART_COLORS[idx % len(CHART_COLORS)],
            text=[f"{row['Avg_LOS']:.1f}d"],
            textposition="outside",
            textfont_size=9,
            name=row["short"],
        ))
    fig_avg_diag.update_layout(
        template=LIGHT_TEMPLATE,
        margin=dict(l=10, r=40, t=10, b=140),
        xaxis_title="Avg LOS (days)", yaxis_title="",
        yaxis=dict(showticklabels=False),
        title="Avg LOS by Top 10 Diagnoses", title_font_size=6,
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.55,
            x=0,
            font=dict(size=7),
            traceorder="reversed",
            itemwidth=30,
            title=dict(text="<b>Diagnoses:</b>", font=dict(size=8, color=TEXT_MUTED)),
        ),
    )

    # ================================================================
    # Summary & Breakdown
    # ================================================================
    short_count = (dff["los_category"] == "Short Stay").sum()
    long_count = (dff["los_category"] == "Long Stay").sum()
    short_pct = short_count / total * 100 if total else 0
    long_pct_cat = long_count / total * 100 if total else 0
    busiest = dff["first_careunit"].value_counts().idxmax() if total else "N/A"

    takeaways = html.Ol(
        [
            html.Li(
                f"Q1: Primary diagnosis strongly influences LOS – avg LOS ranges from "
                f"{avg_by_diag.iloc[-1]['mean']:.1f}d to {avg_by_diag.iloc[0]['mean']:.1f}d across top diagnoses.",
                style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "4px"},
            ),
            html.Li(
                f"Q2: MICU and SICU have distinct disease profiles; "
                f"MICU skews toward medical conditions, SICU toward surgical/trauma.",
                style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "4px"},
            ),
            html.Li(
                f"Q3: {pct_under7:.0f}% stay ≤7d (acute), {pct_over30:.1f}% stay >30d (complex). "
                f"The variation reflects case severity and diagnosis type.",
                style={"color": TEXT_MUTED, "fontSize": "11px"},
            ),
        ],
        style={"paddingLeft": "16px"},
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
        fig_box, diag_key_1, insight_1,
        fig_micu_sicu, insight_2,
        fig_hist, fig_violin, insight_3,
        fig_avg_diag,
        diag_key_table,
        takeaways,
        breakdown,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
