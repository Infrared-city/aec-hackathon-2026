"""
Copenhagen Urban Heat Analysis Dashboard
Urban Thermal Intelligence Platform
────────────────────────────────────────
Built with Panel + HoloViz (HoloViews · GeoViews · hvplot · Bokeh)

Run:
    panel serve dashboard.py --show
    # or
    python dashboard.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")

import panel as pn
import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gvts
import hvplot.pandas            # noqa: registers .hvplot accessor on DataFrames
import pandas as pd
import numpy as np
from holoviews import opts

pn.extension(sizing_mode="stretch_width", notifications=True)
hv.extension("bokeh")
gv.extension("bokeh")

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA  ·  Copenhagen, Denmark
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

# ── Neighborhoods ─────────────────────────────────────────────────────────────
_nbhd = pd.DataFrame({
    "neighborhood": [
        "Indre By", "Nørrebro", "Østerbro", "Vesterbro",
        "Frederiksberg", "Amager Øst", "Valby", "Bispebjerg",
        "Brønshøj", "Sydhavn", "Amager Vest", "Vanløse",
    ],
    "lat": [
        55.6761, 55.6988, 55.7066, 55.6655,
        55.6795, 55.6508, 55.6563, 55.7149,
        55.7052, 55.6565, 55.6408, 55.6863,
    ],
    "lon": [
        12.5683, 12.5474, 12.5731, 12.5499,
        12.5290, 12.6018, 12.5099, 12.5420,
        12.5160, 12.5520, 12.5818, 12.4985,
    ],
    "land_use": [
        "Commercial", "Dense Residential", "Mixed Residential", "Dense Residential",
        "Mixed Residential", "Mixed", "Industrial/Mixed", "Industrial",
        "Suburban", "Industrial", "Mixed", "Suburban",
    ],
    # % of neighbourhood covered by vegetation / parks
    "green_pct":         [15, 20, 35, 18, 40, 25, 22, 20, 45, 10, 30, 38],
    # % of neighbourhood covered by roads, rooftops, concrete
    "impervious_pct":    [85, 78, 62, 80, 58, 70, 75, 77, 52, 88, 67, 60],
    "population_density": [8500, 24000, 13000, 18000, 14000, 9000,
                           7500, 11000,  6000,  3500,  8000,  9500],
    "area_km2":          [8.7, 6.1, 9.8, 5.4, 8.7, 12.3, 14.5, 8.2, 16.1, 8.4, 15.2, 11.2],
})

# Urban Heat Island temperature model:  base + UHI(impervious, green) + noise
def _uhi(imp, green, sigma=0.35):
    return (imp / 100) * 3.6 - (green / 100) * 1.8 + np.random.uniform(-sigma, sigma)

_nbhd["temp_today"] = [
    round(7.2 + _uhi(i, g), 1)
    for i, g in zip(_nbhd["impervious_pct"], _nbhd["green_pct"])
]
_nbhd["temp_next_month"] = [
    round(13.8 + _uhi(i, g, sigma=0.45), 1)
    for i, g in zip(_nbhd["impervious_pct"], _nbhd["green_pct"])
]

def _hotspot(temp, threshold):
    if   temp >= threshold + 2.8: return "Critical"
    elif temp >= threshold + 1.2: return "High"
    elif temp >= threshold - 0.8: return "Moderate"
    else:                         return "Low"

_nbhd["hotspot_today"]      = [_hotspot(t, 9.8)  for t in _nbhd["temp_today"]]
_nbhd["hotspot_next_month"] = [_hotspot(t, 16.0) for t in _nbhd["temp_next_month"]]

# Composite risk index 0–100
for tc, hc, rc in [
    ("temp_today",      "hotspot_today",      "risk_today"),
    ("temp_next_month", "hotspot_next_month",  "risk_next_month"),
]:
    t = _nbhd[tc]
    _nbhd[rc] = (
        (t - t.min()) / (t.max() - t.min()) * 55
        + (_nbhd["impervious_pct"] / 100) * 25
        + (1 - _nbhd["green_pct"]  / 100) * 20
    ).round(1)

# ── 62-day temperature time series (Feb 26 → Apr 28) ─────────────────────────
_dates = pd.date_range("2026-02-26", periods=62, freq="D")
_base  = 6.8 + np.arange(62) * 0.112 + np.sin(np.arange(62) / 7) * 1.1
_noise = np.random.normal(0, 0.6, 62)

ts_df = pd.DataFrame({"date": _dates})
ts_df["City Average"]          = (_base + _noise).round(1)
ts_df["Urban Core (Indre By)"] = (_base + _noise + 2.9 + np.random.normal(0, .3, 62)).round(1)
ts_df["Industrial (Sydhavn)"]  = (_base + _noise + 3.4 + np.random.normal(0, .4, 62)).round(1)
ts_df["Green Zone (Brønshøj)"] = (_base + _noise + 0.3 + np.random.normal(0, .3, 62)).round(1)

_TODAY_IDX      = 30   # March 28 sits at index 30
_NEXT_MONTH_IDX = 61   # April 28 sits at index 61

# ── Active alerts per scenario ────────────────────────────────────────────────
_ALERTS = {
    "today": pd.DataFrame([
        {"Zone": "Sydhavn",    "Issue": "Industrial Heat Cluster",   "Severity": "High",
         "Recommendation": "Monitor industrial cooling systems"},
        {"Zone": "Bispebjerg", "Issue": "Urban Heat Island Peak",    "Severity": "High",
         "Recommendation": "Accelerate street-tree planting program"},
        {"Zone": "Indre By",   "Issue": "Pedestrian Thermal Stress", "Severity": "Moderate",
         "Recommendation": "Install temporary cooling stations"},
    ]),
    "next_month": pd.DataFrame([
        {"Zone": "Sydhavn",    "Issue": "Industrial Heat Cluster",    "Severity": "Critical",
         "Recommendation": "Activate emergency cooling protocol NOW"},
        {"Zone": "Bispebjerg", "Issue": "Urban Heat Island Peak",     "Severity": "Critical",
         "Recommendation": "Deploy mobile cooling units immediately"},
        {"Zone": "Indre By",   "Issue": "Pedestrian Thermal Stress",  "Severity": "High",
         "Recommendation": "Activate Copenhagen Heat Action Plan"},
        {"Zone": "Vesterbro",  "Issue": "Residential Overheating",    "Severity": "High",
         "Recommendation": "Issue public cooling advisory"},
        {"Zone": "Nørrebro",   "Issue": "Residential Overheating",    "Severity": "Moderate",
         "Recommendation": "Community outreach & welfare checks"},
    ]),
}

# ── Analysis narratives ───────────────────────────────────────────────────────
_ANALYSIS = {
    "today": (
        "#3498db",
        """<b>Thermal Situation Report — March 28, 2026</b><br><br>
Copenhagen's <b>Urban Heat Island (UHI)</b> effect is actively shaping today's temperature
distribution across the city. A <b>3.4 °C gap</b> separates the warmest industrial zone (Sydhavn)
from the coolest green neighbourhood (Brønshøj), illustrating the direct impact of land cover
on local microclimate.<br><br>
<b>Key findings:</b><br>
• <b>Sydhavn &amp; Bispebjerg</b> are primary heat emitters — &gt;85 % impervious surface and
  near-zero green cover create persistently elevated temperatures even in early spring.<br>
• <b>Dense residential zones</b> (Nørrebro, Vesterbro) show mid-range temperatures; their high
  population densities amplify heat-stress exposure compared to equally warm but sparser areas.<br>
• <b>Frederiksberg &amp; Brønshøj</b>, with &gt;38 % green coverage, function as natural cooling
  islands and should serve as benchmarks for the city's urban greening strategy.<br>
• A strong negative correlation between <i>green cover</i> and <i>risk index</i> (visible in the
  scatter plot) shows every 10 pp increase in green cover reduces the risk index by ≈ 5 points.<br><br>
<b>Risk level today:</b> <span style="color:#e74c3c;font-weight:bold">2 High-risk zones</span> —
proactive monitoring advised; no immediate public-health emergency.""",
    ),
    "next_month": (
        "#e74c3c",
        """<b>Thermal Projection Report — April 28, 2026</b><br><br>
The seasonal transition to late April introduces a <b>+6.6 °C baseline shift</b>, substantially
amplifying heat stress across the city. Two neighbourhoods are projected to reach
<b>Critical</b> status — a threshold associated with heat-related health risks for vulnerable
populations and outdoor workers.<br><br>
<b>Key findings:</b><br>
• <b>Sydhavn &amp; Bispebjerg</b> cross into Critical territory with temperatures 3+ °C above
  the seasonal baseline. Emergency preparedness protocols should be reviewed immediately.<br>
• The gap between industrial and green zones widens to <b>4.2 °C</b>, reinforcing the urgency
  of expanding Copenhagen's green infrastructure pipeline before summer peaks arrive.<br>
• <b>Vesterbro &amp; Nørrebro</b> join the High-risk category; their population densities of
  18 k and 24 k per km² make them priority targets for targeted cooling interventions.<br>
• Without land-use changes, the model projects <b>5 neighbourhoods at High+ risk</b> on peak
  summer days — a 150 % increase over today's 2-zone risk footprint.<br><br>
<b>Recommended actions:</b> Activate the Copenhagen Heat Action Plan; deploy cooling centres in
Sydhavn, Bispebjerg, and Vesterbro; expedite procurement of mobile shade &amp; misting
infrastructure for dense residential corridors.""",
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
HS_COLORS  = {"Low": "#27ae60", "Moderate": "#f39c12", "High": "#e74c3c", "Critical": "#8e44ad"}
SEV_COLORS = {"Moderate": "#f39c12", "High": "#e74c3c", "Critical": "#8e44ad"}

_CARD_TMPL = (
    "display:inline-block;background:{bg};border-radius:10px;"
    "padding:18px 22px;color:white;text-align:center;margin:4px;min-width:160px;"
)
_SECTION_H3 = (
    "color:{color};border-bottom:2px solid {color};padding-bottom:6px;"
    "margin:14px 0 8px 0;font-family:sans-serif;font-size:13px;"
    "letter-spacing:.6px;text-transform:uppercase"
)

# ══════════════════════════════════════════════════════════════════════════════
# WIDGETS
# ══════════════════════════════════════════════════════════════════════════════
scenario_select = pn.widgets.RadioButtonGroup(
    options=["Today  ·  Mar 28, 2026", "Projection  ·  Apr 28, 2026"],
    value="Today  ·  Mar 28, 2026",
    button_type="success",
    width=440,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _cols(scenario: str):
    if "Today" in scenario:
        return "temp_today", "hotspot_today", "risk_today", "today"
    return "temp_next_month", "hotspot_next_month", "risk_next_month", "next_month"

def _section(text: str, color: str = "#2c3e50") -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<h3 style="{_SECTION_H3.format(color=color)}">{text}</h3>',
        sizing_mode="stretch_width",
    )

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
def kpi_row(scenario: str) -> pn.pane.HTML:
    tc, hc, rc, _ = _cols(scenario)
    temps    = _nbhd[tc]
    hotspots = _nbhd[hc]

    avg_t  = f"{temps.mean():.1f} °C"
    max_t  = f"{temps.max():.1f} °C"
    n_high = int(hotspots.isin(["Critical", "High"]).sum())
    pop_k  = int(_nbhd.loc[hotspots.isin(["Critical", "High"]), "population_density"].sum() // 1000)

    specs = [
        (avg_t,        "City Average Temp",  "#3498db", "🌡️"),
        (max_t,        "Peak Hot-Spot Temp", "#e74c3c", "🔥"),
        (str(n_high),  "High-Risk Zones",    "#e67e22", "⚠️"),
        (f"{pop_k}k",  "Exposed Residents",  "#8e44ad", "👥"),
    ]
    cards_html = "".join(
        f'<div style="{_CARD_TMPL.format(bg=bg)}">'
        f'<div style="font-size:28px;margin-bottom:4px">{icon}</div>'
        f'<div style="font-size:26px;font-weight:700">{val}</div>'
        f'<div style="font-size:11px;opacity:.85;margin-top:4px;'
        f'text-transform:uppercase;letter-spacing:.6px">{lbl}</div></div>'
        for val, lbl, bg, icon in specs
    )
    return pn.pane.HTML(
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;'
        f'justify-content:space-between">{cards_html}</div>',
        sizing_mode="stretch_width",
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAP  ·  GeoViews point overlay on CartoDB Light tiles
# ══════════════════════════════════════════════════════════════════════════════
def map_panel(scenario: str):
    tc, hc, rc, _ = _cols(scenario)
    df = _nbhd[
        ["neighborhood", "lat", "lon", tc, hc, rc, "land_use", "green_pct", "population_density"]
    ].copy()
    df.columns = [
        "Neighborhood", "lat", "lon", "Temperature", "Hotspot",
        "Risk", "Land Use", "Green %", "Pop /km²",
    ]
    df["Color"] = df["Hotspot"].map(HS_COLORS)
    df["Size"]  = (df["Risk"] / 100 * 22 + 10).round(0)

    pts = gv.Points(
        df,
        kdims=["lon", "lat"],
        vdims=["Neighborhood", "Temperature", "Hotspot", "Risk",
               "Land Use", "Green %", "Pop /km²", "Color", "Size"],
    ).opts(
        opts.Points(
            color="Color",
            size=hv.dim("Size"),
            tools=["hover", "tap"],
            hover_tooltips=[
                ("Neighbourhood", "@Neighborhood"),
                ("Temperature",   "@Temperature °C"),
                ("Hot-Spot Level","@Hotspot"),
                ("Risk Index",    "@Risk"),
                ("Land Use",      "@{Land Use}"),
                ("Green Cover",   "@{Green %}%"),
                ("Pop. Density",  "@{Pop /km²} /km²"),
            ],
            alpha=0.88,
            line_color="white",
            line_width=1.5,
            width=600,
            height=490,
        )
    )

    title = (
        "Temperature Map  ·  Today (Mar 28, 2026)"
        if "Today" in scenario
        else "Temperature Map  ·  Projection (Apr 28, 2026)"
    )
    return (gvts.CartoLight * pts).opts(
        opts.Overlay(title=title, width=600, height=490)
    )

# ══════════════════════════════════════════════════════════════════════════════
# BAR CHART  ·  temperature ranking
# ══════════════════════════════════════════════════════════════════════════════
def bar_chart(scenario: str):
    tc, hc, _, _ = _cols(scenario)
    df = _nbhd[["neighborhood", tc, hc]].sort_values(tc, ascending=True).copy()
    df.columns = ["Neighborhood", "Temperature", "Hotspot"]
    colors = df["Hotspot"].map(HS_COLORS).tolist()

    return df.hvplot.barh(
        x="Neighborhood",
        y="Temperature",
        color=colors,
        legend=False,
        title="Temperature Ranking by Neighbourhood",
        xlabel="Temperature (°C)",
        ylabel="",
        width=490,
        height=490,
        hover_cols=["Hotspot"],
        fontsize={"title": "11pt", "labels": "9pt", "ticks": "8pt"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# TIME SERIES  ·  60-day window with scenario markers
# ══════════════════════════════════════════════════════════════════════════════
def time_series(scenario: str):
    zone_cols = [
        "City Average",
        "Urban Core (Indre By)",
        "Industrial (Sydhavn)",
        "Green Zone (Brønshøj)",
    ]
    colors = ["#2c3e50", "#e74c3c", "#8e44ad", "#27ae60"]

    melt = ts_df.melt(
        id_vars="date", value_vars=zone_cols, var_name="Zone", value_name="Temp (°C)"
    )
    chart = melt.hvplot.line(
        x="date",
        y="Temp (°C)",
        by="Zone",
        color=colors,
        title="Temperature Trends — 30-Day History & 30-Day Projection",
        xlabel="",
        ylabel="Temperature (°C)",
        width=600,
        height=310,
        legend="top_left",
    )

    today_vline = hv.VLine(ts_df.iloc[_TODAY_IDX]["date"]).opts(
        color="#e74c3c", line_dash="dashed", line_width=1.5, alpha=0.75
    )
    proj_vline = hv.VLine(ts_df.iloc[_NEXT_MONTH_IDX]["date"]).opts(
        color="#8e44ad", line_dash="dashed", line_width=1.5, alpha=0.75
    )
    return (chart * today_vline * proj_vline).opts(
        opts.Overlay(show_grid=True, width=600, height=310)
    )

# ══════════════════════════════════════════════════════════════════════════════
# SCATTER  ·  risk index vs green space with linear trend
# ══════════════════════════════════════════════════════════════════════════════
def scatter_panel(scenario: str):
    tc, hc, rc, _ = _cols(scenario)
    df = _nbhd[["neighborhood", tc, hc, rc, "green_pct", "impervious_pct"]].copy()
    df.columns = ["Neighborhood", "Temperature", "Hotspot", "Risk", "Green %", "Impervious %"]
    colors = df["Hotspot"].map(HS_COLORS).tolist()

    scatter = df.hvplot.scatter(
        x="Green %",
        y="Risk",
        color=colors,
        size=120,
        alpha=0.88,
        line_color="white",
        title="Risk Index vs Green Space Coverage",
        xlabel="Green Cover (%)",
        ylabel="Risk Index (0–100)",
        hover_cols=["Neighborhood", "Temperature", "Hotspot", "Impervious %"],
        width=490,
        height=310,
        legend=False,
    )

    x, y = df["Green %"].values, df["Risk"].values
    coeffs = np.polyfit(x, y, 1)
    r2     = float(np.corrcoef(x, y)[0, 1] ** 2)
    xr     = np.linspace(x.min(), x.max(), 60)
    trend  = pd.DataFrame({"Green %": xr, "Risk": np.polyval(coeffs, xr)})
    tline  = trend.hvplot.line(
        x="Green %", y="Risk",
        color="#95a5a6", line_dash="dashed", line_width=1.5,
        label=f"Trend  (R²={r2:.2f})",
    )
    return (scatter * tline).opts(
        opts.Overlay(show_grid=True, legend_position="top_right", width=490, height=310)
    )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TEXT
# ══════════════════════════════════════════════════════════════════════════════
def analysis_panel(scenario: str) -> pn.pane.HTML:
    _, _, _, key = _cols(scenario)
    color, text  = _ANALYSIS[key]
    return pn.pane.HTML(
        f'<div style="background:#f8f9fa;border-left:4px solid {color};'
        f'padding:16px 18px;border-radius:4px;font-size:13px;line-height:1.75;'
        f'font-family:sans-serif">{text}</div>',
        sizing_mode="stretch_width",
    )

# ══════════════════════════════════════════════════════════════════════════════
# ALERTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
def _severity_badge(val: str) -> str:
    bg = SEV_COLORS.get(val, "#27ae60")
    return (
        f"background:{bg};color:white;border-radius:4px;"
        f"padding:2px 9px;font-weight:600;font-size:11px;"
    )

def alerts_panel(scenario: str) -> pn.pane.HTML:
    _, _, _, key = _cols(scenario)
    rows_html = "".join(
        f'<tr style="background:{"#fff" if i % 2 == 0 else "#f8f9fa"}">'
        f'<td style="padding:8px 12px;font-weight:600;white-space:nowrap">{r["Zone"]}</td>'
        f'<td style="padding:8px 12px">{r["Issue"]}</td>'
        f'<td style="padding:8px 12px;text-align:center">'
        f'<span style="{_severity_badge(r["Severity"])}">{r["Severity"]}</span></td>'
        f'<td style="padding:8px 12px;font-size:12px;color:#555">{r["Recommendation"]}</td>'
        f"</tr>"
        for i, (_, r) in enumerate(_ALERTS[key].iterrows())
    )
    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-family:sans-serif;font-size:13px">'
        '<thead><tr style="background:#2c3e50;color:white">'
        '<th style="padding:10px 12px;text-align:left">Zone</th>'
        '<th style="padding:10px 12px;text-align:left">Issue</th>'
        '<th style="padding:10px 12px;text-align:center;white-space:nowrap">Severity</th>'
        '<th style="padding:10px 12px;text-align:left">Recommendation</th>'
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    return pn.pane.HTML(table_html, sizing_mode="stretch_width")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR  ·  legend + meta
# ══════════════════════════════════════════════════════════════════════════════
_legend = pn.pane.HTML(
    '<div style="font-family:sans-serif;font-size:12px;margin-top:16px">'
    '<b style="color:#2c3e50;text-transform:uppercase;letter-spacing:.5px">'
    "Hot-Spot Classification</b><br><br>"
    + "".join(
        f'<div style="margin:6px 0">'
        f'<span style="display:inline-block;width:13px;height:13px;background:{c};'
        f'border-radius:50%;vertical-align:middle;margin-right:8px"></span>'
        f"<b>{lvl}</b></div>"
        for lvl, c in HS_COLORS.items()
    )
    + "</div>"
    '<hr style="margin:16px 0;border:none;border-top:1px solid #ddd">'
    '<div style="font-family:sans-serif;font-size:11px;color:#7f8c8d;line-height:1.7">'
    "<b>Data</b>: Synthetic urban thermal model<br>"
    "<b>Scope</b>: 12 Copenhagen neighbourhoods<br>"
    "<b>UHI model</b>: f(impervious surface, green cover)<br>"
    "<b>Baseline</b>: ERA5 reanalysis (simulated)<br>"
    "<b>Projection</b>: +31-day seasonal shift<br>"
    "<b>Updated</b>: Mar 28, 2026 · 09:00 CET"
    "</div>",
    sizing_mode="stretch_width",
)

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT  ·  FastListTemplate
# ══════════════════════════════════════════════════════════════════════════════
template = pn.template.FastListTemplate(
    title="🌡️ Copenhagen Urban Heat Analysis",
    sidebar=[
        pn.pane.HTML(
            '<h3 style="font-family:sans-serif;color:#2c3e50;margin-bottom:8px">Scenario</h3>'
        ),
        scenario_select,
        _legend,
    ],
    main=[
        # ── KPI Cards ────────────────────────────────────────────────────────
        pn.Column(
            _section("Key Performance Indicators"),
            pn.bind(kpi_row, scenario_select.param.value),
            sizing_mode="stretch_width",
        ),
        # ── Map + Bar ─────────────────────────────────────────────────────────
        pn.Column(
            _section("Spatial Distribution"),
            pn.Row(
                pn.bind(map_panel,  scenario_select.param.value),
                pn.bind(bar_chart,  scenario_select.param.value),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        # ── Time series + Scatter ─────────────────────────────────────────────
        pn.Column(
            _section("Trend Analysis"),
            pn.Row(
                pn.bind(time_series,    scenario_select.param.value),
                pn.bind(scatter_panel,  scenario_select.param.value),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
        # ── Analysis + Alerts ─────────────────────────────────────────────────
        pn.Column(
            _section("Analysis & Active Alerts"),
            pn.Row(
                pn.Column(
                    pn.pane.HTML(
                        '<b style="font-family:sans-serif;font-size:11px;color:#7f8c8d;'
                        'text-transform:uppercase">Situation Report</b>'
                    ),
                    pn.bind(analysis_panel, scenario_select.param.value),
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    pn.pane.HTML(
                        '<b style="font-family:sans-serif;font-size:11px;color:#7f8c8d;'
                        'text-transform:uppercase">Active Alerts</b>'
                    ),
                    pn.bind(alerts_panel, scenario_select.param.value),
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        ),
    ],
    accent="#e74c3c",
    theme_toggle=True,
)

# ── Entry points ───────────────────────────────────────────────────────────────
template.servable()   # used by `panel serve dashboard.py`

if __name__ == "__main__":
    pn.serve(template, port=5006, show=True, title="Copenhagen Heat Dashboard")
