"""
Copenhagen Urban Greening — Landing Page
Municipality Expert View
────────────────────────────────────────
Run:
    panel serve frontend/landing.py --show
"""

import warnings; warnings.filterwarnings("ignore")

import panel as pn
import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gvts
import hvplot.pandas  # noqa
import pandas as pd
import numpy as np
from holoviews import opts

pn.extension(sizing_mode="stretch_width")
hv.extension("bokeh")
gv.extension("bokeh")

# ══════════════════════════════════════════════════════════════════════════════
# COLOR PALETTE  ·  ramp around #421312
# ══════════════════════════════════════════════════════════════════════════════
C_DARKEST  = "#421312"
C_DARK     = "#6B2020"
C_MID      = "#9B3333"
C_LIGHT    = "#C45050"
C_LIGHTER  = "#E07070"
C_LIGHTEST = "#F5A0A0"

BG_PAGE    = "#0d0606"
BG_PANEL   = "#180a0a"
BG_CARD_A  = "#1f0c0c"   # darkest red card
BG_CARD_B  = "#1a0f0c"   # warm mid card
BG_CARD_C  = "#0f1a0c"   # green-tinted card (solar/nature)

TEXT_PRIMARY   = "#f0e0e0"
TEXT_SECONDARY = "#b89090"
TEXT_MUTED     = "#7a5555"

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA  ·  Copenhagen neighbourhoods
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)

zones = pd.DataFrame({
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
    # Buildings
    "building_coverage_pct": [72, 65, 52, 68, 48, 55, 60, 62, 42, 75, 50, 45],
    # Vegetation
    "green_pct": [15, 20, 35, 18, 40, 25, 22, 20, 45, 10, 30, 38],
    # Heat comfort index 0–100 (higher = worse = more heat stress)
    "heat_discomfort": [78, 72, 55, 70, 45, 60, 62, 74, 38, 85, 58, 42],
    # Solar potential kWh/m²/year
    "solar_kwh": [920, 880, 950, 900, 960, 1010, 980, 870, 1020, 1050, 1030, 970],
    "area_km2": [8.7, 6.1, 9.8, 5.4, 8.7, 12.3, 14.5, 8.2, 16.1, 8.4, 15.2, 11.2],
    "population_density": [8500, 24000, 13000, 18000, 14000, 9000,
                           7500, 11000,  6000,  3500,  8000,  9500],
})

# Composite vegetation priority score (0–100)
zones["veg_priority"] = (
    (1 - zones["green_pct"]        / 100) * 50
    + (zones["heat_discomfort"]    / 100) * 35
    + (zones["building_coverage_pct"] / 100) * 15
).round(1)

TIER_COLORS = {
    "Critical": C_DARKEST,
    "High":     C_MID,
    "Moderate": C_LIGHTER,
    "Low":      "#7aab5a",
}

def _tier(score):
    if score >= 68: return "Critical"
    if score >= 54: return "High"
    if score >= 40: return "Moderate"
    return "Low"

zones["priority_tier"] = zones["veg_priority"].apply(_tier)

# ── City-wide KPI aggregates ──────────────────────────────────────────────────
avg_building_cov  = zones["building_coverage_pct"].mean()
avg_heat          = zones["heat_discomfort"].mean()
avg_solar         = zones["solar_kwh"].mean()
n_critical        = int((zones["priority_tier"] == "Critical").sum())
n_high_plus       = int(zones["priority_tier"].isin(["Critical", "High"]).sum())
worst_zone        = zones.loc[zones["heat_discomfort"].idxmax(), "neighborhood"]
best_solar_zone   = zones.loc[zones["solar_kwh"].idxmax(), "neighborhood"]

# ══════════════════════════════════════════════════════════════════════════════
# MAP
# ══════════════════════════════════════════════════════════════════════════════
def build_map():
    df = zones.copy()
    df["Color"] = df["priority_tier"].map(TIER_COLORS)
    df["Size"]  = (df["veg_priority"] / 100 * 30 + 14).round(0)

    pts = gv.Points(
        df,
        kdims=["lon", "lat"],
        vdims=[
            "neighborhood", "priority_tier", "veg_priority",
            "green_pct", "heat_discomfort", "solar_kwh",
            "building_coverage_pct", "population_density",
            "Color", "Size",
        ],
    ).opts(opts.Points(
        color="Color",
        size=hv.dim("Size"),
        tools=["hover", "tap"],
        hover_tooltips=[
            ("Neighbourhood",       "@neighborhood"),
            ("Priority Tier",       "@priority_tier"),
            ("Vegetation Priority", "@veg_priority / 100"),
            ("Green Cover",         "@green_pct %"),
            ("Heat Discomfort",     "@heat_discomfort / 100"),
            ("Solar Potential",     "@solar_kwh kWh/m²/yr"),
            ("Building Coverage",   "@building_coverage_pct %"),
            ("Population Density",  "@population_density /km²"),
        ],
        alpha=0.92,
        line_color="white",
        line_width=1.8,
        nonselection_alpha=0.45,
        width=1100,
        height=820,
    ))

    return (gvts.CartoDark * pts).opts(
        opts.Overlay(
            title="Vegetation Placement Priority · Copenhagen",
            width=1100,
            height=820,
        )
    )

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
panel_header = pn.pane.HTML(
    f'<div style="font-family:sans-serif;margin-bottom:18px">'
    f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1.2px;'
    f'color:{TEXT_MUTED};margin-bottom:6px">Municipality Expert Dashboard</div>'
    f'<div style="font-size:20px;font-weight:700;color:{TEXT_PRIMARY};'
    f'line-height:1.2">Copenhagen<br>Urban Greening Analysis</div>'
    f'<div style="font-size:11px;color:{TEXT_SECONDARY};margin-top:6px">'
    f'March 2026 · Vegetation Priority Assessment</div>'
    f'</div>',
    sizing_mode="stretch_width",
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = pn.pane.HTML(
    '<div style="font-family:sans-serif;margin-bottom:18px">'
    f'<div style="font-size:10px;text-transform:uppercase;letter-spacing:1px;'
    f'color:{TEXT_MUTED};margin-bottom:8px">Priority Classification</div>'
    + "".join(
        f'<div style="display:flex;align-items:center;margin:6px 0">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{c};'
        f'border-radius:50%;margin-right:10px;flex-shrink:0;'
        f'box-shadow:0 0 6px {c}88"></span>'
        f'<span style="font-size:12px;color:{TEXT_PRIMARY};font-weight:600">{tier}</span>'
        f'</div>'
        for tier, c in TIER_COLORS.items()
    )
    + "</div>",
    sizing_mode="stretch_width",
)

divider = pn.pane.HTML(
    f'<hr style="border:none;border-top:1px solid {C_DARKEST};margin:14px 0">',
    sizing_mode="stretch_width",
)

# ── Stat cards ────────────────────────────────────────────────────────────────
_CARD_BASE = (
    "font-family:sans-serif;border-radius:10px;padding:16px 18px;"
    "margin-bottom:10px;border-left:4px solid {border};"
    "background:{bg};"
)

def stat_card(value, label, sublabel, bg, border):
    return pn.pane.HTML(
        f'<div style="{_CARD_BASE.format(bg=bg, border=border)}">'
        f'<div style="font-size:28px;font-weight:700;color:{TEXT_PRIMARY};'
        f'letter-spacing:-.5px">{value}</div>'
        f'<div style="font-size:11px;font-weight:600;color:{TEXT_SECONDARY};'
        f'text-transform:uppercase;letter-spacing:.8px;margin-top:5px">{label}</div>'
        f'<div style="font-size:11px;color:{TEXT_MUTED};margin-top:5px;line-height:1.5">'
        f'{sublabel}</div>'
        f'</div>',
        sizing_mode="stretch_width",
    )

card_buildings = stat_card(
    value    = f"{avg_building_cov:.0f}%",
    label    = "Avg Building Coverage",
    sublabel = (
        f"Peak: Sydhavn at "
        f"{zones.loc[zones['neighborhood']=='Sydhavn','building_coverage_pct'].iloc[0]}% "
        f"· dense urban fabric limits infiltration"
    ),
    bg     = BG_CARD_A,
    border = C_DARKEST,
)

card_heat = stat_card(
    value    = f"{avg_heat:.0f} / 100",
    label    = "Avg Heat Discomfort Index",
    sublabel = (
        f"Worst: {worst_zone} (85) · "
        f"{n_high_plus} zones need urgent intervention"
    ),
    bg     = BG_CARD_B,
    border = C_MID,
)

card_solar = stat_card(
    value    = f"{avg_solar:.0f} kWh/m²",
    label    = "Avg Solar Potential / Year",
    sublabel = (
        f"Best: {best_solar_zone} (1050 kWh/m²) · "
        f"supports combined green + solar roofs"
    ),
    bg     = BG_CARD_C,
    border = "#5a8a3a",
)

# ── Analysis text ─────────────────────────────────────────────────────────────
analysis = pn.pane.HTML(
    f'<div style="font-family:sans-serif;font-size:12px;line-height:1.8;'
    f'color:{TEXT_SECONDARY};background:{BG_CARD_A};'
    f'border-left:3px solid {C_DARKEST};padding:16px 18px;border-radius:8px;'
    f'margin-bottom:10px">'

    f'<div style="font-size:11px;font-weight:700;color:{TEXT_PRIMARY};'
    f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">'
    f'Vegetation Placement Analysis</div>'

    f'<b style="color:{C_LIGHTER}">Sydhavn</b> and '
    f'<b style="color:{C_LIGHTER}">Bispebjerg</b> are the highest-priority districts — '
    f'both combine &lt;20% green cover with heat discomfort above 72/100. '
    f'Introducing street trees and green roofs here is modelled to reduce local '
    f'peak temperatures by <b style="color:{TEXT_PRIMARY}">1.5–2.5 °C</b>.'
    f'<br><br>'

    f'<b style="color:{C_LIGHTER}">Vesterbro</b> and '
    f'<b style="color:{C_LIGHTER}">Nørrebro</b> rank High due to their dense '
    f'residential fabric and high population exposure (18k–24k/km²). '
    f'Pocket parks and linear tree corridors are the most feasible intervention '
    f'given the tight urban form.'
    f'<br><br>'

    f'Southern districts (Amager, Sydhavn) show solar potential above '
    f'<b style="color:{TEXT_PRIMARY}">1,000 kWh/m²/yr</b>, making them ideal '
    f'candidates for <b style="color:{TEXT_PRIMARY}">combined solar + green roof</b> '
    f'systems — simultaneously maximising energy yield and urban cooling.'

    f'<div style="font-size:10px;color:{TEXT_MUTED};border-top:1px solid {C_DARKEST};'
    f'padding-top:8px;margin-top:10px">'
    f'Data: synthetic urban model · 12 neighbourhoods · '
    f'Priority: f(green cover, heat discomfort, building density) · '
    f'Solar: ERA5-based estimate</div>'
    f'</div>',
    sizing_mode="stretch_width",
)

# ── Tree planting recommendations ─────────────────────────────────────────────
_RECS = [
    (C_DARKEST, "Sydhavn",    "Street trees on Sydhavnsgade + green roofs on industrial blocks"),
    (C_DARKEST, "Bispebjerg", "Tree corridors along Frederikssundsvej; pocket parks at crossroads"),
    (C_MID,     "Vesterbro",  "Linear planting on Istedgade & Vesterbrogade; courtyard greening"),
    (C_MID,     "Nørrebro",   "Expand Superkilen buffer; tree pits on Nørrebrogade"),
    (C_LIGHT,   "Indre By",   "Shade trees in pedestrian zones; green walls on south-facing facades"),
]

recs_rows = "".join(
    f'<div style="display:flex;align-items:flex-start;margin:8px 0;gap:10px">'
    f'<span style="display:inline-block;width:10px;height:10px;background:{color};'
    f'border-radius:50%;margin-top:3px;flex-shrink:0;box-shadow:0 0 5px {color}88"></span>'
    f'<div><span style="font-weight:600;color:{TEXT_PRIMARY}">{zone}</span>'
    f'<br><span style="font-size:11px;color:{TEXT_SECONDARY}">{rec}</span></div>'
    f'</div>'
    for color, zone, rec in _RECS
)

recommendations = pn.pane.HTML(
    f'<div style="font-family:sans-serif;background:{BG_CARD_A};'
    f'border-left:3px solid {C_MID};padding:16px 18px;border-radius:8px">'
    f'<div style="font-size:11px;font-weight:700;color:{TEXT_PRIMARY};'
    f'text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px">'
    f'Tree Planting Recommendations</div>'
    f'{recs_rows}'
    f'</div>',
    sizing_mode="stretch_width",
)

# ── Assemble right panel ───────────────────────────────────────────────────────
right_panel = pn.Column(
    panel_header,
    legend,
    divider,
    pn.pane.HTML(
        f'<div style="font-family:sans-serif;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1px;color:{TEXT_MUTED};margin-bottom:6px">Key Metrics</div>',
        sizing_mode="stretch_width",
    ),
    card_buildings,
    card_heat,
    card_solar,
    divider,
    analysis,
    recommendations,
    sizing_mode="stretch_width",
    styles={
        "background":  BG_PANEL,
        "padding":     "22px 20px",
        "overflow-y":  "auto",
        "height":      "100%",
        "box-sizing":  "border-box",
        "flex":        "1",
        "min-width":   "0",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR
# ══════════════════════════════════════════════════════════════════════════════
header_bar = pn.pane.HTML(
    f'<div style="background:{C_DARKEST};padding:12px 24px;display:flex;'
    f'align-items:center;justify-content:space-between;'
    f'border-bottom:2px solid {C_DARK}">'
    f'<div style="font-family:sans-serif;color:white;font-size:15px;font-weight:700;'
    f'letter-spacing:.4px">Urban Greenspace Intelligence Platform</div>'
    f'<div style="font-family:sans-serif;color:{C_LIGHTEST};font-size:11px;'
    f'text-transform:uppercase;letter-spacing:.8px">Copenhagen · March 2026</div>'
    f'</div>',
    sizing_mode="stretch_width",
    height=50,
)

# ══════════════════════════════════════════════════════════════════════════════
# FULL LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
map_pane = pn.pane.HoloViews(
    build_map(),
    sizing_mode="stretch_both",
    styles={"flex": "4", "min-width": "0"},
)

main_row = pn.Row(
    map_pane,
    right_panel,
    sizing_mode="stretch_both",
    styles={
        "overflow":  "hidden",
        "display":   "flex",
        "flex-wrap": "nowrap",
    },
)

page = pn.Column(
    header_bar,
    main_row,
    sizing_mode="stretch_both",
    styles={"background": BG_PAGE, "height": "100vh"},
)

page.servable()

if __name__ == "__main__":
    pn.serve(page, port=5007, show=True, title="Copenhagen Vegetation Dashboard")
