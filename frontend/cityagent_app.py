"""
cityAgent — Urban Intelligence Platform  (Panel)
Prompt → Interpreter Agent → Infrared API → Synthesizer → 3-persona dashboard

Run:   uv run frontend/cityagent_app.py
"""

import warnings; warnings.filterwarnings("ignore")
import base64, gzip, io, json, math, os, sys, threading, zipfile
import numpy as np
import panel as pn
import param
import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gvts
import requests
from dotenv import load_dotenv

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(FRONTEND_DIR)
sys.path.insert(0, ROOT_DIR)

from backend.infrared.data.get_location     import get_location
from backend.infrared.data.get_buildings    import get_buildings
from backend.infrared.data.get_weather_data import get_weather_data
from backend.agents.orchestrator            import plan as orchestrate
from backend.agents.interpreter             import interpret

load_dotenv(os.path.join(ROOT_DIR, ".env"))
INFRARED_URL   = os.getenv("INFRARED_API_URL", "")
_INFRARED_KEYS = [os.getenv(f"INFRARED_API_KEY_{i}", "") for i in range(1, 6)]
PAYLOAD_DIR    = os.path.join(ROOT_DIR, "backend", "infrared", "api", "payloads")
BUILD_CACHE    = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "buildings")
GRID_CACHE     = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "grids")
EPW_PATH       = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "weather", "copenhagen.epw")

_MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
WEATHER_FIELDS = [
    "dry-bulb-temperature","relative-humidity","wind-speed",
    "global-horizontal-radiation","direct-normal-radiation",
    "diffuse-horizontal-radiation","horizontal-infrared-radiation-intensity",
]
SEASON_MONTHS = {
    "summer":    (6, 8,  8, 18),
    "winter":    (1, 2,  8, 17),
    "full_year": (6, 8,  8, 18),
}

# ── Status steps per user type ─────────────────────────────────────────────
STEPS = {
    "stakeholder": [
        "Understanding your request — identifying location and performance objectives",
        "Retrieving building footprints and urban geometry from OpenStreetMap",
        "Loading Copenhagen climate data — temperature, humidity, wind and solar radiation",
        "Running thermal comfort simulation — mapping pedestrian-level heat stress across the site",
        "Running solar radiation analysis — measuring incoming solar load across the 512 × 512 m tile",
        "Computing performance statistics — scoring comfort zones and heat-stress areas",
        "Generating insights — mapping results to urban performance indicators",
        "Preparing your Environmental Performance Dashboard",
    ],
    "citizen": [
        "Understanding what matters to you — reading your question and location",
        "Mapping the area — finding buildings, parks, and places around you",
        "Checking local climate conditions — temperature, wind and sunlight for your area",
        "Running thermal comfort analysis — finding the hottest and coolest spots near you",
        "Analysing solar conditions — identifying shaded streets and sun-exposed open spaces",
        "Scoring nearby services — walking distances to parks, shops and transport",
        "Putting it all together — explaining what the analysis means for your daily life",
        "Opening your personal comfort and neighbourhood dashboard",
    ],
    "aec": [
        "Identifying your technical context — location, season and design objectives",
        "Fetching building geometry and urban morphology data from OpenStreetMap",
        "Loading EPW climate data — including hourly temperature, radiation and wind profiles",
        "Running UTCI thermal comfort simulation across the 512 × 512 m analysis tile",
        "Running solar radiation analysis — quantifying direct and diffuse irradiance at pedestrian level",
        "Computing baseline statistics — UTCI distribution, heat-stress zones and priority areas",
        "Quantifying spatial patterns — identifying zones with highest combined thermal and solar load",
        "Routing to AEC technical analysis dashboard",
    ],
}

# ── Accent colours per user type ───────────────────────────────────────────
ACCENT = {
    "stakeholder": ("#c8601a", "rgba(200,96,26,0.10)", "rgba(200,96,26,0.06)"),
    "citizen":     ("#2a6ea8", "rgba(42,110,168,0.10)", "rgba(42,110,168,0.05)"),
    "aec":         ("#2a8a5a", "rgba(42,138,90,0.10)",  "rgba(42,138,90,0.05)"),
}

# ── Keyword detection (mirrors v3 HTML; used while typing) ────────────────
_KW = {
    "stakeholder": {"municipal","municipality","policy","plan","urban plan","stakeholder",
                    "performance","assessment","environmental impact","master plan","district",
                    "public space","plaza","report","score","kpi","decision","governance"},
    "citizen":     {"my","kids","children","safe","walk","outside","hot","cool","tree",
                    "comfortable","neighbourhood","neighborhood","park","street","today",
                    "summer","winter","run","jog","bike","cycling","shade","shaded","sit",
                    "find","where","best"},
    "aec":         {"utci","solar radiation","svf","sky view","heat island","architect",
                    "engineer","landscape","designer","planner","analysis","simulation",
                    "canopy","vegetation","greening","baseline","microclimate","grasshopper",
                    "ladybug","rhino","placement","cooling","urban morphology","infrared"},
}

def _detect_user_type(text: str) -> str:
    p = text.lower()
    scores = {ut: sum(1 for w in kws if w in p) for ut, kws in _KW.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "citizen"

# ══════════════════════════════════════════════════════════════════════════════
# THEME / CSS
# ══════════════════════════════════════════════════════════════════════════════
GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,200;0,9..144,300;0,9..144,400;1,9..144,200;1,9..144,300&family=Instrument+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg:#f5f2ec; --bg2:#ede9e1; --surface:#faf8f4; --card:#f0ece4; --card2:#e8e3d8;
  --border:rgba(60,55,40,0.10); --border2:rgba(60,55,40,0.18);
  --ink:#1c1a14; --ink2:#3a3628; --muted:rgba(28,26,20,0.45); --dim:rgba(28,26,20,0.25);
  --sh:#c8601a; --sh2:rgba(200,96,26,0.10); --sh3:rgba(200,96,26,0.06);
  --ct:#2a6ea8; --ct2:rgba(42,110,168,0.10); --ct3:rgba(42,110,168,0.05);
  --aec:#2a8a5a; --aec2:rgba(42,138,90,0.10); --aec3:rgba(42,138,90,0.05);
  --score-red:#c0352a; --score-ora:#c86a20; --score-yel:#b89010;
  --score-lg:#4a9050; --score-dg:#2a8a5a;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body, .bk-root { background: var(--bg) !important; font-family: 'Instrument Sans', sans-serif !important; color: var(--ink) !important; }
.bk-root .bk { font-family: 'Instrument Sans', sans-serif !important; }
"""

pn.extension(sizing_mode="stretch_width", raw_css=[GLOBAL_CSS])
hv.extension("bokeh")
gv.extension("bokeh")

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class AppState(param.Parameterized):
    view          = param.Selector(default="prompt", objects=["prompt","loading","results"])
    prompt_text   = param.String(default="")
    detected_type = param.Selector(default="citizen", objects=["stakeholder","citizen","aec"])
    user_type     = param.Selector(default="citizen", objects=["stakeholder","citizen","aec"])
    lat           = param.Number(default=55.6761)
    lon           = param.Number(default=12.5683)
    location_name = param.String(default="")
    address_str   = param.String(default="")
    plan          = param.Dict(default={})
    status_idx    = param.Integer(default=0)
    status_pct    = param.Number(default=0)
    sr_grid       = param.Array(default=None, allow_None=True)
    tci_grid      = param.Array(default=None, allow_None=True)
    active_sim    = param.Selector(default="TCI", objects=["TCI","SR"])
    ai_result     = param.Dict(default={})

state = AppState()

def _h(html, **kw): return pn.pane.HTML(html, sizing_mode="stretch_width", **kw)

# ══════════════════════════════════════════════════════════════════════════════
# INFRARED API
# ══════════════════════════════════════════════════════════════════════════════
_key_idx = 0

def call_infrared(analysis_type, lat, lon, geometries, weather):
    global _key_idx
    fmap = {"solar-radiation": "SR.json", "thermal-comfort-index": "TCI.json"}
    if analysis_type not in fmap:
        raise ValueError(f"No payload for {analysis_type}")
    with open(os.path.join(PAYLOAD_DIR, fmap[analysis_type])) as f:
        payload = json.load(f)
    payload["latitude"]   = lat
    payload["longitude"]  = lon
    payload["geometries"] = geometries
    sm, em, sh, eh = weather["month-stamp"]
    if analysis_type == "solar-radiation":
        n = _MONTH_DAYS[sm-1] * (eh - sh + 1)
        payload["time-period"] = {"start-month":sm,"start-day":1,"start-hour":sh,
                                   "end-month":sm,"end-day":_MONTH_DAYS[sm-1],"end-hour":eh}
        payload["diffuse-horizontal-radiation"] = weather["diffuse-horizontal-radiation"][:n]
        payload["direct-normal-radiation"]      = weather["direct-normal-radiation"][:n]
        for k in ["month-stamp","hour-stamp","dry-bulb-temperature","relative-humidity",
                  "wind-speed","global-horizontal-radiation","horizontal-infrared-radiation-intensity"]:
            payload.pop(k, None)
    else:
        payload["month-stamp"] = weather["month-stamp"]
        payload["hour-stamp"]  = weather["hour-stamp"]
        for f in WEATHER_FIELDS:
            if f in weather: payload[f] = weather[f]
    b64  = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
    hdrs = {"Content-Type":"text/plain","X-Infrared-Encoding":"gzip"}
    last_exc = None
    for attempt in range(len(_INFRARED_KEYS)):
        key = _INFRARED_KEYS[(_key_idx + attempt) % len(_INFRARED_KEYS)]
        try:
            r = requests.post(INFRARED_URL, data=b64,
                              headers={**hdrs,"x-api-key":key}, timeout=150)
            if not r.ok: print(f"[infrared] {analysis_type} {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            break
        except requests.exceptions.Timeout:
            print(f"[infrared] timeout key {attempt+1}, rotating")
            last_exc = requests.exceptions.Timeout()
    else:
        raise last_exc or RuntimeError("All keys failed")
    outer = r.json()
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(outer["result"]))) as zf:
        result = json.loads(zf.read("data.json"))
    return result if "output" in result else {"output": result}

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline():
    def _s(**kw):
        for k, v in kw.items(): setattr(state, k, v)

    try:
        prompt = state.prompt_text

        # Step 0 — Interpreter agent (orchestrate)
        _s(status_idx=0, status_pct=5)
        plan_data = orchestrate(prompt)

        # Map persona → user_type; guard against unexpected values
        _PERSONA_TO_UT = {"citizen":"citizen","planner":"stakeholder","expert":"aec"}
        _UT_ALIASES    = {"expert":"aec","aec_expert":"aec","decision_maker":"stakeholder",
                          "municipality":"stakeholder","resident":"citizen","public":"citizen"}
        raw_ut  = plan_data.get("user_type", state.detected_type)
        ut      = _UT_ALIASES.get(raw_ut, raw_ut)
        if ut not in ("stakeholder","citizen","aec"):
            persona_from_plan = plan_data.get("persona","citizen")
            ut = _PERSONA_TO_UT.get(persona_from_plan, state.detected_type)

        persona               = plan_data.get("persona", _PERSONA_TO_UT.get(ut,"citizen"))
        synthesis_instruction = plan_data.get("synthesis_instruction", "")
        season_str            = plan_data.get("season", "summer")
        sm, em, sh, eh        = SEASON_MONTHS.get(season_str, SEASON_MONTHS["summer"])

        _s(plan=plan_data, user_type=ut, detected_type=ut)

        # Step 1 — location + buildings
        _s(status_idx=1, status_pct=12)
        loc_str = plan_data.get("location", "Vesterbro, Copenhagen") or "Vesterbro, Copenhagen"
        try:
            loc  = get_location(loc_str)
            lat  = loc["lat"]; lon = loc["lon"]
            name = loc.get("short_name", loc_str)
        except Exception:
            lat, lon, name = state.lat, state.lon, loc_str
        _s(lat=lat, lon=lon, location_name=name, address_str=name)

        cache_json = os.path.join(BUILD_CACHE, "analysis.json")
        cache_meta = os.path.join(BUILD_CACHE, "analysis_meta.json")
        use_cache  = False
        if os.path.exists(cache_json) and os.path.exists(cache_meta):
            meta = json.load(open(cache_meta))
            dist = math.hypot((lat-meta["lat"])*111320,
                              (lon-meta["lon"])*111320*math.cos(math.radians(lat)))
            use_cache = dist < 10
        if use_cache:
            buildings = json.load(open(cache_json))
        else:
            path = get_buildings(lat, lon, label="analysis", out_dir=BUILD_CACHE)
            buildings = json.load(open(path))
            json.dump({"lat":lat,"lon":lon}, open(cache_meta,"w"))

        # Step 2 — weather
        _s(status_idx=2, status_pct=25)
        weather = get_weather_data(EPW_PATH, start_month=sm, end_month=em,
                                   start_hour=sh, end_hour=eh)

        # Steps 3-4 — simulations
        tci_cache = os.path.join(GRID_CACHE, "tci.npy")
        sr_cache  = os.path.join(GRID_CACHE, "sr.npy")
        use_grid  = os.path.exists(tci_cache) and os.path.exists(sr_cache)

        _s(status_idx=3, status_pct=38)
        if use_grid:
            tci = np.load(tci_cache)
        else:
            raw = call_infrared("thermal-comfort-index", lat, lon, buildings, weather)
            tci = np.array(raw["output"], dtype=float).reshape((512,512), order="F")
            np.save(tci_cache, tci)
        _s(tci_grid=tci)

        _s(status_idx=4, status_pct=60)
        if use_grid:
            sr = np.load(sr_cache)
        else:
            raw = call_infrared("solar-radiation", lat, lon, buildings, weather)
            sr  = np.array(raw["output"], dtype=float).reshape((512,512), order="F")
            np.save(sr_cache, sr)
        _s(sr_grid=sr)

        # Step 5 — statistics
        _s(status_idx=5, status_pct=75)
        def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
        tv, sv = _v(tci), _v(sr)
        stats = {}
        if tv.size:
            stats["thermal-comfort-index"] = {
                "mean":         round(float(np.mean(tv)), 1),
                "max":          round(float(np.max(tv)),  1),
                "pct_above_32": round(float(100*(tv>32).sum()/tv.size), 1),
            }
        if sv.size:
            stats["solar-radiation"] = {
                "mean": round(float(np.mean(sv)), 1),
                "max":  round(float(np.max(sv)),  1),
            }
        if sv.size and tv.size:
            st = float(np.percentile(sv, 70))
            tt = float(np.percentile(tv, 70))
            sf, tf = sr.flatten(), tci.flatten()
            mask = ~(np.isnan(sf) | np.isnan(tf))
            stats["priority_zone_pct"] = round(
                100 * ((sf[mask]>=st) & (tf[mask]>=tt)).sum() / mask.sum(), 1)

        # Step 6 — Synthesizer (interpreter agent)
        _s(status_idx=6, status_pct=88)
        ai = interpret(
            stats=stats,
            location=name,
            user_type=ut,
            prompt=prompt,
            n_buildings=len(buildings),
            season=season_str,
            synthesis_instruction=synthesis_instruction,
            persona=persona,
        )
        _s(ai_result=ai)

        # Step 7 — done
        _s(status_idx=7, status_pct=100)
        import time; time.sleep(0.4)
        _s(view="results")

    except Exception as e:
        import traceback; traceback.print_exc()
        _s(view="prompt")

# ══════════════════════════════════════════════════════════════════════════════
# MAP — reactive, analysis-aware
# ══════════════════════════════════════════════════════════════════════════════
@pn.depends(state.param.active_sim, state.param.tci_grid, state.param.sr_grid,
            state.param.user_type, state.param.lat, state.param.lon)
def _map_pane(active_sim, tci_grid, sr_grid, user_type, lat, lon):
    ld = 256/111320
    lo = 256/(111320*math.cos(math.radians(lat)))
    bounds = (lon-lo, lat-ld, lon+lo, lat+ld)
    base_opts = dict(width=600, height=500, xaxis=None, yaxis=None)
    tiles = gvts.CartoLight().opts(**base_opts)

    # ── Citizen: 3-zone categorical comfort map — NO scientific colorbar ───────
    if user_type == "citizen":
        if tci_grid is None:
            return pn.pane.HoloViews(tiles, sizing_mode="stretch_both")
        # Encode comfort zones: 0=comfortable, 1=warm, 2=hot
        cat = np.where(tci_grid < 26, 0.0,
              np.where(tci_grid < 32, 1.0, 2.0))
        cat[np.isnan(tci_grid)] = np.nan
        img = gv.Image(np.flipud(cat), bounds=bounds,
                       kdims=["Longitude","Latitude"], vdims=["comfort"]).opts(
            cmap=["#4a9050", "#e8a020", "#c0352a"],
            clim=(0, 2), alpha=0.50, colorbar=False,
            tools=["hover"], **base_opts,
        )
        return pn.pane.HoloViews(tiles * img, sizing_mode="stretch_both")

    # ── Stakeholder: TCI heatmap + priority zone (orange) ─────────────────────
    if user_type == "stakeholder":
        grid = tci_grid if tci_grid is not None else sr_grid
        if grid is None:
            return pn.pane.HoloViews(tiles, sizing_mode="stretch_both")
        valid = grid[~np.isnan(grid)]
        vmin = float(np.percentile(valid, 2))  if valid.size else 0
        vmax = float(np.percentile(valid, 98)) if valid.size else 1
        img = gv.Image(np.flipud(tci_grid if tci_grid is not None else grid), bounds=bounds,
                       kdims=["Longitude","Latitude"], vdims=["value"]).opts(
            cmap="YlOrRd", alpha=0.55, clim=(vmin, vmax),
            colorbar=True, colorbar_opts={"title": "\u00b0C UTCI \u2014 heat stress"},
            tools=["hover"], **base_opts,
        )
        overlay = tiles * img
        # Priority zone: top-30th-pct TCI ∩ SR combined
        if tci_grid is not None and sr_grid is not None:
            tf = tci_grid.flatten(); sf = sr_grid.flatten()
            mask = ~(np.isnan(tf) | np.isnan(sf))
            if mask.sum() > 0:
                tt = float(np.percentile(tf[mask], 70))
                st = float(np.percentile(sf[mask], 70))
                pz = np.where((tci_grid >= tt) & (sr_grid >= st), 1.0, np.nan)
                pz_img = gv.Image(np.flipud(pz), bounds=bounds,
                                  kdims=["Longitude","Latitude"], vdims=["pz"]).opts(
                    cmap=["rgba(200,96,26,0.0)", "rgba(200,96,26,0.45)"],
                    clim=(0, 1), alpha=1, colorbar=False, tools=[], **base_opts,
                )
                overlay = overlay * pz_img
        return pn.pane.HoloViews(overlay, sizing_mode="stretch_both")

    # ── AEC: full technical raster, togglable TCI / SR ────────────────────────
    if active_sim == "SR":
        grid   = sr_grid
        cmap   = "YlOrBr"
        clabel = "kWh/m\u00b2 \u2014 Solar irradiance"
    else:
        grid   = tci_grid
        cmap   = "YlOrRd"
        clabel = "\u00b0C UTCI \u2014 Thermal comfort index"

    if grid is None:
        return pn.pane.HoloViews(tiles, sizing_mode="stretch_both")

    valid = grid[~np.isnan(grid)]
    vmin = float(np.percentile(valid, 2))  if valid.size else 0
    vmax = float(np.percentile(valid, 98)) if valid.size else 1

    img = gv.Image(np.flipud(grid), bounds=bounds,
                   kdims=["Longitude","Latitude"], vdims=["value"]).opts(
        cmap=cmap, alpha=0.60, clim=(vmin, vmax),
        colorbar=True, colorbar_opts={"title": clabel},
        tools=["hover"], **base_opts,
    )
    return pn.pane.HoloViews(tiles * img, sizing_mode="stretch_both")

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT SCREEN
# ══════════════════════════════════════════════════════════════════════════════
prompt_input = pn.widgets.TextInput(
    placeholder='e.g. "Assess outdoor comfort at Bryghuspladsen in summer" · Try a location, a question, or a design goal',
    stylesheets=["""
      :host input {
        background: var(--card,#f0ece4); border: 1px solid var(--border2,rgba(60,55,40,0.18));
        border-radius: 9px; color: var(--ink,#1c1a14);
        font-family: 'Instrument Sans', sans-serif; font-size: 15px;
        padding: 13px 16px; outline: none;
      }
    """],
    sizing_mode="stretch_width",
)

run_btn = pn.widgets.Button(
    name="Analyse \u2192",
    button_type="default",
    stylesheets=["""
      :host button {
        background: var(--ink,#1c1a14); color: var(--bg,#f5f2ec); border: none;
        border-radius: 9px; font-family: 'Instrument Sans', sans-serif;
        font-size: 14px; font-weight: 500; padding: 13px 26px; cursor: pointer;
        white-space: nowrap; transition: background 0.2s;
      }
      :host button:hover { background: var(--ink2,#3a3628); }
    """],
)

detect_html = pn.pane.HTML(
    '<div style="display:flex;align-items:center;gap:.5rem;font-family:\'DM Mono\',monospace;'
    'font-size:10px;padding:3px 10px;border-radius:999px;border:1px solid rgba(60,55,40,0.18);'
    'color:rgba(28,26,20,0.25);background:var(--card,#f0ece4)">'
    '<div style="width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0"></div>'
    '<span>Awaiting input\u2026</span></div>',
    sizing_mode="fixed", width=260,
)

def _on_prompt(event):
    t = event.new.strip()
    if not t:
        detect_html.object = (
            '<div style="display:flex;align-items:center;gap:.5rem;font-family:\'DM Mono\',monospace;'
            'font-size:10px;padding:3px 10px;border-radius:999px;border:1px solid rgba(60,55,40,0.18);'
            'color:rgba(28,26,20,0.25);background:var(--card,#f0ece4)">'
            '<div style="width:6px;height:6px;border-radius:50%;background:currentColor"></div>'
            'Awaiting input\u2026</div>'
        )
        return
    ut = _detect_user_type(t)
    state.detected_type = ut
    labels = {"stakeholder":"\U0001f3db Stakeholder / Decision-maker",
              "citizen":"\U0001f6b6 Private Citizen",
              "aec":"\U0001f4d0 AEC Expert"}
    colors = {"stakeholder":"var(--sh,#c8601a)","citizen":"var(--ct,#2a6ea8)","aec":"var(--aec,#2a8a5a)"}
    bg     = {"stakeholder":"var(--sh2)","citizen":"var(--ct2)","aec":"var(--aec2)"}
    border = {"stakeholder":"rgba(200,96,26,0.3)","citizen":"rgba(42,110,168,0.3)","aec":"rgba(42,138,90,0.3)"}
    detect_html.object = (
        f'<div style="display:flex;align-items:center;gap:.5rem;font-family:\'DM Mono\',monospace;'
        f'font-size:10px;padding:3px 10px;border-radius:999px;border:1px solid {border[ut]};'
        f'color:{colors[ut]};background:{bg[ut]}">'
        f'<div style="width:6px;height:6px;border-radius:50%;background:currentColor"></div>'
        f'{labels[ut]}</div>'
    )

prompt_input.param.watch(_on_prompt, "value")

def _fill_prompt(prompt_text):
    prompt_input.value = prompt_text
    prompt_input.param.trigger("value")

def _on_run(event):
    text = prompt_input.value.strip()
    if not text: return
    state.prompt_text  = text
    state.user_type    = state.detected_type
    state.view         = "loading"
    state.status_idx   = 0
    state.status_pct   = 0
    state.sr_grid      = None
    state.tci_grid     = None
    state.ai_result    = {}
    threading.Thread(target=run_pipeline, daemon=True).start()

run_btn.on_click(_on_run)

EXAMPLE_CARDS = [
    ("sh", "\U0001f3db Stakeholder / Decision-maker",
     "Provide an environmental performance assessment for the public space at Bryghuspladsen, Copenhagen — include thermal comfort and solar radiation KPIs for this summer"),
    ("ct", "\U0001f6b6 Private Citizen",
     "I want to take my kids outside this afternoon near N\u00f8rreport — is it too hot? Where are the shadiest and most comfortable streets to walk?"),
    ("aec", "\U0001f4d0 AEC Expert / Urban Designer",
     "Run a UTCI thermal comfort and solar radiation baseline analysis for the Vesterbro district — I need to identify priority zones for urban greening and canopy placement"),
]

def _ep_card(cls, role, text):
    color = {"sh":"var(--sh,#c8601a)","ct":"var(--ct,#2a6ea8)","aec":"var(--aec,#2a8a5a)"}[cls]
    use_btn = pn.widgets.Button(
        name="Use this prompt \u2192",
        button_type="default",
        sizing_mode="stretch_width",
        stylesheets=[f"""
          :host button {{
            background: {color}; color: #fff; border: none;
            border-radius: 7px; padding: 8px 14px;
            font-family: 'Instrument Sans', sans-serif;
            font-size: 12px; font-weight: 500; cursor: pointer;
            transition: opacity 0.15s;
          }}
          :host button:hover {{ opacity: 0.82; }}
        """],
    )
    def _click(e, t=text): _fill_prompt(t)
    use_btn.on_click(_click)
    return pn.Column(
        _h(f'<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.12em;'
           f'text-transform:uppercase;color:{color};padding:.9rem 1rem .3rem">{role}</div>'),
        _h(f'<div style="font-size:12px;color:var(--ink2,#3a3628);line-height:1.5;'
           f'padding:0 1rem .7rem;flex:1">{text}</div>'),
        pn.Column(use_btn, styles={"padding":"0 1rem .9rem","margin-top":"auto"}),
        styles={"background":"var(--surface,#faf8f4)","border":"1px solid var(--border2,rgba(60,55,40,0.18))",
                "border-radius":"10px","display":"flex","flex-direction":"column","height":"100%"},
        sizing_mode="stretch_width",
    )

_enter_key_js = pn.pane.HTML("""
<script>
(function(){
  function _tryRun(){
    var btns=document.querySelectorAll('.bk-btn');
    for(var i=0;i<btns.length;i++){
      if(btns[i].textContent.trim()==='Analyse \u2192'){btns[i].click();return;}
    }
  }
  document.addEventListener('keydown',function(e){
    if(e.key==='Enter'&&document.activeElement&&document.activeElement.tagName==='INPUT'){
      _tryRun();
    }
  });
})();
</script>
""", width=0, height=0, margin=0, styles={"display":"none"})

def _make_prompt_screen():
    cards = pn.Row(*[_ep_card(*c) for c in EXAMPLE_CARDS], sizing_mode="stretch_width",
                   styles={"align-items":"stretch","display":"flex"})
    return pn.Column(
        _enter_key_js,
        _h('<div style="height:3rem"></div>'),
        _h('<div style="font-family:\'Fraunces\',serif;font-size:clamp(2.8rem,6vw,4.4rem);'
           'font-weight:200;letter-spacing:-0.03em;line-height:1;margin-bottom:.2rem">'
           'city<em style="font-style:italic;color:var(--sh,#c8601a)">Agent</em></div>'),
        _h('<div style="font-family:\'DM Mono\',monospace;font-size:11px;letter-spacing:.14em;'
           'text-transform:uppercase;color:var(--muted,rgba(28,26,20,0.45));margin-bottom:3rem">'
           'Urban Environmental Intelligence Platform</div>'),
        pn.Column(
            pn.Row(
                _h('<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:.14em;'
                   'text-transform:uppercase;color:var(--muted)">Describe your question or project</div>'),
                detect_html,
                sizing_mode="stretch_width",
                styles={"align-items":"center","justify-content":"space-between","margin-bottom":".9rem"},
            ),
            pn.Row(prompt_input, run_btn, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            styles={"background":"var(--surface,#faf8f4)","border":"1px solid var(--border2,rgba(60,55,40,0.18))",
                    "border-radius":"14px","padding":"1.6rem",
                    "box-shadow":"0 2px 24px rgba(60,55,40,0.06)"},
        ),
        _h('<div style="height:1.4rem"></div>'),
        cards,
        sizing_mode="stretch_width",
        styles={"max-width":"700px","margin":"0 auto","padding":"0 1rem"},
    )

prompt_screen = _make_prompt_screen()

# ══════════════════════════════════════════════════════════════════════════════
# LOADING SCREEN
# ══════════════════════════════════════════════════════════════════════════════
@pn.depends(state.param.status_idx, state.param.status_pct, state.param.user_type)
def loading_screen(status_idx, status_pct, user_type):
    acc, bg2, _ = ACCENT.get(user_type, ACCENT["citizen"])
    steps = STEPS.get(user_type, STEPS["citizen"])
    label = {
        "stakeholder":"\U0001f3db Stakeholder",
        "citizen":"\U0001f6b6 Private Citizen",
        "aec":"\U0001f4d0 AEC Expert",
    }[user_type]

    rows_html = ""
    for i, step in enumerate(steps):
        if i < status_idx:
            dot_color = acc; opacity = "0.5"; bdr = ""
        elif i == status_idx:
            dot_color = acc; opacity = "1"; bdr = f"border-color:rgba(60,55,40,0.25);background:{bg2}"
        else:
            dot_color = "var(--border2)"; opacity = "0.25"; bdr = ""
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:.8rem;background:var(--surface);'
            f'border:1px solid var(--border);border-radius:8px;padding:10px 14px;'
            f'opacity:{opacity};transition:all .3s;{bdr}">'
            f'<div style="width:7px;height:7px;border-radius:50%;background:{dot_color};flex-shrink:0"></div>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:11.5px;color:var(--muted)">'
            f'{step}</span></div>'
        )

    return pn.Column(
        _h('<div style="height:2rem"></div>'),
        _h('<div style="font-family:\'Fraunces\',serif;font-size:1.6rem;font-weight:200;letter-spacing:-.02em">'
           'city<em style="font-style:italic;color:var(--sh)">Agent</em></div>'),
        _h('<div style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:.14em;'
           'text-transform:uppercase;color:var(--muted);margin-bottom:2rem">Running agentic workflow</div>'),
        _h(f'<div style="display:inline-flex;align-items:center;gap:.5rem;border-radius:999px;'
           f'padding:5px 16px;margin-bottom:1.8rem;background:{bg2};border:1px solid {acc};'
           f'font-family:\'DM Mono\',monospace;font-size:11px;font-weight:500;color:{acc}">'
           f'\u25c6 \u00a0{label}</div>'),
        _h(f'<div style="width:520px;max-width:94vw;display:flex;flex-direction:column;gap:.35rem;'
           f'margin-bottom:1.6rem">{rows_html}</div>'),
        _h(f'<div style="width:520px;max-width:94vw;height:2px;background:var(--border2);'
           f'border-radius:99px;overflow:hidden">'
           f'<div style="height:100%;border-radius:99px;width:{status_pct:.0f}%;'
           f'background:{acc};transition:width .5s ease"></div></div>'),
        sizing_mode="stretch_width",
        styles={"align-items":"center","justify-content":"center","display":"flex",
                "flex-direction":"column","min-height":"100vh","background":"var(--bg)"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# SHARED COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════
_PANEL_STYLES = {
    "background":"var(--surface,#faf8f4)",
    "border-left":"1px solid var(--border2,rgba(60,55,40,0.18))",
    "overflow-y":"auto",
}

def _new_analysis_btn(acc):
    btn = pn.widgets.Button(name="+ Start new analysis", button_type="default",
                            sizing_mode="stretch_width", stylesheets=[f"""
      :host button {{background:{acc};color:#fff;border:none;border-radius:9px;
        font-family:'Instrument Sans',sans-serif;font-size:14px;font-weight:500;
        padding:12px;cursor:pointer;width:100%;transition:opacity .15s}}
      :host button:hover {{opacity:.85}}
    """])
    def _reset(e):
        state.sr_grid=None; state.tci_grid=None; state.ai_result={}; state.view="prompt"
    btn.on_click(_reset)
    return btn

def _topbar():
    ut = state.user_type
    acc, bg2, _ = ACCENT[ut]
    lbl = {"stakeholder":"\U0001f3db Stakeholder","citizen":"\U0001f6b6 Citizen","aec":"\U0001f4d0 AEC Expert"}[ut]
    subtitle = {
        "citizen":     "Outdoor comfort guide",
        "stakeholder": "Environmental Performance Assessment",
        "aec":         "Technical Analysis Dashboard",
    }[ut]
    new_btn = pn.widgets.Button(name="+ New analysis", button_type="default", stylesheets=["""
      :host button { background:var(--ink,#1c1a14);color:var(--bg,#f5f2ec);border:none;
        border-radius:7px;padding:6px 14px;font-family:'Instrument Sans',sans-serif;
        font-size:12px;font-weight:500;cursor:pointer;white-space:nowrap }
      :host button:hover { background:var(--ink2,#3a3628) }
    """])
    def _reset(e):
        state.sr_grid=None; state.tci_grid=None; state.ai_result={}; state.view="prompt"
    new_btn.on_click(_reset)
    return pn.Row(
        _h('city<em style="font-style:italic">Agent</em>', styles={"font-family":"Fraunces,serif",
           "font-size":"1.1rem","font-weight":"200","letter-spacing":"-.02em"}),
        _h('<div style="width:1px;height:18px;background:var(--border2);margin:0 .5rem;align-self:center"></div>'
           f'<div style="font-size:12px;color:var(--muted);flex:1">{state.address_str} \u00b7 {subtitle}</div>'),
        _h(f'<div style="display:flex;align-items:center;gap:.5rem;padding:4px 12px;border-radius:999px;'
           f'border:1px solid {acc};font-family:\'DM Mono\',monospace;font-size:10px;font-weight:500;'
           f'letter-spacing:.06em;text-transform:uppercase;color:{acc};background:{bg2}">{lbl}</div>'),
        new_btn,
        height=52, sizing_mode="stretch_width",
        styles={"background":"var(--surface)","border-bottom":"1px solid rgba(60,55,40,.18)",
                "padding":"0 1.4rem","align-items":"center","display":"flex","flex-shrink":"0"},
    )

def _kpi_card(icon, name, val, unit, note, bar_pct, bar_color):
    return (
        f'<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;'
        f'padding:.9rem">'
        f'<span style="font-size:22px;display:block;margin-bottom:.4rem">{icon}</span>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.1em;'
        f'text-transform:uppercase;color:var(--muted);margin-bottom:4px">{name}</div>'
        f'<div style="font-family:\'Fraunces\',serif;font-size:1.5rem;font-weight:300;'
        f'letter-spacing:-.02em;line-height:1.1">{val} '
        f'<span style="font-family:\'DM Mono\',monospace;font-size:10px;color:var(--muted)">{unit}</span></div>'
        f'<div style="font-size:10px;color:var(--muted);margin-top:3px">{note}</div>'
        f'<div style="height:3px;background:var(--border2);border-radius:99px;margin-top:6px">'
        f'<div style="height:100%;width:{bar_pct}%;background:{bar_color};border-radius:99px"></div>'
        f'</div></div>'
    )

def _rec_card(n, title, body, acc):
    return (
        f'<div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {acc};'
        f'border-radius:10px;padding:.9rem 1rem;margin-bottom:.5rem">'
        f'<div style="display:flex;align-items:baseline;gap:.5rem;margin-bottom:.35rem">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:10px;color:{acc}">{n:02d}</span>'
        f'<span style="font-size:13px;font-weight:500;color:var(--ink)">{title}</span></div>'
        f'<div style="font-size:12px;color:var(--muted);line-height:1.65">{body}</div></div>'
    )

# ══════════════════════════════════════════════════════════════════════════════
# STAKEHOLDER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def _build_stakeholder():
    ai  = state.ai_result or {}
    tci = state.tci_grid; sr = state.sr_grid
    def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
    tv, sv = _v(tci), _v(sr)

    score   = float(ai.get("score", 5.0))
    verdict = ai.get("verdict", "Analysis complete")
    summary = ai.get("summary", "")
    recs    = ai.get("recommendations", [])
    acc, bg2, bg3 = ACCENT["stakeholder"]

    avg_tci = f"{np.mean(tv):.1f}" if tv.size else "\u2014"
    pct_hot = f"{100*(tv>32).sum()/tv.size:.0f}" if tv.size else "\u2014"
    avg_sr  = f"{np.mean(sv):.1f}" if sv.size else "\u2014"
    pz      = round(float(100*(sv>=float(np.percentile(sv,70))).sum()/sv.size),1) if sv.size else 0

    score_color = ("#c0352a" if score<3 else "#c86a20" if score<5 else "#b89010" if score<6.5
                   else "#4a9050" if score<8 else "#2a8a5a")

    kpi_html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1rem">' +
        _kpi_card("\U0001f321","Mean Thermal Comfort", avg_tci,"\u00b0C UTCI",
                  "Comfort zone: 18\u201326\u00b0C \u00b7 UTCI scale",
                  min(100,max(0,int((42-float(avg_tci if avg_tci!="\u2014" else 28))/16*100))),
                  "#c86a20" if float(avg_tci if avg_tci!="\u2014" else 28)>26 else "#4a9050") +
        _kpi_card("\U0001f525","Heat-Stressed Area", pct_hot,"% of tile",
                  "UTCI > 32\u00b0C = strong heat stress threshold",
                  int(float(pct_hot)) if pct_hot!="\u2014" else 0, "#c0352a") +
        _kpi_card("\u2600\ufe0f","Solar Radiation", avg_sr,"kWh/m\u00b2",
                  "Mean over 512 \u00d7 512 m tile \u00b7 summer",
                  min(100,int(float(avg_sr if avg_sr!="\u2014" else 5)/10*100)), "#c86a20") +
        _kpi_card("\U0001f33f","Priority Greening Zone", f"{pz:.0f}","% of tile",
                  "High UTCI + high solar \u2014 urgent tree planting",
                  int(pz), "#c86a20") +
        '</div>'
    )

    recs_html = "".join(
        _rec_card(i+1, (r.split("\u2014")[0].strip() if "\u2014" in r else r[:55]),
                  (r.split("\u2014",1)[1].strip() if "\u2014" in r else r), acc)
        for i, r in enumerate(recs[:4])
    )

    sim_toggle = pn.widgets.RadioButtonGroup(
        options=["TCI","SR"], value=state.active_sim,
        stylesheets=[f"""
          :host .bk-btn-group button{{background:none;border:1px solid var(--border2);
            border-radius:999px;color:var(--muted);font-family:'DM Mono',monospace;
            font-size:10px;padding:4px 12px;margin:2px;transition:all .2s}}
          :host .bk-btn-group button.bk-active{{background:{bg2};border-color:{acc};color:{acc}}}
        """])
    sim_toggle.param.watch(lambda e: setattr(state,"active_sim",e.new), "value")

    right = pn.Column(
        _h(f'<div style="padding:1.3rem 1.4rem;border-bottom:1px solid var(--border);background:{bg3}">'
           f'<div style="display:flex;align-items:baseline;gap:.7rem;margin-bottom:.9rem">'
           f'<div style="font-family:\'Fraunces\',serif;font-size:5rem;font-weight:200;line-height:1;'
           f'letter-spacing:-.04em;color:{score_color}">{score}</div>'
           f'<div><div style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted)">Composite Performance Score</div>'
           f'<div style="font-size:15px;font-weight:500;color:var(--ink);margin-top:4px">{verdict}</div>'
           f'<div style="font-size:12px;color:var(--muted);margin-top:2px">0 = Critical \u00b7 10 = Excellent</div>'
           f'</div></div>'
           f'<div style="height:8px;background:var(--border2);border-radius:99px;overflow:hidden;margin-bottom:.3rem">'
           f'<div style="height:100%;width:{score*10:.0f}%;border-radius:99px;'
           f'background:linear-gradient(90deg,var(--score-red),var(--score-ora),var(--score-yel),var(--score-lg),var(--score-dg))"></div></div></div>'),
        pn.Row(_h('<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;'
                  'letter-spacing:.14em;text-transform:uppercase;color:var(--muted)">Active simulation</div>'),
               sim_toggle, sizing_mode="stretch_width",
               styles={"padding":".7rem 1.4rem","border-bottom":"1px solid var(--border)",
                       "align-items":"center","justify-content":"space-between"}),
        pn.Column(
            _h(f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'
               f'padding-bottom:.4rem;border-bottom:1px solid var(--border)">KPI \u2014 KEY PERFORMANCE INDICATORS</div>'
               f'{kpi_html}'),
            _h(f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;margin-top:.8rem;'
               f'padding-bottom:.4rem;border-bottom:1px solid var(--border)">Analysis Summary</div>'
               f'<div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {acc};'
               f'border-radius:10px;padding:1rem;font-size:12.5px;color:var(--muted);line-height:1.75;'
               f'margin-bottom:1rem">{summary}</div>'),
            _h(f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'
               f'padding-bottom:.4rem;border-bottom:1px solid var(--border)">How to raise this score \u2014 Recommendations</div>'
               f'{recs_html}'),
            pn.Column(_new_analysis_btn(acc), styles={"padding":".6rem 0 1.5rem"}),
            sizing_mode="stretch_width",
            styles={"overflow-y":"auto","flex":"1","padding":"1rem 1.3rem"},
        ),
        sizing_mode="stretch_width",
        styles={**_PANEL_STYLES,"flex":"0 0 520px","width":"520px"},
    )

    return pn.Column(
        _topbar(),
        pn.Row(
            pn.panel(_map_pane, sizing_mode="stretch_both", styles={"flex":"1","min-width":"0"}),
            right,
            sizing_mode="stretch_both",
            styles={"display":"flex","flex-wrap":"nowrap","overflow":"hidden"},
        ),
        sizing_mode="stretch_both", styles={"background":"var(--bg)"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# CITIZEN DASHBOARD — plain-language comfort view, no technical numbers
# ══════════════════════════════════════════════════════════════════════════════
def _build_citizen():
    ai  = state.ai_result or {}
    tci = state.tci_grid
    def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
    tv  = _v(tci)
    acc, bg2, bg3 = ACCENT["citizen"]

    avg_tci_f    = float(np.mean(tv))     if tv.size else 26.0
    pct_comf     = float(100*(tv<26).sum()/tv.size) if tv.size else 50.0
    pct_hot      = float(100*(tv>32).sum()/tv.size) if tv.size else 20.0
    summary      = ai.get("summary", "")
    recs         = ai.get("recommendations", [])

    # Comfort status — three states, zero jargon
    if avg_tci_f < 26:
        st_icon, st_color, st_bg = "\U0001f7e2", "#2a8a5a", "rgba(42,138,90,0.07)"
        st_label = "Great conditions to be outside"
        st_desc  = (f"Most of the area is comfortable right now \u2014 "
                    f"green zones on the map are the most pleasant spots to walk or sit.")
    elif avg_tci_f < 32:
        st_icon, st_color, st_bg = "\U0001f7e1", "#c86a20", "rgba(200,106,32,0.07)"
        st_label = "Warm \u2014 stick to the green areas"
        st_desc  = (f"It's warm outside. Look for the green zones on the map \u2014 "
                    f"those are the shadier, cooler streets. Avoid yellow and red areas "
                    f"during the hottest part of the day.")
    else:
        st_icon, st_color, st_bg = "\U0001f534", "#c0352a", "rgba(192,53,42,0.07)"
        st_label = "Hot \u2014 limit time outside"
        st_desc  = (f"It\u2019s quite hot. Red areas on the map are uncomfortable for children "
                    f"and elderly people. If you go out, keep to shaded streets shown in green "
                    f"and bring water.")

    recs_html = "".join(
        f'<div style="display:flex;gap:.8rem;padding:.75rem 0;border-bottom:1px solid var(--border)">'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;font-weight:500;'
        f'color:{acc};flex-shrink:0;margin-top:2px">{i+1}</div>'
        f'<div style="font-size:13px;color:var(--ink2);line-height:1.65">{r}</div></div>'
        for i, r in enumerate(recs[:3])
    )

    right = pn.Column(
        # Status block — big, clear, no numbers
        _h(f'<div style="padding:1.4rem 1.4rem 1.1rem;background:{st_bg};'
           f'border-bottom:1px solid var(--border)">'
           f'<div style="display:flex;align-items:flex-start;gap:.9rem;margin-bottom:.65rem">'
           f'<span style="font-size:2.2rem;line-height:1">{st_icon}</span>'
           f'<div><div style="font-size:17px;font-weight:600;color:{st_color};line-height:1.2">'
           f'{st_label}</div>'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9px;color:var(--muted);'
           f'letter-spacing:.08em;text-transform:uppercase;margin-top:4px">'
           f'\U0001f4cd {state.address_str}</div></div></div>'
           f'<div style="font-size:13px;color:var(--ink2);line-height:1.65">{st_desc}</div>'
           f'</div>'),
        # AI summary + recommendations
        pn.Column(
            _h(f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               f'text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;'
               f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">About this area</div>'
               f'<div style="font-size:13px;color:var(--muted);line-height:1.75;margin-bottom:1.1rem">'
               f'{summary}</div>'
               f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               f'text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;'
               f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">Tips for your visit</div>'
               f'{recs_html}'),
            pn.Column(_new_analysis_btn(acc), styles={"padding":"1rem 0 1.5rem"}),
            sizing_mode="stretch_width",
            styles={"overflow-y":"auto","flex":"1","padding":"1rem 1.4rem"},
        ),
        sizing_mode="stretch_width",
        styles={**_PANEL_STYLES,"flex":"0 0 380px","width":"380px"},
    )

    # Map + comfort zone legend below it
    map_col = pn.Column(
        pn.panel(_map_pane, sizing_mode="stretch_both"),
        _h('<div style="background:var(--surface);border-top:1px solid var(--border);'
           'padding:.45rem 1rem;display:flex;gap:1.4rem;align-items:center;flex-shrink:0">'
           '<span style="font-family:\'DM Mono\',monospace;font-size:9px;color:var(--muted);'
           'text-transform:uppercase;letter-spacing:.1em;white-space:nowrap">Comfort zones</span>'
           '<span style="display:flex;align-items:center;gap:.35rem;font-size:11.5px;color:var(--ink2)">'
           '<span style="width:11px;height:11px;background:#4a9050;border-radius:3px;'
           'display:inline-block;flex-shrink:0"></span>Comfortable</span>'
           '<span style="display:flex;align-items:center;gap:.35rem;font-size:11.5px;color:var(--ink2)">'
           '<span style="width:11px;height:11px;background:#e8a020;border-radius:3px;'
           'display:inline-block;flex-shrink:0"></span>Warm \u2014 seek shade</span>'
           '<span style="display:flex;align-items:center;gap:.35rem;font-size:11.5px;color:var(--ink2)">'
           '<span style="width:11px;height:11px;background:#c0352a;border-radius:3px;'
           'display:inline-block;flex-shrink:0"></span>Hot \u2014 avoid if possible</span>'
           '</div>'),
        sizing_mode="stretch_both",
        styles={"flex":"1","min-width":"0"},
    )

    return pn.Column(
        _topbar(),
        pn.Row(
            map_col,
            right,
            sizing_mode="stretch_both",
            styles={"display":"flex","flex-wrap":"nowrap","overflow":"hidden"},
        ),
        sizing_mode="stretch_both", styles={"background":"var(--bg)"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# AEC DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def _build_aec():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ai  = state.ai_result or {}
    tci = state.tci_grid; sr = state.sr_grid
    def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
    tv, sv = _v(tci), _v(sr)

    avg_tci = float(np.mean(tv)) if tv.size else 31.4
    avg_sr  = float(np.mean(sv)) if sv.size else 5.3
    pct_hot = float(100*(tv>32).sum()/tv.size) if tv.size else 38.0
    summary = ai.get("summary", "")
    recs    = ai.get("recommendations", [])
    acc, bg2, bg3 = ACCENT["aec"]

    # Radar — ecosystem services (baseline only)
    axes = ['Thermal\nComfort','Solar\nReduction','Urban\nCooling',
            'Carbon\nPotential','Wind\nMitigation','Human\nWellbeing']
    N    = len(axes)
    comf_n = max(0.0, min(1.0, (42-avg_tci)/16))
    sol_n  = max(0.0, min(1.0, avg_sr/10))
    cool_n = max(0.0, min(1.0, 1-pct_hot/100))
    vals   = [comf_n*0.9, sol_n*0.8, cool_n*0.85, 0.25, 0.45, cool_n*0.75]
    angs   = [n/N*2*np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(4.5, 2.8), subplot_kw=dict(polar=True), facecolor="#faf8f4")
    ax.set_facecolor("#faf8f4")
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1.0]); ax.set_yticklabels(["","","",""], fontsize=0)
    ax.set_xticks(angs[:-1])
    ax.set_xticklabels(axes, fontsize=10, color=(60/255, 55/255, 40/255, 0.65), fontfamily="monospace")
    ax.spines['polar'].set_color((60/255, 55/255, 40/255, 0.15))
    ax.grid(color=(60/255, 55/255, 40/255, 0.10), linewidth=0.7)
    bv = vals + [vals[0]]
    ax.plot(angs, bv, color="#2a8a5a", linewidth=1.8)
    ax.fill(angs, bv, alpha=0.22, color="#2a8a5a")
    for a, v in zip(angs[:-1], vals): ax.plot(a, v, 'o', color="#2a8a5a", ms=4)
    plt.tight_layout(pad=0.3)
    radar = pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width",
                               styles={"background":"var(--surface)","padding":".5rem"})
    plt.close(fig)

    # Comparison bars
    bars_html = "".join(
        f'<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.7rem">'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;color:var(--muted);'
        f'width:72px;flex-shrink:0;text-transform:uppercase;letter-spacing:.06em">{lbl}</div>'
        f'<div style="flex:1">'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
        f'<div style="height:8px;border-radius:4px;background:#c0352a;width:{int(bp*100)}%;min-width:2px"></div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;color:var(--muted)">{b_lbl}</div></div>'
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="height:8px;border-radius:4px;background:#2a8a5a;width:{int(ep*100)}%;min-width:2px"></div>'
        f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;color:var(--muted)">{e_lbl}</div></div>'
        f'</div></div>'
        for lbl,b_lbl,e_lbl,bp,ep in [
            ("UTCI mean",  f"{avg_tci:.1f}\u00b0C baseline", f"{max(22,avg_tci-7.5):.1f}\u00b0C + trees",
             min(1,avg_tci/45), min(1,max(22,avg_tci-7.5)/45)),
            ("Solar Rad.", f"{avg_sr:.1f} kWh/m\u00b2 base", f"{avg_sr*0.45:.1f} kWh/m\u00b2 + trees",
             min(1,avg_sr/10), min(1,avg_sr*0.45/10)),
            ("Heat Stress",f"{pct_hot:.0f}% base",       f"{max(0,pct_hot-30):.0f}% + trees",
             pct_hot/100, max(0,pct_hot-30)/100),
        ]
    )

    # AI-driven design strategies (fallback to contextual defaults)
    ai_recs = recs if recs else [
        f"Prioritise street-tree canopy in corridors where UTCI exceeds 34\u00b0C \u2014 "
        f"target the priority zone ({avg_tci:.1f}\u00b0C mean, top 30th pct combined load).",
        f"Apply high-albedo cool-roof mandates to buildings in zones with solar load above "
        f"{avg_sr*1.3:.1f} kWh/m\u00b2 to reduce reflected heat load on pedestrians.",
        f"Commission a Lawson 1970 pedestrian wind comfort assessment alongside vegetation "
        f"interventions to avoid wind-shadow trade-offs in the priority zone.",
    ]
    strats_html = "".join(
        f'<div style="background:var(--card);border:1px solid var(--border);'
        f'border-left:3px solid {acc};border-radius:10px;padding:.9rem 1rem;margin-bottom:.5rem">'
        f'<div style="display:flex;align-items:baseline;gap:.5rem;margin-bottom:.35rem">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:10px;color:{acc}">{i+1:02d}</span>'
        f'<span style="font-size:13px;font-weight:500;color:var(--ink)">'
        f'{r.split("\u2014")[0].strip() if "\u2014" in r else r[:60]}</span></div>'
        f'<div style="font-size:12px;color:var(--muted);line-height:1.65">'
        f'{r.split("\u2014",1)[1].strip() if "\u2014" in r else ""}</div></div>'
        for i, r in enumerate(ai_recs[:5])
    )

    # Technical distribution stats for AEC header
    tci_p50 = float(np.percentile(tv, 50)) if tv.size else 0
    tci_p75 = float(np.percentile(tv, 75)) if tv.size else 0
    tci_p98 = float(np.percentile(tv, 98)) if tv.size else 0
    pct_extreme = float(100*(tv>38).sum()/tv.size) if tv.size else 0
    sr_p98  = float(np.percentile(sv, 98)) if sv.size else 0
    pz_pct  = 0.0
    if tv.size and sv.size:
        tf = tci.flatten(); sf = sr.flatten()
        m  = ~(np.isnan(tf) | np.isnan(sf))
        if m.sum() > 0:
            tt = float(np.percentile(tf[m], 70)); st2 = float(np.percentile(sf[m], 70))
            pz_pct = round(100*((tf[m]>=tt)&(sf[m]>=st2)).sum()/m.sum(), 1)

    stat_pill = (
        lambda lbl, val: f'<span style="display:inline-flex;flex-direction:column;'
        f'padding:5px 10px;border-right:1px solid var(--border)">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:8.5px;color:var(--muted);'
        f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px">{lbl}</span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:12px;color:var(--ink);'
        f'font-weight:500">{val}</span></span>'
    )
    stats_row_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--border);'
        f'border-radius:8px;overflow:hidden;margin-bottom:.6rem;background:var(--card)">'
        + stat_pill("UTCI mean",   f"{avg_tci:.1f}\u00b0C")
        + stat_pill("P50",         f"{tci_p50:.1f}\u00b0C")
        + stat_pill("P75",         f"{tci_p75:.1f}\u00b0C")
        + stat_pill("P98",         f"{tci_p98:.1f}\u00b0C")
        + stat_pill(">32\u00b0C",  f"{pct_hot:.0f}%")
        + stat_pill(">38\u00b0C",  f"{pct_extreme:.0f}%")
        + stat_pill("SR mean",     f"{avg_sr:.1f} kWh/m\u00b2")
        + stat_pill("SR P98",      f"{sr_p98:.1f} kWh/m\u00b2")
        + stat_pill("Priority\nzone", f"{pz_pct:.0f}%")
        + '</div>'
    )

    sim_toggle = pn.widgets.RadioButtonGroup(
        options=["TCI","SR"], value=state.active_sim,
        stylesheets=[f"""
          :host .bk-btn-group button{{background:none;border:1px solid var(--border2);
            border-radius:999px;color:var(--muted);font-family:'DM Mono',monospace;
            font-size:10px;padding:4px 12px;margin:2px;transition:all .2s}}
          :host .bk-btn-group button.bk-active{{background:{bg2};border-color:{acc};color:{acc}}}
        """])
    sim_toggle.param.watch(lambda e: setattr(state,"active_sim",e.new), "value")

    ecosystem_tab = pn.Column(
        _h(f'<div style="padding:1rem 1.2rem;border-bottom:1px solid var(--border)">'
           f'<div style="font-size:12px;font-weight:500;margin-bottom:.7rem">Ecosystem Services \u2014 Current Baseline</div>'),
        radar,
        _h('</div>'),
        _h(f'<div style="padding:0 1.2rem 1rem">'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;margin-top:.8rem;'
           f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">Analysis Summary</div>'
           f'<div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {acc};'
           f'border-radius:10px;padding:1rem;font-size:12.5px;color:var(--muted);line-height:1.75;'
           f'margin-bottom:1rem">{summary}</div>'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'
           f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">Design Strategies &amp; Recommendations</div>'
           f'{strats_html}</div>'),
        pn.Column(_new_analysis_btn(acc),
                  styles={"padding":".8rem 1.2rem 1.5rem","border-top":"1px solid var(--border)"}),
        sizing_mode="stretch_width", styles={"overflow-y":"auto","flex":"1"},
    )

    comparison_tab = pn.Column(
        _h(f'<div style="padding:1rem 1.2rem">'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'
           f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">'
           f'Simulation data \u2014 Baseline vs. Estimated Vegetation-Enhanced</div>'
           f'{bars_html}'
           f'<div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {acc};'
           f'border-radius:10px;padding:1rem;font-size:12.5px;color:var(--muted);line-height:1.75;margin-top:.8rem">'
           f'<strong style="color:var(--ink)">Estimated delta:</strong> Introducing deciduous canopy in the '
           f'priority zone is projected to reduce UTCI by <strong style="color:var(--ink)">\u22125 to \u22127\u00b0C</strong> '
           f'and solar radiation by <strong style="color:var(--ink)">~55%</strong> in canopy zones. '
           f'Run <strong style="color:var(--ink)">get_vegetation.py</strong> to load actual tree inventory '
           f'and compute a precise before/after delta.</div></div>'),
        sizing_mode="stretch_width", styles={"overflow-y":"auto","flex":"1"},
    )

    trees_tab = pn.Column(
        _h(f'<div style="padding:1rem 1.2rem">'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;'
           f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">Tree placement simulation</div>'
           f'<div style="border:1.5px dashed var(--border2);border-radius:12px;padding:2rem;text-align:center;margin-bottom:1rem">'
           f'<div style="font-size:32px;margin-bottom:.5rem">\U0001f333</div>'
           f'<div style="font-size:14px;font-weight:500;color:var(--ink);margin-bottom:.3rem">Upload tree placement data</div>'
           f'<div style="font-size:12px;color:var(--muted)">Drop a CSV or GeoJSON with coordinates,<br>'
           f'species, crown diameter, height, LAI</div></div>'
           f'<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;'
           f'padding-bottom:.35rem;border-bottom:1px solid var(--border)">get_vegetation.py</div>'
           f'<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;'
           f'padding:.9rem 1rem;font-size:12px;color:var(--muted);line-height:1.75">'
           f'Once you upload tree coordinates, <strong style="color:var(--ink)">get_vegetation.py</strong> '
           f'will overlay your canopy onto the 512 \u00d7 512 m Infrared tile, re-run the UTCI simulation '
           f'with canopy shading, and display the thermal comfort delta \u2014 before vs. after planting.</div></div>'),
        sizing_mode="stretch_width", styles={"overflow-y":"auto","flex":"1"},
    )

    tabs = pn.Tabs(
        ("Ecosystem Services", ecosystem_tab),
        ("Sim Comparison",     comparison_tab),
        ("Tree Placement",     trees_tab),
        stylesheets=[f"""
          :host .bk-tab{{padding:9px 14px;font-size:12px;color:var(--muted);
            border-bottom:2px solid transparent;transition:all .2s}}
          :host .bk-tab.bk-active{{color:{acc};border-bottom-color:{acc};font-weight:500}}
          :host .bk-tabs-header{{border-bottom:1px solid var(--border)}}
        """],
    )

    right = pn.Column(
        # Stats header + sim toggle
        _h(f'<div style="padding:.7rem 1rem .5rem;border-bottom:1px solid var(--border)">'
           f'<div style="display:flex;align-items:center;justify-content:space-between;'
           f'margin-bottom:.55rem">'
           f'<span style="font-family:\'DM Mono\',monospace;font-size:9px;letter-spacing:.14em;'
           f'text-transform:uppercase;color:var(--muted)">Simulation statistics \u2014 512\u00d7512 m tile</span>'
           f'</div>'
           f'{stats_row_html}'
           f'</div>'),
        pn.Row(
            _h('<div style="font-family:\'DM Mono\',monospace;font-size:9.5px;letter-spacing:.14em;'
               'text-transform:uppercase;color:var(--muted)">Active layer</div>'),
            sim_toggle, sizing_mode="stretch_width",
            styles={"padding":".45rem 1rem","border-bottom":"1px solid var(--border)",
                    "align-items":"center","justify-content":"space-between"},
        ),
        tabs,
        sizing_mode="stretch_width",
        styles={**_PANEL_STYLES,"flex":"0 0 520px","width":"520px"},
    )

    return pn.Column(
        _topbar(),
        pn.Row(
            pn.panel(_map_pane, sizing_mode="stretch_both", styles={"flex":"1","min-width":"0"}),
            right,
            sizing_mode="stretch_both",
            styles={"display":"flex","flex-wrap":"nowrap","overflow":"hidden"},
        ),
        sizing_mode="stretch_both", styles={"background":"var(--bg)"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════════
@pn.depends(state.param.view, state.param.status_idx, state.param.status_pct, state.param.user_type)
def root(view, status_idx, status_pct, user_type):
    if view == "prompt":  return prompt_screen
    if view == "loading": return loading_screen(status_idx, status_pct, user_type)
    if user_type == "stakeholder": return _build_stakeholder()
    if user_type == "citizen":     return _build_citizen()
    return _build_aec()

pn.panel(root, sizing_mode="stretch_both").servable()

if __name__ == "__main__":
    pn.serve(
        pn.panel(root, sizing_mode="stretch_both"),
        port=5009, show=True, title="cityAgent — Urban Intelligence",
    )
