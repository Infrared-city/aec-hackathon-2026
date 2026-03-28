"""
Urban Greening Analysis — AI-powered prompt flow
Prompt → Orchestrator plans → Map tile selection → Infrared sims → Interpreter → Results
Run:   uv run frontend/app.py
"""

import warnings; warnings.filterwarnings("ignore")
import base64, gzip, io, json, math, os, sys, threading, zipfile
import numpy as np
import pandas as pd
import panel as pn
import param
import holoviews as hv
import geoviews as gv
import geoviews.tile_sources as gvts
import requests
from dotenv import load_dotenv
from holoviews import opts

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(FRONTEND_DIR)
sys.path.insert(0, ROOT_DIR)

from backend.infrared.data.get_location     import get_location
from backend.infrared.data.get_buildings    import get_buildings
from backend.infrared.data.get_weather_data import get_weather_data
from backend.agents.orchestrator            import plan as orchestrate
from backend.agents.interpreter             import interpret

load_dotenv(os.path.join(ROOT_DIR, ".env"))
INFRARED_URL  = os.getenv("INFRARED_API_URL", "")
_INFRARED_KEYS = [os.getenv(f"INFRARED_API_KEY_{i}", "") for i in range(1, 6)]
_key_idx = 0

def _next_infrared_key():
    global _key_idx
    _key_idx = (_key_idx + 1) % len(_INFRARED_KEYS)
    return _INFRARED_KEYS[_key_idx]

INFRARED_KEY = _INFRARED_KEYS[0]
PAYLOAD_DIR  = os.path.join(ROOT_DIR, "backend", "infrared", "api", "payloads")
BUILD_CACHE  = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "buildings")
GRID_CACHE   = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "grids")
EPW_PATH     = os.path.join(ROOT_DIR, "backend", "infrared", "data_cache", "weather", "copenhagen.epw")

_MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]
WEATHER_FIELDS = [
    "dry-bulb-temperature","relative-humidity","wind-speed",
    "global-horizontal-radiation","direct-normal-radiation",
    "diffuse-horizontal-radiation","horizontal-infrared-radiation-intensity",
]
SEASON_MONTHS = {
    "summer":    (6, 8,  8, 18),
    "winter":    (1, 2,  8, 17),   # Jan–Feb (avoids year-wrap, EPW-safe)
    "full_year": (6, 8,  8, 18),   # cap to summer — API times out on full-year payloads
}

pn.extension(sizing_mode="stretch_width", raw_css=["""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=DM+Mono:wght@300;400&display=swap');
:root{--bg:#141412;--surface:#1c1c1a;--card:#242422;--border:rgba(255,255,255,0.08);
      --border2:rgba(255,255,255,0.14);--white:#f0ede6;--muted:rgba(240,237,230,0.45);
      --dim:rgba(240,237,230,0.18);--green:#2ecc9a;--green2:rgba(46,204,154,0.15);
      --green3:rgba(46,204,154,0.08)}
body,.bk-root{background:var(--bg)!important;font-family:'Inter',sans-serif!important}
"""])
hv.extension("bokeh")
gv.extension("bokeh")

# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════
class AppState(param.Parameterized):
    view          = param.Selector(default="prompt",
                                   objects=["prompt","map","loading","results"])
    prompt_text   = param.String(default="")
    user_type     = param.Selector(default="citizen", objects=["citizen","expert"])
    lat           = param.Number(default=55.6761)
    lon           = param.Number(default=12.5683)
    location_name = param.String(default="")
    address_str   = param.String(default="")
    plan          = param.Dict(default={})           # orchestrator output
    status_step   = param.String(default="")
    status_pct    = param.Number(default=0)
    sr_grid       = param.Array(default=None, allow_None=True)
    tci_grid      = param.Array(default=None, allow_None=True)
    buildings_raw = param.Dict(default={})
    active_sim    = param.Selector(default="TCI", objects=["TCI","SR","Run"])
    ai_result     = param.Dict(default={})           # interpreter output
    network_df    = param.Parameter(default=None)    # comfort network segments

state = AppState()

GLOBAL_CSS = """<style>
:root{--bg:#141412;--surface:#1c1c1a;--card:#242422;--border:rgba(255,255,255,0.08);
      --border2:rgba(255,255,255,0.14);--white:#f0ede6;--muted:rgba(240,237,230,0.45);
      --dim:rgba(240,237,230,0.18);--green:#2ecc9a;--green2:rgba(46,204,154,0.15)}
*{box-sizing:border-box}
</style>"""

LOGO = """<div style="font-family:'DM Mono',monospace;font-size:1.5rem;font-weight:400;
  letter-spacing:.04em;margin-bottom:2.5rem;color:rgba(240,237,230,.45)">
  urban<span style="color:#2ecc9a">GREEN</span></div>"""

def _css(html, **kw):
    return pn.pane.HTML(html, sizing_mode="stretch_width", **kw)

def _divider():
    return _css('<hr style="border:none;border-top:.5px solid rgba(255,255,255,.08);margin:12px 0">')

# ══════════════════════════════════════════════════════════════════════════════
# INFRARED API
# ══════════════════════════════════════════════════════════════════════════════
def call_infrared(analysis_type, lat, lon, geometries, weather):
    fmap = {"solar-radiation":"SR.json","thermal-comfort-index":"TCI.json"}
    if analysis_type not in fmap:
        raise ValueError(f"No payload for {analysis_type}")
    with open(os.path.join(PAYLOAD_DIR, fmap[analysis_type])) as f:
        payload = json.load(f)
    payload["latitude"]   = lat
    payload["longitude"]  = lon
    payload["geometries"] = geometries
    sm, em = weather["month-stamp"]
    sh, eh = weather["hour-stamp"]
    if analysis_type == "solar-radiation":
        sr_month = sm; sr_days = _MONTH_DAYS[sr_month-1]; n_sr = sr_days*(eh-sh+1)
        payload["time-period"] = {"start-month":sr_month,"start-day":1,"start-hour":sh,
                                   "end-month":sr_month,"end-day":sr_days,"end-hour":eh}
        payload["diffuse-horizontal-radiation"] = weather["diffuse-horizontal-radiation"][:n_sr]
        payload["direct-normal-radiation"]       = weather["direct-normal-radiation"][:n_sr]
        for k in ["month-stamp","hour-stamp","dry-bulb-temperature","relative-humidity",
                  "wind-speed","global-horizontal-radiation","horizontal-infrared-radiation-intensity"]:
            payload.pop(k, None)
    else:
        payload["month-stamp"] = weather["month-stamp"]
        payload["hour-stamp"]  = weather["hour-stamp"]
        for f in WEATHER_FIELDS:
            if f in weather: payload[f] = weather[f]
    b64 = base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode()
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
            print(f"[infrared] timeout on attempt {attempt+1}, rotating key")
            last_exc = requests.exceptions.Timeout()
    else:
        raise last_exc
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
        lat, lon  = state.lat, state.lon
        plan_data = state.plan
        season    = plan_data.get("season", "summer")
        _SUPPORTED = {"solar-radiation", "thermal-comfort-index"}
        sims      = [s for s in plan_data.get("simulations", ["solar-radiation","thermal-comfort-index"])
                     if s in _SUPPORTED]
        if not sims:
            sims = ["solar-radiation", "thermal-comfort-index"]
        sm, em, sh, eh = SEASON_MONTHS.get(season, SEASON_MONTHS["summer"])

        # ── Buildings ──────────────────────────────────────────────────────
        _s(status_step="get_buildings() → OSM geometry", status_pct=10)
        cache_json = os.path.join(BUILD_CACHE, "analysis.json")
        cache_meta = os.path.join(BUILD_CACHE, "analysis_meta.json")
        use_cache  = False
        if os.path.exists(cache_json) and os.path.exists(cache_meta):
            meta = json.load(open(cache_meta))
            dist = math.hypot((lat-meta["lat"])*111320,
                              (lon-meta["lon"])*111320*math.cos(math.radians(lat)))
            use_cache = dist < 10
        if use_cache:
            _s(status_step="get_buildings() → using cache", status_pct=20)
            buildings = json.load(open(cache_json))
        else:
            path = get_buildings(lat, lon, label="analysis", out_dir=BUILD_CACHE)
            buildings = json.load(open(path))
            json.dump({"lat":lat,"lon":lon}, open(cache_meta,"w"))
        _s(buildings_raw=buildings, status_pct=30)

        # ── Weather ────────────────────────────────────────────────────────
        _s(status_step="get_weather_data() → EPW fields", status_pct=40)
        weather = get_weather_data(EPW_PATH, start_month=sm, end_month=em,
                                   start_hour=sh, end_hour=eh)

        # ── Simulations ────────────────────────────────────────────────────
        sr, tci = None, None
        tci_cache = os.path.join(GRID_CACHE, "tci.npy")
        sr_cache  = os.path.join(GRID_CACHE, "sr.npy")
        use_grid_cache = os.path.exists(tci_cache) and os.path.exists(sr_cache)

        pct_step = 15
        pct_now  = 45
        for sim in sims:
            if use_grid_cache:
                _s(status_step=f"Loading cached {sim} grid", status_pct=pct_now)
                if sim == "thermal-comfort-index":
                    tci = np.load(tci_cache); _s(tci_grid=tci)
                elif sim == "solar-radiation":
                    sr  = np.load(sr_cache);  _s(sr_grid=sr)
            else:
                _s(status_step=f"run_{sim.replace('-','_')}() -> Infrared API", status_pct=pct_now)
                raw  = call_infrared(sim, lat, lon, buildings, weather)
                grid = np.array(raw["output"], dtype=float).reshape((512,512), order="F")
                if sim == "solar-radiation":
                    sr  = grid; _s(sr_grid=sr)
                    np.save(sr_cache, sr)
                elif sim == "thermal-comfort-index":
                    tci = grid; _s(tci_grid=tci)
                    np.save(tci_cache, tci)
            pct_now = min(85, pct_now + pct_step)

        # ── Interpreter ────────────────────────────────────────────────────
        _s(status_step="Synthesiser agent — generating insights", status_pct=88)
        def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
        sv, tv = _v(sr), _v(tci)
        stats = {}
        if tv.size: stats["thermal-comfort-index"] = {
            "mean":round(float(np.mean(tv)),1), "max":round(float(np.max(tv)),1),
            "pct_above_32":round(float(100*(tv>32).sum()/tv.size),1)}
        if sv.size: stats["solar-radiation"] = {
            "mean":round(float(np.mean(sv)),0), "max":round(float(np.max(sv)),0)}
        if sv.size and tv.size:
            tt = float(np.percentile(tv,70)); st = float(np.percentile(sv,70))
            sf, tf = sr.flatten(), tci.flatten()
            mask = ~(np.isnan(sf)|np.isnan(tf))
            stats["priority_zone_pct"] = round(
                100*((sf[mask]>=st)&(tf[mask]>=tt)).sum()/mask.sum(), 1)

        ai = interpret(
            stats       = stats,
            location    = state.location_name,
            user_type   = plan_data.get("user_type", state.user_type),
            prompt      = state.prompt_text,
            n_buildings = len(buildings),
            season      = season,
        )
        # ── Comfort network ────────────────────────────────────────────────
        if tci is not None:
            _s(status_step="Building comfort run network → OSM streets", status_pct=95)
            net = _build_comfort_network(tci, lat, lon)
            _s(network_df=net)

        _s(ai_result=ai, status_pct=100, view="results", status_step="Done.")

    except Exception as e:
        import traceback; traceback.print_exc()
        _s(view="map", status_step=str(e))

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def tile_bounds(lat, lon, half_m=256):
    ld = half_m/111320
    lo = half_m/(111320*math.cos(math.radians(lat)))
    return (lon-lo, lat-ld, lon+lo, lat+ld)

def _sample_grid(grid, lat, lon, clat, clon):
    """Sample a 512×512 grid at a geographic point."""
    mla = 111320
    mlo = 111320 * math.cos(math.radians(clat))
    i = int((lon - clon) * mlo + 256)
    j = int((lat - clat) * mla + 256)
    if 0 <= i < 512 and 0 <= j < 512:
        return float(grid[j, i])
    return float("nan")

def _build_comfort_network(tci_grid, clat, clon):
    """Fetch OSM walk network, sample TCI at each edge midpoint, return DataFrame."""
    try:
        import osmnx as ox
        ox.settings.log_console = False
        G = ox.graph_from_point((clat, clon), dist=300, network_type="walk", simplify=True)
        rows = []
        for u, v, _ in G.edges(data=True):
            ud, vd = G.nodes[u], G.nodes[v]
            mid_lat = (ud["y"] + vd["y"]) / 2
            mid_lon = (ud["x"] + vd["x"]) / 2
            utci = _sample_grid(tci_grid, mid_lat, mid_lon, clat, clon)
            rows.append({"x0": ud["x"], "y0": ud["y"],
                         "x1": vd["x"], "y1": vd["y"],
                         "UTCI": utci if not math.isnan(utci) else 30.0})
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        print(f"[network] {e}")
        return None

def merc_to_wgs84(x, y):
    return (math.degrees(2*math.atan(math.exp(y/20037508.34*math.pi))-math.pi/2),
            x/20037508.34*180)

def _metric_card(name, val, unit, pct, color="var(--green,#2ecc9a)"):
    return f"""<div style="background:#242422;border:.5px solid rgba(255,255,255,.08);
      border-radius:8px;padding:.7rem .9rem">
      <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                  color:rgba(240,237,230,.45);margin-bottom:4px;
                  font-family:'DM Mono',monospace">{name}</div>
      <div style="font-size:1.2rem;font-weight:300;color:#f0ede6">{val}
        <span style="font-size:10px;color:rgba(240,237,230,.45);margin-left:2px">{unit}</span></div>
      <div style="height:2px;background:rgba(255,255,255,.08);border-radius:99px;margin-top:6px">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:99px"></div>
      </div></div>"""

def _rec(n, text):
    return f"""<div style="display:flex;gap:.7rem;padding:.6rem 0;
      border-bottom:.5px solid rgba(255,255,255,.08)">
      <span style="color:#2ecc9a;font-size:11px;flex-shrink:0;padding-top:2px;
                   font-family:'DM Mono',monospace">{n:02d}</span>
      <span style="font-size:12px;color:#f0ede6;line-height:1.65">{text}</span></div>"""

_PANEL_STYLES = {
    "background":"#1c1c1a","border-left":".5px solid rgba(255,255,255,.08)",
    "overflow-y":"auto","height":"100%","box-sizing":"border-box",
    "flex":"1","min-width":"0",
}

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — PROMPT
# ══════════════════════════════════════════════════════════════════════════════
EXAMPLES = [
    "Is Nørrebro safe for my kids to play outside in summer?",
    "Which streets in Vesterbro get the most heat? I want more trees.",
    "Analyse thermal comfort and solar potential in Sydhavn for green roof prioritisation.",
    "Run a summer heat stress analysis for Bispebjerg to identify vegetation gaps.",
    "My neighbourhood in Amager feels too hot. Where should the city plant trees?",
    "Assess UTCI and solar radiation in Frederiksberg for urban greening strategy.",
]

prompt_input = pn.widgets.TextAreaInput(
    placeholder="Describe what you want to analyse…",
    rows=3,
    stylesheets=["""
      :host textarea{width:100%;background:#242422;border:.5px solid rgba(255,255,255,.14);
        border-radius:8px;color:#f0ede6;font-family:'Inter',sans-serif;font-size:14px;
        padding:13px 16px;outline:none;resize:none;box-sizing:border-box;
        transition:border-color .2s;line-height:1.6}
      :host textarea:focus{border-color:#2ecc9a}
      :host textarea::placeholder{color:rgba(240,237,230,.18)}
    """],
    sizing_mode="stretch_width",
)

analyse_btn = pn.widgets.Button(
    name="Analyse →",
    stylesheets=["""
      :host button{width:100%;background:#2ecc9a;color:#0a120d;border:none;
        border-radius:8px;font-family:'Inter',sans-serif;font-size:14px;
        font-weight:500;padding:14px;cursor:pointer;letter-spacing:.02em;margin-top:8px}
      :host button:hover{opacity:.88}
    """],
    sizing_mode="stretch_width",
)

prompt_error = _css("")

_example_pills = "".join(
    f'<div onclick="fillPrompt(this)" style="cursor:pointer;padding:7px 14px;'
    f'border:.5px solid rgba(255,255,255,.14);border-radius:8px;font-size:12px;'
    f'color:rgba(240,237,230,.55);margin-bottom:6px;transition:all .2s;'
    f'font-family:Inter,sans-serif;line-height:1.4" '
    f'onmouseover="this.style.borderColor=\'#2ecc9a\';this.style.color=\'#f0ede6\'" '
    f'onmouseout="this.style.borderColor=\'rgba(255,255,255,.14)\';this.style.color=\'rgba(240,237,230,.55)\'"'
    f'>{ex}</div>'
    for ex in EXAMPLES
)
example_pane = pn.pane.HTML(f"""
    <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                color:rgba(240,237,230,.35);margin-bottom:.5rem;
                font-family:'DM Mono',monospace">Example prompts</div>
    {_example_pills}
    <script>
    function fillPrompt(el){{
      var ta = document.querySelector('textarea');
      if(ta) ta.value = el.innerText;
    }}
    </script>""", sizing_mode="stretch_width")

def _on_analyse(event):
    prompt = prompt_input.value.strip()
    if not prompt:
        prompt_error.object = '<div style="color:#ff6b6b;font-size:12px;margin-bottom:8px">Please enter a prompt.</div>'
        return
    prompt_error.object = '<div style="color:rgba(240,237,230,.45);font-size:12px;margin-bottom:8px;font-family:\'DM Mono\',monospace">Planner agent — reading prompt…</div>'
    state.prompt_text = prompt

    def _plan_and_go():
        try:
            p = orchestrate(prompt)
            state.plan = p
            # geocode the location the planner extracted
            loc = get_location(p.get("location", "Copenhagen, Denmark"))
            state.lat          = loc["lat"]
            state.lon          = loc["lon"]
            state.location_name = loc.get("short_name", p.get("location",""))
            state.address_str  = p.get("location","")
            tap.event(x=loc["lon"], y=loc["lat"])
            state.view = "map"
        except Exception as e:
            import traceback; traceback.print_exc()
            prompt_error.object = f'<div style="color:#ff6b6b;font-size:12px;margin-bottom:8px">{e}</div>'

    threading.Thread(target=_plan_and_go, daemon=True).start()

analyse_btn.on_click(_on_analyse)

prompt_screen = pn.Column(
    _css(GLOBAL_CSS),
    _css(f"""
    <div style="background:#141412;min-height:100vh;display:flex;flex-direction:column;
                align-items:center;justify-content:center;padding:2rem">
      {LOGO}
      <div style="background:#1c1c1a;border:.5px solid rgba(255,255,255,.08);
                  border-radius:12px;padding:2rem;width:600px;max-width:94vw">
        <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
                    color:#2ecc9a;margin-bottom:.6rem;font-family:'DM Mono',monospace">
          Urban Environmental Analysis</div>
        <div style="font-size:1.25rem;font-weight:400;color:#f0ede6;
                    margin-bottom:.4rem;line-height:1.4">
          What would you like to analyse?</div>
        <div style="font-size:12px;color:rgba(240,237,230,.45);margin-bottom:1.4rem;
                    line-height:1.6">
          Describe your question in plain language. The AI will detect your role and
          select the right simulations, fetch real building data, and run the Infrared API.</div>
    """),
    prompt_error,
    prompt_input,
    analyse_btn,
    _css("</div>"),  # close card
    _css(f"""
        <div style="margin-top:1.5rem;width:600px;max-width:94vw">
    """),
    example_pane,
    _css("</div></div>"),
    sizing_mode="stretch_both",
    styles={"background":"#141412"},
)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — MAP TILE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
tap      = hv.streams.Tap(x=state.lon, y=state.lat)
coord_html = pn.pane.HTML("", sizing_mode="stretch_width")

def _upd_coord():
    coord_html.object = (
        f'<div style="font-family:\'DM Mono\',monospace;font-size:11px;'
        f'color:rgba(240,237,230,.45);padding:6px 0">'
        f'{state.lat:.5f}°N  {state.lon:.5f}°E</div>'
    )
_upd_coord()

def _tap_overlay(x, y):
    lat, lon = merc_to_wgs84(x, y) if abs(x) > 180 else (y, x)
    state.lon, state.lat = lon, lat
    _upd_coord()
    ld = 256/(111320*math.cos(math.radians(lat))); la = 256/111320
    box = [(lon-ld,lat-la),(lon+ld,lat-la),(lon+ld,lat+la),(lon-ld,lat+la),(lon-ld,lat-la)]
    return (
        gv.Points([(lon,lat)],kdims=["lon","lat"]).opts(
            opts.Points(color="#2ecc9a",size=14,line_color="white",line_width=2,tools=[]))
        * gv.Path([box],kdims=["lon","lat"]).opts(
            opts.Path(color="#2ecc9a",line_width=2,line_dash="dashed",alpha=0.85))
    )

sel_overlay = hv.DynamicMap(_tap_overlay, streams=[tap])
sel_map     = (gvts.CartoDark * sel_overlay).opts(
    opts.Overlay(title="Click to place 512 × 512 m analysis tile", width=1100, height=820))

run_btn = pn.widgets.Button(name="Run analysis →", stylesheets=["""
  :host button{width:100%;background:#2ecc9a;color:#0a120d;border:none;border-radius:8px;
    font-family:'Inter',sans-serif;font-size:14px;font-weight:500;padding:14px;
    cursor:pointer;letter-spacing:.02em}
  :host button:hover{opacity:.88}
"""], sizing_mode="stretch_width")

back_map_btn = pn.widgets.Button(name="← Edit prompt", stylesheets=["""
  :host button{width:100%;background:none;border:.5px solid rgba(255,255,255,.14);
    border-radius:8px;color:rgba(240,237,230,.45);font-family:'Inter',sans-serif;
    font-size:13px;padding:10px;cursor:pointer;margin-top:6px}
  :host button:hover{border-color:#2ecc9a;color:#2ecc9a}
"""], sizing_mode="stretch_width")

plan_summary_pane = pn.pane.HTML("", sizing_mode="stretch_width")

def _refresh_plan_summary():
    p = state.plan
    if not p: return
    sims = ", ".join(f'<span style="color:#2ecc9a;font-family:\'DM Mono\',monospace;font-size:11px">{s}</span>'
                     for s in p.get("simulations",[]))
    plan_summary_pane.object = f"""
    <div style="background:#242422;border:.5px solid rgba(255,255,255,.08);
                border-radius:8px;padding:12px 14px;margin-bottom:12px">
      <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                  color:rgba(240,237,230,.35);margin-bottom:6px;
                  font-family:'DM Mono',monospace">Planner selected</div>
      <div style="margin-bottom:6px;line-height:1.8">{sims}</div>
      <div style="font-size:11px;color:rgba(240,237,230,.45);line-height:1.6">
        {p.get('plan_text','')}
      </div>
    </div>"""

def _on_run(event):
    state.view = "loading"
    threading.Thread(target=run_pipeline, daemon=True).start()

def _on_back_map(event):
    state.view = "prompt"

run_btn.on_click(_on_run)
back_map_btn.on_click(_on_back_map)

map_right = pn.Column(
    _css(f"""
    <div style="font-family:'Inter',sans-serif;margin-bottom:14px">
      <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;
                  color:#2ecc9a;margin-bottom:.5rem;font-family:'DM Mono',monospace">
        Step 2 of 2</div>
      <div style="font-size:1.1rem;font-weight:400;color:#f0ede6;line-height:1.4;
                  margin-bottom:.5rem">Confirm analysis tile</div>
      <div style="font-size:12px;color:rgba(240,237,230,.45);line-height:1.7">
        The planner has centred the map on <b style="color:#f0ede6">{state.location_name}</b>.
        Click to adjust the 512 × 512 m tile if needed, then run.
      </div>
    </div>
    """),
    plan_summary_pane,
    coord_html,
    _css('<hr style="border:none;border-top:.5px solid rgba(255,255,255,.08);margin:10px 0">'),
    run_btn,
    back_map_btn,
    _css("""<div style="display:flex;gap:8px;margin-top:1.5rem">
      <div style="width:8px;height:8px;border-radius:50%;background:rgba(255,255,255,.14)"></div>
      <div style="width:8px;height:8px;border-radius:50%;background:#2ecc9a"></div>
    </div>"""),
    sizing_mode="stretch_width",
    styles={**_PANEL_STYLES, "padding":"28px 22px"},
)

map_screen = pn.Row(
    pn.pane.HoloViews(sel_map, sizing_mode="stretch_both",
                      styles={"flex":"4","min-width":"0"}),
    map_right,
    sizing_mode="stretch_both",
    styles={"display":"flex","flex-wrap":"nowrap","overflow":"hidden"},
)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — LOADING
# ══════════════════════════════════════════════════════════════════════════════
STEPS_DEF = [
    "Planner agent — selecting simulations",
    "get_buildings() → OSM geometry",
    "get_weather_data() → EPW fields",
    "run_solar_radiation() → Infrared API",
    "run_thermal_comfort() → Infrared API",
    "Synthesiser agent — generating insights",
    "Building comfort run network → OSM streets",
]

def _pct_to_step(pct):
    for i, t in enumerate([0,10,30,40,55,80,88,100][1:], 1):
        if pct <= t: return i-1
    return len(STEPS_DEF)-1

@pn.depends(state.param.status_step, state.param.status_pct)
def loading_panel(step_text, pct):
    active = _pct_to_step(pct)
    rows = ""
    for i, label in enumerate(STEPS_DEF):
        done, cur = i < active, i == active
        dot = "#2ecc9a" if cur else ("rgba(46,204,154,.4)" if done else "rgba(255,255,255,.14)")
        sty = ("opacity:1;border-color:rgba(46,204,154,.35);background:rgba(46,204,154,.08)" if cur
               else "opacity:.7" if done else "opacity:.28")
        rows += f"""<div style="display:flex;align-items:center;gap:.8rem;background:#1c1c1a;
          border:.5px solid rgba(255,255,255,.08);border-radius:8px;padding:11px 16px;
          margin-bottom:.5rem;font-family:'DM Mono',monospace;font-size:12px;
          color:rgba(240,237,230,.45);{sty}">
          <div style="width:8px;height:8px;border-radius:50%;background:{dot};flex-shrink:0"></div>
          {label}</div>"""
    return pn.pane.HTML(f"""
    {GLOBAL_CSS}
    <div style="background:#141412;min-height:100vh;display:flex;flex-direction:column;
                align-items:center;justify-content:center;padding:2rem">
      {LOGO}
      <div style="font-size:1.15rem;font-weight:400;color:rgba(240,237,230,.45);
                  margin-bottom:2rem">Running agentic workflow</div>
      <div style="width:480px;max-width:94vw">{rows}</div>
      <div style="width:480px;max-width:94vw;height:3px;background:rgba(255,255,255,.08);
                  border-radius:99px;overflow:hidden;margin-top:1.5rem">
        <div style="height:100%;width:{pct}%;background:#2ecc9a;border-radius:99px;
                    transition:width .5s ease"></div>
      </div>
    </div>""", sizing_mode="stretch_both")

loading_screen = pn.Column(
    pn.panel(loading_panel, sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
    styles={"background":"#141412"},
)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — RESULTS MAP
# ══════════════════════════════════════════════════════════════════════════════
def _tree_points(tci, sr, clat, clon, n=40):
    if tci is None or sr is None: return None
    tv, sv = tci[~np.isnan(tci)], sr[~np.isnan(sr)]
    if not tv.size or not sv.size: return None
    tt, st = float(np.percentile(tv,70)), float(np.percentile(sv,70))
    tn = (tci-np.nanmin(tci))/(np.nanmax(tci)-np.nanmin(tci)+1e-9)
    sn = (sr -np.nanmin(sr)) /(np.nanmax(sr) -np.nanmin(sr) +1e-9)
    sc = tn*.6+sn*.4
    mask = (tci>=tt)&(sr>=st)&~np.isnan(tci)&~np.isnan(sr)
    if not mask.any(): return None
    ys,xs = np.where(mask); order = np.argsort(sc[ys,xs])[::-1]
    ys,xs = ys[order], xs[order]
    sel=[]
    for k in range(len(xs)):
        cx,cy=int(xs[k]),int(ys[k])
        if not any(abs(cx-sx)<20 and abs(cy-sy)<20 for sx,sy,*_ in sel):
            sel.append((cx,cy,tci[cy,cx],sr[cy,cx]))
        if len(sel)>=n: break
    if not sel: return None
    mla=111320; mlo=111320*math.cos(math.radians(clat))
    return pd.DataFrame({
        "lon":[clon+(i-256+.5)/mlo for i,j,*_ in sel],
        "lat":[clat+(j-256+.5)/mla for i,j,*_ in sel],
        "UTCI (°C)":[round(float(t),1) for _,_,t,_ in sel],
        "Solar (kWh/m²)":[round(float(s),1) for _,_,_,s in sel],
        "Suggested planting":["Tree / green roof zone"]*len(sel),
    })

def _build_results_map():
    from matplotlib.colors import LinearSegmentedColormap
    lat, lon = state.lat, state.lon
    base = gvts.CartoDark.opts(width=1100, height=820)

    if state.active_sim == "Run":
        layers = base
        net = state.network_df
        if net is not None and not net.empty:
            run_cmap = LinearSegmentedColormap.from_list(
                "run", ["#2ecc9a","#c8a040","#e85d04","#ff4444"], 256)
            # comfortable streets (UTCI < 28) — thick green
            for label, mask, color, lw, alpha in [
                ("Comfortable run", net["UTCI"] < 28,      "#2ecc9a", 3.5, 0.95),
                ("Moderate",        (net["UTCI"] >= 28) & (net["UTCI"] < 32), "#c8a040", 2.5, 0.75),
                ("Avoid (heat)",    net["UTCI"] >= 32,     "#e85d04", 1.5, 0.55),
            ]:
                sub = net[mask]
                if sub.empty: continue
                paths = [[(r.x0,r.y0),(r.x1,r.y1)] for _,r in sub.iterrows()]
                layers = layers * gv.Path(paths, kdims=["Longitude","Latitude"],
                    label=label).opts(opts.Path(
                        color=color, line_width=lw, alpha=alpha, tools=["hover"]))
        return layers.opts(opts.Overlay(
            title=f"Comfortable running routes · {state.location_name}", width=1100, height=820))

    # Heatmap views (TCI / SR)
    grid = state.tci_grid if state.active_sim=="TCI" else state.sr_grid
    cmap = (LinearSegmentedColormap.from_list("tci",["#141412","#1c2e22","#1f4a32","#2ecc9a","#a8f0d8"],256)
            if state.active_sim=="TCI" else
            LinearSegmentedColormap.from_list("sr", ["#141412","#1e1c0e","#3a3010","#7a6020","#c8a040"],256))
    lbl  = "UTCI (°C)" if state.active_sim=="TCI" else "Solar Radiation (kWh/m²)"
    layers = base
    if grid is not None:
        v = grid[~np.isnan(grid)]
        vmin,vmax = (float(np.percentile(v,5)),float(np.percentile(v,95))) if v.size else (0,1)
        img = gv.Image(grid.T, bounds=tile_bounds(lat,lon),
                       kdims=["Longitude","Latitude"],vdims=[lbl]).opts(
            opts.Image(cmap=cmap,alpha=0.75,colorbar=True,clim=(vmin,vmax),
                       tools=["hover"],width=1100,height=820))
        layers = layers * img
    df = _tree_points(state.tci_grid, state.sr_grid, lat, lon)
    if df is not None:
        layers = layers * gv.Points(df,kdims=["lon","lat"],
            vdims=["UTCI (°C)","Solar (kWh/m²)","Suggested planting"]).opts(
            opts.Points(color="#2ecc9a",size=9,line_color="#141412",line_width=1.5,
                        alpha=0.92,tools=["hover"],
                        hover_tooltips=[("","@{Suggested planting}"),
                                        ("UTCI","@{UTCI (°C)} °C"),
                                        ("Solar","@{Solar (kWh/m²)} kWh/m²")]))
    return layers.opts(opts.Overlay(
        title=f"Vegetation priority · {state.location_name}", width=1100, height=820))

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 4 — RESULTS PANEL
# ══════════════════════════════════════════════════════════════════════════════
def _build_results_panel():
    sr, tci = state.sr_grid, state.tci_grid
    ai      = state.ai_result
    def _v(a): return a[~np.isnan(a)] if a is not None else np.array([])
    sv, tv = _v(sr), _v(tci)

    avg_tci  = f"{np.mean(tv):.1f}"  if tv.size else "N/A"
    max_tci  = f"{np.max(tv):.1f}"   if tv.size else "N/A"
    avg_sr   = f"{np.mean(sv):.0f}"  if sv.size else "N/A"
    max_sr   = f"{np.max(sv):.0f}"   if sv.size else "N/A"
    pct_hot  = f"{100*(tv>32).sum()/tv.size:.0f}" if tv.size else "0"
    pct_comf = f"{100*(tv<=32).sum()/tv.size:.0f}" if tv.size else "0"

    # AI-generated content (with fallbacks)
    score   = ai.get("score", 5.0)
    verdict = ai.get("verdict", "Analysis complete")
    summary = ai.get("summary", "")
    recs    = ai.get("recommendations", [])

    score_bar = int(min(100, max(0, score*10)))
    user_tag  = state.plan.get("user_type", state.user_type)

    # sim toggle
    sim_toggle = pn.widgets.RadioButtonGroup(
        options=["TCI","SR","Run"], value=state.active_sim,
        stylesheets=["""
          :host .bk-btn-group button{background:none;border:.5px solid rgba(255,255,255,.14);
            border-radius:999px;color:rgba(240,237,230,.45);font-family:'DM Mono',monospace;
            font-size:11px;padding:4px 14px;margin:2px;transition:all .2s}
          :host .bk-btn-group button.bk-active{background:rgba(46,204,154,.15);
            border-color:#2ecc9a;color:#2ecc9a}
        """],
    )
    def _on_sim(e): state.active_sim = e.new
    sim_toggle.param.watch(_on_sim, "value")

    back_btn = pn.widgets.Button(name="← New analysis", stylesheets=["""
      :host button{background:#1c1c1a;border:.5px solid rgba(255,255,255,.14);
        color:rgba(240,237,230,.45);font-family:'Inter',sans-serif;font-size:12px;
        padding:6px 14px;cursor:pointer;border-radius:6px;transition:all .2s;margin:12px 14px}
      :host button:hover{border-color:#2ecc9a;color:#2ecc9a}
    """])
    def _on_back(e):
        state.sr_grid=None; state.tci_grid=None; state.ai_result={}; state.view="prompt"
    back_btn.on_click(_on_back)

    metrics = (
        _metric_card("Thermal Comfort", avg_tci, "°C UTCI",
                     max(0,min(100,int((38-float(avg_tci))/18*100))) if avg_tci!="N/A" else 50)
        + _metric_card("Solar Radiation", avg_sr, "kWh/m²",
                       min(100,int(float(avg_sr)/12)) if avg_sr!="N/A" else 50, "#c8a040")
        + _metric_card("Heat-Stress Area", pct_hot, "% of tile",
                       int(pct_hot) if pct_hot!="N/A" else 0, "#ff6b6b")
        + _metric_card("Comfortable Area", pct_comf, "% of tile",
                       int(pct_comf) if pct_comf!="N/A" else 0)
    )

    recs_html = "".join(_rec(i+1, r) for i, r in enumerate(recs[:3]))
    if not recs_html:
        recs_html = _rec(1,"Prioritise street tree planting in high-TCI corridors.")

    persona_badge = (
        '<span style="background:rgba(46,204,154,.15);border:.5px solid #2ecc9a;'
        'color:#2ecc9a;font-size:10px;font-family:\'DM Mono\',monospace;'
        'padding:2px 8px;border-radius:999px;margin-left:8px">'
        + user_tag + '</span>'
    )

    return pn.Column(
        # header
        _css(f"""
        <div style="padding:1.2rem 1.4rem .8rem;border-bottom:.5px solid rgba(255,255,255,.08)">
          <div style="font-size:1.05rem;font-weight:500;color:#f0ede6;margin-bottom:2px">
            {state.location_name}{persona_badge}</div>
          <div style="font-size:12px;color:rgba(240,237,230,.45)">
            📍 {state.address_str} · {state.lat:.4f}°N {state.lon:.4f}°E</div>
          <div style="font-size:11px;color:rgba(240,237,230,.35);margin-top:4px;
                      font-family:'DM Mono',monospace;font-style:italic">
            "{state.prompt_text[:80]}{'…' if len(state.prompt_text)>80 else ''}"</div>
        </div>
        """),
        # score
        _css(f"""
        <div style="padding:1.2rem 1.4rem;border-bottom:.5px solid rgba(255,255,255,.08)">
          <div style="display:flex;align-items:baseline;gap:.5rem;margin-bottom:.6rem">
            <div style="font-size:2.8rem;font-weight:300;color:#2ecc9a;line-height:1">{score}</div>
            <div>
              <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                          color:rgba(240,237,230,.45)">Comfort Score / 10</div>
              <div style="font-size:12px;color:#f0ede6;margin-top:2px">{verdict}</div>
            </div>
          </div>
          <div style="height:4px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden">
            <div style="height:100%;width:{score_bar}%;background:#2ecc9a;border-radius:99px"></div>
          </div>
        </div>
        """),
        # sim toggle + legend
        _css(f"""
        <div style="padding:.8rem 1.4rem;border-bottom:.5px solid rgba(255,255,255,.08)">
        """),
        sim_toggle,
        _css(f"""
          <div style="padding-top:.5rem;font-size:11px;color:rgba(240,237,230,.35);line-height:2">
            {"".join([
              f'<span style="display:inline-block;width:22px;height:3px;background:{c};vertical-align:middle;border-radius:2px;margin-right:5px"></span>{lbl}<br>'
              for c,lbl in [("#2ecc9a","Comfortable (UTCI < 28 °C)"),
                            ("#c8a040","Moderate (28–32 °C)"),
                            ("#e85d04","Avoid — heat stress (> 32 °C)")]
            ]) if state.active_sim == "Run" else
            '<span style="display:inline-block;width:10px;height:10px;background:#2ecc9a;border-radius:50%;vertical-align:middle;margin-right:5px;border:1.5px solid #141412"></span>Suggested tree planting locations'}
          </div>
        </div>
        """),
        # metrics
        _css(f"""
        <div style="padding:.8rem 1.4rem;border-bottom:.5px solid rgba(255,255,255,.08)">
          <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                      color:rgba(240,237,230,.45);margin-bottom:.5rem">Key metrics</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem">{metrics}</div>
        </div>
        """),
        # AI summary
        _css(f"""
        <div style="padding:.9rem 1.4rem;border-bottom:.5px solid rgba(255,255,255,.08)">
          <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                      color:rgba(240,237,230,.45);margin-bottom:.5rem">Analysis summary</div>
          <div style="font-size:12px;line-height:1.75;color:rgba(240,237,230,.6)">{summary}</div>
        </div>
        """),
        # AI recommendations
        _css(f"""
        <div style="padding:.9rem 1.4rem">
          <div style="font-size:10px;letter-spacing:.12em;text-transform:uppercase;
                      color:rgba(240,237,230,.45);margin-bottom:.6rem">Recommendations</div>
          {recs_html}
        </div>
        """),
        back_btn,
        sizing_mode="stretch_width",
        styles=_PANEL_STYLES,
    )

def _build_results_view():
    return pn.Row(
        pn.pane.HoloViews(_build_results_map(), sizing_mode="stretch_both",
                          styles={"flex":"4","min-width":"0"}),
        _build_results_panel(),
        sizing_mode="stretch_both",
        styles={"display":"flex","flex-wrap":"nowrap","overflow":"hidden"},
    )

# ══════════════════════════════════════════════════════════════════════════════
# REACTIVE ROOT
# ══════════════════════════════════════════════════════════════════════════════
def _on_view_to_map(event):
    if event.new == "map":
        _refresh_plan_summary()

state.param.watch(_on_view_to_map, "view")

@pn.depends(state.param.view, state.param.active_sim,
            state.param.status_step, state.param.status_pct)
def root(view, active_sim, status_step, status_pct):
    if view == "prompt":  return prompt_screen
    if view == "map":     return map_screen
    if view == "loading": return loading_screen
    return _build_results_view()

pn.panel(root, sizing_mode="stretch_both").servable()

if __name__ == "__main__":
    pn.serve(pn.panel(root, sizing_mode="stretch_both"),
             port=5008, show=True, title="Urban Greening Analysis")
