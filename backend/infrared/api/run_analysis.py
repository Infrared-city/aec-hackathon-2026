import base64
import gzip
import io
import json
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from matplotlib.colors import ListedColormap

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ENV_PATH     = os.path.join(SCRIPT_DIR, "..", "..", "..", ".env")
PAYLOAD_PATH = os.path.join(SCRIPT_DIR, "payloads", "WS.json")

load_dotenv(dotenv_path=ENV_PATH)

# ── Credentials ────────────────────────────────────────────────────────────

INFRARED_URL     = os.getenv("INFRARED_API_URL")
INFRARED_API_KEY = os.getenv("INFRARED_API_KEY_1")

# ── Colour maps keyed by analysis-type from payload ────────────────────────

COLORMAPS = {
    "wind-speed":                  ("Blues",    "Wind speed (m/s)"),
    "solar-radiation":             ("YlOrRd",   "Solar radiation (kWh/m²)"),
    "sky-view-factors":            ("GnBu_r",   "Sky view factor (0–1)"),
    "daylight-availability":       ("YlOrBr",   "Daylight autonomy (%)"),
    "direct-sun-hours":            ("viridis",  "Direct sun hours (h/day)"),
    "thermal-comfort-index":       ("magma",    "UTCI (°C)"),
    "thermal-comfort-statistics":  ("coolwarm", "Comfortable hours (%)"),
    "pedestrian-wind-comfort":     None,
}

PWC_CATEGORIES = ["A", "B", "C", "D", "E"]
PWC_COLORS     = {
    "A": "#005000",
    "B": "#00FF00",
    "C": "#FFFF00",
    "D": "#FF9900",
    "E": "#FF0000",
    "U": "#bdbdbd",
}
PWC_ORDER = PWC_CATEGORIES + ["U"]


# ── 1. Load payload ────────────────────────────────────────────────────────

with open(PAYLOAD_PATH, "r", encoding="utf-8") as f:
    analysis_settings = json.load(f)

sim_type = analysis_settings.get("analysis-type", "unknown")
print(f"Analysis type: {sim_type}")

# ── 2. Compress and post ───────────────────────────────────────────────────

compressed     = gzip.compress(json.dumps(analysis_settings).encode("utf-8"))
base64_encoded = base64.b64encode(compressed).decode("utf-8")

headers = {
    "x-api-key":           INFRARED_API_KEY,
    "Content-Type":        "text/plain",
    "X-Infrared-Encoding": "gzip",
}

print(f"Posting {os.path.basename(PAYLOAD_PATH)} ...")

try:
    response = requests.post(INFRARED_URL, data=base64_encoded, headers=headers)
    response.raise_for_status()
    print(f"Status: {response.status_code}")

except requests.RequestException as e:
    status    = e.response.status_code if e.response is not None else None
    body_text = e.response.text        if e.response is not None else ""

    print(f"\n[ERROR] Request failed — status: {status}")
    print(f"Response body: {body_text}")

    if e.response is not None:
        print(f"Response headers: {dict(e.response.headers)}")
        try:
            body_json      = e.response.json()
            print(f"Parsed JSON: {body_json}")
            parsed_message = body_json.get("message")
            if parsed_message:
                print(f"Server message: {parsed_message}")
            if "error" in body_json:
                print(f"Error details: {body_json['error']}")
        except ValueError:
            print(f"(Response is not JSON)")

    print(f"Exception type: {type(e).__name__}")
    print(f"Exception: {e}")
    raise SystemExit(1)

# ── 3. Decode response ─────────────────────────────────────────────────────

def decode_response(response):
    outer            = response.json()
    compressed_bytes = base64.b64decode(outer["result"])
    zip_buffer       = io.BytesIO(compressed_bytes)
    with zipfile.ZipFile(zip_buffer, "r") as zf:
        with zf.open("data.json") as f:
            result = json.loads(f.read())
    if isinstance(result, dict) and "output" in result:
        return result
    return {"output": result}


result = decode_response(response)
data   = result["output"]
print(f"Data points: {len(data)}")

# ── 4. Plot ────────────────────────────────────────────────────────────────

if sim_type == "pedestrian-wind-comfort":
    arr        = np.array(data, dtype=object)
    cat_to_num = {label: idx + 1 for idx, label in enumerate(PWC_CATEGORIES)}
    cat_to_num["U"]  = len(PWC_ORDER)
    cat_to_num[None] = np.nan

    numeric_grid = np.vectorize(
        lambda x: cat_to_num.get(x, np.nan), otypes=[float]
    )(arr)

    colors = [PWC_COLORS[c] for c in PWC_ORDER]
    cmap   = ListedColormap(colors)
    cmap.set_bad(color="#f0f0f0")

    plt.figure(figsize=(8, 8))
    im   = plt.imshow(numeric_grid, cmap=cmap, origin="lower",
                      vmin=1, vmax=len(PWC_ORDER))
    cbar = plt.colorbar(im, ticks=range(1, len(PWC_ORDER) + 1))
    cbar.ax.set_yticklabels(PWC_ORDER)

else:
    cmap, label = COLORMAPS.get(sim_type, ("viridis", "Value"))
    grid        = np.array(data, dtype=float).reshape((512, 512), order="F")
    valid       = grid[~np.isnan(grid)]

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap, origin="lower")
    plt.colorbar(label=label)

    if valid.size > 0:
        plt.clim(np.min(valid), np.max(valid))
        print(f"min={np.min(valid):.3f}  max={np.max(valid):.3f}  "
              f"mean={np.mean(valid):.3f}")

plt.title(sim_type)
plt.tight_layout()
plt.show()