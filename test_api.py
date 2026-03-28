"""
Quick Infrared API health check.
Run: uv run test_api.py
"""
import base64, gzip, io, json, os, sys, time, zipfile
import requests
from dotenv import load_dotenv

load_dotenv()

URL  = os.getenv("INFRARED_API_URL", "")
KEYS = [os.getenv(f"INFRARED_API_KEY_{i}", "") for i in range(1, 6)]

# Tiny synthetic payload — no real buildings, just a lat/lon ping
MINI_PAYLOAD = {
    "latitude": 55.6761,
    "longitude": 12.5683,
    "geometries": [],               # empty → API runs on open field
    "month-stamp": [6, 6],
    "hour-stamp":  [10, 12],
    "dry-bulb-temperature":              [18.0, 19.0, 20.0],
    "relative-humidity":                 [60.0, 60.0, 60.0],
    "wind-speed":                        [3.0,  3.0,  3.0],
    "global-horizontal-radiation":       [400.0,500.0,450.0],
    "direct-normal-radiation":           [300.0,380.0,350.0],
    "diffuse-horizontal-radiation":      [100.0,120.0,110.0],
    "horizontal-infrared-radiation-intensity": [300.0,310.0,305.0],
    "analysis-type": "thermal-comfort-index",
    "resolution": 512,
}

print(f"URL : {URL}")
print(f"Keys: {len([k for k in KEYS if k])} configured\n")

for i, key in enumerate(KEYS):
    if not key:
        print(f"Key {i+1}: (empty, skipping)")
        continue
    print(f"Key {i+1}: {key[:8]}…  ", end="", flush=True)
    b64 = base64.b64encode(gzip.compress(json.dumps(MINI_PAYLOAD).encode())).decode()
    t0  = time.time()
    try:
        r = requests.post(URL, data=b64,
                          headers={"x-api-key": key,
                                   "Content-Type": "text/plain",
                                   "X-Infrared-Encoding": "gzip"},
                          timeout=30)
        elapsed = time.time() - t0
        if r.ok:
            try:
                outer = r.json()
                with zipfile.ZipFile(io.BytesIO(base64.b64decode(outer["result"]))) as zf:
                    data = json.loads(zf.read("data.json"))
                n = len(data.get("output", data) if "output" not in data else data["output"])
                print(f"OK  ({elapsed:.1f}s)  output length={n}")
            except Exception as e:
                print(f"OK  ({elapsed:.1f}s)  but parse error: {e}")
                print(f"      raw: {r.text[:200]}")
        else:
            print(f"HTTP {r.status_code}  ({elapsed:.1f}s)  {r.text[:200]}")
    except requests.exceptions.Timeout:
        print(f"TIMEOUT after 30s")
    except Exception as e:
        print(f"ERROR: {e}")
