import os
import time
import requests

# ── Config ─────────────────────────────────────────────────────────────────

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT    = "simRAG/1.0" 
MAX_RETRIES   = 3
RETRY_DELAY   = 2


# ── Public API ─────────────────────────────────────────────────────────────

def get_location(address: str) -> dict:

    params = {
        "q":              address,
        "format":         "json",
        "limit":          1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[get_location] Geocoding: '{address}' (attempt {attempt}) ...")
            r = requests.get(
                NOMINATIM_URL,
                params=params,
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            results = r.json()

            if not results:
                raise ValueError(f"No location found for: '{address}'")

            hit  = results[0]
            lat  = float(hit["lat"])
            lon  = float(hit["lon"])
            name = hit.get("display_name", address)

            # bbox from Nominatim is [south, north, west, east]
            raw_bbox = hit.get("boundingbox", [])
            if len(raw_bbox) == 4:
                bbox = (
                    float(raw_bbox[2]),   # min_lon (west)
                    float(raw_bbox[0]),   # min_lat (south)
                    float(raw_bbox[3]),   # max_lon (east)
                    float(raw_bbox[1]),   # max_lat (north)
                )
            else:
                bbox = None

            print(f"[get_location] Found: {name}")
            print(f"[get_location] lat={lat:.6f}  lon={lon:.6f}")

            # Shorten display name to first two components
            short_name = ", ".join(hit.get("display_name", address).split(", ")[:2])

            return {
                "lat":          lat,
                "lon":          lon,
                "display_name": name,
                "short_name":   short_name,
                "bbox":         bbox,
            }

        except ValueError:
            raise
        except Exception as e:
            print(f"  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Geocoding failed after {MAX_RETRIES} attempts: '{address}'")


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_addresses = [
        "Bloxhub, Copenhagen",
    ]

    for address in test_addresses:
        print("-" * 60)
        loc = get_location(address)
        print(f"  lat={loc['lat']}  lon={loc['lon']}")
        print(f"  display: {loc['display_name']}")
        print()
        time.sleep(1)  