import json
import math
import os
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import requests
from shapely.geometry import Polygon

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data_cache", "buildings")

# ── Config ─────────────────────────────────────────────────────────────────

TILE_SIZE   = 512
MAX_RETRIES = 3
RETRY_DELAY = 5

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEIGHT_BY_TYPE = {
    "house": 6.0, "residential": 9.0, "apartments": 15.0,
    "commercial": 4.0, "retail": 4.0, "office": 12.0,
    "industrial": 8.0, "warehouse": 10.0, "garage": 3.0,
    "shed": 3.0, "roof": 1.0,
}


# ── Bounding box ───────────────────────────────────────────────────────────

def get_bbox(center_lat: float, center_lon: float, size_m: float = TILE_SIZE) -> dict:
    half    = size_m / 2
    lat_d   = half / 111320
    lon_d   = half / (111320 * math.cos(math.radians(center_lat)))
    return {
        "north": center_lat + lat_d,
        "south": center_lat - lat_d,
        "east":  center_lon + lon_d,
        "west":  center_lon - lon_d,
    }


# ── Overpass query ─────────────────────────────────────────────────────────

def query_overpass(bbox: dict) -> dict:
    """
    Query OSM for buildings. Uses out geom so node coords are
    returned inline — no separate node lookup needed.
    Retries across mirrors on failure.
    """
    query = f"""
    [out:json][timeout:30];
    (
        way["building"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[get_buildings] Querying {mirror} (attempt {attempt}) ...")
                r = requests.post(mirror, data=query, timeout=60)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        print(f"  All retries failed for {mirror}, trying next mirror ...")

    raise RuntimeError("All Overpass mirrors failed. Try again later.")


# ── Height extraction ──────────────────────────────────────────────────────

def parse_height(tags: dict) -> float:
    for key in ("height", "building:height"):
        if key in tags:
            try:
                s = str(tags[key])
                if "ft" in s or "'" in s:
                    return max(float(s.replace("ft","").replace("'","").strip()) * 0.3048, 0.5)
                return max(float(s.replace("m","").strip()), 0.5)
            except (ValueError, TypeError):
                continue

    if "building:levels" in tags:
        try:
            return max(float(tags["building:levels"]) * 3.0, 1.0)
        except (ValueError, TypeError):
            pass

    return HEIGHT_BY_TYPE.get(tags.get("building", ""), 3.0)


# ── Local coordinate conversion ────────────────────────────────────────────

def to_local(coords: list, center_lat: float, center_lon: float) -> list:
    k_lon = 111320 * math.cos(math.radians(center_lat))
    k_lat = 111320
    return [
        [(lon - center_lon) * k_lon, (lat - center_lat) * k_lat]
        for lon, lat in coords
    ]


# ── Mesh extrusion ─────────────────────────────────────────────────────────

def extrude(local_coords: list, height: float):
    """Extrude a 2D polygon to a 3D triangle mesh."""
    if len(local_coords) < 3:
        return None, None
    try:
        poly = Polygon(local_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                return None, None

        pts = list(poly.exterior.coords[:-1])
        n   = len(pts)
        if n < 3:
            return None, None

        verts = []
        for x, y in pts:
            verts.extend([x, y, 0.0])
        for x, y in pts:
            verts.extend([x, y, float(height)])

        if len(verts) != n * 6:
            return None, None

        idx = []
        for i in range(1, n - 1):
            idx.extend([0, i, i + 1])
        for i in range(1, n - 1):
            idx.extend([n, n + i + 1, n + i])
        for i in range(n):
            ni = (i + 1) % n
            idx.extend([i, ni, i + n])
            idx.extend([ni, ni + n, i + n])

        return verts, idx
    except Exception as e:
        print(f"[extrude] {e}")
        return None, None


# ── Coordinate shift ───────────────────────────────────────────────────────

def shift_to_positive(dotbim: dict, shift: float = 256.0) -> dict:
    """Shift X/Y into positive quadrant — required by Infrared engine."""
    import numpy as np
    out = {}
    for name, mesh in dotbim.items():
        coords = np.array(mesh["coordinates"]).reshape(-1, 3)
        if coords[:, 2].max() <= 1.0:
            continue
        coords[:, 0] += shift
        coords[:, 1] += shift
        out[name] = {**mesh, "coordinates": coords.flatten().tolist()}
    return out


# ── Public API ─────────────────────────────────────────────────────────────

def get_buildings(
    lat:     float,
    lon:     float,
    label:   str  = "tile",
    out_dir: str  = OUTPUT_DIR,
    plot:    bool = False,
) -> str:
    """
    Fetch OSM buildings for a 512x512m area centred on (lat, lon),
    convert to dotBIM format, save as JSON, return the file path.

    Args:
        lat:     Centre latitude
        lon:     Centre longitude
        label:   Output filename prefix
        out_dir: Directory to write the JSON file
        plot:    If True, show a 2D matplotlib plot of the footprints

    Returns:
        Absolute path to the saved dotBIM JSON file.
    """
    os.makedirs(out_dir, exist_ok=True)

    bbox     = get_bbox(lat, lon)
    osm_data = query_overpass(bbox)
    elements = osm_data.get("elements", [])
    print(f"[get_buildings] {len(elements)} elements returned from OSM")

    buildings_2d = []
    dotbim       = {}

    for idx, el in enumerate(elements):
        if el["type"] != "way" or "geometry" not in el:
            continue

        coords = [(n["lon"], n["lat"]) for n in el["geometry"]]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue

        polygon_ll  = coords[:-1]
        local       = to_local(polygon_ll, lat, lon)
        height      = parse_height(el.get("tags", {}))
        osm_id      = el.get("id", idx)

        buildings_2d.append({"coords": local, "height": height})

        verts, inds = extrude(local, height)
        if verts and inds:
            name = "".join(
                c if c.isalnum() or c in "-_" else "-"
                for c in f"building-{osm_id}-{idx}"
            )
            dotbim[name] = {
                "mesh_id":     len(dotbim),
                "coordinates": verts,
                "indices":     inds,
            }

    print(f"[get_buildings] {len(buildings_2d)} polygons → {len(dotbim)} meshes")

    dotbim = shift_to_positive(dotbim)

    # ── Optional 2D plot ──────────────────────────────────────────────────
    if plot:
        half = TILE_SIZE / 2
        fig, ax = plt.subplots(figsize=(8, 8))
        for b in buildings_2d:
            if len(b["coords"]) >= 3:
                ax.add_patch(patches.Polygon(
                    b["coords"], closed=True,
                    facecolor="lightblue", edgecolor="black",
                    alpha=0.7, linewidth=0.5,
                ))
        ax.plot(0, 0, "ro", markersize=8, label="Centre")
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{label} — {len(buildings_2d)} buildings")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"{label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dotbim, f, indent=2, ensure_ascii=False)

    print(f"[get_buildings] saved → {out_path} ({len(dotbim)} meshes)")
    return out_path


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    LAT   = 55.6729
    LON   = 12.5784
    LABEL = "bloxhub"

    path = get_buildings(lat=LAT, lon=LON, label=LABEL, plot=True)
    print(f"Done: {path}")