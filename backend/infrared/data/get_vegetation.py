import json
import math
import os
import time
import requests
from shapely.geometry import Polygon

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data_cache", "vegetation")

# ── Config ─────────────────────────────────────────────────────────────────

TILE_SIZE   = 512
MAX_RETRIES = 3
RETRY_DELAY = 5

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Default crown radius by genus/species tag (metres)
CROWN_RADIUS_BY_GENUS = {
    "quercus":    6.0,   # oak
    "tilia":      5.0,   # linden / lime
    "acer":       5.0,   # maple
    "betula":     4.0,   # birch
    "populus":    5.0,   # poplar
    "fraxinus":   5.0,   # ash
    "prunus":     3.5,   # cherry / plum
    "platanus":   7.0,   # plane
    "pinus":      4.0,   # pine
    "picea":      3.0,   # spruce
    "fagus":      6.0,   # beech
}
DEFAULT_CROWN_RADIUS = 3.0  # fallback
DEFAULT_TREE_HEIGHT  = 8.0


# ── Bounding box ───────────────────────────────────────────────────────────

def get_bbox(center_lat: float, center_lon: float, size_m: float = TILE_SIZE) -> dict:
    half  = size_m / 2
    lat_d = half / 111320
    lon_d = half / (111320 * math.cos(math.radians(center_lat)))
    return {
        "north": center_lat + lat_d,
        "south": center_lat - lat_d,
        "east":  center_lon + lon_d,
        "west":  center_lon - lon_d,
    }


# ── Overpass query ─────────────────────────────────────────────────────────

def query_overpass(bbox: dict) -> dict:
    """
    Query OSM for individual trees (nodes) and tree rows/areas (ways/relations).
    Returns raw Overpass JSON.
    """
    query = f"""
    [out:json][timeout:30];
    (
        node["natural"="tree"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
        way["natural"="tree_row"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
        way["landuse"="forest"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
        way["natural"="wood"]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
    );
    out geom;
    """
    for mirror in OVERPASS_MIRRORS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[get_vegetation] Querying {mirror} (attempt {attempt}) ...")
                r = requests.post(mirror, data=query, timeout=60)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        print(f"  All retries failed for {mirror}, trying next mirror ...")

    raise RuntimeError("All Overpass mirrors failed. Try again later.")


# ── Attribute helpers ──────────────────────────────────────────────────────

def _parse_height(tags: dict) -> float:
    for key in ("height", "tree:height"):
        if key in tags:
            try:
                return max(float(str(tags[key]).replace("m", "").strip()), 0.5)
            except (ValueError, TypeError):
                continue
    return DEFAULT_TREE_HEIGHT


def _parse_crown_radius(tags: dict) -> float:
    for key in ("diameter_crown", "crown_diameter"):
        if key in tags:
            try:
                return max(float(str(tags[key]).replace("m", "").strip()) / 2.0, 0.5)
            except (ValueError, TypeError):
                continue
    genus = tags.get("genus", tags.get("species", "")).lower().split()[0]
    return CROWN_RADIUS_BY_GENUS.get(genus, DEFAULT_CROWN_RADIUS)


# ── Local coordinate conversion ────────────────────────────────────────────

def to_local_pt(lat: float, lon: float, center_lat: float, center_lon: float):
    k_lon = 111320 * math.cos(math.radians(center_lat))
    k_lat = 111320
    return (lon - center_lon) * k_lon, (lat - center_lat) * k_lat


def to_local(coords: list, center_lat: float, center_lon: float) -> list:
    return [list(to_local_pt(lat, lon, center_lat, center_lon)) for lon, lat in coords]


# ── Crown circle → polygon ─────────────────────────────────────────────────

def crown_polygon(cx: float, cy: float, radius: float, n_pts: int = 16) -> list:
    return [
        [cx + radius * math.cos(2 * math.pi * i / n_pts),
         cy + radius * math.sin(2 * math.pi * i / n_pts)]
        for i in range(n_pts)
    ]


# ── Extrusion (flat disk at canopy height) ─────────────────────────────────

def extrude_crown(local_coords: list, height: float):
    """Extrude crown footprint to a flat top-surface mesh at canopy height."""
    if len(local_coords) < 3:
        return None, None
    try:
        poly = Polygon(local_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        pts = list(poly.exterior.coords[:-1])
        n   = len(pts)
        if n < 3:
            return None, None

        # Top face only (canopy disk)
        verts = []
        for x, y in pts:
            verts.extend([x, y, float(height)])

        idx = []
        for i in range(1, n - 1):
            idx.extend([0, i, i + 1])

        return verts, idx
    except Exception as e:
        print(f"[extrude_crown] {e}")
        return None, None


# ── Coordinate shift ───────────────────────────────────────────────────────

def shift_to_positive(dotbim: dict, shift: float = 256.0) -> dict:
    import numpy as np
    out = {}
    for name, mesh in dotbim.items():
        coords = np.array(mesh["coordinates"]).reshape(-1, 3)
        coords[:, 0] += shift
        coords[:, 1] += shift
        out[name] = {**mesh, "coordinates": coords.flatten().tolist()}
    return out


# ── Public API ─────────────────────────────────────────────────────────────

def get_vegetation(
    lat:     float,
    lon:     float,
    label:   str  = "tile",
    out_dir: str  = OUTPUT_DIR,
) -> str:
    """
    Fetch OSM trees and wooded areas for a 512x512m area centred on (lat, lon).
    Converts each tree to a crown-disk mesh (flat canopy polygon at tree height).
    Saves as dotBIM-style JSON and returns the file path.

    Args:
        lat:     Centre latitude
        lon:     Centre longitude
        label:   Output filename prefix
        out_dir: Directory to write the JSON file

    Returns:
        Absolute path to the saved JSON file.
    """
    os.makedirs(out_dir, exist_ok=True)

    bbox     = get_bbox(lat, lon)
    osm_data = query_overpass(bbox)
    elements = osm_data.get("elements", [])
    print(f"[get_vegetation] {len(elements)} elements returned from OSM")

    dotbim = {}
    n_trees = 0

    for idx, el in enumerate(elements):
        tags   = el.get("tags", {})
        el_type = el.get("type")

        if el_type == "node":
            # Individual tree point → approximate as crown circle
            node_lat = el.get("lat")
            node_lon = el.get("lon")
            if node_lat is None or node_lon is None:
                continue

            cx, cy  = to_local_pt(node_lat, node_lon, lat, lon)
            height  = _parse_height(tags)
            radius  = _parse_crown_radius(tags)
            crown   = crown_polygon(cx, cy, radius)
            verts, inds = extrude_crown(crown, height)

        elif el_type == "way" and "geometry" in el:
            # Tree row or wood polygon — use the raw footprint as crown area
            coords = [(n["lon"], n["lat"]) for n in el["geometry"]]
            if len(coords) < 4:
                continue
            if coords[0] == coords[-1]:
                coords = coords[:-1]

            local  = to_local(coords, lat, lon)
            height = _parse_height(tags)
            verts, inds = extrude_crown(local, height)

        else:
            continue

        if verts and inds:
            osm_id = el.get("id", idx)
            name = "".join(
                c if c.isalnum() or c in "-_" else "-"
                for c in f"tree-{osm_id}-{idx}"
            )
            dotbim[name] = {
                "mesh_id":     len(dotbim),
                "coordinates": verts,
                "indices":     inds,
            }
            n_trees += 1

    print(f"[get_vegetation] {n_trees} vegetation meshes generated")

    dotbim = shift_to_positive(dotbim)

    out_path = os.path.join(out_dir, f"{label}_vegetation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dotbim, f, indent=2, ensure_ascii=False)

    print(f"[get_vegetation] saved → {out_path} ({n_trees} meshes)")
    return out_path


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    LAT   = 55.70318
    LON   = 12.54799
    LABEL = "norrebro"

    path = get_vegetation(lat=LAT, lon=LON, label=LABEL)
    print(f"Done: {path}")
