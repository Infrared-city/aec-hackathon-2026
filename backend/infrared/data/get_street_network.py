import json
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import osmnx as ox

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data_cache", "streets")

# ── Config ─────────────────────────────────────────────────────────────────

TILE_SIZE = 512     # metres — matches Infrared tile size

# ── Highway type → mode classification ────────────────────────────────────
# An edge can belong to multiple modes (e.g. a residential road is both
# drive and cycle). Priority for colouring: drive > cycle > walk.

DRIVE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
    "living_street", "service",
}

CYCLE_HIGHWAYS = {
    "cycleway", "residential", "living_street", "unclassified",
    "primary", "secondary", "tertiary", "path", "track",
}

WALK_HIGHWAYS = {
    "footway", "pedestrian", "path", "steps", "track",
    "living_street", "residential",
}

# Colours per mode
COLORS = {
    "drive": "#E05C2A",   # coral / orange
    "cycle": "#2E86AB",   # blue
    "walk":  "#4CAF50",   # green
    "other": "#AAAAAA",   # grey fallback
}


def classify(highway) -> str:
    """Return the display mode for a highway tag value."""
    # highway can be a string or a list (osmnx sometimes returns lists)
    if isinstance(highway, list):
        highway = highway[0]
    highway = str(highway)

    if highway in DRIVE_HIGHWAYS:
        return "drive"
    if highway in CYCLE_HIGHWAYS:
        return "cycle"
    if highway in WALK_HIGHWAYS:
        return "walk"
    return "other"


# ── Public API ─────────────────────────────────────────────────────────────

def get_street_network(
    lat:     float,
    lon:     float,
    label:   str  = "tile",
    out_dir: str  = OUTPUT_DIR,
    plot:    bool = False,
) -> str:
   
    os.makedirs(out_dir, exist_ok=True)

    # ── Cache check ────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, f"{label}.geojson")
    if os.path.exists(out_path):
        print(f"[get_street_network] loaded from cache → {out_path}")
        return out_path

    # ── 1. Fetch full graph ────────────────────────────────────────────────
    print(f"[get_street_network] Fetching all streets within "
          f"{TILE_SIZE // 2}m of ({lat:.5f}, {lon:.5f}) ...")

    G = ox.graph_from_point(
        (lat, lon),
        dist=TILE_SIZE // 2,
        network_type="all",
        retain_all=True,
    )

    nodes, edges = ox.graph_to_gdfs(G)
    print(f"[get_street_network] {len(nodes)} nodes, {len(edges)} edges")

    # ── 2. Local coordinate conversion ────────────────────────────────────
    k_lon = 111320 * math.cos(math.radians(lat))
    k_lat = 111320

    def to_local(lon_, lat_):
        return (lon_ - lon) * k_lon, (lat_ - lat) * k_lat

    # ── 3. Build GeoJSON with mode classification ──────────────────────────
    features = []
    for _, row in edges.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        local_coords = [to_local(x, y) for x, y in geom.coords]
        highway      = row.get("highway", "other")
        mode         = classify(highway)

        feature = {
            "type": "Feature",
            "geometry": {
                "type":        "LineString",
                "coordinates": [[x, y] for x, y in local_coords],
            },
            "properties": {
                "name":     row.get("name", ""),
                "highway":  highway if isinstance(highway, str) else highway[0],
                "mode":     mode,
                "oneway":   row.get("oneway", False),
                "length_m": round(row.get("length", 0.0), 2),
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "center_lat":  lat,
            "center_lon":  lon,
            "tile_size_m": TILE_SIZE,
            "node_count":  len(nodes),
            "edge_count":  len(edges),
        },
    }

    # ── 4. Optional colour-coded plot ──────────────────────────────────────
    if plot:
        half = TILE_SIZE / 2
        fig, ax = plt.subplots(figsize=(8, 8))

        for feat in features:
            coords = feat["geometry"]["coordinates"]
            mode   = feat["properties"]["mode"]
            xs     = [c[0] for c in coords]
            ys     = [c[1] for c in coords]
            ax.plot(xs, ys, color=COLORS[mode], linewidth=1.2, alpha=0.85)

        ax.plot(0, 0, "ro", markersize=8, zorder=5)

        # Legend
        legend_handles = [
            mpatches.Patch(color=COLORS["drive"], label="Drive"),
            mpatches.Patch(color=COLORS["cycle"], label="Cycle"),
            mpatches.Patch(color=COLORS["walk"],  label="Walk"),
            mpatches.Patch(color=COLORS["other"], label="Other"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

        # Mode counts for title
        counts = {}
        for feat in features:
            m = feat["properties"]["mode"]
            counts[m] = counts.get(m, 0) + 1

        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"{label} — {len(features)} segments\n"
            + "  ".join(f"{m}: {n}" for m, n in sorted(counts.items()))
        )
        plt.tight_layout()
        plt.show()

    # ── 5. Save ────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    print(f"[get_street_network] saved → {out_path}")
    return out_path


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    LAT   = 55.6729
    LON   = 12.5784
    LABEL = "bloxhub"

    path = get_street_network(lat=LAT, lon=LON, label=LABEL, plot=True)
    print(f"Done: {path}")