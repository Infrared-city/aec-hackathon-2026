import csv
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EPW = os.path.join(
    SCRIPT_DIR, "..", "data_cache", "weather", "copenhagen.epw"
)


COL = {
    "dry-bulb-temperature":                      6,
    "relative-humidity":                         8,
    "horizontal-infrared-radiation-intensity":   12,
    "global-horizontal-radiation":               13,
    "direct-normal-radiation":                   14,
    "diffuse-horizontal-radiation":              15,
    "wind-speed":                                22,
}


# ── EPW parser ─────────────────────────────────────────────────────────────

def _parse_epw(epw_path: str):
    """
    Parse an EPW file and return:
      - location: {latitude, longitude, timezone, elevation, city}
      - rows: list of 8760 dicts, one per hour
    """
    with open(epw_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # ── Location from row 1 ────────────────────────────────────────────────
    # Format: LOCATION,City,State,Country,Source,WMO,Lat,Lon,TZ,Elev
    loc_parts = lines[0].strip().split(",")
    location  = {
        "city":      loc_parts[1] if len(loc_parts) > 1 else "",
        "country":   loc_parts[3] if len(loc_parts) > 3 else "",
        "latitude":  float(loc_parts[6]) if len(loc_parts) > 6 else 0.0,
        "longitude": float(loc_parts[7]) if len(loc_parts) > 7 else 0.0,
        "timezone":  float(loc_parts[8]) if len(loc_parts) > 8 else 0.0,
        "elevation": float(loc_parts[9]) if len(loc_parts) > 9 else 0.0,
    }

    # ── Hourly data starts at row 9 (index 8) ─────────────────────────────
    rows = []
    reader = csv.reader(lines[8:])
    for row in reader:
        if len(row) < 23:
            continue
        rows.append(row)

    return location, rows


# ── Hour-of-year index ─────────────────────────────────────────────────────

# Days per month (non-leap year)
_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def _hoy(month: int, day: int, hour: int) -> int:
    """Convert month/day/hour (1-based) to 0-based hour-of-year index."""
    doy = sum(_DAYS[:month - 1]) + (day - 1)
    return doy * 24 + (hour - 1)   # EPW hours are 1-24


def _get_hoys(start_month: int, end_month: int,
              start_hour: int, end_hour: int) -> list[int]:
    """
    Return list of 0-based HOY indices for the given period.
    start_hour / end_hour are 0-based (0–23), end_hour inclusive.
    """
    hoys = []
    for m in range(start_month, end_month + 1):
        for d in range(1, _DAYS[m - 1] + 1):
            for h in range(1, 25):              # EPW hours: 1–24
                if start_hour <= (h - 1) <= end_hour:
                    idx = _hoy(m, d, h)
                    if idx < 8760:
                        hoys.append(idx)
    return hoys


# ── Public API ─────────────────────────────────────────────────────────────

def get_weather_data(
    epw_path:    str = DEFAULT_EPW,
    start_month: int = 1,
    end_month:   int = 12,
    start_hour:  int = 0,
    end_hour:    int = 23,
) -> dict:
    """
    Extract weather fields from an EPW file for a given period.

    Args:
        epw_path:    Path to the .epw file
        start_month: Start month (1–12)
        end_month:   End month (1–12)
        start_hour:  Start hour, 0-based (0–23)
        end_hour:    End hour, 0-based (0–23), inclusive

    Returns:
        Dict with weather fields ready to merge into an Infrared payload.
    """
    print(f"[get_weather_data] Parsing {os.path.basename(epw_path)} ...")
    location, rows = _parse_epw(epw_path)
    hoys            = _get_hoys(start_month, end_month, start_hour, end_hour)

    print(f"[get_weather_data] {len(hoys)} hours "
          f"(months {start_month}–{end_month}, hours {start_hour}–{end_hour})")

    data = {
        "latitude":    location["latitude"],
        "longitude":   location["longitude"],
        "city":        location["city"],
        "month-stamp": [start_month, end_month],
        "hour-stamp":  [start_hour, end_hour],
    }

    for field, col_idx in COL.items():
        data[field] = [float(rows[h][col_idx]) for h in hoys if h < len(rows)]

    print(f"[get_weather_data] Done — {len(data['dry-bulb-temperature'])} values per field")
    return data


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = get_weather_data(epw_path=DEFAULT_EPW)

    print(f"\nCity:      {data['city']}")
    print(f"Location:  {data['latitude']}, {data['longitude']}")
    print(f"Hours:     {len(data['global-horizontal-radiation'])}")
    print(f"Temp range: "
          f"{min(data['dry-bulb-temperature']):.1f} – "
          f"{max(data['dry-bulb-temperature']):.1f} °C")
    print(f"Wind range: "
          f"{min(data['wind-speed']):.1f} – "
          f"{max(data['wind-speed']):.1f} m/s")

    # Save to JSON for inspection
    out = os.path.join(SCRIPT_DIR, "..", "data_cache", "weather", "weather_data.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved → {out}")