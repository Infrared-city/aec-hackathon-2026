"""
Synthesiser / Interpreter Agent
─────────────────────────────────
Reads simulation results + benchmarks from ontology.yaml.
Returns a score, plain-language summary, and 3 recommendations.
Tailors language for expert vs citizen user.
Falls back to rule-based output if Anthropic API is unavailable.
"""

import json
import os
import yaml
import anthropic

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(SCRIPT_DIR, "..", "..", "ontology.yaml")

def _load_ontology() -> dict:
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Rule-based fallback ────────────────────────────────────────────────────────

def _rule_based_interpret(stats: dict, location: str, user_type: str,
                           n_buildings: int, season: str) -> dict:
    tci = stats.get("thermal-comfort-index", {})
    sr  = stats.get("solar-radiation", {})
    pz  = stats.get("priority_zone_pct", 0)

    avg_tci  = tci.get("mean", 28.0)
    pct_hot  = tci.get("pct_above_32", 0)
    avg_sr   = sr.get("mean", 5.0)

    # Score: lower UTCI and lower pct_hot = better comfort → higher score
    # 32°C = stress threshold. Score 10 if avg<26, 0 if avg>42
    comfort_score = max(0.0, min(10.0, (42 - avg_tci) / 1.6))
    score = round(comfort_score, 1)

    if score >= 8:   verdict = "Excellent thermal comfort"
    elif score >= 6: verdict = "Moderate — some heat stress areas"
    elif score >= 4: verdict = "Poor — significant heat stress"
    else:            verdict = "Critical — urgent greening needed"

    if user_type == "expert":
        summary = (
            f"Analysis of {n_buildings} buildings in {location} ({season}) shows a mean UTCI of "
            f"{avg_tci:.1f}°C, with {pct_hot:.0f}% of the tile exceeding the 32°C heat-stress "
            f"threshold. Mean solar irradiance is {avg_sr:.0f} kWh/m². "
            f"Approximately {pz:.0f}% of the area qualifies as a priority vegetation zone "
            f"(high TCI + high solar potential), indicating significant greening opportunities."
        )
        recs = [
            f"Prioritise street tree canopy in corridors where UTCI exceeds 34°C — "
            f"target the {pz:.0f}% priority zone identified in the solar-TCI overlay.",
            f"Apply cool-roof mandates to buildings with solar radiation above "
            f"{avg_sr*1.3:.0f} kWh/m² to reduce reflected heat load on pedestrians.",
            f"Commission a Lawson pedestrian wind comfort assessment alongside "
            f"vegetation interventions to avoid wind-shadow trade-offs.",
        ]
    else:
        hot_desc = "very hot" if pct_hot > 40 else "somewhat hot" if pct_hot > 20 else "mostly comfortable"
        summary = (
            f"{location} feels {hot_desc} in {season}. About {pct_hot:.0f}% of streets and open "
            f"spaces exceed safe outdoor comfort levels for children and elderly residents. "
            f"Green dots on the map show where new trees would make the biggest difference."
        )
        recs = [
            f"Ask the city to plant trees on the highlighted streets — "
            f"shade reduces pavement temperature by up to 10°C on hot days.",
            f"Avoid prolonged outdoor activity between 11:00 and 15:00 in the marked "
            f"hot zones until more tree cover is in place.",
            f"Contact your local neighbourhood council about installing drinking fountains "
            f"and misting stations at the green-dot locations on the map.",
        ]

    return {"score": score, "verdict": verdict, "summary": summary, "recommendations": recs}


# ── Public API ─────────────────────────────────────────────────────────────────

def interpret(
    stats:       dict,   # {sim_key: {mean, max, min, pct_hot, pct_priority, ...}}
    location:    str,
    user_type:   str,    # "expert" | "citizen"
    prompt:      str,
    n_buildings: int,
    season:      str,
) -> dict:
    """
    Returns:
    {
        "score":           float,   # 0–10
        "verdict":         str,     # short verdict label
        "summary":         str,     # 2–3 sentence plain language summary
        "recommendations": [str, str, str],
    }
    """
    # Try LLM first
    try:
        ontology   = _load_ontology()
        ctx        = ontology["agent"]["synthesiser_context"]
        scale      = ontology["agent"]["score_scale"]
        benchmarks = {k: ontology["simulations"][k]["benchmarks"]
                      for k in stats if k in ontology["simulations"]}

        tone = (
            "Expert user (urban planner / municipality). Use technical language, "
            "reference UTCI thresholds, Lawson categories, kWh/m² values, and policy implications."
            if user_type == "expert"
            else
            "Citizen user (resident, non-expert). Use simple everyday language. "
            "Avoid jargon. Relate findings to lived experience — how hot it feels, "
            "where their children can play safely, where more trees would help."
        )

        stats_yaml = yaml.dump(stats, allow_unicode=True)
        bench_yaml = yaml.dump(benchmarks, allow_unicode=True)
        scale_yaml = yaml.dump(scale, allow_unicode=True)

        system = f"""{ctx}

User type: {tone}

Score scale reference:
{scale_yaml}

Benchmark thresholds:
{bench_yaml}

Simulation statistics for {location} ({season}), {n_buildings} buildings analysed:
{stats_yaml}

Original user question: "{prompt}"

Return ONLY valid JSON with keys:
  score (float 0-10),
  verdict (short label matching the score scale),
  summary (2-3 sentences, adapted to user type),
  recommendations (array of exactly 3 strings, each starting with an action verb,
                   adapted to user type — citizen recs should be concrete and local,
                   expert recs should reference policy levers and spatial priorities).
"""

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content":
                       f"Interpret the simulation results for: {prompt}"}],
        )

        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        result = json.loads(raw.strip())
        print(f"[interpreter] score={result.get('score')} verdict={result.get('verdict')} (LLM)")
        return result

    except anthropic.APIStatusError as e:
        if e.status_code in (400, 402, 429):
            print(f"[interpreter] Anthropic API unavailable ({e.status_code}), using rule-based fallback")
        else:
            raise
    except Exception as e:
        print(f"[interpreter] LLM error, using rule-based fallback: {e}")

    result = _rule_based_interpret(stats, location, user_type, n_buildings, season)
    print(f"[interpreter] score={result.get('score')} verdict={result.get('verdict')} (rules)")
    return result
