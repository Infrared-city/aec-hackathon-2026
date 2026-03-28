"""
Synthesizer Agent  (pipeline position: Infrared results → [THIS] → Dashboard)
───────────────────────────────────────────────────────────────────────────────
Receives simulation statistics + the synthesis_instruction written by the
Interpreter agent (orchestrator.py) for this specific user and prompt.

Returns:
  {
    "score":           float   0–10
    "verdict":         str     short label
    "summary":         str     2–3 sentence plain-language summary
    "recommendations": [str, str, str]
  }

Tailors vocabulary and depth to the persona supplied by the Interpreter.
Falls back to rule-based output if the Anthropic API is unavailable.
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

def _rule_based_interpret(stats: dict, location: str, persona: str,
                           n_buildings: int, season: str) -> dict:
    tci = stats.get("thermal-comfort-index", {})
    sr  = stats.get("solar-radiation", {})
    pz  = stats.get("priority_zone_pct", 0)

    avg_tci  = tci.get("mean", 28.0)
    pct_hot  = tci.get("pct_above_32", 0)
    avg_sr   = sr.get("mean", 5.0)

    # Score: lower UTCI → better comfort → higher score
    comfort_score = max(0.0, min(10.0, (42 - avg_tci) / 1.6))
    score = round(comfort_score, 1)

    if score >= 8:   verdict = "Excellent thermal comfort"
    elif score >= 6: verdict = "Moderate — some heat stress areas"
    elif score >= 4: verdict = "Poor — significant heat stress"
    else:            verdict = "Critical — urgent greening needed"

    if persona in ("planner", "expert"):
        summary = (
            f"Analysis of {n_buildings} buildings in {location} ({season}) shows a mean UTCI of "
            f"{avg_tci:.1f}°C, with {pct_hot:.0f}% of the tile exceeding the 32°C heat-stress "
            f"threshold. Mean solar irradiance is {avg_sr:.0f} kWh/m². "
            f"Approximately {pz:.0f}% of the area qualifies as a priority vegetation zone "
            f"(high TCI + high solar potential), indicating significant greening opportunities."
        )
        if persona == "expert":
            recs = [
                f"Prioritise street-tree canopy in corridors where UTCI exceeds 34°C — "
                f"target the {pz:.0f}% priority zone identified in the solar-TCI overlay. "
                f"Expected UTCI reduction: −5 to −7°C with deciduous canopy at 8 m spacing.",
                f"Apply cool-roof mandates (albedo ≥ 0.40) to buildings with solar radiation above "
                f"{avg_sr*1.3:.0f} kWh/m² to reduce reflected heat load on pedestrians. "
                f"Reference EN 17037 and EU Solar Atlas regional benchmarks.",
                f"Commission a Lawson 1970 pedestrian wind comfort assessment alongside "
                f"vegetation interventions to avoid wind-shadow trade-offs in the priority zone.",
            ]
        else:  # planner
            recs = [
                f"Prioritise tree planting in the {pz:.0f}% priority zone (high UTCI + high solar). "
                f"A continuous deciduous buffer on the main pedestrian axis reduces solar exposure "
                f"by 55–70% while preserving winter solar access.",
                f"Apply cool-roof policy for buildings in the highest solar-load zones "
                f"(above {avg_sr*1.3:.0f} kWh/m²) — reduces stored-heat re-radiation by ~22%.",
                f"Link the greening strategy to the district green-corridor network to amplify "
                f"UHI mitigation beyond the 512 × 512 m analysis tile.",
            ]
    else:  # citizen
        hot_desc = "very hot" if pct_hot > 40 else "quite warm" if pct_hot > 20 else "mostly comfortable"
        summary = (
            f"{location} feels {hot_desc} in {season}. About {pct_hot:.0f}% of streets and open "
            f"spaces exceed safe outdoor comfort levels for children and elderly residents. "
            f"Green dots on the map show where new trees would make the biggest difference."
        )
        recs = [
            f"Ask the city to plant trees on the highlighted streets — "
            f"shade can reduce pavement temperature by up to 10°C on hot days.",
            f"Avoid prolonged outdoor activity between 11:00 and 15:00 in the marked "
            f"hot zones until more tree cover is in place.",
            f"Contact your local neighbourhood council about installing drinking fountains "
            f"and misting stations at the green-dot locations on the map.",
        ]

    return {"score": score, "verdict": verdict, "summary": summary, "recommendations": recs}


# ── Public API ─────────────────────────────────────────────────────────────────

def interpret(
    stats:                dict,
    location:             str,
    user_type:            str,
    prompt:               str,
    n_buildings:          int,
    season:               str,
    synthesis_instruction: str = "",
    persona:              str  = "",
) -> dict:
    """
    Synthesize simulation results into a human-readable analysis.

    Parameters
    ----------
    stats                 : {sim_key: {mean, max, pct_above_32, ...}}
    location              : human-readable location name
    user_type             : "citizen" | "stakeholder" | "aec"  (app routing)
    prompt                : original user prompt
    n_buildings           : number of buildings analysed
    season                : "summer" | "winter" | "full_year"
    synthesis_instruction : directive written by the Interpreter agent (may be empty)
    persona               : "citizen" | "planner" | "expert"  (from Interpreter)
    """
    # Resolve persona — prefer Interpreter's persona, fall back from user_type
    _UT_TO_PERSONA = {"citizen": "citizen", "stakeholder": "planner", "aec": "expert"}
    if not persona:
        persona = _UT_TO_PERSONA.get(user_type, "citizen")

    try:
        ontology   = _load_ontology()
        ctx        = ontology["agent"]["synthesiser_context"]
        scale      = ontology["agent"]["score_scale"]
        benchmarks = {k: ontology["simulations"][k]["benchmarks"]
                      for k in stats if k in ontology.get("simulations", {})}

        # Build persona-aware tone block
        _TONE = {
            "citizen": (
                "Citizen user (resident, non-expert). Use plain everyday language. "
                "No acronyms without immediate explanation. Frame everything in terms of lived "
                "experience — how hot it feels, where children can play safely, which streets "
                "are most comfortable. Recommendations must be concrete and local."
            ),
            "planner": (
                "Municipal planner / urban designer. Use metric values (UTCI °C, kWh/m²). "
                "Reference UTCI comfort categories (ISO 15743) and Lawson 1970 criteria. "
                "Rank findings spatially and frame recommendations as policy-level actions "
                "with spatial and temporal specificity."
            ),
            "expert": (
                "Environmental engineer / researcher. Full scientific precision: UTCI percentile "
                "distributions, solar irradiance statistics, priority zone methodology. "
                "Reference EN 17037, ISO 15743, EU Solar Atlas. Quantify improvement targets "
                "and note confidence ranges where relevant."
            ),
        }
        tone = _TONE.get(persona, _TONE["citizen"])

        stats_yaml = yaml.dump(stats,      allow_unicode=True)
        bench_yaml = yaml.dump(benchmarks, allow_unicode=True)
        scale_yaml = yaml.dump(scale,      allow_unicode=True)

        # synthesis_instruction from Interpreter takes priority as the directing voice
        directive_block = (
            f"\nSPECIFIC DIRECTIVE FROM THE INTERPRETER AGENT FOR THIS ANALYSIS:\n{synthesis_instruction}\n"
            if synthesis_instruction else ""
        )

        system = f"""{ctx}
{directive_block}
Persona tone: {tone}

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
  summary (2-3 sentences, adapted to persona and directive above),
  recommendations (array of exactly 3 strings, each starting with an action verb;
                   adapted to persona — citizen recs must be concrete and local,
                   planner/expert recs must reference policy levers and spatial priorities).
"""

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            system=system,
            messages=[{"role": "user",
                       "content": f"Synthesize the simulation results for: {prompt}"}],
        )

        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        print(f"[synthesizer] score={result.get('score')} verdict={result.get('verdict')} "
              f"persona={persona} (LLM)")
        return result

    except anthropic.APIStatusError as e:
        if e.status_code in (400, 402, 429):
            print(f"[synthesizer] API unavailable ({e.status_code}), rule-based fallback")
        else:
            raise
    except Exception as e:
        print(f"[synthesizer] LLM error ({e}), rule-based fallback")

    result = _rule_based_interpret(stats, location, persona, n_buildings, season)
    print(f"[synthesizer] score={result.get('score')} verdict={result.get('verdict')} "
          f"persona={persona} (rules)")
    return result
