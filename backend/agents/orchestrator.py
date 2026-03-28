"""
Interpreter Agent  (pipeline position: USER PROMPT → [THIS] → Infrared API → Synthesizer)
────────────────────────────────────────────────────────────────────────────────────────────
Uses interpreter_instructions.yml as the LLM system prompt.

Returns a flat plan dict that run_pipeline() in the app can execute:
  {
    status                 — "ready" | "inferred_defaults" | "needs_clarification"
    user_type              — "citizen" | "stakeholder" | "aec"  (mapped from persona)
    persona                — "citizen" | "planner" | "expert"
    expert_score           — int 0-10
    location               — str (natural-language address, never None)
    season                 — "summer" | "winter" | "full_year"
    synthesis_instruction  — str (directive for the Synthesizer agent)
    simulations            — ["thermal-comfort-index", "solar-radiation"]
    clarifications         — list of {field, question} dicts
    inferred               — list of {field, value, reasoning, confidence} dicts
  }

Falls back to rule-based parsing if Anthropic API is unavailable.
"""

import json
import os
import re
import anthropic

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
INTERP_PATH   = os.path.join(SCRIPT_DIR, "..", "..", "interpreter_instructions.yml")

# Persona → app user_type
_PERSONA_TO_UT = {"citizen": "citizen", "planner": "stakeholder", "expert": "aec"}

_COPENHAGEN_NEIGHBORHOODS = [
    "nørrebro","vesterbro","østerbro","amager","frederiksberg","sydhavn",
    "valby","bispebjerg","brønshøj","vanløse","kongens enghave","indre by",
    "christianshavn","islands brygge","carlsberg","nordhavn",
]


def _season_from_month(sm: int) -> str:
    if sm in (12, 1, 2): return "winter"
    if sm in (6, 7, 8):  return "summer"
    return "full_year"


# ── Synthesis instructions per persona (used by rule-based AND as LLM fallback baseline) ─
_SYNTH_INSTR = {
    "citizen": (
        "You are advising a non-expert user. Use plain, everyday language — no acronyms without "
        "immediate explanation. Frame all findings in terms of lived experience: how hot it feels, "
        "whether children or elderly people can be outside safely, which streets or squares feel "
        "most comfortable. Avoid numbers unless they add immediate meaning. "
        "Give 3 concrete, local recommendations the user can act on today or share with their council."
    ),
    "planner": (
        "You are reporting to a municipal planner or urban designer evaluating environmental "
        "performance at neighbourhood scale. Use metric values (UTCI °C, kWh/m²) and reference "
        "UTCI comfort categories (ISO 15743) and Lawson 1970 pedestrian wind comfort criteria. "
        "Rank findings by spatial priority, identify the priority intervention zone (top-30th-pct "
        "combined UTCI + solar load), and structure the 3 recommendations as policy-level actions "
        "with spatial and temporal specificity."
    ),
    "expert": (
        "You are producing a technical brief for an environmental engineer, climate scientist, or "
        "researcher. Use full scientific precision: UTCI percentile distributions, solar irradiance "
        "statistics (mean, P98), priority zone extent (% of tile), and priority zone identification "
        "methodology. Reference EN 17037, ISO 15743 UTCI comfort categories, and EU Solar Atlas "
        "benchmarks. Structure the 3 recommendations as technical interventions with quantified "
        "improvement targets and confidence ranges."
    ),
}


# ── Rule-based fallback ────────────────────────────────────────────────────────────────────

def _rule_based_plan(prompt: str) -> dict:
    p = prompt.lower()

    # Location
    location = None
    for nb in _COPENHAGEN_NEIGHBORHOODS:
        if nb in p:
            location = f"{nb.title()}, Copenhagen"
            break
    if not location:
        m = re.search(r'\bin ([A-ZÆØÅ][a-zæøå]+(?:\s[A-ZÆØÅ][a-zæøå]+)?)', prompt)
        if m:
            location = m.group(1) + ", Copenhagen"

    # Persona / expert_score
    expert_kw  = {"utci","kwh","epw","svf","sda","infrared","microclimate","morphology",
                  "baseline","simulation","solar radiation","vegetation gap","thermal"}
    planner_kw = {"municipality","municipal","policy","master plan","kpi","assessment",
                  "governance","district","planner","architect","engineer","landscape",
                  "report","stakeholder","performance","environmental impact"}
    exp_score  = min(10, sum(2 for w in expert_kw if w in p))
    plan_score = sum(1 for w in planner_kw if w in p)

    if exp_score >= 8:
        persona, expert_score = "expert", exp_score
    elif plan_score > 0 or exp_score >= 4:
        persona, expert_score = "planner", max(5, exp_score)
    else:
        persona, expert_score = "citizen", max(0, exp_score)

    # Season
    if any(w in p for w in ["winter","cold","snow","january","february","december"]):
        season = "winter"
    elif any(w in p for w in ["year","annual","full","spring","autumn"]):
        season = "full_year"
    else:
        season = "summer"

    user_type   = _PERSONA_TO_UT.get(persona, "citizen")
    synth_instr = _SYNTH_INSTR[persona]

    if location is None:
        return {
            "status":                "needs_clarification",
            "user_type":             user_type,
            "persona":               persona,
            "expert_score":          expert_score,
            "location":              "Copenhagen, Denmark",
            "season":                season,
            "synthesis_instruction": synth_instr,
            "simulations":           ["thermal-comfort-index", "solar-radiation"],
            "clarifications":        [{"field": "location.address",
                                       "question": "Which neighbourhood or address would you like me to analyse?"}],
            "inferred":              [],
        }

    return {
        "status":                "inferred_defaults" if persona == "citizen" else "ready",
        "user_type":             user_type,
        "persona":               persona,
        "expert_score":          expert_score,
        "location":              location,
        "season":                season,
        "synthesis_instruction": synth_instr,
        "simulations":           ["thermal-comfort-index", "solar-radiation"],
        "clarifications":        [],
        "inferred":              [],
    }


# ── Public API ────────────────────────────────────────────────────────────────────────────

def plan(prompt: str) -> dict:
    """
    Interpret a user prompt using the full interpreter_instructions.yml spec.
    Returns a flat plan dict — see module docstring for schema.
    """
    try:
        instructions_raw = open(INTERP_PATH, encoding="utf-8").read()

        system = f"""{instructions_raw}

═══ DEPLOYMENT CONSTRAINTS ═══
The following constraints apply to this specific deployment and override any
conflicting guidance in the instructions above:

1. Only these data scripts are currently wired:
   get_location, get_buildings, get_weather_data

2. Only these Infrared analysis types have working API payloads:
   thermal-comfort-index, solar-radiation

3. Always include BOTH thermal-comfort-index AND solar-radiation in analysis_tasks,
   regardless of the user's specific request.

4. If location.address cannot be determined, set status="needs_clarification"
   and populate clarifications[] with a warm follow-up question.

5. Return ONLY a valid JSON object matching the output_contract. No markdown fence,
   no prose outside the JSON.
"""

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1800,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        # ── Extract flat fields the app needs ───────────────────────────────
        profile     = result.get("user_profile", {})
        persona     = profile.get("persona", "citizen")
        exp_score   = int(profile.get("expert_score", 3))
        user_type   = _PERSONA_TO_UT.get(persona, "citizen")
        location    = (result.get("location") or {}).get("address")
        status      = result.get("status", "ready")
        clarifs     = result.get("clarifications", [])

        # Infer season from data_tasks → get_weather_data → start_month
        season = "summer"
        for task in result.get("data_tasks", []):
            if task.get("script") == "get_weather_data":
                sm = task.get("args", {}).get("start_month", 6)
                season = _season_from_month(int(sm))
                break

        synth_instr = result.get("synthesis", {}).get("synthesis_instruction", "")
        # Fall back to our built-in instruction if LLM returned empty
        if not synth_instr:
            synth_instr = _SYNTH_INSTR.get(persona, _SYNTH_INSTR["citizen"])

        # Always have a usable location
        if not location:
            location = "Copenhagen, Denmark"

        plan_result = {
            "status":                status,
            "user_type":             user_type,
            "persona":               persona,
            "expert_score":          exp_score,
            "location":              location,
            "season":                season,
            "synthesis_instruction": synth_instr,
            "simulations":           ["thermal-comfort-index", "solar-radiation"],
            "clarifications":        clarifs,
            "inferred":              result.get("inferred", []),
        }

        print(f"[interpreter-agent] LLM: persona={persona} expert_score={exp_score} "
              f"user_type={user_type} status={status} location={location} season={season}")
        return plan_result

    except anthropic.APIStatusError as e:
        if e.status_code in (400, 402, 429):
            print(f"[interpreter-agent] API unavailable ({e.status_code}), rule-based fallback")
        else:
            raise
    except Exception as e:
        print(f"[interpreter-agent] LLM error ({e}), rule-based fallback")

    result = _rule_based_plan(prompt)
    print(f"[interpreter-agent] rules: persona={result['persona']} user_type={result['user_type']} "
          f"status={result['status']}")
    return result
