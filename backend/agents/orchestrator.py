"""
Planner / Orchestrator Agent
─────────────────────────────
Reads a plain-language prompt and ontology.yaml.
Returns: location string, detected simulations, user_type, and a plan summary.
Falls back to rule-based parsing if Anthropic API is unavailable.
"""

import json
import os
import re
import yaml
import anthropic

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(SCRIPT_DIR, "..", "..", "ontology.yaml")

def _load_ontology() -> dict:
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── Rule-based fallback ────────────────────────────────────────────────────────

_COPENHAGEN_NEIGHBORHOODS = [
    "nørrebro","vesterbro","østerbro","amager","frederiksberg","sydhavn",
    "valby","bispebjerg","brønshøj","vanløse","kongens enghave","indre by",
    "christianshavn","islands brygge","carlsberg","nordhavn",
]

def _rule_based_plan(prompt: str) -> dict:
    p = prompt.lower()

    # Location extraction
    location = "Copenhagen, Denmark"
    for nb in _COPENHAGEN_NEIGHBORHOODS:
        if nb in p:
            location = f"{nb.title()}, Copenhagen"
            break
    # Also try to catch quoted or capitalised places
    m = re.search(r'\bin ([A-ZÆØÅ][a-zæøå]+(?:\s[A-ZÆØÅ][a-zæøå]+)?)', prompt)
    if m:
        location = m.group(1) + ", Copenhagen"

    # User type
    expert_words = {"planner","architect","engineer","municipality","utci","kWh","policy",
                    "thermal","solar potential","vegetation gap","infrared","strategy"}
    user_type = "expert" if any(w in p for w in expert_words) else "citizen"

    # Season
    if any(w in p for w in ["winter","cold","snow","january","february","december"]):
        season = "winter"
    elif any(w in p for w in ["year","annual","spring","autumn","full"]):
        season = "full_year"
    else:
        season = "summer"

    # Simulations
    sims = ["thermal-comfort-index", "solar-radiation"]

    # Plan text
    plan_text = (
        f"Running thermal comfort and solar radiation analysis for {location} "
        f"({season}) to identify vegetation priority zones."
    )

    return {
        "location":    location,
        "simulations": sims,
        "workflow":    None,
        "user_type":   user_type,
        "season":      season,
        "plan_text":   plan_text,
    }


def plan(prompt: str) -> dict:
    """
    Given a user prompt, return:
    {
        "location":    str,          # extracted location string
        "simulations": [str, ...],   # list of infrared_key values to run
        "workflow":    str | null,   # matched workflow id if any
        "user_type":   "expert" | "citizen",
        "season":      "summer" | "winter" | "full_year",
        "plan_text":   str,          # one-sentence plan for display
    }
    """
    # Try LLM first
    try:
        ontology = _load_ontology()
        sims_yaml  = yaml.dump({"simulations": ontology["simulations"]}, allow_unicode=True)
        wf_yaml    = yaml.dump({"workflows":   ontology["workflows"]},   allow_unicode=True)
        ctx        = ontology["agent"]["planner_context"]

        system = f"""{ctx}

Below is the simulation ontology you must reason over.

{sims_yaml}

{wf_yaml}

IMPORTANT RULES:
- Extract the most specific location you can from the prompt (neighbourhood, address, or city).
  If no location is given, default to "Copenhagen, Denmark".
- Detect whether the user sounds like an expert (planner, architect, engineer, municipality)
  or a citizen (resident, worried about heat, asking about their street, child, elderly).
- Select the minimum set of simulations that answer the question.
  Always include thermal-comfort-index if the question is about heat, comfort, or vegetation.
  Always include solar-radiation if the question involves greening, trees, or solar potential.
- Detect the season from words like "summer", "winter", "spring", "autumn", "this year".
  Default to "summer" if unclear and the topic is heat/comfort.
- Return ONLY valid JSON with keys:
  location, simulations (array of infrared_key strings), workflow (string or null),
  user_type ("expert" or "citizen"), season ("summer" | "winter" | "full_year"), plan_text (string).
"""

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        # Keep only simulations we have payloads for
        _SUPPORTED = {"solar-radiation", "thermal-comfort-index"}
        sims = [s for s in result.get("simulations", []) if s in _SUPPORTED]
        if "thermal-comfort-index" not in sims:
            sims.append("thermal-comfort-index")
        if "solar-radiation" not in sims:
            sims.append("solar-radiation")
        result["simulations"] = sims

        print(f"[orchestrator] plan (LLM): {result}")
        return result

    except anthropic.APIStatusError as e:
        if e.status_code in (400, 402, 429):
            print(f"[orchestrator] Anthropic API unavailable ({e.status_code}), using rule-based fallback")
        else:
            raise
    except Exception as e:
        print(f"[orchestrator] LLM error, using rule-based fallback: {e}")

    result = _rule_based_plan(prompt)
    print(f"[orchestrator] plan (rules): {result}")
    return result
