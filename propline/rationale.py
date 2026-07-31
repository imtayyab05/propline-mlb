"""Groq — plain-English reasons for the top picks.

The LLM does NOT rank anything. Scores come from propline/scoring.py; Groq only turns
an already-decided shortlist into readable sentences. That keeps the betting logic
inspectable and means a bad model day can never reorder the board.

Cost control: one batched request per run covering every pick, rather than one call
per player. Comfortably inside Groq's free tier.
"""

from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
TIMEOUT = 120
MAX_PER_CALL = 25          # keep each prompt small enough to stay reliable

FIELD_GLOSSARY = """Field meanings:
- matchup_est_woba / matchup_est_slg: the hitter's expected production against THIS
  starter's specific pitch mix (xwOBA ~.320 is average, .400+ is excellent)
- recent_barrel_pct: share of batted balls hit at ideal speed+angle over the last 10
  games (league average ~8%)
- recent_hard_hit: share of batted balls at 95+ mph
- best_pitch / best_pitch_for_batter: the pitch in this starter's arsenal the hitter
  handles best
- primary_pitch: what the starter throws most often
- recent_k_per_game / recent_k_pct / recent_whiff_pct: the pitcher's recent strikeout form
- opp_lineup_k_pct: how often the opposing lineup strikes out
- park_runs: park run factor, 100 = neutral, higher favours hitters
- recent_games: how many games the recent numbers cover (fewer = less reliable)"""

SYSTEM = (
    "You write one-sentence explanations for baseball prop shortlists.\n"
    "The picks were ALREADY ranked by a statistical model. Never re-rank them, never "
    "contradict the numbers, never invent a statistic that is not in the input.\n\n"
    + FIELD_GLOSSARY +
    "\n\nRules:\n"
    "- ONE complete sentence per pick, 12-22 words.\n"
    "- Name the player, then cite two concrete numbers as evidence.\n"
    "- Never output a bare field name or a raw key:value pair.\n"
    "- Plain language. No hype, no betting advice, no guarantees, no 'lock' or 'smash'.\n"
    "- If a pick has few recent games, say the sample is small.\n\n"
    # NB: deliberately a made-up player. Using a real one risks the model echoing the
    # example verbatim when a genuine pick happens to match it, which would hide a
    # failure to actually read the data.
    "Example input:  {\"id\": 0, \"player_name\": \"Sample Hitter\", \"opp_starter\": "
    "\"Sample Pitcher\", \"matchup_est_woba\": 0.377, \"recent_barrel_pct\": 9.4, "
    "\"best_pitch\": \"Slider\"}\n"
    "Example output: {\"id\": 0, \"text\": \"Sample Hitter projects at a .377 xwOBA "
    "versus Sample Pitcher's mix and is barrelling 9.4% of batted balls.\"}\n\n"
    "Return strict JSON: {\"rationales\": [{\"id\": <id>, \"text\": \"...\"}]}"
)


def _payload(rows: list[dict]) -> str:
    return json.dumps({"picks": rows}, default=str)


RATE_LIMIT_RETRIES = 4
PAUSE_BETWEEN_CALLS = 2.0   # free tier is 12k tokens/min; spacing avoids most 429s


def _retry_after(resp) -> float:
    """How long Groq wants us to wait. It says so in the header and the message."""
    hdr = resp.headers.get("retry-after")
    if hdr:
        try:
            return float(hdr)
        except ValueError:
            pass
    m = re.search(r"try again in ([\d.]+)s", resp.text)
    return float(m.group(1)) if m else 5.0


def _call(rows: list[dict], api_key: str) -> dict[int, str]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _payload(rows)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {api_key}"},
                          json=body, timeout=TIMEOUT)
        if r.status_code == 429 and attempt < RATE_LIMIT_RETRIES:
            wait = _retry_after(r) + 0.5
            print(f"    rate limited, waiting {wait:.1f}s")
            time.sleep(wait)
            continue
        if not r.ok:
            raise RuntimeError(f"groq {r.status_code}: {r.text[:300]}")
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        return {int(item["id"]): item.get("text", "") for item in data.get("rationales", [])}
    raise RuntimeError("groq: rate limited after retries")


def add_rationales(df: pd.DataFrame, fields: list[str], label: str,
                   top_n: int = 15, api_key: str | None = None) -> pd.DataFrame:
    """Attach a `rationale` column to the top N rows of a scored frame.

    Only the shortlist gets sent — there is no value in explaining pick #180, and it
    keeps the request small. Everything below top_n simply has no rationale.
    """
    df = df.copy()
    if "rationale" not in df.columns:
        df["rationale"] = None
    if df.empty:
        return df

    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  WARN  GROQ_API_KEY missing — picks will have no written reasons")
        return df

    top = df.sort_values("score", ascending=False).head(top_n)
    rows = []
    for i, (_, r) in enumerate(top.iterrows()):
        item = {"id": i, "prop": label}
        for f in fields:
            if f in r.index and pd.notna(r[f]):
                v = r[f]
                item[f] = round(float(v), 3) if isinstance(v, (int, float)) else str(v)
        rows.append(item)

    texts: dict[int, str] = {}
    for i in range(0, len(rows), MAX_PER_CALL):
        chunk = rows[i:i + MAX_PER_CALL]
        try:
            got = _call(chunk, api_key)
        except Exception as exc:  # noqa: BLE001 — never let this break the pipeline
            print(f"  WARN  rationale generation failed for {label}: {exc}")
            return df
        texts.update({k + i: v for k, v in got.items()})
        time.sleep(PAUSE_BETWEEN_CALLS)

    for pos, idx in enumerate(top.index):
        if pos in texts:
            df.at[idx, "rationale"] = texts[pos]
    return df
