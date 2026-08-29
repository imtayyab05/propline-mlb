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
# A small model on purpose: this job is "restate two supplied numbers in a sentence",
# not reasoning, and the larger models burn the free daily token quota within a single
# day of scheduled runs.
#
# Model history, because this WILL happen again — Groq retires models with no notice:
#   llama-3.3-70b-versatile  -> exhausted the daily quota, then retired
#   llama-3.1-8b-instant     -> retired 2026-08; every call 404'd mid-slate
#   openai/gpt-oss-20b       -> current
#
# If this 404s, list the models the key can actually see and pick the smallest
# instruction-following one:
#   GET https://api.groq.com/openai/v1/models
#
# Rejected alternatives when choosing this one: qwen3.6-27b could not hold to the JSON
# schema (400s), and groq/compound-mini was accurate but cost ~2x the tokens for the
# same two sentences.
MODEL = "openai/gpt-oss-20b"
TIMEOUT = 120
# Twelve, established by testing against a real slate rather than guessed.
#
# The cap is the model's OUTPUT length, not the input. A batch of 25 real picks is
# ~8,000 characters in and needs 25 sentences back; gpt-oss-20b silently returns an
# empty completion, which Groq reports as `json_validate_failed` with an empty
# `failed_generation` — a confusing error that reads like a prompt problem and is not
# one. Synthetic test rows are far shorter than real ones, so 25 passes in a toy test
# and fails in production. Re-measure with real data if the model ever changes.
MAX_PER_CALL = 12

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
- recent_games: how many games the recent numbers cover (fewer = less reliable)

Team and game total fields are INTERNAL INDEXES on arbitrary scales. Never quote their
raw values — they mean nothing to a reader. Describe what they imply instead:
- combined_offense / lineup_matchup_woba: how well the lineup(s) project against the
  starting pitching they face. Say "both lineups project well against tonight's
  starters", not "combined offense of 0.674".
- combined_bullpen_tired / opp_bullpen_tired: 0 = fully rested pen, 1 = most arms
  worked recently. Say "the pen is short-handed" or "the pen is rested".
- opp_starter_weak: how much the opposing starter gives up. Say "a starter who has
  been hittable", not the number.
park_runs and named pitchers ARE real and may be quoted directly."""

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
    "Return strict JSON: {\"rationales\": [{\"id\": <id>, \"text\": \"...\"}]}\n\n"
    # Smaller models paraphrase these rates into the wrong denominator, which turns a
    # correct number into a false statement. Naming the exact mistakes fixes it; the
    # general glossary above on its own did not.
    "STRICT WORDING — these are the mistakes models actually make here:\n"
    "- barrel and hard-hit rates are a share of BATTED BALLS. Never write 'of his "
    "swings', 'of his hits', 'of his at-bats' or 'of his plate appearances'.\n"
    "- xwOBA is a rate, not a count. Never write 'hits .400 xwOBA'; write 'projects at "
    "a .400 xwOBA'.\n"
    "- If recent_games is below 7, you MUST say the sample is small."
)


# Game and team totals have no player, no opposing starter and no rate stats — but the
# player prompt above demands all three. Asked to explain a game total with it, the
# model replied "No player data available for this pick.", then began returning empty
# completions, which Groq surfaces as `json_validate_failed`. That reads like a JSON
# bug and is really a prompt that does not match the rows being sent.
TOTALS_SYSTEM = (
    "You write one-sentence explanations for baseball GAME TOTAL and TEAM TOTAL picks.\n"
    "These are about run scoring across a whole game, not about an individual player. "
    "There is deliberately no player in the input - never ask for one, and never say "
    "data is missing.\n\n"
    "Field meanings:\n"
    "- teams / team: the matchup, or the team the pick is on\n"
    "- venue: the ballpark\n"
    "- park_runs: park run factor, 100 = neutral, higher favours scoring\n"
    "- combined_offense_desc / lineup_matchup_woba_desc: how the lineup(s) project "
    "against the starting pitching they face (quiet / average / strong)\n"
    "- combined_bullpen_tired_desc / opp_bullpen_tired_desc: bullpen condition "
    "(rested / moderately worked / short-handed)\n"
    "- opp_starter_weak_desc: how hittable the opposing starter has been "
    "(tough / average / hittable)\n\n"
    "Rules:\n"
    "- ONE complete sentence per pick, 12-22 words.\n"
    "- Name the teams or the team, then give two reasons from the fields provided.\n"
    "- The _desc fields are already plain English. Use those words; never invent "
    "numbers for them. park_runs is a real number and may be quoted.\n"
    "- Plain language. No hype, no betting advice, no guarantees.\n\n"
    "Example output: {\"id\": 0, \"text\": \"Both lineups project strongly at Truist "
    "Park and the Braves bullpen is short-handed after heavy use.\"}\n\n"
    "Return strict JSON: {\"rationales\": [{\"id\": <id>, \"text\": \"...\"}]}"
)


def _payload(rows: list[dict]) -> str:
    return json.dumps({"picks": rows}, default=str)


def label_internal_indexes(df: pd.DataFrame, mapping: dict[str, tuple[str, str, str]]
                          ) -> pd.DataFrame:
    """Replace arbitrary internal indexes with plain-English bands.

    Telling the model "don't quote this number" does not reliably work — it quoted
    "0.674 combined offense" anyway. So the number never gets sent: each index is
    converted to a word based on where it sits across today's slate, and only the word
    is passed on. Deterministic, and impossible for the model to misread.

    mapping: {column: (low_label, mid_label, high_label)}
    """
    out = df.copy()
    for col, (lo, mid, hi) in mapping.items():
        if col not in out.columns:
            continue
        pct = out[col].rank(pct=True)
        out[col + "_desc"] = pct.map(
            lambda p: lo if pd.isna(p) or p < 0.34 else (mid if p < 0.67 else hi))
    return out


RATE_LIMIT_RETRIES = 6

# Groq's free tier caps TOKENS PER MINUTE, not per request. Eight categories fired two
# seconds apart all land inside the same minute and blow the ceiling together — which
# is how Total Bases ended up with no rationale at all while its neighbours had 25.
# Spreading the calls across the minute keeps each one under the limit, at the cost of
# about a minute on a run that already takes several.
PAUSE_BETWEEN_CALLS = 7.0

# Groq reports two different 429s with the same status code: a per-MINUTE limit that
# clears in seconds, and a per-DAY quota that clears in tens of minutes. Waiting out
# the daily one would stall a scheduled run past its 45-minute timeout, five times a
# day — so anything longer than this is treated as "no rationale today" instead.
# Picks are still fully ranked and usable; only the written sentence is missing.
MAX_WAIT_SECONDS = 90


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


def _call(rows: list[dict], api_key: str, system: str = None) -> dict[int, str]:
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system or SYSTEM},
            {"role": "user", "content": _payload(rows)},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {api_key}"},
                          json=body, timeout=TIMEOUT)
        if r.status_code == 429:
            wait = _retry_after(r) + 0.5
            if wait > MAX_WAIT_SECONDS:
                raise RuntimeError(
                    f"daily quota reached (Groq asked for {wait / 60:.0f} min) — "
                    f"skipping rationale text for this run"
                )
            if attempt < RATE_LIMIT_RETRIES:
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
                   top_n: int = 15, api_key: str | None = None,
                   system: str | None = None) -> pd.DataFrame:
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
                if isinstance(v, (int, float)):
                    # keep whole numbers whole, or the model writes "a 104.0 park"
                    item[f] = int(v) if float(v).is_integer() else round(float(v), 3)
                else:
                    item[f] = str(v)
        rows.append(item)

    # Degrade gracefully. Groq is the least reliable dependency in the pipeline — models
    # get retired, quotas bite, and this one intermittently returns an empty completion —
    # so a failure must cost only the sentences it actually lost. Previously one bad
    # chunk returned early and discarded every rationale already collected for that
    # category, turning a partial failure into a total one.
    texts: dict[int, str] = {}
    split = 0
    for i in range(0, len(rows), MAX_PER_CALL):
        chunk = rows[i:i + MAX_PER_CALL]
        try:
            got = _call(chunk, api_key, system)
            texts.update({k + i: v for k, v in got.items()})
        except Exception as exc:  # noqa: BLE001 — never let this break the pipeline
            # Retry at half size: an empty completion is usually the model running out
            # of output room, which a smaller batch fixes.
            split += 1
            half = max(1, len(chunk) // 2)
            for j in range(0, len(chunk), half):
                sub = chunk[j:j + half]
                try:
                    got = _call(sub, api_key, system)
                    texts.update({k + i + j: v for k, v in got.items()})
                except Exception:
                    print(f"  WARN  {label}: {len(sub)} picks unexplained ({exc})")
                time.sleep(PAUSE_BETWEEN_CALLS)
        time.sleep(PAUSE_BETWEEN_CALLS)

    if texts:
        note = f", {split} chunk(s) split" if split else ""
        print(f"  ok    {label}: {len(texts)}/{len(rows)} explained{note}")

    for pos, idx in enumerate(top.index):
        if pos in texts:
            df.at[idx, "rationale"] = texts[pos]
    return df
