"""Sportsbook lines — game totals, team totals, and pitcher strikeout props.

The client wants the model's grade shown next to the market's number, so he can see
where they disagree rather than reading a score in isolation.

CREDIT BUDGET — read before changing anything here
--------------------------------------------------
The Odds API's free tier is 500 credits a month, and the two market families cost
wildly different amounts:

  totals / team totals   one call covers the whole slate      ~1-2 credits
  pitcher strikeouts     a per-EVENT endpoint, so 15 games    ~15 credits

At one pull a day that is roughly 480 credits a month — inside the free tier, but
with almost no headroom. At three pulls a day it is ~1,350 and the key dies in the
second week, silently, mid-slate.

So odds are fetched ONCE per slate and cached to disk. Later runs of the same day
reuse the file. If the cache is present this module makes no requests at all, which
is what keeps five scheduled runs a day affordable.
"""

from __future__ import annotations

import json
import os
import unicodedata
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
REGION = "us"
TIMEOUT = 45

# Bookmaker-agnostic: the median across books is steadier than any single one, which
# can be stale or shaded. Devin compares the model against "the market", not against
# one book's number.
# Slate-wide markets: one call returns every game, so this is almost free.
#
# ONLY "totals" works here. The betting-markets documentation lists team_totals
# alongside it as a featured market, but the slate endpoint rejects it with
# INVALID_MARKET — team totals are per-event, and therefore cost per game like a
# player prop. Verified against the live API, which returns 422 at no credit cost.
SLATE_MARKETS = "totals"

# Player props live on a per-EVENT endpoint and are charged per market per event.
# Every category the client scores has a market key, so the full set is listed here —
# but pulling all six across a 15-game slate is ~90 credits a day, or 2,700 a month
# against a 500-credit free tier. PROP_MARKETS is therefore what we ACTUALLY request,
# and it is deliberately just the strikeout line: it is the category he already rates
# highest, and it is the one his spec asked for by name.
AVAILABLE_PROP_MARKETS = {
    "strikeouts": "pitcher_strikeouts",
    "hits": "batter_hits",
    "total_bases": "batter_total_bases",
    "home_runs": "batter_home_runs",
    "rbis": "batter_rbis",
    "runs": "batter_runs_scored",
}
# team_totals sits here rather than in SLATE_MARKETS for the reason above. Adding it
# doubles the per-event cost, so it is off by default; the client gets game totals
# free and can turn team totals on if he decides they are worth the credits.
PROP_MARKETS = ["pitcher_strikeouts"]

# Rough credit cost, so a change here is a deliberate decision rather than a surprise
# invoice: one slate call, plus one per game per prop market.
def estimate_credits(games: int, markets: list[str] | None = None) -> int:
    return 2 + games * len(markets if markets is not None else PROP_MARKETS)


class OddsError(RuntimeError):
    pass


def _key() -> str | None:
    return os.getenv("ODDS_API_KEY")


def _get(path: str, **params):
    key = _key()
    if not key:
        raise OddsError("ODDS_API_KEY is not set")
    params["apiKey"] = key
    r = requests.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
    if r.status_code == 401:
        raise OddsError("odds API rejected the key")
    if r.status_code == 429:
        raise OddsError("odds API quota exhausted for this key")
    r.raise_for_status()
    # every response carries the running budget; worth surfacing rather than guessing
    left = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    return r.json(), (left, used)


def fetch_slate_odds(slate_date, cache_dir, force: bool = False) -> dict:
    """Everything we need for one slate, fetched once and cached.

    Returns {"totals": [...], "strikeouts": [...], "fetched_at": ..., "credits": ...}.
    """
    cache = Path(cache_dir) / "odds.json"
    if cache.exists() and not force:
        with open(cache, encoding="utf-8") as fh:
            data = json.load(fh)
        print(f"  ok    odds: reusing today's cache ({len(data.get('totals', []))} games)")
        return data

    out = {"totals": [], "strikeouts": [], "fetched_at":
           datetime.now(timezone.utc).isoformat(), "credits": {}}

    # --- game and team totals: one call for the whole slate ----------------------
    events, budget = _get(f"/sports/{SPORT}/odds", regions=REGION,
                          markets=SLATE_MARKETS, oddsFormat="american")
    out["totals"] = events
    out["credits"]["after_totals"] = budget[0]

    # --- strikeout props: one call PER EVENT, the expensive half -----------------
    # Only for games that have actually started listing props; a request for an event
    # with no prop market still costs a credit, so failures are counted and reported
    # rather than retried.
    # Player props only exist for games that have not started. Requesting them for a
    # game already under way returns nothing AND still costs a credit, so the filter
    # here is a saving, not just tidiness: on a slate that is half played it halves
    # the expensive half of the bill.
    now = datetime.now(timezone.utc)
    upcoming = []
    for ev in events:
        try:
            start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            upcoming.append(ev)
            continue
        if start > now:
            upcoming.append(ev)
    skipped = len(events) - len(upcoming)

    misses = 0
    for ev in upcoming:
        try:
            detail, budget = _get(f"/sports/{SPORT}/events/{ev['id']}/odds",
                                  regions=REGION, markets=",".join(PROP_MARKETS),
                                  oddsFormat="american")
            out["strikeouts"].append(detail)
        except Exception:  # noqa: BLE001 — one missing market must not lose the rest
            misses += 1
        time.sleep(0.3)
    out["credits"]["remaining"] = budget[0]

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(out, fh)

    print(f"  ok    odds: {len(out['totals'])} games, "
          f"{len(out['strikeouts'])} with strikeout props"
          + (f", {misses} without" if misses else "")
          + (f", {skipped} already started" if skipped else "")
          + f" | credits left: {out['credits'].get('remaining')}")
    return out


# --- shaping -------------------------------------------------------------------

def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2, 2)


def game_totals(raw: dict) -> pd.DataFrame:
    """Median over/under line per game, with the two team names to join on."""
    rows = []
    for ev in raw.get("totals", []):
        points = []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "totals":
                    continue
                for oc in mk.get("outcomes", []):
                    if oc.get("point") is not None:
                        points.append(float(oc["point"]))
        if not points:
            continue
        rows.append({
            "odds_event_id": ev.get("id"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "commence_time": ev.get("commence_time"),
            "vegas_total": _median(points),
            "books": len(ev.get("bookmakers", [])),
        })
    return pd.DataFrame(rows)


def strikeout_lines(raw: dict) -> pd.DataFrame:
    """Median strikeout line per pitcher."""
    rows = []
    for ev in raw.get("strikeouts", []):
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "pitcher_strikeouts":
                    continue
                for oc in mk.get("outcomes", []):
                    if oc.get("point") is None:
                        continue
                    rows.append({
                        "pitcher_name": oc.get("description") or oc.get("name"),
                        "line": float(oc["point"]),
                        "odds_event_id": ev.get("id"),
                    })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return (df.groupby(["pitcher_name", "odds_event_id"], as_index=False)
              .agg(vegas_k_line=("line", lambda s: _median(list(s))),
                   books=("line", "size")))


def _norm_name(n) -> str:
    """Fold a name for matching: strip accents, punctuation and case.

    The odds feed writes "Cristopher Sanchez" where the MLB API writes
    "Cristopher Sánchez". Exact matching drops those silently, which would look like
    a missing line rather than a spelling difference.
    """
    if not isinstance(n, str):
        return ""
    stripped = "".join(c for c in unicodedata.normalize("NFKD", n)
                       if not unicodedata.combining(c))
    return stripped.lower().replace(".", "").replace("'", "").replace("-", " ").strip()


def attach_game_totals(schedule: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    """Map each odds event to a game_pk on this slate.

    Joined on the two team names, which match the MLB API exactly — no fuzzy logic
    needed. The odds feed covers a wider date window than one slate, so anything that
    does not match a scheduled game today is simply a different day's fixture and is
    dropped rather than force-fitted.
    """
    if schedule.empty or totals.empty:
        return pd.DataFrame()

    sched = schedule[["game_pk", "home_team", "away_team"]].copy()
    sched["_h"] = sched["home_team"].map(_norm_name)
    sched["_a"] = sched["away_team"].map(_norm_name)

    t = totals.copy()
    t["_h"] = t["home_team"].map(_norm_name)
    t["_a"] = t["away_team"].map(_norm_name)

    out = sched.merge(t.drop(columns=["home_team", "away_team"]),
                      on=["_h", "_a"], how="inner")

    # Doubleheaders break a team-name join: two games between the same two teams on
    # the same day produce a cartesian match, and the board silently gains duplicate
    # rows with different totals. Disambiguate on first pitch — keep, for each
    # scheduled game, the odds event that starts closest to it.
    if "game_time_utc" in schedule.columns and "commence_time" in out.columns:
        times = schedule[["game_pk", "game_time_utc"]]
        out = out.merge(times, on="game_pk", how="left")
        start = pd.to_datetime(out["game_time_utc"], utc=True, errors="coerce")
        book = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
        out["_gap"] = (start - book).abs()
        out = (out.sort_values("_gap")
                  .drop_duplicates("game_pk", keep="first")
                  .drop_duplicates("odds_event_id", keep="first")
                  .drop(columns=["_gap", "game_time_utc"]))

    return out.drop(columns=["_h", "_a"])


def attach_strikeout_lines(schedule: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Map each pitcher's market line to the probable starter he is."""
    if schedule.empty or lines.empty:
        return pd.DataFrame()

    probables = []
    for _, g in schedule.iterrows():
        for side in ("home", "away"):
            pid, name = g.get(f"{side}_probable_id"), g.get(f"{side}_probable")
            if pd.notna(pid) and isinstance(name, str):
                probables.append({"player_id": int(pid), "_n": _norm_name(name)})
    if not probables:
        return pd.DataFrame()

    p = pd.DataFrame(probables).drop_duplicates("player_id")
    l = lines.copy()
    l["_n"] = l["pitcher_name"].map(_norm_name)
    out = p.merge(l, on="_n", how="inner")
    return out.drop(columns=["_n"])
