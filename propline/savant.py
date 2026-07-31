"""Baseball Savant downloader.

Replaces the client's manual ~20 minute routine of clicking Download CSV across
15-20 leaderboard pages. Writes one CSV per pull into data/raw/<date>/.
"""

from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .config import (BASE, HAND_KEY, PULLS, STATCAST_SEARCH, statcast_search_params)

USER_AGENT = "PropLine-MLB/0.1 (data collection for private analytics)"
TIMEOUT = 120
RETRIES = 3
PAUSE = 1.0  # polite gap between requests


class SavantError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_csv(session, path, params, allow_empty=False) -> pd.DataFrame:
    """GET a Savant CSV and parse it. Follows redirects (batted-ball returns 301).

    An empty result is normally a bug worth retrying, but for date-chunked pulls it is
    legitimate — the All-Star break and off-days genuinely have no regular-season
    games — so callers can opt out of that check.
    """
    url = f"{BASE}{path}"
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT, allow_redirects=True)
            r.raise_for_status()
            text = r.content.decode("utf-8-sig")
            if text.lstrip().startswith("<"):
                raise SavantError(f"HTML returned instead of CSV for {path} — check params")
            try:
                df = pd.read_csv(io.StringIO(text))
            except pd.errors.EmptyDataError:
                if allow_empty:
                    return pd.DataFrame()
                raise
            if df.empty and not allow_empty:
                raise SavantError(f"empty CSV for {path}")
            return df
        except Exception as exc:  # noqa: BLE001 - retry any transport/parse failure
            last = exc
            if attempt < RETRIES:
                time.sleep(2 * attempt)
    raise SavantError(f"failed after {RETRIES} attempts: {path} -> {last}")


def pull_leaderboards(year, out_dir, date_start=None, date_end=None, hand=None,
                      only=None, session=None):
    """Download every configured leaderboard into out_dir.

    date_start/date_end and hand are applied ONLY to endpoints that genuinely support
    them (the bat-tracking family). Everything else is season-to-date by definition —
    see docs/savant_endpoints.md.
    """
    session = session or _session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for pull in PULLS:
        if only and pull.name not in only:
            continue

        player_type = "pitcher" if pull.name.startswith("pitcher") else "batter"
        params = pull.url_params(
            year=year,
            date_start=date_start,
            date_end=date_end,
            hand=hand,
            hand_key=HAND_KEY[player_type],
        )

        suffix = ""
        if pull.windowed and date_start and date_end:
            suffix += f"_{date_start}_to_{date_end}"
        if pull.handed and hand:
            suffix += f"_vs{hand}"

        name = f"{pull.name}{suffix}"
        try:
            df = fetch_csv(session, pull.path, params)
        except SavantError as exc:
            print(f"  FAIL  {name}: {exc}")
            manifest.append({"pull": name, "rows": 0, "ok": False, "error": str(exc)})
            continue

        dest = out_dir / f"{name}.csv"
        df.to_csv(dest, index=False)
        windowed = "windowed" if (pull.windowed and date_start) else "season"
        print(f"  ok    {name:52} {len(df):>5} rows  ({windowed})")
        manifest.append({"pull": name, "rows": len(df), "ok": True, "file": str(dest)})
        time.sleep(PAUSE)

    return pd.DataFrame(manifest)


PARK_FACTORS = "/leaderboard/statcast-park-factors"


def pull_park_factors(year, out_dir, years_rolling=3, session=None) -> pd.DataFrame:
    """Park factors, indexed to 100 (above = hitter friendly).

    This page has no CSV export — the table is rendered client-side from a JSON blob
    embedded in the HTML, so we read that directly rather than scraping the table.
    Fragile by nature: if Savant changes the page, this is the first thing to break.
    """
    session = session or _session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r = session.get(f"{BASE}{PARK_FACTORS}", params={
        "type": "year", "year": year, "batSide": "", "stat": "index_wOBA",
        "condition": "All", "rolling": "", "parks": "mlb",
    }, timeout=TIMEOUT)
    r.raise_for_status()

    match = re.search(r"var data\s*=\s*(\[.*?\]);", r.text, re.S)
    if not match:
        raise SavantError("park factors: embedded `var data` block not found — "
                          "Savant likely changed the page layout")

    df = pd.DataFrame(json.loads(match.group(1)))
    keep = ["venue_id", "venue_name", "main_team_id", "name_display_club", "year_range",
            "index_runs", "index_hr", "index_hits", "index_so", "index_woba", "index_obp"]
    df = df[[c for c in keep if c in df.columns]]

    # the embedded JSON ships every index as a string ("104"), which silently breaks
    # any downstream comparison or sort
    for c in df.columns:
        if c.startswith("index_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("venue_id", "main_team_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    dest = out_dir / "park_factors.csv"
    df.to_csv(dest, index=False)
    print(f"  ok    park_factors: {len(df)} venues")
    return df


ROW_CAP = 25_000   # Savant hard-truncates statcast_search at this many rows
CHUNK_DAYS = 3     # ~2.5-4.5k rows per game day, so 3 days stays well clear of the cap


def pull_statcast_search(date_start, date_end, season, out_dir, player_type="batter",
                         session=None, chunk_days=CHUNK_DAYS):
    """Raw pitch-level data for a date range — the source for L5/L10 and any split.

    Savant silently truncates this endpoint at 25,000 rows: ask for 21 days and it
    returns the most recent ~7 with no error and a partial day at the boundary. So we
    request it in small date chunks and stitch the results together, asserting that no
    individual chunk came back at the cap.
    """
    session = session or _session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(date_start, "%Y-%m-%d").date()
    end = datetime.strptime(date_end, "%Y-%m-%d").date()

    frames, cursor = [], start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        params = statcast_search_params(cursor.isoformat(), chunk_end.isoformat(),
                                        season, player_type)
        df = fetch_csv(session, STATCAST_SEARCH, params, allow_empty=True)

        if len(df) >= ROW_CAP:
            raise SavantError(
                f"chunk {cursor}..{chunk_end} hit the {ROW_CAP} row cap — data would be "
                f"silently truncated. Lower chunk_days and re-run."
            )

        if not df.empty:
            frames.append(df)
        print(f"    chunk {cursor}..{chunk_end}: {len(df)} rows")
        cursor = chunk_end + timedelta(days=1)
        time.sleep(PAUSE)

    if not frames:
        raise SavantError(f"no rows at all for {date_start}..{date_end}")
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    dest = out_dir / f"statcast_raw_{player_type}_{date_start}_to_{date_end}.csv"
    out.to_csv(dest, index=False)
    days = out["game_date"].nunique() if "game_date" in out.columns else 0
    print(f"  ok    statcast_search {player_type} {date_start}..{date_end}: "
          f"{len(out)} rows across {days} game days")
    return out
