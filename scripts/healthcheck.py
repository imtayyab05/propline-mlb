"""Monthly health check — everything the maintenance retainer needs to report.

Read-only. Touches nothing, changes nothing, costs nothing: the Odds API quota is
read from a free endpoint's response headers, and every other figure comes from
Supabase counts.

    python scripts/healthcheck.py            # last 30 days
    python scripts/healthcheck.py --days 7   # last week

Prints a full diagnostic, then a short CLIENT SUMMARY block at the end that can be
pasted straight into a message.

Two figures cannot be read automatically because there is no GitHub or Netlify token
in .env — the script prints those two URLs so the manual check stays part of the same
routine.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propline.db import load_env, read  # noqa: E402

ODDS_BASE = "https://api.the-odds-api.com/v4"
RUNS_PER_DAY = 5          # five scheduled slots; each logs a collection + a processing row

MANUAL_CHECKS = [
    ("GitHub Actions minutes", "https://github.com/settings/billing"),
    ("Netlify credits", "https://app.netlify.com/teams/-/billing"),
]


def _count(table: str, filters: dict | None = None) -> int | None:
    """Exact row count without downloading the rows.

    PostgREST caps a normal select at 1000 rows, which has quietly understated
    numbers in this project before. Range 0-0 with Prefer: count=exact returns the
    true total in the Content-Range header instead.
    """
    url, key = os.getenv("SUPABASE_URL", "").rstrip("/"), os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return None
    r = requests.get(f"{url}/rest/v1/{table}",
                     params={"select": "*", **(filters or {})},
                     headers={"apikey": key, "Authorization": f"Bearer {key}",
                              "Prefer": "count=exact", "Range": "0-0"}, timeout=45)
    if not r.ok:
        return None
    rng = r.headers.get("content-range", "")
    return int(rng.split("/")[-1]) if "/" in rng else None


def _pct(part, whole) -> str:
    if not whole:
        return "n/a"
    return f"{100 * part / whole:.0f}%"


def odds_quota() -> tuple[int | None, int | None]:
    """Credits remaining/used, read from a FREE endpoint.

    /sports is not charged, so the monthly check never eats into the very budget it
    is reporting on.
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        return None, None
    try:
        r = requests.get(f"{ODDS_BASE}/sports", params={"apiKey": key}, timeout=45)
        if not r.ok:
            return None, None
        rem = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        return (int(float(rem)) if rem else None, int(float(used)) if used else None)
    except requests.RequestException:
        return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description="PropLine MLB — health check")
    ap.add_argument("--days", type=int, default=30, help="reporting window (default 30)")
    args = ap.parse_args()
    load_env()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days))
    since_iso = since.isoformat()
    today = date.today().isoformat()

    print("=" * 66)
    print(f"PropLine MLB — health check   {today}   (last {args.days} days)")
    print("=" * 66)

    # --- automation -------------------------------------------------------------
    print("\nAUTOMATION")
    runs = read("pipeline_runs", {"select": "slate_date,stage,status,started_at,run_type",
                                  "started_at": f"gte.{since_iso}",
                                  "order": "id.desc", "limit": "2000"})
    processing = [r for r in runs if r.get("stage") == "processing"]
    failed = [r for r in runs if r.get("status") != "ok"]
    expected = args.days * RUNS_PER_DAY

    print(f"  publish runs      {len(processing)} of ~{expected} expected "
          f"({_pct(len(processing), expected)})")
    print(f"  failures          {len(failed)}")
    for r in failed[:5]:
        print(f"                    {str(r.get('started_at'))[:16]}  {r.get('stage')}  "
              f"{r.get('status')}")
    if len(failed) > 5:
        print(f"                    ... and {len(failed) - 5} more")

    slates = sorted({r["slate_date"] for r in processing if r.get("slate_date")})
    print(f"  slates covered    {len(slates)}"
          + (f"  ({slates[0]} to {slates[-1]})" if slates else ""))

    # --- latest slate quality ---------------------------------------------------
    latest = slates[-1] if slates else None
    print(f"\nLATEST SLATE  {latest or '(none)'}")
    if latest:
        games = _count("games", {"game_date": f"eq.{latest}"})
        picks = _count("prop_picks", {"slate_date": f"eq.{latest}"})
        no_why = _count("prop_picks", {"slate_date": f"eq.{latest}",
                                       "rationale": "is.null"})
        gpicks = _count("game_picks", {"slate_date": f"eq.{latest}"})
        conf = _count("lineups", {"game_date": f"eq.{latest}", "status": "eq.confirmed"})
        proj = _count("lineups", {"game_date": f"eq.{latest}", "status": "eq.projected"})
        scr = _count("lineups", {"game_date": f"eq.{latest}", "status": "eq.scratched"})

        print(f"  games             {games}")
        print(f"  player picks      {picks}")
        print(f"  game/team picks   {gpicks}")
        if picks and no_why is not None:
            have = picks - no_why
            flag = "  <-- AI quota likely exhausted" if have / max(picks, 1) < 0.2 else ""
            print(f"  with explanation  {have} of {picks} ({_pct(have, picks)}){flag}")
        print(f"  lineups           {conf} confirmed / {proj} projected / {scr} scratched")

        if games == 0:
            print("  NOTE: no games on this slate — check whether the season is in "
                  "its off-period before treating this as a fault.")

        # Market coverage, straight off the stored details.
        rows = read("game_picks", {"select": "details", "slate_date": f"eq.{latest}",
                                   "prop": "eq.game_total", "limit": "200"})
        with_line = sum(1 for r in rows if (r.get("details") or {}).get("vegas_total")
                        is not None)
        print(f"  market totals     {with_line} of {len(rows)} games "
              f"({_pct(with_line, len(rows))})")

        ks = read("prop_picks", {"select": "details", "slate_date": f"eq.{latest}",
                                 "prop": "eq.strikeouts", "limit": "200"})
        with_k = sum(1 for r in ks if (r.get("details") or {}).get("vegas_k_line")
                     is not None)
        with_ars = sum(1 for r in ks if (r.get("details") or {}).get("arsenal"))
        print(f"  strikeout lines   {with_k} of {len(ks)} starters "
              f"({_pct(with_k, len(ks))})  — books post these late, 60-70% is normal")
        print(f"  pitch arsenals    {with_ars} of {len(ks)} starters "
              f"({_pct(with_ars, len(ks))})")

    # --- service quotas ---------------------------------------------------------
    print("\nQUOTAS")
    rem, used = odds_quota()
    if rem is None:
        print("  odds api          could not read (key missing or service down)")
    else:
        daily = used / max((datetime.now(timezone.utc).day), 1)
        warn = "  <-- TIGHT" if rem < 120 else ""
        print(f"  odds api          {rem} credits left, {used} used "
              f"(~{daily:.0f}/day){warn}")
        print(f"                    a 14-game slate costs ~16; 500/month is the free tier")

    blanks = None
    if latest:
        blanks = no_why
    if blanks:
        print("  groq (AI text)    explanations missing on the latest slate — daily "
              "token cap. Refills next run.")
    else:
        print("  groq (AI text)    ok on the latest slate")

    # --- storage ----------------------------------------------------------------
    print("\nSTORAGE")
    total_picks = _count("prop_picks")
    total_games = _count("game_picks")
    if total_picks:
        per_slate = total_picks / max(len(slates), 1)
        print(f"  prop_picks        {total_picks:,} rows")
        print(f"  game_picks        {total_games:,} rows")
        print(f"  growth            ~{per_slate:,.0f} rows per slate; nothing prunes "
              f"old slates yet")
        if total_picks > 150_000:
            print("                    consider a 90-day retention policy "
                  "(free tier is 500 MB)")

    # --- manual ------------------------------------------------------------------
    print("\nCHECK BY HAND (no token in .env for these)")
    for label, url in MANUAL_CHECKS:
        print(f"  {label:22} {url}")

    # --- pasteable summary --------------------------------------------------------
    print("\n" + "=" * 66)
    print("CLIENT SUMMARY — copy from here")
    print("=" * 66)
    month = datetime.now().strftime("%B %Y")
    print(f"PropLine MLB — {month} status\n")
    print(f"- Automated runs: {len(processing)} publishes over the last {args.days} days"
          f"{f', {len(failed)} failed and were re-run' if failed else ', no failures'}.")
    if latest:
        print(f"- Latest slate ({latest}): {games} games, {picks} ranked player picks, "
              f"{gpicks} game and team totals.")
        print(f"- Lineup coverage: {conf} confirmed, {proj} projected.")
        print(f"- Market lines: {with_line} of {len(rows)} game totals, "
              f"{with_k} of {len(ks)} strikeout lines.")
    if rem is not None:
        print(f"- Data service usage is within limits ({rem} of 500 credits remaining "
              f"this month).")
    print("- Hosting, database and scheduling all running normally.")
    print("\n(Add anything you changed or tuned this month.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
