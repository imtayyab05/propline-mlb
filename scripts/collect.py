"""Phase 1 entry point — the full daily collection run.

Replaces the client's manual routine end to end:
  Savant leaderboards + windowed splits + raw pitch data
  + schedule + lineups (projected/confirmed) + bullpen availability
  -> one clean intermediate Excel workbook for the processing stage.

Examples
--------
  python scripts/collect.py                       # today, L5 + L10, vs L and R
  python scripts/collect.py --date 2026-07-31
  python scripts/collect.py --skip-raw            # leaderboards only, much faster
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propline.intermediate import build_intermediate  # noqa: E402
from propline.mlb import (get_bullpen_usage, get_lineups, get_schedule)  # noqa: E402
from propline.rolling import rolling_batter_splits  # noqa: E402
from propline.savant import pull_leaderboards, pull_statcast_search  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PropLine MLB — daily collection")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--windows", type=int, nargs="*", default=[5, 10])
    ap.add_argument("--hands", nargs="*", default=["L", "R"], choices=["L", "R"])
    ap.add_argument("--raw-days", type=int, default=21,
                    help="days of raw pitch data to pull for rolling splits")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    day = args.date
    as_of = datetime.strptime(day, "%Y-%m-%d").date()
    year = as_of.year
    raw_dir = Path("data/raw") / day
    out_xlsx = Path("data/intermediate") / f"collection_{day}.xlsx"

    print(f"\n{'='*66}\nPropLine collection — {day}\n{'='*66}")

    # 1. Savant season leaderboards
    print("\n[1/5] Savant leaderboards (season)")
    manifest = pull_leaderboards(year, raw_dir)

    # 2. Windowed + handedness splits (bat-tracking family only)
    print("\n[2/5] Windowed splits")
    for days in args.windows:
        start = (as_of - timedelta(days=days)).isoformat()
        for hand in args.hands:
            print(f"  -- last {days}d vs {hand}HP")
            pull_leaderboards(year, raw_dir, date_start=start, date_end=day, hand=hand)

    # 3. Raw pitch-level data -> rolling L5/L10
    rolling = pd.DataFrame()
    if not args.skip_raw:
        print(f"\n[3/5] Raw pitch data (last {args.raw_days}d) + rolling splits")
        start = (as_of - timedelta(days=args.raw_days)).isoformat()
        raw = pull_statcast_search(start, day, year, raw_dir)
        rolling = rolling_batter_splits(raw, windows=tuple(args.windows))
        print(f"  ok    rolling splits: {len(rolling)} rows")
    else:
        print("\n[3/5] Raw pitch data — SKIPPED (no L5/L10 this run)")

    # 4. Schedule, lineups, bullpen
    print("\n[4/5] Schedule / lineups / bullpen")
    schedule = get_schedule(day)
    print(f"  ok    schedule: {len(schedule)} games")
    lineups = get_lineups(day, schedule)
    if lineups.empty:
        print("  WARN  no lineups resolved — downstream picks will be empty")
    else:
        counts = lineups.status.value_counts().to_dict()
        print(f"  ok    lineups: {len(lineups)} slots, {lineups.team_id.nunique()} teams {counts}")
    bullpen = get_bullpen_usage(day, days=3)
    print(f"  ok    bullpen: {len(bullpen)} relievers")

    # 5. Intermediate workbook
    print("\n[5/5] Intermediate workbook")
    extra = {"schedule": schedule, "lineups": lineups, "bullpen_status": bullpen}
    if not rolling.empty:
        extra["rolling_splits"] = rolling
    sheets = build_intermediate(raw_dir, out_xlsx, extra=extra)
    print(f"  ok    {out_xlsx}  ({len(sheets)} sheets)")

    failed = manifest[~manifest.ok]
    print(f"\n{'='*66}")
    print(f"season pulls: {int(manifest.ok.sum())}/{len(manifest)} | "
          f"lineups: {len(lineups)} | bullpen: {len(bullpen)} | sheets: {len(sheets)}")
    if len(failed):
        for _, r in failed.iterrows():
            print(f"  FAILED {r['pull']}: {r['error']}")
        return 1
    print("collection OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
