"""Collection entry point — run the daily Savant pull.

Examples
--------
  # everything, season-to-date, into data/raw/<today>/
  python scripts/pull_savant.py

  # add the last-10-days windowed splits vs LHP and vs RHP
  python scripts/pull_savant.py --windows 10 --hands L R

  # also grab raw pitch-level data for the last 10 days (source for L5/L10)
  python scripts/pull_savant.py --raw-days 10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propline.savant import pull_leaderboards, pull_statcast_search  # noqa: E402


def main() -> int:
    today = date.today()
    ap = argparse.ArgumentParser(description="Pull Baseball Savant data")
    ap.add_argument("--year", type=int, default=today.year)
    ap.add_argument("--out", default=None, help="output dir (default data/raw/<today>)")
    ap.add_argument("--windows", type=int, nargs="*", default=[],
                    help="rolling day-windows for bat-tracking pulls, e.g. 5 10")
    ap.add_argument("--hands", nargs="*", default=[], choices=["L", "R"],
                    help="handedness splits to pull alongside the combined numbers")
    ap.add_argument("--raw-days", type=int, default=0,
                    help="also pull raw pitch-level data for the last N days")
    args = ap.parse_args()

    out = Path(args.out or Path("data/raw") / today.isoformat())

    print(f"\n=== season-to-date leaderboards -> {out} ===")
    manifest = pull_leaderboards(args.year, out)

    for days in args.windows:
        start = (today - timedelta(days=days)).isoformat()
        for hand in (args.hands or [None]):
            label = f"L{days}" + (f" vs {hand}HP" if hand else "")
            print(f"\n=== {label} ===")
            pull_leaderboards(args.year, out, date_start=start,
                              date_end=today.isoformat(), hand=hand)

    if args.raw_days:
        start = (today - timedelta(days=args.raw_days)).isoformat()
        print(f"\n=== raw pitch-level {start}..{today} ===")
        pull_statcast_search(start, today.isoformat(), args.year, out)

    failed = manifest[~manifest.ok]
    print(f"\n{int(manifest.ok.sum())}/{len(manifest)} season pulls succeeded")
    if len(failed):
        for _, row in failed.iterrows():
            print(f"  FAILED {row['pull']}: {row['error']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
