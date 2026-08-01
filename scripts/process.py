"""Phase 2 entry point — turn a collection run into ranked prop picks.

Reads the intermediate workbook produced by scripts/collect.py, builds matchups,
scores every prop category, and writes the client-facing Excel file.

  python scripts/process.py --date 2026-07-31
"""

from __future__ import annotations

import argparse
import glob
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propline.db import load_env, log_run  # noqa: E402
from propline.matchup import build_matchups  # noqa: E402
from propline.publish import publish_slate  # noqa: E402
from propline.storage import upload_workbook  # noqa: E402
from propline.rationale import add_rationales, label_internal_indexes  # noqa: E402
from propline.mlb import get_player_names  # noqa: E402
from propline.output import build_picks_workbook  # noqa: E402
from propline.rolling import rolling_pitcher_splits  # noqa: E402
from propline.scoring import (score_batter_props, score_game_totals,
                              score_pitcher_strikeouts)  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PropLine MLB — processing")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--window", default="L10", choices=["L5", "L10"])
    ap.add_argument("--top", type=int, default=40,
                    help="rows per tab in the Excel workbook, where a long tab is just "
                         "harder to read")
    ap.add_argument("--publish-top", type=int, default=0,
                    help="rows per category written to the database; 0 means all. The "
                         "dashboard filters what it is given, so publishing only the top "
                         "40 made 'Confirmed only' near-useless mid-afternoon — with 3 "
                         "teams confirmed, just 3 of those 27 hitters ranked inside the "
                         "top 40. Searching a player outside the top 40 came back empty "
                         "for the same reason.")
    ap.add_argument("--no-rationale", action="store_true", help="skip the Groq step")
    ap.add_argument("--explain-top", type=int, default=25,
                    help="how many picks per category get a written reason. The "
                         "dashboard lists 40, so anything lower leaves visible blanks "
                         "in the Why column.")
    ap.add_argument("--publish", action="store_true", help="write results to Supabase")
    ap.add_argument("--run-kind", default="manual",
                    help="scheduled_morning | scheduled_afternoon | manual — shown on the dashboard")
    args = ap.parse_args()
    load_env()

    started = datetime.now()
    day = args.date
    raw_dir = Path("data/raw") / day
    inter = Path("data/intermediate") / f"collection_{day}.xlsx"
    if not inter.exists():
        print(f"missing {inter} — run scripts/collect.py --date {day} first")
        return 1

    print(f"\n{'='*66}\nPropLine processing — {day}\n{'='*66}")

    x = pd.ExcelFile(inter)
    lineups = pd.read_excel(x, "lineups")
    schedule = pd.read_excel(x, "schedule")
    rolling = pd.read_excel(x, "rolling_splits")

    ba = pd.read_csv(raw_dir / "batter_pitch_arsenal_stats.csv")
    pa = pd.read_csv(raw_dir / "pitcher_pitch_arsenal_stats.csv")
    season = pd.read_csv(raw_dir / "batter_expected_stats.csv")

    # --- matchups ---------------------------------------------------------------
    print("\n[1/7] Matchup engine")
    matchups = build_matchups(lineups, schedule, ba, pa)
    print(f"  ok    {len(matchups)} hitter-vs-starter matchups "
          f"({int(matchups.reliable.sum())} with reliable arsenal coverage)")

    # --- pitcher rolling (needed for K props) ------------------------------------
    print("\n[2/7] Pitcher rolling form")
    raw_files = glob.glob(str(raw_dir / "statcast_raw_*.csv"))
    if raw_files:
        raw = pd.read_csv(raw_files[0], low_memory=False)
        names = get_player_names(raw.pitcher.dropna().unique())
        pitcher_rolling = rolling_pitcher_splits(raw, windows=(5, 10), name_map=names)
        print(f"  ok    {pitcher_rolling.player_id.nunique()} pitchers")
    else:
        pitcher_rolling = pd.DataFrame()
        print("  WARN  no raw pitch data — strikeout props will be skipped")

    # --- scoring ----------------------------------------------------------------
    print("\n[3/7] Scoring")
    batter_scores = score_batter_props(matchups, rolling, season, window=args.window)
    print(f"  ok    batters: {len(batter_scores)} rows across "
          f"{batter_scores.prop.nunique()} categories")

    if not pitcher_rolling.empty:
        pitcher_scores = score_pitcher_strikeouts(schedule, pitcher_rolling, lineups,
                                                  rolling, window=args.window)
        print(f"  ok    strikeouts: {len(pitcher_scores)} starters")
    else:
        pitcher_scores = pd.DataFrame()

    # --- game + team totals -----------------------------------------------------
    print("\n[4/7] Game and team totals")
    bullpen = pd.read_excel(x, "bullpen_status")
    pf_path = raw_dir / "park_factors.csv"
    park = pd.read_csv(pf_path) if pf_path.exists() else pd.DataFrame(columns=["venue_name"])
    if not pf_path.exists():
        print("  WARN  no park_factors.csv — all parks treated as neutral")
    team_totals, game_totals = score_game_totals(schedule, matchups, bullpen, park,
                                                 rolling, window=args.window)
    print(f"  ok    totals: {len(game_totals)} games, {len(team_totals)} team totals")

    # --- rationale --------------------------------------------------------------
    if not args.no_rationale:
        print("\n[5/7] Written reasons (Groq)")
        bfields = ["player_name", "team", "opp_starter", "matchup_est_woba",
                   "matchup_est_slg", "recent_barrel_pct", "recent_hard_hit",
                   "best_pitch_for_batter", "primary_pitch", "recent_games"]
        parts = []
        for prop, grp in batter_scores.groupby("prop"):
            parts.append(add_rationales(grp, bfields, prop, top_n=args.explain_top))
        batter_scores = pd.concat(parts, ignore_index=True)
        if not pitcher_scores.empty:
            pitcher_scores = add_rationales(
                pitcher_scores, ["player_name", "team", "opponent", "recent_k_per_game",
                                 "recent_k_pct", "recent_whiff_pct", "opp_lineup_k_pct",
                                 "recent_games"], "strikeouts", top_n=args.explain_top)
        # Team/game indexes are banded into words BEFORE the model sees them. Simply
        # instructing it not to quote the raw values did not work — it wrote
        # "combined offense of 0.674", which means nothing to a reader.
        game_totals = label_internal_indexes(game_totals, {
            "combined_offense": ("quiet", "average", "strong"),
            "combined_bullpen_tired": ("rested", "moderately worked", "short-handed"),
        })
        game_totals = add_rationales(
            game_totals, ["teams", "venue", "park_runs", "combined_offense_desc",
                          "combined_bullpen_tired_desc"], "game_total", top_n=args.explain_top)

        team_totals = label_internal_indexes(team_totals, {
            "lineup_matchup_woba": ("unfavourable", "even", "favourable"),
            "opp_bullpen_tired": ("rested", "moderately worked", "short-handed"),
            "opp_starter_weak": ("tough", "average", "hittable"),
        })
        team_totals = add_rationales(
            team_totals, ["team", "opponent", "opp_starter", "park_runs",
                          "lineup_matchup_woba_desc", "opp_bullpen_tired_desc",
                          "opp_starter_weak_desc"], "team_total", top_n=args.explain_top)
        done = int(batter_scores.rationale.notna().sum())
        print(f"  ok    {done} batter picks explained")
    else:
        print("\n[5/7] Written reasons — SKIPPED")

    # --- output -----------------------------------------------------------------
    print("\n[6/7] Picks workbook")
    statuses = lineups.status.value_counts().to_dict() if not lineups.empty else {}
    meta = {
        "Slate date": day,
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Games": len(schedule),
        "Lineup status": ", ".join(f"{k}: {v}" for k, v in statuses.items()),
        "Form window": args.window,
        "Picks per category": args.top,
        "Note": "Projected lineups are an early read; confirmed lineups post 2-4h before first pitch.",
    }
    out = build_picks_workbook(batter_scores, pitcher_scores,
                               Path("data/picks") / f"props_{day}.xlsx",
                               meta, top_n=args.top,
                               team_totals=team_totals, game_totals=game_totals)
    print(f"  ok    {out}")

    # --- publish ----------------------------------------------------------------
    if args.publish:
        print("\n[7/7] Publishing to Supabase")
        try:
            written = publish_slate(day, schedule, lineups, bullpen, batter_scores,
                                    pitcher_scores, team_totals, game_totals,
                                    top_n=args.publish_top or None)
            for table, n in written.items():
                print(f"  ok    {table:16} {n} rows")

            # The workbook is generated on a runner that disappears when the job ends,
            # so it goes to storage — that is what the dashboard's Download Excel
            # button points at. A failed upload must not fail an otherwise good run.
            try:
                link = upload_workbook(out, day)
                print(f"  ok    workbook uploaded")
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN  workbook upload failed: {exc}")

            log_run(day, args.run_kind, "processing", "ok",
                    detail=", ".join(f"{k}={v}" for k, v in written.items()),
                    started_at=started)
        except Exception as exc:
            log_run(day, args.run_kind, "processing", "failed", detail=str(exc)[:400],
                    started_at=started)
            raise
    else:
        print("\n[7/7] Publishing — SKIPPED (pass --publish)")

    print(f"\n{'='*66}\nprocessing OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
