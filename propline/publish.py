"""Shape pipeline output into database rows and push to Supabase.

Kept separate from db.py so that module stays a thin generic client: this file owns
the mapping between our DataFrames and the schema in db/schema.sql.
"""

from __future__ import annotations

import pandas as pd

from .db import upsert

# signals stored alongside each pick so the dashboard can show the "why" without
# recomputing anything
BATTER_DETAIL = ["matchup_est_woba", "matchup_est_slg", "matchup_whiff", "matchup_k_pct",
                 "recent_barrel_pct", "recent_hard_hit", "recent_games", "recent_pa",
                 "primary_pitch", "best_pitch_for_batter", "best_pitch_est_woba",
                 "arsenal_coverage", "batting_order"]
PITCHER_DETAIL = ["recent_k_per_game", "recent_k_pct", "recent_whiff_pct",
                  "opp_lineup_k_pct", "recent_pitches_per_game", "recent_games"]
GAME_DETAIL = ["combined_offense", "combined_starter_weak", "combined_bullpen_tired"]
TEAM_DETAIL = ["lineup_matchup_woba", "opp_starter_weak", "opp_bullpen_tired",
               "recent_team_form"]


# Columns the database types as bigint/int. Pandas widens any column containing a
# missing value to float, so an id arrives as "701542.0" and Postgres rejects it.
# Casting to the nullable Int64 dtype keeps whole numbers whole and NULLs as NULL.
INT_COLS = {
    "game_pk", "team_id", "opponent_id", "opp_team_id", "player_id", "subject_id",
    "home_team_id", "away_team_id", "home_probable_id", "away_probable_id",
    "batting_order", "rank", "appearances", "pitches", "park_runs",
}


def _ints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in INT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def _details(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    return df[present].apply(
        lambda r: {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                   for k, v in r.items()}, axis=1)


def publish_slate(slate_date, schedule, lineups, bullpen,
                  batter_scores, pitcher_scores, team_totals, game_totals,
                  top_n: int = 40) -> dict[str, int]:
    """Write the whole slate. Returns rows written per table."""
    written: dict[str, int] = {}

    if schedule is not None and not schedule.empty:
        cols = ["game_pk", "game_date", "game_time_utc", "status", "venue",
                "home_team", "home_team_id", "away_team", "away_team_id",
                "home_probable_id", "home_probable", "away_probable_id", "away_probable"]
        written["games"] = upsert("games", _ints(schedule[[c for c in cols if c in schedule]]),
                                  on_conflict="game_pk")

    if lineups is not None and not lineups.empty:
        cols = ["game_pk", "game_date", "team_id", "team", "opponent_id", "home_away",
                "batting_order", "player_id", "player_name", "position", "status"]
        written["lineups"] = upsert("lineups", _ints(lineups[[c for c in cols if c in lineups]]),
                                    on_conflict="game_pk,team_id,batting_order")

    if bullpen is not None and not bullpen.empty:
        bp = bullpen.rename(columns={"as_of": "as_of"}).copy()
        bp["as_of"] = str(slate_date)
        cols = ["as_of", "team_id", "team", "player_id", "player_name",
                "last_appearance", "appearances", "pitches", "availability"]
        written["bullpen_status"] = upsert("bullpen_status",
                                           _ints(bp[[c for c in cols if c in bp]]),
                                           on_conflict="as_of,team_id,player_id")

    # --- player props -----------------------------------------------------------
    picks = []
    if batter_scores is not None and not batter_scores.empty:
        for prop, grp in batter_scores.groupby("prop"):
            g = grp.sort_values("score", ascending=False).head(top_n).copy()
            picks.append(pd.DataFrame({
                "slate_date": str(slate_date), "prop": prop,
                "subject_id": g["player_id"], "subject_name": g["player_name"],
                "team": g["team"], "opponent": g.get("opp_starter"),
                "rank": g["rank"], "score": g["score"],
                "lineup_status": g.get("status"),
                "rationale": g.get("rationale"),
                "details": _details(g, BATTER_DETAIL),
            }))

    if pitcher_scores is not None and not pitcher_scores.empty:
        g = pitcher_scores.sort_values("score", ascending=False).head(top_n).copy()
        picks.append(pd.DataFrame({
            "slate_date": str(slate_date), "prop": "strikeouts",
            "subject_id": g["player_id"], "subject_name": g["player_name"],
            "team": g["team"], "opponent": g["opponent"],
            "rank": g["rank"], "score": g["score"],
            "lineup_status": "n/a", "rationale": g.get("rationale"),
            "details": _details(g, PITCHER_DETAIL),
        }))

    if picks:
        allp = pd.concat(picks, ignore_index=True).drop_duplicates(
            subset=["slate_date", "prop", "subject_id"])
        written["prop_picks"] = upsert("prop_picks", _ints(allp),
                                       on_conflict="slate_date,prop,subject_id")

    # --- game / team totals -----------------------------------------------------
    frames = []
    if game_totals is not None and not game_totals.empty:
        g = game_totals.copy()
        frames.append(pd.DataFrame({
            "slate_date": str(slate_date), "game_pk": g["game_pk"],
            "prop": "game_total", "subject": g["teams"], "rank": g["rank"],
            "score": g["score"], "venue": g["venue"], "park_runs": g["park_runs"],
            "park_matched": g.get("park_matched"), "rationale": g.get("rationale"),
            "details": _details(g, GAME_DETAIL),
        }))
    if team_totals is not None and not team_totals.empty:
        g = team_totals.copy()
        frames.append(pd.DataFrame({
            "slate_date": str(slate_date), "game_pk": g["game_pk"],
            "prop": "team_total", "subject": g["team"], "rank": g["rank"],
            "score": g["score"], "venue": g["venue"], "park_runs": g["park_runs"],
            "park_matched": g.get("park_matched"), "rationale": g.get("rationale"),
            "details": _details(g, TEAM_DETAIL),
        }))
    if frames:
        allg = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["slate_date", "prop", "game_pk", "subject"])
        written["game_picks"] = upsert("game_picks", _ints(allg),
                                       on_conflict="slate_date,prop,game_pk,subject")

    return written
