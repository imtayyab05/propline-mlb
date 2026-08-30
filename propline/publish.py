"""Shape pipeline output into database rows and push to Supabase.

Kept separate from db.py so that module stays a thin generic client: this file owns
the mapping between our DataFrames and the schema in db/schema.sql.
"""

from __future__ import annotations

import pandas as pd

from .db import check_json, delete_where, read, upsert

# signals stored alongside each pick so the dashboard can show the "why" without
# recomputing anything
# Stored alongside each pick so the dashboard can show the "why" without recomputing.
# Per prop, because in v2 the props are scored on genuinely different inputs and a
# shared list would either omit what matters or carry dead weight on every row.
PROP_DETAIL = {
    "hits": ["recent_xwoba", "contact_rate", "contact_rate_recent", "line_drive_rate",
             "sweet_spot_pct", "ground_ball_rate", "gb_penalty", "starter_whip",
             "starter_whiff_matchup", "lineup_position"],
    "total_bases": ["tb_path", "tb_power_score", "tb_volume_score", "tb_strict",
                    "iso_recent_14day", "iso_season", "recent_barrel_pct",
                    "starter_slg_allowed_vs_hand", "pitch_matchup_grade"],
    "home_runs": ["hr_matchup_rv", "recent_barrel_pct", "fly_ball_rate",
                  "ground_ball_rate", "iso_recent_14day", "recent_hard_hit",
                  "pitcher_barrel_suppression_flag", "hr_suppression_capped",
                  "pitch_matchup_grade"],
    "rbis": ["table_setter_obp", "context_mult", "matchup_est_slg", "recent_tb_rate",
             "pitch_matchup_grade", "lineup_position"],
    "runs": ["top_order_obp", "table_setter_obp", "context_mult", "matchup_est_woba",
             "pitch_matchup_grade", "lineup_position"],
}

# Always useful, whatever the prop.
COMMON_DETAIL = ["recent_games", "recent_pa", "primary_pitch", "best_pitch_for_batter",
                 "arsenal_coverage"]

PITCHER_DETAIL = ["split_k_matchup", "split_k_rate_matchup", "whiff_14day",
                  "vegas_k_line",
                  "opp_lineup_k_pct", "pitcher_whip", "whip_efficiency",
                  "expected_pitch_limit", "leash_penalty", "recent_games",
                  "recent_k_per_game", "recent_k_pct",
                  # Nested list: the starter's top 3 pitches to each side. Rides here
                  # rather than in a table of its own — see propline/arsenal.py.
                  "arsenal"]

GAME_DETAIL = ["combined_offense", "combined_starter_weak",
               "combined_pen_workload", "pen_status_home", "pen_status_away",
               "starter_whip_k9", "combined_starter_k9", "starters_resolved",
               "temp_f", "wind", "precip_pct", "weather_mult", "roof_type",
               "vegas_total", "vs_vegas", "market_edge"]
TEAM_DETAIL = ["lineup_matchup_woba", "opp_starter_weak", "opp_pen_status",
               "opp_pen_workload", "opp_pen_pitches_3d", "opp_pen_innings_3d",
               "starter_whip_k9", "opp_starter_whip", "opp_starter_k9",
               "temp_f", "wind", "weather_mult", "recent_team_form"]


# Columns the database types as bigint/int. Pandas widens any column containing a
# missing value to float, so an id arrives as "701542.0" and Postgres rejects it.
# Casting to the nullable Int64 dtype keeps whole numbers whole and NULLs as NULL.
INT_COLS = {
    "game_pk", "team_id", "opponent_id", "opp_team_id", "player_id", "subject_id",
    "home_team_id", "away_team_id", "home_probable_id", "away_probable_id",
    "batting_order", "rank", "appearances", "pitches", "park_runs",
    "pen_pitches_3d", "pen_unavailable", "pen_relievers",
}


def _ints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in INT_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


def _clean(v):
    """One detail value, made JSON-safe.

    Lists and dicts are passed through untouched and MUST be tested first: pd.isna on a
    multi-element list returns an array, and `None if <array>` raises "truth value of an
    array with more than one element is ambiguous". The nested pitch arsenal is six
    entries long, so it hit that on every publish.
    """
    if isinstance(v, (list, dict)):
        return v
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def _details(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    return df[present].apply(
        lambda r: {k: _clean(v) for k, v in r.items()}, axis=1)


def publish_slate(slate_date, schedule, lineups, bullpen,
                  batter_scores, pitcher_scores, team_totals, game_totals,
                  top_n: int | None = None) -> dict[str, int]:
    """Write the whole slate. Returns rows written per table.

    top_n=None publishes every scored row. The dashboard can only filter what it has
    been given, so truncating here silently breaks its filters: with three teams
    confirmed, only three of those 27 hitters fell inside a top-40 cut, making
    "Confirmed only" look empty rather than selective. The Excel workbook is trimmed
    separately, where a shorter tab genuinely does read better.
    """
    written: dict[str, int] = {}
    # a plain slice is clearer than threading None through every .head() call
    cut = (lambda df: df) if top_n is None else (lambda df: df.head(top_n))

    if schedule is not None and not schedule.empty:
        cols = ["game_pk", "game_date", "game_time_utc", "status", "venue",
                "home_team", "home_team_id", "away_team", "away_team_id",
                "home_probable_id", "home_probable", "away_probable_id", "away_probable"]
        written["games"] = upsert("games", _ints(schedule[[c for c in cols if c in schedule]]),
                                  on_conflict="game_pk")

    if lineups is not None and not lineups.empty:
        cols = ["game_pk", "game_date", "team_id", "team", "opponent_id", "home_away",
                "batting_order", "player_id", "player_name", "position", "status"]
        incoming = _ints(lineups[[c for c in cols if c in lineups]])

        # A confirmed lineup must never be replaced by a projection. The collection
        # step already respects that, but the guarantee has to hold at this layer too:
        # re-running processing against an earlier intermediate workbook would
        # otherwise quietly downgrade teams that have since been confirmed, putting
        # "projected" badges on picks that are actually settled — the single most
        # misleading thing this tool could show.
        existing = read("lineups", {"select": "team_id,status",
                                    "game_date": f"eq.{slate_date}", "limit": "2000"})
        locked = {r["team_id"] for r in existing if r["status"] == "confirmed"}
        if locked:
            downgrade = incoming["team_id"].isin(locked) & (incoming["status"] != "confirmed")
            if downgrade.any():
                teams = incoming.loc[downgrade, "team"].nunique()
                print(f"  keep  {teams} confirmed lineup(s) — refusing to overwrite "
                      f"with projections")
                incoming = incoming[~downgrade]

        if not incoming.empty:
            # Clear only the teams being rewritten, so the protected ones survive.
            for tid in incoming["team_id"].dropna().unique():
                delete_where("lineups", {"game_date": f"eq.{slate_date}",
                                         "team_id": f"eq.{int(tid)}"})
            written["lineups"] = upsert("lineups", incoming,
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
            g = cut(grp.sort_values("score", ascending=False)).copy()
            picks.append(pd.DataFrame({
                "slate_date": str(slate_date), "prop": prop,
                "subject_id": g["player_id"], "subject_name": g["player_name"],
                "team": g["team"], "opponent": g.get("opp_starter"),
                "rank": g["rank"], "score": g["score"],
                "lineup_status": g.get("status"),
                "rationale": g.get("rationale"),
                "details": _details(g, PROP_DETAIL.get(prop, []) + COMMON_DETAIL),
            }))

    if pitcher_scores is not None and not pitcher_scores.empty:
        g = cut(pitcher_scores.sort_values("score", ascending=False)).copy()
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
        # Clear the slate's board first. Upserting alone leaves anyone who was in the
        # top N earlier but dropped out since, keeping their old rank and score — the
        # board then reads as a blend of runs, with duplicate ranks, scores out of
        # order, and stale "projected" badges on a confirmed slate.
        # Prove it serialises BEFORE clearing the slate, so a bad payload fails
        # loudly with the old board still intact.
        check_json(_ints(allp))
        delete_where("prop_picks", {"slate_date": f"eq.{slate_date}"})
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
        check_json(_ints(allg))
        delete_where("game_picks", {"slate_date": f"eq.{slate_date}"})
        written["game_picks"] = upsert("game_picks", _ints(allg),
                                       on_conflict="slate_date,prop,game_pk,subject")

    return written
