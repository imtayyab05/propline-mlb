"""Matchup engine — every hitter in today's lineups vs the opposing starter's arsenal.

The core idea, and the thing the client was doing by eye:

  a pitcher throws a specific MIX of pitches (usage %)
  a hitter has specific results AGAINST each pitch type

So a hitter's expected performance against a given starter is his per-pitch numbers
weighted by that starter's actual usage. A hitter who mashes fastballs but whiffs on
sliders is a different proposition against a 60%-fastball starter than against a
slider-heavy one, even though his season line is identical in both cases.
"""

from __future__ import annotations

import pandas as pd

# Minimum share of a pitcher's arsenal we need batter data for before trusting the
# weighted number. Below this the hitter simply has not faced enough of the mix.
MIN_COVERAGE = 0.50


def _weighted(group: pd.DataFrame, value_col: str) -> float | None:
    """Usage-weighted mean of a batter metric across the pitcher's arsenal."""
    d = group.dropna(subset=[value_col, "pitch_usage"])
    if d.empty or d["pitch_usage"].sum() == 0:
        return None
    return float((d[value_col] * d["pitch_usage"]).sum() / d["pitch_usage"].sum())


def build_matchups(lineups: pd.DataFrame, schedule: pd.DataFrame,
                   batter_arsenal: pd.DataFrame, pitcher_arsenal: pd.DataFrame,
                   pitcher_hand: dict[int, str] | None = None) -> pd.DataFrame:
    """One row per (hitter, opposing starter) for today's slate."""

    # --- who does each lineup slot actually face today? -------------------------
    starters = []
    for _, g in schedule.iterrows():
        # the home team's hitters face the away probable, and vice versa
        starters.append({"game_pk": g["game_pk"], "team_id": g["home_team_id"],
                         "opp_starter_id": g["away_probable_id"],
                         "opp_starter": g["away_probable"]})
        starters.append({"game_pk": g["game_pk"], "team_id": g["away_team_id"],
                         "opp_starter_id": g["home_probable_id"],
                         "opp_starter": g["home_probable"]})
    starters = pd.DataFrame(starters)

    df = lineups.merge(starters, on=["game_pk", "team_id"], how="left")
    df = df[df["opp_starter_id"].notna()].copy()
    df["opp_starter_id"] = df["opp_starter_id"].astype("int64")
    if pitcher_hand:
        df["opp_starter_throws"] = df["opp_starter_id"].map(pitcher_hand)

    # --- the arsenal join -------------------------------------------------------
    pa = pitcher_arsenal[["player_id", "pitch_type", "pitch_name", "pitch_usage",
                          "est_woba", "whiff_percent", "k_percent"]].rename(columns={
        "player_id": "opp_starter_id",
        "est_woba": "pitcher_est_woba_with_pitch",
        "whiff_percent": "pitcher_whiff_with_pitch",
        "k_percent": "pitcher_k_with_pitch",
    })
    ba = batter_arsenal[["player_id", "pitch_type", "est_woba", "est_slg",
                         "whiff_percent", "k_percent", "hard_hit_percent", "pa"]].rename(columns={
        "est_woba": "batter_est_woba_vs_pitch",
        "est_slg": "batter_est_slg_vs_pitch",
        "whiff_percent": "batter_whiff_vs_pitch",
        "k_percent": "batter_k_vs_pitch",
        "hard_hit_percent": "batter_hard_hit_vs_pitch",
        "pa": "batter_pa_vs_pitch",
    })

    # every pitch the starter throws, crossed with this hitter's record against it
    pairs = df.merge(pa, on="opp_starter_id", how="left")
    pairs = pairs.merge(ba, left_on=["player_id", "pitch_type"],
                        right_on=["player_id", "pitch_type"], how="left")

    rows = []
    keys = ["game_pk", "team", "team_id", "player_id", "player_name", "position",
            "batting_order", "status", "opp_starter_id", "opp_starter"]
    if "opp_starter_throws" in pairs.columns:
        keys.append("opp_starter_throws")

    for key, g in pairs.groupby(keys, dropna=False):
        total_usage = g["pitch_usage"].sum()
        covered = g.loc[g["batter_est_woba_vs_pitch"].notna(), "pitch_usage"].sum()
        coverage = (covered / total_usage) if total_usage else 0.0

        # what the starter leans on most
        top = g.nlargest(1, "pitch_usage")
        best = g.dropna(subset=["batter_est_woba_vs_pitch"]).nlargest(1, "batter_est_woba_vs_pitch")

        row = dict(zip(keys, key))
        row.update({
            "arsenal_pitches": int(g["pitch_type"].notna().sum()),
            "arsenal_coverage": round(coverage, 2),
            "reliable": coverage >= MIN_COVERAGE,
            "primary_pitch": top["pitch_name"].iloc[0] if len(top) else None,
            "primary_pitch_usage": round(float(top["pitch_usage"].iloc[0]), 1) if len(top) else None,
            # hitter's expected output against THIS mix
            "matchup_est_woba": _weighted(g, "batter_est_woba_vs_pitch"),
            "matchup_est_slg": _weighted(g, "batter_est_slg_vs_pitch"),
            "matchup_whiff": _weighted(g, "batter_whiff_vs_pitch"),
            "matchup_k_pct": _weighted(g, "batter_k_vs_pitch"),
            "matchup_hard_hit": _weighted(g, "batter_hard_hit_vs_pitch"),
            # what the starter gives up with that same mix
            "starter_est_woba_allowed": _weighted(g, "pitcher_est_woba_with_pitch"),
            "starter_whiff": _weighted(g, "pitcher_whiff_with_pitch"),
            # the hitter's single best pitch to look for in this arsenal
            "best_pitch_for_batter": best["pitch_name"].iloc[0] if len(best) else None,
            "best_pitch_est_woba": round(float(best["batter_est_woba_vs_pitch"].iloc[0]), 3) if len(best) else None,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    for c in ("matchup_est_woba", "matchup_est_slg", "matchup_whiff",
              "matchup_k_pct", "matchup_hard_hit", "starter_est_woba_allowed",
              "starter_whiff"):
        if c in out.columns:
            out[c] = out[c].astype(float).round(3)

    return out.sort_values(["game_pk", "team", "batting_order"]).reset_index(drop=True)
