"""Arsenal tabs — what each starter actually throws, and how each hitter handles it.

The client called this one of the main goals of v2. Two views:

  * PITCHER: for every starter on today's slate, his top pitches to left-handed and to
    right-handed hitters separately, with usage, velocity, whiff rate and the damage
    done against each.
  * BATTER: for every hitter in today's lineups, how he performs against each pitch
    type — and crucially, whether the man he faces tonight actually throws it.

A NOTE ON WHERE THE SPLIT COMES FROM
------------------------------------
Savant's pitch-arsenal-stats leaderboard cannot split by batter side. hand, pitchHand
and batSide all return byte-identical files, so the vs-LHB/vs-RHB breakdown simply does
not exist there. It does exist in the raw pitch data, where every pitch carries both its
type and the side the batter stood on, so the split is computed here.

The consequence, stated plainly because it changes how the numbers should be read: the
hand-split columns cover the raw collection window (about three weeks), not the season.
Season-long per-pitch numbers that are NOT split by hand come from the leaderboard and
are labelled as season. The two are kept in separate columns rather than blended.
"""

from __future__ import annotations

import pandas as pd

# Consistent with propline/rolling.py so whiff rates agree across the project.
WHIFFS = ["swinging_strike", "swinging_strike_blocked"]
SWINGS = ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
          "hit_into_play"]

# Rate stats (whiff%, xwOBA allowed) need a sample before they mean anything. Below
# this the pitch still APPEARS — the mix is the point of the tab — but its rates are
# blanked rather than shown as fact.
#
# Filtering these rows out entirely was the first attempt and it was wrong: over a
# three-week raw window a genuine third pitch often sits at 15-20 pitches to one side,
# so the filter quietly turned "top 3" into "top 2" for most of the slate.
MIN_PITCHES_VS_HAND = 25

# How many pitches per side make the tab. The client asked for the top 3.
TOP_PITCHES = 3

HAND_LABEL = {"L": "vs LHB", "R": "vs RHB"}


def _rate(part: pd.Series, whole: pd.Series) -> pd.Series:
    """Percentage, guarding division by zero.

    The to_numeric is load-bearing: replace(0, pd.NA) on an integer column yields an
    OBJECT series, and .round() on object dtype silently does nothing — which is how
    whiff rates first came out as 30.434783.
    """
    out = 100 * part / whole.replace(0, pd.NA)
    return pd.to_numeric(out, errors="coerce").round(1)


def pitch_effectiveness_by_hand(raw: pd.DataFrame,
                                min_pitches: int = MIN_PITCHES_VS_HAND) -> pd.DataFrame:
    """Per pitcher, per pitch type, per batter side: usage, whiff% and xwOBA allowed.

    Computed from raw pitch data because this split exists nowhere else.
    """
    cols = ["pitcher", "stand", "pitch_type", "description",
            "estimated_woba_using_speedangle"]
    df = raw[[c for c in cols if c in raw.columns]].dropna(
        subset=["pitcher", "stand", "pitch_type"])
    if df.empty:
        return pd.DataFrame()

    df = df.assign(
        _whiff=df["description"].isin(WHIFFS),
        _swing=df["description"].isin(SWINGS),
    )
    agg = (df.groupby(["pitcher", "stand", "pitch_type"], as_index=False)
             .agg(pitches=("description", "size"),
                  whiffs=("_whiff", "sum"),
                  swings=("_swing", "sum"),
                  xwoba_allowed=("estimated_woba_using_speedangle", "mean")))

    # Usage is the share of that pitcher's pitches to that side, so one pitcher/side
    # sums to 100. Computed BEFORE the min-pitches filter, or dropping a rare pitch
    # would inflate everything else.
    agg["usage_pct"] = _rate(agg["pitches"],
                             agg.groupby(["pitcher", "stand"])["pitches"]
                                .transform("sum"))
    agg["whiff_pct"] = _rate(agg["whiffs"], agg["swings"])
    agg["xwoba_allowed"] = agg["xwoba_allowed"].round(3)

    # Keep every pitch, but only claim a rate where the sample supports one.
    thin = agg["pitches"] < min_pitches
    agg.loc[thin, ["whiff_pct", "xwoba_allowed"]] = pd.NA
    agg["rates_reliable"] = ~thin

    return agg.rename(columns={"pitcher": "player_id", "stand": "vs_hand"})


def pitcher_velocities(arsenals: pd.DataFrame) -> pd.DataFrame:
    """Reshape the wide velocity export (ff_avg_speed, sl_avg_speed, ...) to long."""
    if arsenals is None or arsenals.empty:
        return pd.DataFrame(columns=["player_id", "pitch_type", "avg_speed"])

    id_col = "pitcher" if "pitcher" in arsenals.columns else "player_id"
    speed_cols = [c for c in arsenals.columns if c.endswith("_avg_speed")]
    if not speed_cols:
        return pd.DataFrame(columns=["player_id", "pitch_type", "avg_speed"])

    long = arsenals.melt(id_vars=[id_col], value_vars=speed_cols,
                         var_name="pitch_type", value_name="avg_speed").dropna(
        subset=["avg_speed"])
    long["pitch_type"] = long["pitch_type"].str.replace("_avg_speed", "", regex=False
                                                        ).str.upper()
    return long.rename(columns={id_col: "player_id"})


def pitcher_arsenal_table(schedule: pd.DataFrame, raw: pd.DataFrame,
                          arsenals: pd.DataFrame | None = None,
                          arsenal_stats: pd.DataFrame | None = None,
                          names: dict | None = None,
                          pitcher_hand: dict | None = None,
                          top_n: int = TOP_PITCHES) -> pd.DataFrame:
    """Today's starters: their top `top_n` pitches to each side of the plate."""
    if schedule is None or schedule.empty or raw is None or raw.empty:
        return pd.DataFrame()

    eff = pitch_effectiveness_by_hand(raw)
    if eff.empty:
        return pd.DataFrame()

    # Only today's probable starters, with the fixture attached so the tab reads as a
    # slate preview rather than a leaderboard.
    starters = []
    for _, g in schedule.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            pid = g.get(f"{side}_probable_id")
            if pd.isna(pid):
                continue
            starters.append({
                "player_id": int(pid),
                "pitcher": g.get(f"{side}_probable"),
                "team": g.get(f"{side}_team"),
                "opponent": g.get(f"{opp}_team"),
                "game_pk": g.get("game_pk"),
            })
    if not starters:
        return pd.DataFrame()
    sp = pd.DataFrame(starters).drop_duplicates("player_id")

    out = sp.merge(eff, on="player_id", how="inner")
    if out.empty:
        return pd.DataFrame()

    if pitcher_hand:
        out["throws"] = out["player_id"].map(pitcher_hand)
    if names:
        out["pitcher"] = out["pitcher"].fillna(out["player_id"].map(names))

    # Velocity, and the season-long per-pitch numbers that are not hand-split.
    velo = pitcher_velocities(arsenals)
    if not velo.empty:
        out = out.merge(velo, on=["player_id", "pitch_type"], how="left")

    if arsenal_stats is not None and not arsenal_stats.empty:
        keep = [c for c in ("player_id", "pitch_type", "pitch_name",
                            "run_value_per_100") if c in arsenal_stats.columns]
        if len(keep) > 2:
            out = out.merge(arsenal_stats[keep].drop_duplicates(
                ["player_id", "pitch_type"]), on=["player_id", "pitch_type"], how="left")

    out["side"] = out["vs_hand"].map(HAND_LABEL).fillna(out["vs_hand"])

    # Top N by usage within each pitcher/side.
    out = out.sort_values(["player_id", "vs_hand", "usage_pct"],
                          ascending=[True, True, False])
    out["pitch_rank"] = out.groupby(["player_id", "vs_hand"]).cumcount() + 1
    out = out[out["pitch_rank"] <= top_n]

    return out.sort_values(["team", "pitcher", "vs_hand", "pitch_rank"]
                           ).reset_index(drop=True)


def batter_arsenal_table(matchups: pd.DataFrame, batter_arsenal: pd.DataFrame,
                         pitch_mix: pd.DataFrame | None = None,
                         min_pa: int = 10) -> pd.DataFrame:
    """Every hitter's per-pitch-type performance, tied to tonight's starter.

    The matchup engine already blends these into one number per hitter. This tab shows
    the parts, and marks which pitch types the starter he faces actually throws — a
    hitter's weakness against a cutter does not matter if tonight's man has none.
    """
    if matchups is None or matchups.empty or batter_arsenal is None \
            or batter_arsenal.empty:
        return pd.DataFrame()

    keep = [c for c in ("player_id", "pitch_type", "pitch_name", "pa", "est_woba",
                        "est_slg", "whiff_percent", "run_value_per_100", "ba", "slg")
            if c in batter_arsenal.columns]
    ba = batter_arsenal[keep].copy()
    if "pa" in ba.columns:
        ba = ba[pd.to_numeric(ba["pa"], errors="coerce").fillna(0) >= min_pa]

    base_cols = [c for c in ("player_id", "player_name", "team", "opp_starter",
                             "opp_starter_id", "stands", "bats", "batting_order",
                             "status") if c in matchups.columns]
    out = matchups[base_cols].drop_duplicates("player_id").merge(
        ba, on="player_id", how="inner")
    if out.empty:
        return pd.DataFrame()

    # Does tonight's starter throw this pitch to this hitter's side, and how often?
    if pitch_mix is not None and not pitch_mix.empty and "opp_starter_id" in out.columns:
        side_col = "stands" if "stands" in out.columns else "bats"
        mix = pitch_mix.rename(columns={"player_id": "opp_starter_id",
                                        "vs_hand": side_col,
                                        "usage_pct": "sp_usage_pct"})
        cols = ["opp_starter_id", side_col, "pitch_type", "sp_usage_pct"]
        out = out.merge(mix[[c for c in cols if c in mix.columns]].drop_duplicates(
            [c for c in ("opp_starter_id", side_col, "pitch_type") if c in mix.columns]),
            on=[c for c in ("opp_starter_id", side_col, "pitch_type") if c in mix.columns],
            how="left")
        out["sp_throws_it"] = out["sp_usage_pct"].notna()
    else:
        out["sp_usage_pct"] = pd.NA
        out["sp_throws_it"] = False

    # Most relevant pitch first: what he will actually see tonight, then his own volume.
    sort_cols = [c for c in ("team", "batting_order", "player_name") if c in out.columns]
    out = out.sort_values(sort_cols + ["sp_usage_pct", "pa"],
                          ascending=[True] * len(sort_cols) + [False, False])
    return out.reset_index(drop=True)


def arsenal_summary(pitcher_arsenal: pd.DataFrame) -> pd.DataFrame:
    """Collapse the pitcher tab into one compact list per starter, for the dashboard.

    Kept deliberately small. This rides inside the existing `details` JSON on the 27
    strikeout picks rather than in a table of its own: at roughly 500 bytes a starter
    it costs about 13 KB a slate, needs no new table, and therefore needs no schema
    migration run by hand against the live database.

    Shape: [{side, pitch, usage, velo, whiff, xwoba, ok}, ...] — six entries, the top
    three pitches to each side, already ordered.
    """
    if pitcher_arsenal is None or pitcher_arsenal.empty:
        return pd.DataFrame(columns=["player_id", "arsenal"])

    df = pitcher_arsenal.sort_values(["player_id", "vs_hand", "pitch_rank"])

    def _one(g: pd.DataFrame) -> list[dict]:
        out = []
        for _, r in g.iterrows():
            def _n(col, nd):
                v = r.get(col)
                return None if pd.isna(v) else round(float(v), nd)
            # `or` is WRONG here: a missing pitch_name arrives as NaN, and NaN is
            # truthy in Python, so `nan or "FC"` returns the NaN. That NaN reached
            # json.dumps, which emits a bare NaN, and PostgREST rejected the whole
            # prop_picks batch with "Empty or invalid json" — after publish had
            # already deleted the slate, leaving the board empty.
            name = r.get("pitch_name")
            if name is None or (not isinstance(name, str) and pd.isna(name)):
                name = r.get("pitch_type")
            out.append({
                "side": r.get("side"),
                "pitch": None if name is None or (not isinstance(name, str)
                                                  and pd.isna(name)) else str(name),
                "usage": _n("usage_pct", 1),
                "velo": _n("avg_speed", 1),
                "whiff": _n("whiff_pct", 1),
                "xwoba": _n("xwoba_allowed", 3),
                # False means the sample behind whiff/xwOBA was too thin to trust; the
                # pitch is still real and still shown.
                "ok": bool(r.get("rates_reliable", False)),
            })
        return out

    rows = [{"player_id": int(pid), "arsenal": _one(g)}
            for pid, g in df.groupby("player_id")]
    return pd.DataFrame(rows)


def attach_arsenal(pitcher_scores: pd.DataFrame,
                   pitcher_arsenal: pd.DataFrame) -> pd.DataFrame:
    """Attach the compact arsenal list to the strikeout board."""
    if pitcher_scores is None or pitcher_scores.empty:
        return pitcher_scores
    summary = arsenal_summary(pitcher_arsenal)
    if summary.empty:
        return pitcher_scores
    return pitcher_scores.merge(summary, on="player_id", how="left")
