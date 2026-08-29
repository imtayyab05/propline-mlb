"""Rolling-window aggregation from raw pitch-level Statcast data.

This exists because Savant's leaderboards ignore date filters (see
docs/savant_endpoints.md) — L5/L10 cannot be downloaded, it has to be computed.

Upside: once raw pitch data is stored, ANY split is a groupby rather than another
HTTP request — last 5 games, vs LHP, vs sliders, with two strikes, whatever the
client asks for later.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

# Statcast marks a barrel as launch_speed_angle == 6
BARREL = 6
HARD_HIT_MPH = 95.0

HIT_EVENTS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# Plate appearances that are not at-bats. ISO and slugging are per AB, so counting
# walks in the denominator would quietly punish patient hitters — the exact profile
# the client is trying to separate from genuine extra-base power.
NON_AB_EVENTS = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt",
                 "sac_fly_double_play", "sac_bunt_double_play", "catcher_interf"}
# Events that end a plate appearance (used for PA counting)
PA_EVENTS_EXCLUDED: set[str] = set()


# Statcast returns 68 columns; the aggregates below use 13. Reading the rest costs
# roughly five times the memory for nothing, and on a three-week pull that was enough
# to exhaust the heap mid-run on a normal desktop.
RAW_COLUMNS = [
    "game_date", "game_pk", "batter", "pitcher", "player_name",
    "stand", "p_throws", "events", "description",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle",
]


def read_raw(path) -> pd.DataFrame:
    """Load a raw Statcast export with only the columns the aggregates need."""
    header = pd.read_csv(path, nrows=0).columns
    cols = [c for c in RAW_COLUMNS if c in header]
    return pd.read_csv(path, usecols=cols, low_memory=False)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for c in ("launch_speed", "launch_angle", "launch_speed_angle",
              "estimated_woba_using_speedangle", "woba_value", "woba_denom"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _agg(g: pd.DataFrame) -> pd.Series:
    pa = g["events"].notna().sum()
    bbe = g["launch_speed"].notna().sum()
    ev = g["launch_speed"]
    hits = g["events"].map(HIT_EVENTS).fillna(0)

    ab = int(pa - g["events"].isin(NON_AB_EVENTS).sum())
    tb = int(hits.sum())

    return pd.Series({
        "games": g["game_pk"].nunique(),
        "pa": int(pa),
        "ab": ab,
        # ISO is slugging minus average: extra bases per at-bat, stripped of singles.
        # This is the signal that separates a doubles hitter from a slap hitter, which
        # a raw xwOBA cannot do.
        "iso": round((tb - int((hits > 0).sum())) / ab, 3) if ab else None,
        "slg": round(tb / ab, 3) if ab else None,
        "bbe": int(bbe),
        "hits": int((hits > 0).sum()),
        "total_bases": int(hits.sum()),
        "home_runs": int((g["events"] == "home_run").sum()),
        "strikeouts": int(g["events"].isin(["strikeout", "strikeout_double_play"]).sum()),
        "walks": int(g["events"].isin(["walk", "intent_walk"]).sum()),
        "k_pct": round(100 * g["events"].isin(["strikeout", "strikeout_double_play"]).sum() / pa, 1) if pa else None,
        "bb_pct": round(100 * g["events"].isin(["walk", "intent_walk"]).sum() / pa, 1) if pa else None,
        "avg_ev": round(ev.mean(), 1) if bbe else None,
        "max_ev": round(ev.max(), 1) if bbe else None,
        "hard_hit_pct": round(100 * (ev >= HARD_HIT_MPH).sum() / bbe, 1) if bbe else None,
        "barrel_pct": round(100 * (g["launch_speed_angle"] == BARREL).sum() / bbe, 1) if bbe else None,
        "xwoba_contact": round(g["estimated_woba_using_speedangle"].mean(), 3) if bbe else None,
    })


def _pitcher_agg(g: pd.DataFrame) -> pd.Series:
    bf = g["events"].notna().sum()          # batters faced
    bbe = g["launch_speed"].notna().sum()
    ev = g["launch_speed"]
    ks = g["events"].isin(["strikeout", "strikeout_double_play"]).sum()
    swstr = g["description"].isin(["swinging_strike", "swinging_strike_blocked"]).sum()
    swings = g["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
    ]).sum()
    hits = g["events"].map(HIT_EVENTS).fillna(0)

    ab_against = int(bf - g["events"].isin(NON_AB_EVENTS).sum())
    tb_against = int(hits.sum())

    return pd.Series({
        "games": g["game_pk"].nunique(),
        "batters_faced": int(bf),
        # Slugging allowed, which is the thing the client wants split by batter hand:
        # how badly this starter suppresses extra-base contact from lefties vs righties.
        "slg_allowed": round(tb_against / ab_against, 3) if ab_against else None,
        "iso_allowed": round((tb_against - int((hits > 0).sum())) / ab_against, 3)
                       if ab_against else None,
        "pitches": int(len(g)),
        "strikeouts": int(ks),
        "walks": int(g["events"].isin(["walk", "intent_walk"]).sum()),
        "hits_allowed": int((hits > 0).sum()),
        "home_runs_allowed": int((g["events"] == "home_run").sum()),
        "k_pct": round(100 * ks / bf, 1) if bf else None,
        "bb_pct": round(100 * g["events"].isin(["walk", "intent_walk"]).sum() / bf, 1) if bf else None,
        "k_per_game": round(ks / g["game_pk"].nunique(), 1) if g["game_pk"].nunique() else None,
        "pitches_per_game": round(len(g) / g["game_pk"].nunique(), 1) if g["game_pk"].nunique() else None,
        "whiff_pct": round(100 * swstr / swings, 1) if swings else None,
        "avg_ev_allowed": round(ev.mean(), 1) if bbe else None,
        "hard_hit_pct_allowed": round(100 * (ev >= HARD_HIT_MPH).sum() / bbe, 1) if bbe else None,
        "barrel_pct_allowed": round(100 * (g["launch_speed_angle"] == BARREL).sum() / bbe, 1) if bbe else None,
        "xwoba_contact_allowed": round(g["estimated_woba_using_speedangle"].mean(), 3) if bbe else None,
    })


def _windowed(df: pd.DataFrame, id_col: str, window: int) -> pd.DataFrame:
    """Keep only each player's most recent `window` games."""
    gm = (df[[id_col, "game_pk", "game_date"]].drop_duplicates()
            .sort_values([id_col, "game_date"], ascending=[True, False]))
    gm["gm_rank"] = gm.groupby(id_col).cumcount() + 1
    keep = gm[gm.gm_rank <= window][[id_col, "game_pk"]]
    return df.merge(keep, on=[id_col, "game_pk"], how="inner")


def _day_window(df: pd.DataFrame, days: int, as_of=None) -> pd.DataFrame:
    """Keep rows inside the last `days` CALENDAR days.

    The L5/L10 windows above count a player's own games, which is the right unit for
    "recent form" — a hitter who sat two days still gets a full five games. The
    client's v2 spec asks specifically for 14-DAY figures, which is a different
    question and a genuinely different number for anyone who has been rested or
    platooned. Both are offered rather than quietly substituting one for the other.
    """
    if df.empty:
        return df
    end = as_of or df["game_date"].max()
    start = end - timedelta(days=days - 1)
    return df[(df["game_date"] >= start) & (df["game_date"] <= end)]


def rolling_batter_days(raw: pd.DataFrame, days: int = 14, by_hand: bool = False,
                        as_of=None) -> pd.DataFrame:
    """Per-batter aggregates over the last N calendar days.

    Feeds iso_recent_14day and the other 14-day columns in the v2 spec.
    """
    df = _day_window(_prep(raw), days, as_of)
    if df.empty:
        return pd.DataFrame()

    splits = [("all", df)]
    if by_hand:
        splits += [("vsL", df[df.p_throws == "L"]), ("vsR", df[df.p_throws == "R"])]

    frames = []
    for label, part in splits:
        if part.empty:
            continue
        out = (part.groupby(["batter", "player_name"], as_index=False)
                   .apply(_agg, include_groups=False).reset_index(drop=True))
        ids = part[["batter", "player_name"]].drop_duplicates().reset_index(drop=True)
        if "batter" not in out.columns:
            out = pd.concat([ids, out], axis=1)
        out["window"] = f"D{days}"
        out["split"] = label
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    res = pd.concat(frames, ignore_index=True).rename(columns={"batter": "player_id"})
    for c in ("player_id", "games", "pa", "ab", "bbe", "hits", "total_bases",
              "home_runs", "strikeouts", "walks"):
        if c in res.columns:
            res[c] = res[c].fillna(0).astype("int64")

    front = ["player_id", "player_name", "window", "split", "games", "pa"]
    cols = front + [c for c in res.columns if c not in front]
    return res[cols].sort_values("pa", ascending=False).reset_index(drop=True)


def rolling_pitcher_splits(raw: pd.DataFrame, windows=(5, 10), by_hand=True,
                           name_map: dict | None = None) -> pd.DataFrame:
    """Per-pitcher aggregates over their last N appearances.

    Uses the same raw pitch data as the batter side — every row carries both a batter
    and a pitcher id. `player_name` in the raw feed follows the batter, so pitcher
    names come from `name_map` (built from the Savant pitcher files).

    Note `games` counts appearances, not starts: a reliever's L5 is their last five
    outings. The scoring layer filters to probable starters for strikeout props.
    """
    df = _prep(raw)
    frames = []

    for window in windows:
        sub = _windowed(df, "pitcher", window)
        splits = [("all", sub)]
        if by_hand:
            # for a pitcher the meaningful split is the BATTER's handedness
            splits += [("vsL", sub[sub.stand == "L"]), ("vsR", sub[sub.stand == "R"])]

        for label, part in splits:
            if part.empty:
                continue
            out = (part.groupby("pitcher").apply(_pitcher_agg, include_groups=False)
                       .reset_index())
            out["window"] = f"L{window}"
            out["split"] = label
            frames.append(out)

    if not frames:
        return pd.DataFrame()

    res = pd.concat(frames, ignore_index=True).rename(columns={"pitcher": "player_id"})
    res["player_name"] = res["player_id"].map(name_map or {})

    int_cols = ["player_id", "games", "batters_faced", "pitches", "strikeouts",
                "walks", "hits_allowed", "home_runs_allowed"]
    for c in int_cols:
        if c in res.columns:
            res[c] = res[c].fillna(0).astype("int64")

    front = ["player_id", "player_name", "window", "split", "games", "batters_faced"]
    cols = front + [c for c in res.columns if c not in front]
    return res[cols].sort_values(["window", "split", "batters_faced"],
                                ascending=[True, True, False])


def rolling_batter_splits(raw: pd.DataFrame, windows=(5, 10), by_hand=True) -> pd.DataFrame:
    """Per-batter aggregates over the last N *games played by that batter*.

    Windows are counted in the player's own games, not calendar days — a hitter who
    sat two days still gets a true 'last 5 games', which is what the client means.
    """
    df = _prep(raw)
    frames = []

    for window in windows:
        # rank each player's games most-recent-first, then keep the last N
        gm = (df[["batter", "game_pk", "game_date"]].drop_duplicates()
                .sort_values(["batter", "game_date"], ascending=[True, False]))
        gm["gm_rank"] = gm.groupby("batter").cumcount() + 1
        keep = gm[gm.gm_rank <= window][["batter", "game_pk"]]
        sub = df.merge(keep, on=["batter", "game_pk"], how="inner")

        splits = [("all", sub)]
        if by_hand:
            splits += [("vsL", sub[sub.p_throws == "L"]), ("vsR", sub[sub.p_throws == "R"])]

        for label, part in splits:
            if part.empty:
                continue
            out = (part.groupby(["batter", "player_name"], as_index=False)
                       .apply(_agg, include_groups=False)
                       .reset_index(drop=True))
            ids = (part[["batter", "player_name"]].drop_duplicates()
                       .reset_index(drop=True))
            out = pd.concat([ids, out.drop(columns=[c for c in ("batter", "player_name") if c in out])], axis=1) \
                if "batter" not in out.columns else out
            out["window"] = f"L{window}"
            out["split"] = label
            frames.append(out)

    if not frames:
        return pd.DataFrame()

    res = pd.concat(frames, ignore_index=True)
    res = res.rename(columns={"batter": "player_id"})

    # _agg returns a mixed-type Series, which pandas widens to float across the board.
    # Counts must stay integers or they surface as "3.0 games" in the client's Excel.
    int_cols = ["player_id", "games", "pa", "ab", "bbe", "hits", "total_bases",
                "home_runs", "strikeouts", "walks"]
    for c in int_cols:
        if c in res.columns:
            res[c] = res[c].fillna(0).astype("int64")

    front = ["player_id", "player_name", "window", "split", "games", "pa"]
    cols = front + [c for c in res.columns if c not in front]
    return res[cols].sort_values(["window", "split", "pa"], ascending=[True, True, False])
