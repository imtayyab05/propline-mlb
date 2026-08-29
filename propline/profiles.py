"""Batted-ball and contact profiles — the v2 inputs.

v1 scored hits mostly on expected on-base quality, which surfaced high-OBP hitters
who collect singles and walks. The client's v2 spec adds contact quality so that
ground-ball and slap-hitter profiles stop cluttering the top of the board.

Everything here comes from files the collector already downloads. Nothing new is
fetched.
"""

from __future__ import annotations

import pandas as pd

# A hitter above this ground-ball rate has his hit score docked: balls on the ground
# turn into outs and double plays rather than multi-hit games. The client's spec asks
# for "10 to 15 points"; the midpoint is used, scaled by how far above the line he is
# so a 51% hitter is not treated like a 65% one.
GB_PENALTY_THRESHOLD = 50.0
GB_PENALTY_MAX = 15.0


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def _as_pct(s: pd.Series) -> pd.Series:
    """Savant mixes fractions and percentages between files. Return percentages."""
    s = pd.to_numeric(s, errors="coerce")
    # a rate column whose maximum is <= 1 is expressed as a fraction
    return s * 100 if s.notna().any() and s.max() <= 1.0 else s


def batter_profiles(batted_ball: pd.DataFrame, exit_velocity: pd.DataFrame,
                    bat_tracking: pd.DataFrame, stats: pd.DataFrame,
                    day14: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per batter: contact quality signals used by the v2 hit model.

    Savant spells the id column differently in every file, so each is normalised to
    `player_id` before joining — see docs/savant_endpoints.md.
    """
    def norm(df):
        df = df.copy()
        for alias in ("id", "player_id"):
            if alias in df.columns:
                return df.rename(columns={alias: "player_id"})
        return df

    bb, ev, bt, st = (norm(d) for d in (batted_ball, exit_velocity, bat_tracking, stats))

    out = bb[["player_id"]].drop_duplicates().copy()

    # --- batted-ball shape ------------------------------------------------------
    # NOTE: the client's spec calls for "fbld" as a line-drive rate. In Savant's
    # exit-velocity file `fbld` is the average EXIT VELOCITY on fly balls and line
    # drives, not a rate at all. The real rates live in the batted-ball file, so
    # ld_rate/gb_rate are used here.
    out = out.merge(bb[["player_id", "ld_rate", "gb_rate", "fb_rate"]], on="player_id", how="left")
    # The batted-ball file gives these as fractions (0.42), while every other rate in
    # the pipeline is a percentage (42.0). Mixing the two silently disabled the
    # ground-ball penalty entirely — a 62% ground-ball hitter scored a 0.0 deduction
    # against a 50.0 threshold. Normalise on the way in.
    out["line_drive_rate"] = _as_pct(_num(out, "ld_rate"))
    out["ground_ball_rate"] = _as_pct(_num(out, "gb_rate"))

    # --- sweet spot -------------------------------------------------------------
    if "anglesweetspotpercent" in ev.columns:
        out = out.merge(ev[["player_id", "anglesweetspotpercent"]], on="player_id", how="left")
        out["sweet_spot_pct"] = _num(out, "anglesweetspotpercent")
    elif "sweet_spot_percent" in st.columns:
        out = out.merge(st[["player_id", "sweet_spot_percent"]], on="player_id", how="left")
        out["sweet_spot_pct"] = _num(out, "sweet_spot_percent")

    # --- contact ----------------------------------------------------------------
    # Bat-tracking gives whiff per swing on competitive swings, which is a cleaner
    # read on bat-to-ball than the season whiff% over all swings.
    if "whiff_per_swing" in bt.columns:
        bt = bt.copy()
        bt["contact_rate"] = (1 - pd.to_numeric(bt["whiff_per_swing"], errors="coerce")) * 100
        out = out.merge(bt[["player_id", "contact_rate"]], on="player_id", how="left")
    if "contact_rate" not in out.columns and "whiff_percent" in st.columns:
        st = st.copy()
        st["contact_rate"] = 100 - pd.to_numeric(st["whiff_percent"], errors="coerce")
        out = out.merge(st[["player_id", "contact_rate"]], on="player_id", how="left")

    # --- isolated power ---------------------------------------------------------
    # ISO = slugging minus average: extra bases per at-bat with singles stripped out.
    # This is the number that separates a doubles hitter from a slap hitter, and it is
    # the core of the v2 total-bases model.
    if {"slg_percent", "batting_avg"} <= set(st.columns):
        iso = st[["player_id"]].copy()
        iso["iso_season"] = (_num(st, "slg_percent") - _num(st, "batting_avg")).round(3)
        out = out.merge(iso, on="player_id", how="left")

    if day14 is not None and not day14.empty:
        d = day14[day14.get("split", "all") == "all"] if "split" in day14 else day14
        cols = [c for c in ("player_id", "iso", "ab", "games") if c in d.columns]
        d = d[cols].rename(columns={"iso": "iso_recent_14day",
                                    "ab": "ab_14day", "games": "games_14day"})
        out = out.merge(d, on="player_id", how="left")

    keep = ["player_id", "line_drive_rate", "ground_ball_rate", "sweet_spot_pct",
            "contact_rate", "iso_season", "iso_recent_14day", "ab_14day",
            "games_14day"]
    return out[[c for c in keep if c in out.columns]]


def _ip_to_float(v) -> float | None:
    """Savant writes innings as 75.1 meaning 75 and one third, not 75.1 innings.

    Treating it as a decimal understates WHIP by roughly a third of an inning per
    start, which is small but wrong in a consistent direction.
    """
    try:
        s = str(v)
        whole, _, frac = s.partition(".")
        outs = {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac[:1], 0.0) if frac else 0.0
        return float(whole) + outs
    except (TypeError, ValueError):
        return None


def pitcher_profiles(pitcher_stats: pd.DataFrame,
                     pitcher_rolling: pd.DataFrame | None = None,
                     window: str = "L10") -> pd.DataFrame:
    """Per-starter WHIP, which Savant does not publish directly.

    WHIP = (hits + walks) / innings pitched. All three parts are in the statistics
    export, so this is arithmetic rather than another request.
    """
    df = pitcher_stats.copy()
    if "player_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "player_id"})

    ip = df["p_formatted_ip"].map(_ip_to_float) if "p_formatted_ip" in df.columns else None
    if ip is None:
        return pd.DataFrame(columns=["player_id", "whip", "innings"])

    hits = _num(df, "hit")
    walks = _num(df, "walk")
    out = pd.DataFrame({
        "player_id": df["player_id"],
        "innings": ip,
        "whip": ((hits + walks) / ip.replace(0, pd.NA)).round(3),
    })
    out = out.dropna(subset=["player_id"])

    # Slugging allowed to left- and right-handed hitters, from the raw pitch data we
    # already aggregate. The client asked for pitcher SLG-against by handedness; this
    # is that, computed rather than downloaded, because no Savant leaderboard splits
    # a pitcher's results by the batter's side.
    if pitcher_rolling is not None and not pitcher_rolling.empty:
        pr = pitcher_rolling[pitcher_rolling.window == window]
        for side in ("L", "R"):
            part = pr[pr.split == f"vs{side}"]
            if part.empty or "slg_allowed" not in part.columns:
                continue
            out = out.merge(
                part[["player_id", "slg_allowed"]].rename(
                    columns={"slg_allowed": f"slg_allowed_vs_{side}"}),
                on="player_id", how="left")

    # Barrels allowed, regardless of hand — the input behind the suppression flag and,
    # later, the home-run suppression cap.
    if pitcher_rolling is not None and not pitcher_rolling.empty:
        allsp = pitcher_rolling[(pitcher_rolling.window == window)
                                & (pitcher_rolling.split == "all")]
        if "barrel_pct_allowed" in allsp.columns:
            out = out.merge(
                allsp[["player_id", "barrel_pct_allowed"]].rename(
                    columns={"barrel_pct_allowed": "starter_barrel_pct_allowed"}),
                on="player_id", how="left")

    return out


def gb_penalty(ground_ball_rate: pd.Series) -> pd.Series:
    """Points deducted from a hit score for a ground-ball-heavy profile.

    Scaled rather than a cliff: a hitter one point over the line should not be
    docked the same as one fifteen points over.
    """
    over = (pd.to_numeric(ground_ball_rate, errors="coerce") - GB_PENALTY_THRESHOLD)
    return (over.clip(lower=0) / 15.0 * GB_PENALTY_MAX).clip(upper=GB_PENALTY_MAX).fillna(0)
