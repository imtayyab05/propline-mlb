"""Batted-ball and contact profiles — the v2 inputs.

v1 scored hits mostly on expected on-base quality, which surfaced high-OBP hitters
who collect singles and walks. The client's v2 spec adds contact quality so that
ground-ball and slap-hitter profiles stop cluttering the top of the board.

Everything here comes from files the collector already downloads. Nothing new is
fetched.
"""

from __future__ import annotations

import pandas as pd

from .mlb import effective_bat_side

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
    # Lift. The home-run model needs this explicitly: a hitter can have elite
    # exit velocity and still never clear a fence if he hits everything on the ground.
    out["fly_ball_rate"] = _as_pct(_num(out, "fb_rate"))

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

    keep = ["player_id", "line_drive_rate", "ground_ball_rate", "fly_ball_rate",
            "sweet_spot_pct",
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


def pitcher_pitch_mix(raw: pd.DataFrame, min_pitches: int = 30) -> pd.DataFrame:
    """How often each starter throws each pitch type, split by the batter's side.

    The client asked for "pitcher pitch-mix % vs LHB/RHB". Savant's
    pitch-arsenal-stats leaderboard cannot do this: hand, pitchHand and batSide all
    return byte-identical results, so the split does not exist there. It does exist in
    the raw pitch data, where every pitch carries both its type and the side the batter
    stood on.

    Usage is the share of that pitcher's pitches to that side, so the rows for one
    pitcher and one side sum to 100.
    """
    df = raw[["pitcher", "stand", "pitch_type"]].dropna()
    if df.empty:
        return pd.DataFrame()

    counts = (df.groupby(["pitcher", "stand", "pitch_type"])
                .size().rename("pitches").reset_index())
    totals = counts.groupby(["pitcher", "stand"])["pitches"].transform("sum")

    # Below this a "mix" is noise: one changeup in a September mop-up inning is not a
    # 33% changeup rate.
    counts = counts[totals >= min_pitches].copy()
    counts["usage_pct"] = (100 * counts["pitches"] / totals[counts.index]).round(1)
    return counts.rename(columns={"pitcher": "player_id", "stand": "vs_hand"})


def hr_matchup_matrix(matchups: pd.DataFrame, batter_arsenal: pd.DataFrame,
                      pitch_mix: pd.DataFrame) -> pd.DataFrame:
    """Weighted matchup matrix: hitter's value per pitch x pitcher's usage of it.

    This is the client's core home-run idea. A hitter who punishes fastballs but is
    helpless against breaking balls is a different proposition against a 60%-fastball
    starter than against a slider-heavy one, and a single blended matchup number hides
    exactly that. Weighting the hitter's per-pitch run value by how often this starter
    actually throws each pitch TO HIS SIDE keeps the distinction.

    Returns one row per (player_id, opp_starter_id) with the weighted value and the
    coverage behind it.
    """
    need = {"player_id", "opp_starter_id", "stands"}
    if not need <= set(matchups.columns) or pitch_mix.empty:
        return pd.DataFrame()

    rv_col = "run_value_per_100" if "run_value_per_100" in batter_arsenal.columns else None
    if rv_col is None:
        return pd.DataFrame()

    ba = batter_arsenal.rename(columns={"player_id": "player_id"})[
        ["player_id", "pitch_type", rv_col]].dropna()

    pairs = matchups[["player_id", "opp_starter_id", "stands"]].dropna().drop_duplicates()
    mix = pitch_mix.rename(columns={"player_id": "opp_starter_id"})

    # every pitch this starter throws to this hitter's side, joined to the hitter's
    # own record against that pitch type
    grid = pairs.merge(mix, left_on=["opp_starter_id", "stands"],
                       right_on=["opp_starter_id", "vs_hand"], how="left")
    grid = grid.merge(ba, on=["player_id", "pitch_type"], how="left")

    def _agg(g):
        d = g.dropna(subset=[rv_col, "usage_pct"])
        total = g["usage_pct"].sum()
        if d.empty or total == 0:
            return pd.Series({"hr_matchup_rv": None, "hr_matchup_coverage": 0.0})
        return pd.Series({
            "hr_matchup_rv": round(
                float((d[rv_col] * d["usage_pct"]).sum() / d["usage_pct"].sum()), 3),
            "hr_matchup_coverage": round(float(d["usage_pct"].sum() / total), 2),
        })

    out = (grid.groupby(["player_id", "opp_starter_id"])
               .apply(_agg, include_groups=False).reset_index())
    return out


# How many hitters ahead of a batter count as his table-setters. Three is the client's
# number: the men most likely to still be on base when he comes up.
TABLE_SETTERS = 3

# The context multiplier discounts; it never inflates. Two reasons.
#
# The client's spec is worded as a discount — "if table-setters project poorly,
# automatically discount the RBI score" — so a hitter with a good lineup around him is
# simply not penalised, rather than being pushed above what his own bat earned.
#
# It also keeps every prop on the same 0-100 scale. A multiplier above 1.0 pushed RBI
# scores past 100 while hits and home runs stayed inside it, and clipping them back
# created ties at the very top of the board, which is the worst place to have them.
CONTEXT_MIN, CONTEXT_MAX = 0.85, 1.00


def lineup_context(lineups: pd.DataFrame, batter_stats: pd.DataFrame) -> pd.DataFrame:
    """Who bats around each hitter, and how good they are at reaching base.

    The client's point: runs and RBIs are bottlenecked by teammates. An elite hitter
    with nobody on base in front of him has nobody to drive in, and no amount of
    individual quality fixes that.

    Two numbers per hitter:
      table_setter_obp  the men batting immediately AHEAD of him — who will be on base
                        when he bats, so this is what drives RBIs
      top_order_obp     his team's 1-2 hitters, a proxy for how often the lineup turns
                        over and scores at all

    The order wraps: the leadoff man's table-setters are the 7-8-9 hitters, because
    that is who is on base when he comes up in the later innings.
    """
    st = batter_stats.copy()
    if "player_id" not in st.columns and "id" in st.columns:
        st = st.rename(columns={"id": "player_id"})
    if "on_base_percent" not in st.columns:
        return pd.DataFrame()

    obp = dict(zip(st["player_id"], pd.to_numeric(st["on_base_percent"], errors="coerce")))
    league_obp = pd.Series(list(obp.values())).median()

    # Only the nine who are actually batting: scratched hitters carry slot+100 and are
    # not in the order at all.
    lu = lineups[lineups.get("status", "") != "scratched"].copy()
    lu = lu[lu["batting_order"].between(1, 9)]

    rows = []
    for (game_pk, team_id), team in lu.groupby(["game_pk", "team_id"]):
        team = team.sort_values("batting_order")
        order = team["player_id"].tolist()
        n = len(order)
        if n == 0:
            continue

        vals = [obp.get(p) for p in order]
        vals = [v if pd.notna(v) else league_obp for v in vals]
        top = sum(vals[:2]) / 2 if n >= 2 else vals[0]

        for i, pid in enumerate(order):
            # the TABLE_SETTERS hitters immediately before him, wrapping the order
            ahead = [vals[(i - k) % n] for k in range(1, TABLE_SETTERS + 1)]
            rows.append({
                "game_pk": game_pk, "team_id": team_id, "player_id": pid,
                "table_setter_obp": round(sum(ahead) / len(ahead), 3),
                "top_order_obp": round(top, 3),
            })

    return pd.DataFrame(rows)


def context_multiplier(values: pd.Series) -> pd.Series:
    """Turn a lineup-context number into a bounded multiplier around 1.0.

    Percentile-based, so it says "good for today's slate" rather than depending on
    where league OBP happens to sit this season.
    """
    pct = values.rank(pct=True).fillna(0.5)
    return (CONTEXT_MIN + pct * (CONTEXT_MAX - CONTEXT_MIN)).round(3)


# The client's WHIP bands for strikeout props. His reasoning: a pitcher who allows
# traffic runs up his pitch count and gets pulled early, so high WHIP costs outs even
# though it can briefly inflate strikeouts per inning.
WHIP_EFFICIENT_LO, WHIP_EFFICIENT_HI = 1.00, 1.25
WHIP_PENALTY_ABOVE = 1.40

# Leash risk. A starter who is not going five innings cannot reach a strikeout number
# regardless of his rate stats, so the deduction is applied to the score rather than
# folded into a weight.
SHORT_LEASH_PITCHES = 80
LEASH_PENALTY_MAX = 20.0


def opposing_lineup_k(lineups: pd.DataFrame, batter_rolling: pd.DataFrame,
                      window: str = "L10") -> pd.DataFrame:
    """Aggregate strikeout tendency of each team's actual starting nine.

    v1 averaged whatever hitters happened to have recent data. This uses the confirmed
    or projected 1-9 only, so a lineup full of contact hitters reads differently from
    one stacked with three-true-outcome bats — which is the client's "vacuum fallacy":
    a pitcher's strikeout ceiling is set as much by who is standing in the box as by
    his own stuff.
    """
    r = batter_rolling[(batter_rolling.window == window)
                       & (batter_rolling.split == "all")][["player_id", "k_pct", "pa"]]
    lu = lineups[lineups.get("status", "") != "scratched"]
    lu = lu[lu["batting_order"].between(1, 9)]

    merged = lu[["team_id", "player_id"]].merge(r, on="player_id", how="left")
    if merged.empty:
        return pd.DataFrame()

    # Weighted by plate appearances: a hitter with 40 recent PA describes the lineup
    # better than one with four.
    def _agg(g):
        d = g.dropna(subset=["k_pct"])
        if d.empty:
            return pd.Series({"opp_lineup_k_pct": None, "lineup_k_coverage": 0.0})
        w = d["pa"].fillna(1).clip(lower=1)
        return pd.Series({
            "opp_lineup_k_pct": round(float((d["k_pct"] * w).sum() / w.sum()), 1),
            "lineup_k_coverage": round(len(d) / len(g), 2),
        })

    out = merged.groupby("team_id").apply(_agg, include_groups=False).reset_index()
    return out.rename(columns={"team_id": "opp_team_id"})


def lineup_handedness(lineups: pd.DataFrame, bats: dict[int, str],
                      pitcher_throws: str | None = None) -> pd.DataFrame:
    """What share of each team's starting nine bats left and right.

    Feeds the split-K matchup: a pitcher with a large platoon split facing a lineup
    stacked against him is a different proposition from the same pitcher facing one
    stacked in his favour, and a season-long K% cannot express that.
    """
    lu = lineups[lineups.get("status", "") != "scratched"]
    lu = lu[lu["batting_order"].between(1, 9)]

    rows = []
    for team_id, team in lu.groupby("team_id"):
        sides = [effective_bat_side(bats.get(int(p)), pitcher_throws)
                 for p in team["player_id"]]
        sides = [s for s in sides if s in ("L", "R")]
        if not sides:
            continue
        n = len(sides)
        rows.append({"opp_team_id": team_id,
                     "lineup_share_L": round(sides.count("L") / n, 2),
                     "lineup_share_R": round(sides.count("R") / n, 2)})
    return pd.DataFrame(rows)


def gb_penalty(ground_ball_rate: pd.Series) -> pd.Series:
    """Points deducted from a hit score for a ground-ball-heavy profile.

    Scaled rather than a cliff: a hitter one point over the line should not be
    docked the same as one fifteen points over.
    """
    over = (pd.to_numeric(ground_ball_rate, errors="coerce") - GB_PENALTY_THRESHOLD)
    return (over.clip(lower=0) / 15.0 * GB_PENALTY_MAX).clip(upper=GB_PENALTY_MAX).fillna(0)
