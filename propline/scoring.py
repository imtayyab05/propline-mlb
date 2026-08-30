"""Formula-based prop scoring — the actual decision engine.

Deliberately NOT an LLM. Every pick's score traces back to numbers you can point at,
which matters for a betting tool: when a pick loses, the client can see exactly which
signal was wrong and tune it. Groq only writes the English afterwards.

How it works
------------
1. Each signal is converted to a percentile (0-1) ACROSS TODAY'S SLATE. Percentiles,
   not raw values, because the question is never "is .350 good" in the abstract — it's
   "who are the best plays on this particular slate".
2. Signals are combined with the weights in WEIGHTS below.
3. Small samples get shrunk toward the season baseline, so a hitter with 3 recent games
   cannot ride a hot streak to the top of the board.

WEIGHTS is v1 and is meant to be argued with — it is the one part of this system that
encodes betting opinion rather than fact.
"""

from __future__ import annotations

import pandas as pd

from .mlb import effective_bat_side
from .profiles import gb_penalty

# --- tunable ------------------------------------------------------------------

WEIGHTS = {
    # v2, to the client's spec after three days of live testing. v1 leaned on expected
    # on-base quality, which floated high-OBP hitters who walk and single rather than
    # collect multiple hits, and it ignored who was pitching beyond the arsenal match.
    # Contact quality and pitcher WHIP replace that. A ground-ball penalty is applied
    # separately, after scoring — see gb_penalty in profiles.py.
    "hits": {
        "recent_xwoba": 0.35,        # current form, not season-to-date
        "contact_rate": 0.25,        # bat-to-ball: you cannot single without contact
        "ld_sweet_index": 0.25,      # line drives and sweet-spot contact fall in
        "starter_whip": 0.15,        # a pitcher who allows traffic allows hits
    },
    # v2 total bases: a DUAL PATH, not one blended number.
    #
    # There are two ways to reach two total bases and they suit different hitters: a
    # contact hitter gets there with two singles, a slugger with one double. Averaging
    # the two signals scored both types as mediocre and let high-OBP singles hitters
    # ride expected-on-base to the top, which is exactly what the client complained
    # about. Each path is scored on its own and the better one wins, so a hitter only
    # has to be good at one of them.
    "total_bases_power": {
        "iso_recent_14day": 0.30,          # extra bases per at-bat, recent form
        "iso_season": 0.20,                # the same, as a stable anchor
        "recent_barrel_pct": 0.25,         # contact hit at the ideal speed and angle
        "recent_hard_hit": 0.10,
        "starter_slg_allowed_vs_hand": 0.15,   # what this arm gives up to his side
    },
    "total_bases_volume": {
        "recent_hit_rate": 0.35,           # two singles is two bases
        "contact_rate": 0.25,
        "matchup_est_woba": 0.25,
        "lineup_spot": 0.15,               # more plate appearances, more chances
    },
    # v2 home runs. The client's complaint: elite hitters scored well even when they
    # hit the ball on the ground or into doubles rather than over the fence. Two fixes
    # — reward LIFT explicitly, and replace the blended arsenal number with a weighted
    # matchup matrix (his term) that keeps per-pitch detail instead of averaging it
    # away. A hitter who mashes fastballs but flails at breaking balls should not look
    # the same against a slider-heavy arm as against a fastball-heavy one.
    "home_runs": {
        "hr_matchup_rv": 0.30,        # run value per pitch x this starter's usage
        "recent_barrel_pct": 0.25,    # barrels are the shape of a home run
        "fly_ball_rate": 0.20,        # you cannot homer on the ground
        "iso_recent_14day": 0.15,
        "recent_hard_hit": 0.10,
    },
    "rbis": {
        "matchup_est_slg": 0.30,
        "recent_tb_rate": 0.20,
        "rbi_spot": 0.25,            # 3-4-5 hitters bat with men on
        "team_offense": 0.15,
        "season_est_slg": 0.10,
    },
    "runs": {
        "matchup_est_woba": 0.30,
        "run_spot": 0.25,            # 1-2-3 hitters score most
        "recent_hit_rate": 0.20,
        "team_offense": 0.15,
        "matchup_k_pct": -0.10,
    },
    # team runs in a single game
    "team_total": {
        "lineup_matchup_woba": 0.35,   # how the lineup fares vs the opposing starter
        "opp_starter_weak": 0.25,      # what that starter gives up generally
        "opp_bullpen_tired": 0.20,     # a depleted pen means worse innings 6-9
        "park_runs": 0.10,
        "recent_team_form": 0.10,
    },
    # combined runs, both teams
    "game_total": {
        "combined_offense": 0.40,
        "combined_starter_weak": 0.25,
        "combined_bullpen_tired": 0.20,
        "park_runs": 0.15,
    },
    # pitcher side
    "strikeouts": {
        "recent_k_per_game": 0.30,
        "recent_k_pct": 0.25,
        "recent_whiff_pct": 0.20,
        "opp_lineup_k_pct": 0.15,    # facing a strikeout-prone lineup
        "recent_pitches_per_game": 0.10,
    },
}

# Which WEIGHTS entries are per-hitter. The rest are pitcher- or team-level and are
# scored by their own functions — looping over all of WEIGHTS would score a game total
# as if it were a batting prop.
# Scored by the plain weighted-percentile loop. total_bases is absent on purpose: it
# takes the better of two competing paths, which the loop cannot express.
BATTER_PROPS = ("hits", "home_runs", "rbis", "runs")

# A pick at or above this score is a "strict" 2+ total bases play, per the client's
# spec. Below it the model is describing a good hitter, not a two-base expectation.
TB_STRICT_THRESHOLD = 90.0

# Facing a starter in the best quartile at preventing barrels, a home-run score is
# capped rather than merely nudged. The client asked for a strict cap: no matter how
# good the hitter looks on paper, a pitcher who does not allow barrels is a hard
# ceiling on the outcome the bet needs.
HR_SUPPRESSION_CAP = 78.0

# How much a lineup slot is worth for each prop family (index 0 = leadoff)
SPOT_PA = [1.00, 0.97, 0.94, 0.90, 0.86, 0.82, 0.78, 0.74, 0.70]   # plate appearances
SPOT_RBI = [0.55, 0.75, 0.95, 1.00, 0.95, 0.80, 0.65, 0.55, 0.50]  # men on base
SPOT_RUN = [1.00, 0.95, 0.90, 0.80, 0.70, 0.62, 0.58, 0.55, 0.55]  # driven in by others

FULL_WINDOW = 10   # games that count as a complete L10
MIN_SHRINK = 0.35  # a 1-game sample still keeps this much of its own signal


def _pct(s: pd.Series) -> pd.Series:
    """Percentile rank across the slate, 0-1. Missing values land mid-pack (0.5)."""
    return s.rank(pct=True).fillna(0.5)


def _shrink(recent: pd.Series, baseline: pd.Series, games: pd.Series) -> pd.Series:
    """Pull thin samples toward the season number.

    A hitter with 10 recent games is trusted fully; one with 2 games is mostly judged
    on his season profile. Prevents a single hot night from topping the board.
    """
    w = (games.fillna(0) / FULL_WINDOW).clip(MIN_SHRINK, 1.0)
    base = baseline.fillna(recent.median())
    return recent.fillna(base) * w + base * (1 - w)


def _weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Percentile-weight a set of columns into a 0-100 score.

    Columns absent from the frame are skipped and the divisor shrinks with them, so a
    missing input dilutes the score rather than silently scoring everyone zero.
    """
    score = pd.Series(0.0, index=df.index)
    used = 0.0
    for col, w in weights.items():
        if col not in df.columns:
            continue
        score += _pct(df[col]) * w
        used += abs(w)
    return (100 * score / used).round(1) if used else pd.Series(0.0, index=df.index)


def _score_total_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Better of the power path and the volume path — see the WEIGHTS note."""
    block = df.copy()
    power = _weighted_score(block, WEIGHTS["total_bases_power"])
    volume = _weighted_score(block, WEIGHTS["total_bases_volume"])

    block["tb_power_score"] = power
    block["tb_volume_score"] = volume
    # Which route this hitter is actually taking, so the client can see whether a pick
    # is a slugger or a contact bat rather than inferring it from the name.
    block["tb_path"] = (power >= volume).map({True: "power", False: "volume"})
    block["score"] = pd.concat([power, volume], axis=1).max(axis=1).round(1)
    block["tb_strict"] = block["score"] >= TB_STRICT_THRESHOLD
    block["prop"] = "total_bases"
    return block


def score_batter_props(matchups: pd.DataFrame, rolling: pd.DataFrame,
                       season: pd.DataFrame, team_offense: pd.DataFrame | None = None,
                       window: str = "L10", profiles: pd.DataFrame | None = None,
                       pitchers: pd.DataFrame | None = None,
                       hr_matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score hits / total bases / home runs / RBIs / runs for every hitter today."""

    r = rolling[(rolling.window == window) & (rolling.split == "all")].copy()
    r["recent_hit_rate"] = r["hits"] / r["pa"].replace(0, pd.NA)
    r["recent_tb_rate"] = r["total_bases"] / r["pa"].replace(0, pd.NA)
    keep = ["player_id", "games", "pa", "recent_hit_rate", "recent_tb_rate",
            "barrel_pct", "hard_hit_pct", "xwoba_contact", "home_runs"]
    df = matchups.merge(r[keep].rename(columns={
        "games": "recent_games", "pa": "recent_pa",
        "barrel_pct": "recent_barrel_pct", "hard_hit_pct": "recent_hard_hit",
        "home_runs": "recent_hr"}), on="player_id", how="left")

    s = season.rename(columns={"est_ba": "season_est_ba", "est_slg": "season_est_slg",
                               "est_woba": "season_est_woba"})
    df = df.merge(s[["player_id", "season_est_ba", "season_est_slg", "season_est_woba"]],
                  on="player_id", how="left")

    if team_offense is not None:
        df = df.merge(team_offense, on="team_id", how="left")
    if "team_offense" not in df.columns:
        df["team_offense"] = df.get("season_est_woba")

    # shrink recent form toward season profile where samples are thin
    df["recent_hit_rate"] = _shrink(df["recent_hit_rate"], df["season_est_ba"], df["recent_games"])
    df["recent_tb_rate"] = _shrink(df["recent_tb_rate"], df["season_est_slg"], df["recent_games"])

    # --- v2 hit-model inputs -------------------------------------------------
    # Recent xwOBA on contact is the form signal; the season figure anchors it when
    # the recent sample is thin.
    df["recent_xwoba"] = _shrink(df.get("xwoba_contact"), df["season_est_woba"],
                                 df["recent_games"])

    if profiles is not None and not profiles.empty:
        df = df.merge(profiles, on="player_id", how="left")

    # Line drives and sweet-spot contact both describe balls that fall in. Combined
    # as percentiles rather than raw numbers because they sit on different scales.
    if {"line_drive_rate", "sweet_spot_pct"} <= set(df.columns):
        df["ld_sweet_index"] = (_pct(df["line_drive_rate"])
                                + _pct(df["sweet_spot_pct"])) / 2

    # A starter who puts runners on allows hits. Joined on the opposing starter so
    # every hitter in a lineup shares the pitcher they actually face.
    if pitchers is not None and not pitchers.empty:
        w = pitchers.rename(columns={"player_id": "opp_starter_id",
                                     "whip": "starter_whip"})
        # Take every pitcher column, not just WHIP: the handedness splits live here
        # too, and an explicit column list silently dropped them.
        df = df.merge(w, on="opp_starter_id", how="left")

    # --- pitcher split matched to the side this hitter actually stands on --------
    # A switch hitter bats opposite the arm he faces, so his platoon split is decided
    # by the starter rather than by him. Roughly one lineup slot in eight is a switch
    # hitter, and applying a fixed side would hand all of them the wrong half.
    if {"bats", "opp_starter_throws"} <= set(df.columns):
        df["stands"] = [effective_bat_side(b, t)
                        for b, t in zip(df["bats"], df["opp_starter_throws"])]
        if {"slg_allowed_vs_L", "slg_allowed_vs_R"} <= set(df.columns):
            df["starter_slg_allowed_vs_hand"] = df["slg_allowed_vs_R"].where(
                df["stands"] == "R", df["slg_allowed_vs_L"])

    # Weighted matchup matrix, joined on the hitter AND the starter he faces — the
    # value is specific to that pairing, not a property of either alone.
    if hr_matrix is not None and not hr_matrix.empty:
        df = df.merge(hr_matrix, on=["player_id", "opp_starter_id"], how="left")

    # --- client-facing grades ----------------------------------------------------
    # A letter for how the hitter handles this starter's actual mix. The underlying
    # number is an expected wOBA against a weighted arsenal, which is precise but not
    # scannable; the client wants to glance down a column and see the shape of a slate.
    if "matchup_est_woba" in df.columns:
        q = _pct(df["matchup_est_woba"])
        df["pitch_matchup_grade"] = pd.cut(
            q, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=["D", "C", "B", "A", "A+"]).astype("object")

    # A starter in the top tier at preventing barrels is a warning on any power pick,
    # so it is surfaced as a flag rather than left buried inside the matchup number.
    if "starter_barrel_pct_allowed" in df.columns:
        low = _pct(df["starter_barrel_pct_allowed"]) <= 0.25
        df["pitcher_barrel_suppression_flag"] = low.map(
            {True: "SUPPRESSES BARRELS", False: ""})

    # lineup-slot values
    idx = (df["batting_order"].fillna(9).astype(int) - 1).clip(0, 8)
    df["lineup_spot"] = idx.map(lambda i: SPOT_PA[i])
    df["rbi_spot"] = idx.map(lambda i: SPOT_RBI[i])
    df["run_spot"] = idx.map(lambda i: SPOT_RUN[i])

    # a starter who gives up hard contact is good news for HR props
    df["starter_hr_prone"] = df["starter_est_woba_allowed"]

    out = [_score_total_bases(df)]
    for prop in BATTER_PROPS:
        weights = WEIGHTS[prop]
        score = pd.Series(0.0, index=df.index)
        for col, w in weights.items():
            if col not in df.columns:
                continue
            score += _pct(df[col]) * w
        total_w = sum(abs(w) for c, w in weights.items() if c in df.columns)
        block = df.copy()
        block["prop"] = prop
        block["score"] = (100 * score / total_w).round(1) if total_w else 0.0

        # Ground-ball hitters bleed multi-hit upside into outs and double plays.
        # Applied after weighting, as a deduction, so it stays visible as its own
        # number rather than disappearing inside a percentile.
        # A pitcher who does not allow barrels caps the home-run outcome regardless of
        # how the hitter grades. Applied as a ceiling, not a deduction: the point is
        # that the ceiling is lower, not that the hitter is worse.
        if prop == "home_runs" and "starter_barrel_pct_allowed" in block.columns:
            suppresses = _pct(block["starter_barrel_pct_allowed"]) <= 0.25
            block["hr_suppression_capped"] = suppresses
            block.loc[suppresses, "score"] = block.loc[suppresses, "score"].clip(
                upper=HR_SUPPRESSION_CAP)

        if prop == "hits" and "ground_ball_rate" in block.columns:
            block["gb_penalty"] = gb_penalty(block["ground_ball_rate"]).round(1)
            block["score"] = (block["score"]
                              - block["gb_penalty"]).clip(lower=0).round(1)

        out.append(block)

    res = pd.concat(out, ignore_index=True)
    res["rank"] = res.groupby("prop")["score"].rank(ascending=False, method="min").astype(int)
    return res.sort_values(["prop", "score"], ascending=[True, False]).reset_index(drop=True)


def score_pitcher_strikeouts(schedule: pd.DataFrame, pitcher_rolling: pd.DataFrame,
                             lineups: pd.DataFrame, batter_rolling: pd.DataFrame,
                             window: str = "L10") -> pd.DataFrame:
    """Score strikeout props for today's probable starters."""

    starters = []
    for _, g in schedule.iterrows():
        for side, opp in (("home", "away"), ("away", "home")):
            pid = g[f"{side}_probable_id"]
            if pd.isna(pid):
                continue
            starters.append({
                "game_pk": g["game_pk"], "player_id": int(pid),
                "player_name": g[f"{side}_probable"], "team": g[f"{side}_team"],
                "team_id": g[f"{side}_team_id"], "opp_team_id": g[f"{opp}_team_id"],
                "opponent": g[f"{opp}_team"],
            })
    df = pd.DataFrame(starters)
    if df.empty:
        return df

    p = pitcher_rolling[(pitcher_rolling.window == window)
                        & (pitcher_rolling.split == "all")].copy()
    df = df.merge(p[["player_id", "games", "k_per_game", "k_pct", "whiff_pct",
                     "pitches_per_game", "batters_faced"]].rename(columns={
        "games": "recent_games", "k_per_game": "recent_k_per_game",
        "k_pct": "recent_k_pct", "whiff_pct": "recent_whiff_pct",
        "pitches_per_game": "recent_pitches_per_game"}), on="player_id", how="left")

    # how strikeout-prone is the lineup he faces?
    b = batter_rolling[(batter_rolling.window == window)
                       & (batter_rolling.split == "all")][["player_id", "k_pct"]]
    opp_k = (lineups.merge(b, on="player_id", how="left")
                    .groupby("team_id")["k_pct"].mean()
                    .rename("opp_lineup_k_pct").reset_index())
    df = df.merge(opp_k, left_on="opp_team_id", right_on="team_id",
                  how="left", suffixes=("", "_drop"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])

    weights = WEIGHTS["strikeouts"]
    score = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col in df.columns:
            score += _pct(df[col]) * w
    total_w = sum(abs(w) for c, w in weights.items() if c in df.columns)

    df["prop"] = "strikeouts"
    df["score"] = (100 * score / total_w).round(1) if total_w else 0.0
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def score_game_totals(schedule: pd.DataFrame, matchups: pd.DataFrame,
                      bullpen: pd.DataFrame, park_factors: pd.DataFrame,
                      rolling: pd.DataFrame, window: str = "L10") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score team totals and combined game totals.

    Returns (team_totals, game_totals).

    Unlike player props this is built bottom-up from the same matchup numbers: a team's
    run expectation is its own hitters against the opposing starter, adjusted for how
    much bullpen that opponent has left and how the park plays.
    """
    # --- offence: average matchup quality of each lineup ------------------------
    off = (matchups.groupby(["game_pk", "team_id", "team"], as_index=False)
                   .agg(lineup_matchup_woba=("matchup_est_woba", "mean"),
                        opp_starter_weak=("starter_est_woba_allowed", "mean"),
                        slots=("player_id", "count")))

    # --- recent team form from rolling player data ------------------------------
    r = rolling[(rolling.window == window) & (rolling.split == "all")]
    form = matchups[["team_id", "player_id"]].merge(
        r[["player_id", "total_bases", "pa"]], on="player_id", how="left")
    form = (form.groupby("team_id", as_index=False)
                .apply(lambda g: pd.Series({
                    "recent_team_form": g["total_bases"].sum() / g["pa"].sum()
                    if g["pa"].sum() else None}), include_groups=False)
                .reset_index(drop=True))
    form["team_id"] = sorted(matchups["team_id"].unique())
    off = off.merge(form, on="team_id", how="left")

    # --- bullpen: how many arms does each team have left ------------------------
    if bullpen is not None and not bullpen.empty:
        pen = (bullpen.assign(avail=lambda d: d.availability.eq("available"))
                      .groupby("team_id", as_index=False)
                      .agg(relievers=("player_id", "count"),
                           available=("avail", "sum")))
        # fewer available arms -> more runs allowed later in the game
        pen["bullpen_tired"] = 1 - (pen["available"] / pen["relievers"].replace(0, pd.NA))
    else:
        pen = pd.DataFrame(columns=["team_id", "bullpen_tired"])

    # --- assemble per-team rows, attaching the OPPONENT's pen and the park ------
    rows = []
    park = park_factors.copy()
    for _, g in schedule.iterrows():
        pf = park[park.venue_name == g["venue"]]
        # Savant has no factors for brand-new or temporary venues (e.g. the Athletics'
        # Sutter Health Park). Falling back to neutral is reasonable, but it must be
        # visible in the output rather than silently pretending the park is average.
        park_runs = int(pf["index_runs"].iloc[0]) if len(pf) else 100
        park_hr = int(pf["index_hr"].iloc[0]) if len(pf) else 100
        park_matched = bool(len(pf))
        for side, opp in (("home", "away"), ("away", "home")):
            rows.append({
                "game_pk": g["game_pk"], "venue": g["venue"],
                "team_id": g[f"{side}_team_id"], "team": g[f"{side}_team"],
                "opponent": g[f"{opp}_team"], "opp_team_id": g[f"{opp}_team_id"],
                "opp_starter": g[f"{opp}_probable"],
                "park_runs": park_runs, "park_hr": park_hr,
                "park_matched": park_matched,
                "home_away": side,
            })
    tt = pd.DataFrame(rows).merge(off.drop(columns=["team"]), on=["game_pk", "team_id"], how="left")
    tt = tt.merge(pen[["team_id", "bullpen_tired"]].rename(
        columns={"team_id": "opp_team_id", "bullpen_tired": "opp_bullpen_tired"}),
        on="opp_team_id", how="left")

    w = WEIGHTS["team_total"]
    s = pd.Series(0.0, index=tt.index)
    for col, weight in w.items():
        if col in tt.columns:
            s += _pct(tt[col]) * weight
    tw = sum(abs(v) for c, v in w.items() if c in tt.columns)
    tt["prop"] = "team_total"
    tt["score"] = (100 * s / tw).round(1) if tw else 0.0
    tt["rank"] = tt["score"].rank(ascending=False, method="min").astype(int)
    tt = tt.sort_values("score", ascending=False).reset_index(drop=True)

    # --- combine the two halves of each game ------------------------------------
    gt = (tt.groupby(["game_pk", "venue", "park_runs", "park_hr"], as_index=False)
            .agg(combined_offense=("lineup_matchup_woba", "sum"),
                 combined_starter_weak=("opp_starter_weak", "sum"),
                 combined_bullpen_tired=("opp_bullpen_tired", "sum"),
                 park_matched=("park_matched", "first")))

    # Build the "Away @ Home" label from the schedule itself. Deriving it from group
    # order silently reverses fixtures — it put the Braves away at their own park.
    label = schedule.assign(
        teams=schedule["away_team"] + " @ " + schedule["home_team"]
    )[["game_pk", "teams", "game_time_utc"]]
    gt = gt.merge(label, on="game_pk", how="left")

    w = WEIGHTS["game_total"]
    s = pd.Series(0.0, index=gt.index)
    for col, weight in w.items():
        if col in gt.columns:
            s += _pct(gt[col]) * weight
    tw = sum(abs(v) for c, v in w.items() if c in gt.columns)
    gt["prop"] = "game_total"
    gt["score"] = (100 * s / tw).round(1) if tw else 0.0
    gt["rank"] = gt["score"].rank(ascending=False, method="min").astype(int)
    gt = gt.sort_values("score", ascending=False).reset_index(drop=True)

    return tt, gt
