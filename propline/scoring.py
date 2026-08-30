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
from .profiles import context_multiplier, gb_penalty
from .weather import wind_description

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
    # v2 runs and RBIs. The client's point: both are bottlenecked by teammates. An
    # elite hitter with nobody on base in front of him has nobody to drive in, and no
    # amount of individual quality fixes that. So the weights below score the hitter,
    # and a bounded lineup-context multiplier is applied afterwards — his spec asks for
    # a multiplier, not another weighted term, because the effect is conditional rather
    # than additive.
    "rbis": {
        "matchup_est_slg": 0.35,     # driving runners in is a slugging skill
        "recent_tb_rate": 0.25,
        "rbi_spot": 0.25,            # 3-4-5 hitters bat with men on
        "season_est_slg": 0.15,
    },
    "runs": {
        "matchup_est_woba": 0.35,    # you cannot score without reaching base
        "run_spot": 0.25,            # 1-2-3 hitters score most
        "recent_hit_rate": 0.25,
        "matchup_k_pct": -0.15,
    },
    # team runs in a single game
    "team_total": {
        "lineup_matchup_woba": 0.32,   # how the lineup fares vs the opposing starter
        "opp_starter_weak": 0.23,      # what that starter gives up generally
        "opp_pen_workload": 0.20,      # a worked pen means worse innings 6-9
        "opp_starter_whip": 0.05,      # baserunners allowed -> more scoring chances
        "park_runs": 0.10,
        "recent_team_form": 0.10,
    },
    # combined runs, both teams
    #
    # The negative K/9 term is the client's "pitching duel" fix. Two aces facing each
    # other suppress a total in a way the offence and park terms cannot see: strikeouts
    # remove balls in play entirely, so the lineups never get the contact that the
    # matchup numbers assume. Weighted modestly — a duel is a real trap but it is one
    # signal among several, and starters leave in the sixth.
    "game_total": {
        "combined_offense": 0.34,
        "combined_starter_weak": 0.22,
        "combined_pen_workload": 0.20,
        "park_runs": 0.12,
        "combined_starter_k9": -0.12,
    },
    # pitcher side
    # v2 strikeouts, to the client's spec. His diagnosis was the "vacuum fallacy":
    # v1 judged a pitcher on his own numbers and treated every opposing lineup as
    # league average, so control arms facing swing-happy lineups were underrated
    # (his examples: Logan Webb and Shane Bieber, both graded low and both delivering)
    # while raw stuff was overrated against contact lineups.
    "strikeouts": {
        "split_k_matchup": 0.35,      # his K% by hand, weighted by TODAY's lineup
        "whiff_14day": 0.25,          # current stuff, not a season average
        "opp_lineup_k_pct": 0.25,     # who is actually standing in the box
        "whip_efficiency": 0.15,      # traffic means pitch count means an early hook
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

# WHIP bands, from the client's spec. Below the upper bound a starter is efficient
# enough to work deep; above WHIP_PENALTY_ABOVE the pitch count climbs and the hook
# comes early, which costs outs regardless of strikeout rate.
WHIP_EFFICIENT_HI = 1.25
WHIP_PENALTY_ABOVE = 1.40

# Leash risk, applied as a deduction rather than a weight: a starter who will not see
# the fifth inning cannot reach a strikeout number however good his rates are.
SHORT_LEASH_PITCHES = 80
LEASH_PENALTY_MAX = 20.0

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
                       hr_matrix: pd.DataFrame | None = None,
                       lineup_ctx: pd.DataFrame | None = None) -> pd.DataFrame:
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

    # Who bats around each hitter. Joined on the lineup slot he occupies today, not on
    # the player, because the same hitter has different men in front of him if the
    # manager moves him.
    if lineup_ctx is not None and not lineup_ctx.empty:
        df = df.merge(lineup_ctx, on=["game_pk", "team_id", "player_id"], how="left")

    # His spec asks for the batting slot under its own name, and for the starter's
    # vulnerability to contact as a word rather than a percentage — both are for
    # scanning a sheet quickly, not for the maths.
    df["lineup_position"] = df["batting_order"]
    if "starter_whiff" in df.columns:
        q = _pct(df["starter_whiff"])
        # low whiff rate = easier to make contact against = favourable for a hit
        df["starter_whiff_matchup"] = pd.cut(
            q, [-0.01, 0.33, 0.66, 1.01],
            labels=["Favorable", "Neutral", "Tough"]).astype("object")

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
        # Lineup context. RBIs depend on who is ON BASE ahead of the hitter; runs
        # depend on the lineup turning over and scoring at all, which the top of the
        # order drives. Bounded to +/-15%: teammates matter, but they cannot make a
        # hitter a different player.
        if prop == "rbis" and "table_setter_obp" in block.columns:
            block["context_mult"] = context_multiplier(block["table_setter_obp"])
            block["score"] = (block["score"] * block["context_mult"]).round(1)
        elif prop == "runs" and "top_order_obp" in block.columns:
            block["context_mult"] = context_multiplier(block["top_order_obp"])
            block["score"] = (block["score"] * block["context_mult"]).round(1)

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


def _whip_efficiency(whip: pd.Series) -> pd.Series:
    """Turn WHIP into a 0-1 efficiency score for strikeout purposes.

    The client's reasoning: traffic on base runs up the pitch count, and a starter at
    95 pitches in the fifth gets pulled before he can accumulate strikeouts. So being
    efficient helps here, even though a high WHIP briefly inflates strikeouts per
    inning. Flat at the top — no extra credit for a 0.90 WHIP over a 1.20 — then
    falling away above his threshold.
    """
    w = pd.to_numeric(whip, errors="coerce")
    eff = 1.0 - ((w - WHIP_EFFICIENT_HI) / 0.5)   # 1.25 -> 1.0, ~1.75 -> 0
    return eff.clip(lower=0.0, upper=1.0).fillna(0.5)


def _leash_penalty(pitches_per_game: pd.Series) -> pd.Series:
    """Points deducted for a starter who will not be out there long enough.

    Scaled rather than a cliff: 79 pitches an outing is not the same risk as 55.
    """
    p = pd.to_numeric(pitches_per_game, errors="coerce")
    short = (SHORT_LEASH_PITCHES - p).clip(lower=0)
    return (short / SHORT_LEASH_PITCHES * LEASH_PENALTY_MAX).clip(
        upper=LEASH_PENALTY_MAX).fillna(0)


def score_pitcher_strikeouts(schedule: pd.DataFrame, pitcher_rolling: pd.DataFrame,
                             lineups: pd.DataFrame, batter_rolling: pd.DataFrame,
                             window: str = "L10",
                             pitcher_days: pd.DataFrame | None = None,
                             pitchers: pd.DataFrame | None = None,
                             opp_k: pd.DataFrame | None = None,
                             lineup_hand: pd.DataFrame | None = None,
                             pitcher_hand: dict[int, str] | None = None) -> pd.DataFrame:
    """Score strikeout props for today's probable starters.

    v2 rebuilds this around the client's "vacuum fallacy" point: v1 judged a pitcher on
    his own rate stats and implicitly treated every opposing lineup as league average.
    That underrated control arms facing swing-happy lineups — his examples were Logan
    Webb and Shane Bieber, both graded low and both delivering — and overrated raw stuff
    against contact lineups.
    """
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

    if pitcher_hand:
        df["throws"] = df["player_id"].map(pitcher_hand)

    # --- the pitcher's own recent form ------------------------------------------
    p = pitcher_rolling[(pitcher_rolling.window == window)
                        & (pitcher_rolling.split == "all")]
    df = df.merge(p[["player_id", "games", "k_per_game", "k_pct", "whiff_pct",
                     "pitches_per_game", "batters_faced"]].rename(columns={
        "games": "recent_games", "k_per_game": "recent_k_per_game",
        "k_pct": "recent_k_pct", "whiff_pct": "recent_whiff_pct",
        "pitches_per_game": "recent_pitches_per_game"}), on="player_id", how="left")

    # --- his K% by batter hand, weighted by the lineup he actually faces ---------
    for side in ("L", "R"):
        part = pitcher_rolling[(pitcher_rolling.window == window)
                               & (pitcher_rolling.split == f"vs{side}")]
        if part.empty:
            continue
        df = df.merge(part[["player_id", "k_pct"]].rename(
            columns={"k_pct": f"k_pct_vs{side}"}), on="player_id", how="left")

    if lineup_hand is not None and not lineup_hand.empty:
        df = df.merge(lineup_hand, on="opp_team_id", how="left")

    if {"k_pct_vsL", "k_pct_vsR", "lineup_share_L", "lineup_share_R"} <= set(df.columns):
        # A platoon-heavy arm facing a lineup stacked against him is a different bet
        # from the same arm facing one stacked in his favour, and a single season K%
        # cannot say so.
        df["split_k_matchup"] = (
            df["k_pct_vsL"].fillna(df["recent_k_pct"]) * df["lineup_share_L"].fillna(0.5)
            + df["k_pct_vsR"].fillna(df["recent_k_pct"]) * df["lineup_share_R"].fillna(0.5)
        ).round(1)

    # --- 14-day whiff, per the spec ---------------------------------------------
    if pitcher_days is not None and not pitcher_days.empty:
        d = pitcher_days[pitcher_days.split == "all"]
        df = df.merge(d[["player_id", "whiff_pct"]].rename(
            columns={"whiff_pct": "whiff_14day"}), on="player_id", how="left")
    if "whiff_14day" not in df.columns:
        df["whiff_14day"] = df.get("recent_whiff_pct")

    # --- WHIP and the opposing lineup -------------------------------------------
    if pitchers is not None and not pitchers.empty:
        df = df.merge(pitchers[["player_id", "whip"]].rename(
            columns={"whip": "pitcher_whip"}), on="player_id", how="left")
        df["whip_efficiency"] = _whip_efficiency(df["pitcher_whip"])

    if opp_k is not None and not opp_k.empty:
        df = df.merge(opp_k, on="opp_team_id", how="left")
    else:
        b = batter_rolling[(batter_rolling.window == window)
                           & (batter_rolling.split == "all")][["player_id", "k_pct"]]
        agg = (lineups.merge(b, on="player_id", how="left")
                      .groupby("team_id")["k_pct"].mean()
                      .rename("opp_lineup_k_pct").reset_index()
                      .rename(columns={"team_id": "opp_team_id"}))
        df = df.merge(agg, on="opp_team_id", how="left")

    # --- score -------------------------------------------------------------------
    df["score"] = _weighted_score(df, WEIGHTS["strikeouts"])

    # Leash risk is a deduction, not a weight: no strikeout rate survives being pulled
    # in the fourth inning.
    if "recent_pitches_per_game" in df.columns:
        df["leash_penalty"] = _leash_penalty(df["recent_pitches_per_game"]).round(1)
        df["expected_pitch_limit"] = pd.to_numeric(
            df["recent_pitches_per_game"], errors="coerce").round(0)
        df["score"] = (df["score"] - df["leash_penalty"]).clip(lower=0).round(1)

    # A letter for the handedness matchup, so a slate can be scanned rather than read.
    if "split_k_matchup" in df.columns:
        q = _pct(df["split_k_matchup"])
        df["split_k_rate_matchup"] = pd.cut(
            q, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
            labels=["D", "C", "B", "A", "A+"]).astype("object")

    df["prop"] = "strikeouts"
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
def score_game_totals(schedule: pd.DataFrame, matchups: pd.DataFrame,
                      bullpen: pd.DataFrame, park_factors: pd.DataFrame,
                      rolling: pd.DataFrame, window: str = "L10",
                      pen_workload: pd.DataFrame | None = None,
                      pitchers: pd.DataFrame | None = None,
                      weather: pd.DataFrame | None = None,
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score team totals and combined game totals.

    Returns (team_totals, game_totals).

    Unlike player props this is built bottom-up from the same matchup numbers: a team's
    run expectation is its own hitters against the opposing starter, adjusted for how
    much bullpen that opponent has left and how the park plays.

    v2 addresses the client's diagnosis that macro totals get "thrown off by late-inning
    bullpen collapses or elite pitching duels":

      * bullpen fatigue is now three-day pitch counts and innings for the relief unit
        (propline.profiles.bullpen_workload) rather than v1's count of arms that
        happened to pitch recently. Counting bodies treated a pen that threw 350
        pitches the same as one that threw 100 across the same number of appearances.
      * pitching duels are caught by a negative combined-K/9 term.
      * weather enters as a multiplier on the finished score, per the spec's wording,
        so it nudges an ordering built on baseball rather than competing with it.
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

    # --- bullpen: how hard has each relief unit been worked ---------------------
    # Prefer the v2 workload model. Fall back to v1's arm count only if it is missing,
    # so a slate can still be scored when the boxscore pull comes up short.
    if pen_workload is not None and not pen_workload.empty:
        pen = pen_workload.copy()
    elif bullpen is not None and not bullpen.empty:
        from .profiles import bullpen_workload
        pen = bullpen_workload(bullpen)
    else:
        pen = pd.DataFrame(columns=["team_id", "pen_workload", "pen_status"])

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
                "opp_starter_id": g.get(f"{opp}_probable_id"),
                "park_runs": park_runs, "park_hr": park_hr,
                "park_matched": park_matched,
                "home_away": side,
            })
    tt = pd.DataFrame(rows).merge(off.drop(columns=["team"]), on=["game_pk", "team_id"], how="left")

    # The pen that matters to a team's total is the OPPONENT's — they are the arms it
    # will face from the sixth inning on.
    pen_cols = [c for c in ("pen_workload", "pen_status", "pen_pitches_3d",
                            "pen_innings_3d", "pen_unavailable") if c in pen.columns]
    if pen_cols:
        renames = {"team_id": "opp_team_id"}
        renames.update({c: f"opp_{c}" for c in pen_cols})
        tt = tt.merge(pen[["team_id"] + pen_cols].rename(columns=renames),
                      on="opp_team_id", how="left")

    # Opposing starter's WHIP and K/9, the "pitching duel" inputs.
    if pitchers is not None and not pitchers.empty and "opp_starter_id" in tt.columns:
        keep = [c for c in ("whip", "k_per_9") if c in pitchers.columns]
        if keep:
            tt = tt.merge(
                pitchers[["player_id"] + keep].rename(columns={
                    "player_id": "opp_starter_id", "whip": "opp_starter_whip",
                    "k_per_9": "opp_starter_k9"}),
                on="opp_starter_id", how="left")

    # Weather, attached before scoring so both boards share one source.
    wx_cols = ["game_pk", "temp_f", "wind_mph", "wind_dir_deg", "precip_pct",
               "weather_mult", "indoor", "roof_type"]
    if weather is not None and not weather.empty:
        tt = tt.merge(weather[[c for c in wx_cols if c in weather.columns]],
                      on="game_pk", how="left")
        tt["weather_mult"] = pd.to_numeric(
            tt.get("weather_mult"), errors="coerce").fillna(1.0)
        tt["wind"] = [wind_description(a, b)
                      for a, b in zip(tt.get("wind_mph", pd.Series(dtype=float)),
                                      tt.get("wind_dir_deg", pd.Series(dtype=float)))]
    else:
        tt["weather_mult"] = 1.0

    # The opposing starter, as the client asked to see him: WHIP and K/9 together, so
    # a duel is obvious at a glance without opening another tab.
    tt["starter_whip_k9"] = [
        _whip_k9_label(a, b)
        for a, b in zip(tt.get("opp_starter_whip", pd.Series(dtype=float)),
                        tt.get("opp_starter_k9", pd.Series(dtype=float)))]

    tt["prop"] = "team_total"
    tt["score"] = _apply_weather(_weighted_score(tt, WEIGHTS["team_total"]),
                                 tt["weather_mult"])
    tt["rank"] = tt["score"].rank(ascending=False, method="min").astype(int)
    tt = tt.sort_values("score", ascending=False).reset_index(drop=True)

    # --- combine the two halves of each game ------------------------------------
    agg = {"combined_offense": ("lineup_matchup_woba", "sum"),
           "combined_starter_weak": ("opp_starter_weak", "sum"),
           "park_matched": ("park_matched", "first"),
           "weather_mult": ("weather_mult", "first")}
    # Summing each side's OPPONENT pen covers both relief units exactly once.
    if "opp_pen_workload" in tt.columns:
        agg["combined_pen_workload"] = ("opp_pen_workload", "sum")
    if "opp_starter_k9" in tt.columns:
        # MEAN, not sum. A sum treats an unresolved starter as a zero-strikeout pitcher:
        # rookies below the 20-IP cutoff in pitcher_stats have no K/9, which halved the
        # combined figure and — because this term is weighted negatively — pushed those
        # games UP the board. Houston @ Mets ranked first on exactly that artefact.
        # Averaging the starters we do have leaves the estimate honest either way.
        agg["combined_starter_k9"] = ("opp_starter_k9", "mean")
        agg["starters_resolved"] = ("opp_starter_k9", "count")
    if "opp_starter_whip" in tt.columns:
        agg["combined_starter_whip"] = ("opp_starter_whip", "mean")
    gt = tt.groupby(["game_pk", "venue", "park_runs", "park_hr"],
                    as_index=False).agg(**agg)

    # Build the "Away @ Home" label from the schedule itself. Deriving it from group
    # order silently reverses fixtures — it put the Braves away at their own park.
    label = schedule.assign(
        teams=schedule["away_team"] + " @ " + schedule["home_team"]
    )[["game_pk", "teams", "game_time_utc"]]
    gt = gt.merge(label, on="game_pk", how="left")

    if weather is not None and not weather.empty:
        gt = gt.merge(weather[[c for c in wx_cols if c in weather.columns
                               and c != "weather_mult"]], on="game_pk", how="left")
        gt["wind"] = [wind_description(a, b)
                      for a, b in zip(gt.get("wind_mph", pd.Series(dtype=float)),
                                      gt.get("wind_dir_deg", pd.Series(dtype=float)))]

    # Per-side detail the client asked for by name: pen status for each club, and both
    # starters' WHIP/K9 side by side.
    gt = _attach_game_sides(gt, tt, schedule)

    gt["prop"] = "game_total"
    gt["score"] = _apply_weather(_weighted_score(gt, WEIGHTS["game_total"]),
                                 gt["weather_mult"])
    gt["rank"] = gt["score"].rank(ascending=False, method="min").astype(int)
    gt = gt.sort_values("score", ascending=False).reset_index(drop=True)

    return tt, gt


def _whip_k9_label(whip, k9) -> str:
    """"1.12 WHIP / 9.4 K9" — one glanceable cell instead of two columns."""
    if pd.isna(whip) and pd.isna(k9):
        return ""
    w = "?" if pd.isna(whip) else f"{float(whip):.2f}"
    k = "?" if pd.isna(k9) else f"{float(k9):.1f}"
    return f"{w} WHIP / {k} K9"


def _apply_weather(score: pd.Series, mult) -> pd.Series:
    """Nudge a finished score by the weather multiplier, keeping the 0-100 scale.

    Clipped at 100: the multiplier can only reorder and shade the board, never push a
    game off the top of the scale that every other prop is read against.
    """
    m = pd.to_numeric(mult, errors="coerce").fillna(1.0)
    return (score * m).clip(0, 100).round(1)


def _attach_game_sides(gt: pd.DataFrame, tt: pd.DataFrame,
                       schedule: pd.DataFrame) -> pd.DataFrame:
    """Home/away bullpen status and both starters' WHIP-K9 line."""
    side = tt[["game_pk", "team_id", "home_away"]].copy()
    for col, out in (("opp_pen_status", "pen_status"),
                     ("starter_whip_k9", "starter_whip_k9")):
        if col in tt.columns:
            side[out] = tt[col].values

    for home_away in ("home", "away"):
        part = side[side["home_away"] == home_away]
        if part.empty:
            continue
        # A team's own pen status is carried on the OTHER side's row (each row holds
        # its opponent's pen), so flip the label back to the club it belongs to.
        flip = {"home": "away", "away": "home"}[home_away]
        src = side[side["home_away"] == flip]
        if "pen_status" in src.columns:
            gt = gt.merge(src[["game_pk", "pen_status"]].rename(
                columns={"pen_status": f"pen_status_{home_away}"}),
                on="game_pk", how="left")
        if "starter_whip_k9" in part.columns:
            # The starter a side FACES is that side's opponent's starter, which is the
            # value already on this row.
            gt = gt.merge(part[["game_pk", "starter_whip_k9"]].rename(
                columns={"starter_whip_k9": f"starter_{flip}_whip_k9"}),
                on="game_pk", how="left")

    home_col, away_col = "starter_home_whip_k9", "starter_away_whip_k9"
    if home_col in gt.columns and away_col in gt.columns:
        gt["starter_whip_k9"] = [
            " vs ".join([x for x in (h, a) if x]) or ""
            for h, a in zip(gt[home_col].fillna(""), gt[away_col].fillna(""))]

    # Bullpen status spelled out with the club attached. The rationale model is given
    # this rather than the bare home/away labels: handed pen_status_home alongside a
    # "Away @ Home" string it has no way to tell which club is which, and it duly
    # reported a rested Nationals bullpen on a night Washington's pen was overworked.
    if {"pen_status_home", "pen_status_away"} <= set(gt.columns):
        names = schedule[["game_pk", "home_team", "away_team"]]
        gt = gt.merge(names, on="game_pk", how="left")
        gt["pen_summary"] = [
            "; ".join(part for part in (
                f"{ht} bullpen {hs}" if pd.notna(hs) and str(hs) != "nan" else "",
                f"{at} bullpen {as_}" if pd.notna(as_) and str(as_) != "nan" else "",
            ) if part)
            for ht, hs, at, as_ in zip(gt["home_team"], gt["pen_status_home"],
                                       gt["away_team"], gt["pen_status_away"])]
    return gt


def attach_market_edge(totals: pd.DataFrame, vegas: pd.DataFrame) -> pd.DataFrame:
    """Put the market line beside the model score and measure the disagreement.

    The client asked for a direct variance — "model projects 9.4 runs vs Vegas 8.5,
    +0.9 edge". This model does not produce a run projection: the score is a
    percentile ranking across the slate, not an expected number of runs, and
    inventing a runs figure by rescaling a percentile would look precise while
    meaning nothing.

    What IS honest, and answers the same question, is comparing RANKS: where does
    this game sit on the model board, versus where its total sits on the market
    board. A game the model ranks in the 90th percentile that the market ranks in
    the 40th is a disagreement worth looking at, and that is the signal he is after.

    market_edge is positive when the model likes a game more than the market does.
    """
    if totals.empty or vegas is None or vegas.empty:
        return totals

    cols = [c for c in ("game_pk", "vegas_total", "books") if c in vegas.columns]
    out = totals.merge(vegas[cols].drop_duplicates("game_pk"), on="game_pk", how="left")

    if "vegas_total" in out.columns and out["vegas_total"].notna().any():
        model_pct = _pct(out["score"])
        market_pct = _pct(out["vegas_total"])
        out["market_edge"] = (100 * (model_pct - market_pct)).round(0)
        # A word, because a signed percentile difference is not self-explanatory.
        out["vs_vegas"] = pd.cut(
            out["market_edge"], [-101, -20, 20, 101],
            labels=["market higher", "agree", "model higher"]).astype("object")
    return out
