"""Savant pull definitions.

Every entry here was probed live against the 2026 season and returns headers matching
the client's manually downloaded CSVs. See docs/savant_endpoints.md for the evidence.

The important distinction encoded below is `windowed`: only the bat-tracking family
honours dateStart/dateEnd. Everything else silently returns season-to-date regardless
of what dates you pass, so we never pretend otherwise.
"""

from __future__ import annotations

BASE = "https://baseballsavant.mlb.com"

# Column set for the /leaderboard/custom "statistics tab" export, mirroring the
# client's batter stats / pitcher stats files.
CUSTOM_SELECTIONS = ",".join([
    "player_age", "p_game", "p_formatted_ip", "pa", "ab", "hit", "single", "double",
    "triple", "home_run", "strikeout", "walk", "k_percent", "bb_percent", "batting_avg",
    "slg_percent", "on_base_percent", "on_base_plus_slg", "xba", "xslg", "woba", "xwoba",
    "xobp", "xiso", "avg_swing_speed", "fast_swing_rate", "blasts_contact", "blasts_swing",
    "squared_up_contact", "squared_up_swing", "avg_swing_length", "swords", "attack_angle",
    "attack_direction", "ideal_angle_rate", "vertical_swing_path", "exit_velocity_avg",
    "launch_angle_avg", "sweet_spot_percent", "barrel_batted_rate", "hard_hit_percent",
    "avg_best_speed", "avg_hyper_speed", "whiff_percent", "swing_percent",
])


class Pull:
    """One downloadable Savant leaderboard."""

    def __init__(self, name, path, params, windowed=False, handed=False):
        self.name = name
        self.path = path
        self.params = params
        # windowed: endpoint genuinely respects dateStart/dateEnd
        self.windowed = windowed
        # handed: endpoint accepts pitchHand / batSide splits
        self.handed = handed

    def url_params(self, year, date_start=None, date_end=None, hand=None, hand_key=None):
        p = {"csv": "true", **self.params}
        p.setdefault("year", year)
        if self.windowed and date_start and date_end:
            p["dateStart"] = date_start
            p["dateEnd"] = date_end
        if self.handed and hand and hand_key:
            p[hand_key] = hand
        return p


# --- Season-level leaderboards -------------------------------------------------

PULLS: list[Pull] = [
    # statistics tab
    Pull("batter_stats", "/leaderboard/custom",
         {"type": "batter", "filter": "", "min": "q", "selections": CUSTOM_SELECTIONS,
          "sort": "pa", "sortDir": "desc"}),
    Pull("pitcher_stats", "/leaderboard/custom",
         {"type": "pitcher", "filter": "", "min": "q", "selections": CUSTOM_SELECTIONS,
          "sort": "pa", "sortDir": "desc"}),

    # expected stats — player and team level
    Pull("batter_expected_stats", "/leaderboard/expected_statistics",
         {"type": "batter", "position": "", "team": "", "filterType": "bip", "min": "q"}),
    Pull("pitcher_expected_stats", "/leaderboard/expected_statistics",
         {"type": "pitcher", "position": "", "team": "", "filterType": "bip", "min": "q"}),
    Pull("league_batter_expected_stats", "/leaderboard/expected_statistics",
         {"type": "batter-team", "filterType": "bip", "min": "q"}),
    Pull("league_pitcher_expected_stats", "/leaderboard/expected_statistics",
         {"type": "pitcher-team", "filterType": "bip", "min": "q"}),

    # exit velocity / barrels
    Pull("batter_exit_velocity", "/leaderboard/statcast",
         {"type": "batter", "position": "", "team": "", "min": "q"}),
    Pull("pitcher_exit_velocity", "/leaderboard/statcast",
         {"type": "pitcher", "position": "", "team": "", "min": "q"}),

    # percentile rankings
    Pull("batter_percentile_rankings", "/leaderboard/percentile-rankings", {"type": "batter"}),
    Pull("pitcher_percentile_rankings", "/leaderboard/percentile-rankings", {"type": "pitcher"}),

    # batted ball profiles (NOTE: 301 redirect — session must follow redirects)
    Pull("batter_batted_ball", "/leaderboard/batted-ball", {"type": "batter", "min": "q", "team": ""}),
    Pull("pitcher_batted_ball", "/leaderboard/batted-ball", {"type": "pitcher", "min": "q", "team": ""}),

    # pitch arsenals
    Pull("pitcher_pitch_arsenals", "/leaderboard/pitch-arsenals",
         {"min": "100", "type": "avg_speed", "hand": ""}),
    Pull("pitcher_pitch_arsenal_stats", "/leaderboard/pitch-arsenal-stats",
         {"type": "pitcher", "pitchType": "", "team": "", "min": "10"}),
    Pull("batter_pitch_arsenal_stats", "/leaderboard/pitch-arsenal-stats",
         {"type": "batter", "pitchType": "", "team": "", "min": "10"}),

    # bat tracking family — the only endpoints that honour dates + handedness.
    # minSwings/minGroupSwings replace `min` here, and pitchType="" means ALL pitch
    # types, which is the checkbox panel the client currently ticks by hand.
    Pull("batter_bat_tracking", "/leaderboard/bat-tracking",
         {"type": "batter", "minSwings": "q", "minGroupSwings": "1", "pitchType": "",
          "gameType": "", "groupBy": "", "team": ""},
         windowed=True, handed=True),
    Pull("pitcher_bat_tracking", "/leaderboard/bat-tracking",
         {"type": "pitcher", "minSwings": "q", "minGroupSwings": "1", "pitchType": "",
          "gameType": "", "groupBy": "", "team": ""},
         windowed=True, handed=True),
    Pull("batter_swing_path", "/leaderboard/bat-tracking/swing-path-attack-angle",
         {"type": "batter", "minSwings": "q", "pitchType": "", "team": ""},
         windowed=True, handed=True),
    # NOTE: this endpoint spells team grouping "batting-team"/"pitching-team", while
    # /leaderboard/expected_statistics above spells it "batter-team"/"pitcher-team".
    # Savant is inconsistent here; both spellings are verified correct for their own
    # endpoint and passing the wrong one silently returns player rows instead.
    Pull("team_batting_swing_path", "/leaderboard/bat-tracking/swing-path-attack-angle",
         {"type": "batting-team", "minSwings": "q", "pitchType": "", "team": ""},
         windowed=True, handed=True),
    Pull("team_pitching_swing_path", "/leaderboard/bat-tracking/swing-path-attack-angle",
         {"type": "pitching-team", "minSwings": "q", "pitchType": "", "team": ""},
         windowed=True, handed=True),
]

# For batters we split by the pitcher's hand; for pitchers by the batter's hand.
HAND_KEY = {"batter": "pitchHand", "pitcher": "batSide"}

# Raw pitch-level search — the route to L5/L10 and any arbitrary split.
STATCAST_SEARCH = "/statcast_search/csv"


def statcast_search_params(date_start, date_end, season, player_type="batter"):
    return {
        "all": "true",
        "hfSea": f"{season}|",
        "hfGT": "R|",
        "game_date_gt": date_start,
        "game_date_lt": date_end,
        "player_type": player_type,
        "min_pitches": "0",
        "min_results": "0",
        "type": "details",
    }
