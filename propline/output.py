"""Client-facing Excel output — one tab per prop category.

This is the file the client actually opens each morning, so it is written for a human
reading it over coffee, not for a machine: plain column names, best picks at the top,
and the lineup status visible on every row so a projected pick is never mistaken for
a confirmed one.

Each prop gets its OWN columns. v1 showed the same seven fields everywhere, which was
fine when every category was scored the same way. In v2 they are not: total bases
takes the better of two paths, home runs weigh a matchup matrix and a barrel cap,
strikeouts are about the lineup a pitcher faces. Showing a shared column set would
hide precisely the numbers that drive each board.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Shown on every player-prop tab, before and after the prop-specific columns.
HEAD = [
    ("rank", "#"),
    ("player_name", "Player"),
    ("team", "Team"),
    ("lineup_position", "Bat"),
    ("opp_starter", "Opposing SP"),
    ("score", "Score"),
    ("status", "Lineup"),
    ("recent_games", "Recent G"),
]
TAIL = [("rationale", "Why")]

# The prop-specific middle. Names match the client's requested column list where he
# gave one, so his spec and the sheet can be read side by side.
PROP_MIDDLE = {
    "hits": [
        ("recent_xwoba", "Recent xwOBA"),
        ("contact_rate_recent", "Contact% (14d)"),
        ("contact_rate", "Contact%"),
        ("line_drive_rate", "Line Drive%"),
        ("sweet_spot_pct", "Sweet Spot%"),
        ("ground_ball_rate", "Ground Ball%"),
        ("gb_penalty", "GB Penalty"),
        ("starter_whip", "SP WHIP"),
        ("starter_whiff_matchup", "SP Whiff Matchup"),
    ],
    "total_bases": [
        ("tb_path", "Path"),
        ("tb_power_score", "Power Score"),
        ("tb_volume_score", "Volume Score"),
        ("tb_strict", "Strict 2+"),
        ("iso_recent_14day", "ISO (14d)"),
        ("iso_season", "ISO (season)"),
        ("recent_barrel_pct", "Barrel%"),
        ("starter_slg_allowed_vs_hand", "SP SLG Allowed (his side)"),
        ("pitch_matchup_grade", "Pitch Matchup"),
    ],
    "home_runs": [
        ("hr_matchup_rv", "Arsenal Run Value"),
        ("recent_barrel_pct", "Barrel%"),
        ("fly_ball_rate", "Fly Ball%"),
        ("ground_ball_rate", "Ground Ball%"),
        ("iso_recent_14day", "ISO (14d)"),
        ("recent_hard_hit", "Hard Hit%"),
        ("pitcher_barrel_suppression_flag", "Barrel Suppression"),
        ("hr_suppression_capped", "Capped"),
        ("pitch_matchup_grade", "Pitch Matchup"),
    ],
    "rbis": [
        ("table_setter_obp", "Table-Setter OBP"),
        ("context_mult", "Lineup Context"),
        ("matchup_est_slg", "vs Arsenal xSLG"),
        ("recent_tb_rate", "TB/PA"),
        ("pitch_matchup_grade", "Pitch Matchup"),
    ],
    "runs": [
        ("top_order_obp", "Top-Order OBP"),
        ("table_setter_obp", "Table-Setter OBP"),
        ("context_mult", "Lineup Context"),
        ("matchup_est_woba", "vs Arsenal xwOBA"),
        ("pitch_matchup_grade", "Pitch Matchup"),
    ],
}

# Strikeouts is a pitcher board, so it does not share the hitter header at all.
PITCHER_COLS = [
    ("rank", "#"),
    ("player_name", "Pitcher"),
    ("team", "Team"),
    ("opponent", "Opponent"),
    ("score", "Score"),
    ("split_k_rate_matchup", "Split K Matchup"),
    ("split_k_matchup", "Split K%"),
    ("whiff_14day", "Whiff% (14d)"),
    ("opp_lineup_k_pct", "Opp Lineup K%"),
    ("pitcher_whip", "WHIP"),
    ("whip_efficiency", "WHIP Efficiency"),
    ("vegas_k_line", "Vegas K Line"),
    ("expected_pitch_limit", "Expected Pitches"),
    ("leash_penalty", "Leash Penalty"),
    ("recent_games", "Recent G"),
    ("rationale", "Why"),
]

TEAM_TOTAL_COLS = [
    ("rank", "#"), ("team", "Team"), ("opponent", "Opponent"),
    ("opp_starter", "Opposing SP"), ("score", "Score"),
    ("lineup_matchup_woba", "Lineup vs SP xwOBA"),
    ("opp_starter_weak", "SP xwOBA Allowed"),
    ("opp_bullpen_tired", "Opp Pen Tired"),
    ("park_runs", "Park Runs"), ("venue", "Venue"), ("rationale", "Why"),
]

GAME_TOTAL_COLS = [
    ("rank", "#"), ("teams", "Matchup"), ("venue", "Venue"), ("score", "Score"),
    ("vegas_total", "Vegas Total"),
    ("vs_vegas", "vs Vegas"),
    ("market_edge", "Model-Market Gap"),
    ("park_runs", "Park Runs"), ("park_matched", "Park Data"),
    ("combined_offense", "Combined Offense"),
    ("combined_starter_weak", "Combined SP Weakness"),
    ("combined_bullpen_tired", "Combined Pen Tired"), ("rationale", "Why"),
]

SHEET_TITLES = {
    "hits": "Hits",
    "total_bases": "Total Bases",
    "home_runs": "Home Runs",
    "rbis": "RBIs",
    "runs": "Runs",
    "strikeouts": "Strikeouts",
}


def columns_for(prop: str) -> list[tuple[str, str]]:
    """Full column spec for one prop tab."""
    if prop == "strikeouts":
        return PITCHER_COLS
    return HEAD + PROP_MIDDLE.get(prop, []) + TAIL


def _shape(df: pd.DataFrame, spec, top_n=None) -> pd.DataFrame:
    cols = [(src, label) for src, label in spec if src in df.columns]
    out = df[[src for src, _ in cols]].copy()
    out.columns = [label for _, label in cols]
    if top_n:
        out = out.head(top_n)
    return out


def build_picks_workbook(batter_scores: pd.DataFrame, pitcher_scores: pd.DataFrame,
                         out_path, run_meta: dict, top_n: int = 40,
                         team_totals: pd.DataFrame | None = None,
                         game_totals: pd.DataFrame | None = None) -> Path:
    """Write the finished picks workbook. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        # Summary first — the client sees run freshness before any picks
        meta = pd.DataFrame(list(run_meta.items()), columns=["Field", "Value"])
        meta.to_excel(xl, sheet_name="Summary", index=False)

        for prop, title in SHEET_TITLES.items():
            if prop == "strikeouts":
                src = pitcher_scores
            else:
                src = (batter_scores[batter_scores.prop == prop]
                       if not batter_scores.empty else batter_scores)
            if src is None or src.empty:
                continue
            _shape(src.sort_values("score", ascending=False),
                   columns_for(prop), top_n).to_excel(xl, sheet_name=title, index=False)

        # game/team totals are slate-wide, so never truncated to top_n
        for frame, spec, title in ((game_totals, GAME_TOTAL_COLS, "Game Totals"),
                                   (team_totals, TEAM_TOTAL_COLS, "Team Totals")):
            if frame is not None and not frame.empty:
                shaped = _shape(frame.sort_values("score", ascending=False), spec)
                shaped.to_excel(xl, sheet_name=title, index=False)

        # auto-size columns so nothing is cut off on open
        for ws in xl.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 9), 46)

    return out_path
