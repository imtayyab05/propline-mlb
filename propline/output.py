"""Client-facing Excel output — one tab per prop category.

This is the file the client actually opens each morning, so it is written for a human
reading it over coffee, not for a machine: plain column names, best picks at the top,
and the lineup status visible on every row so a projected pick is never mistaken for
a confirmed one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BATTER_COLS = [
    ("rank", "#"),
    ("player_name", "Player"),
    ("team", "Team"),
    ("batting_order", "Bat"),
    ("opp_starter", "Opposing SP"),
    ("score", "Score"),
    ("status", "Lineup"),
    ("recent_games", "Recent G"),
    ("matchup_est_woba", "vs Arsenal xwOBA"),
    ("matchup_est_slg", "vs Arsenal xSLG"),
    ("recent_barrel_pct", "Barrel%"),
    ("recent_hard_hit", "HardHit%"),
    ("primary_pitch", "SP Main Pitch"),
    ("best_pitch_for_batter", "Best Pitch For Him"),
    ("best_pitch_est_woba", "xwOBA On It"),
    ("arsenal_coverage", "Data Cover"),
    ("rationale", "Why"),
]

PITCHER_COLS = [
    ("rank", "#"),
    ("player_name", "Pitcher"),
    ("team", "Team"),
    ("opponent", "Opponent"),
    ("score", "Score"),
    ("recent_games", "Recent G"),
    ("recent_k_per_game", "K/Game"),
    ("recent_k_pct", "K%"),
    ("recent_whiff_pct", "Whiff%"),
    ("opp_lineup_k_pct", "Opp Lineup K%"),
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
                spec = PITCHER_COLS
            else:
                src = batter_scores[batter_scores.prop == prop] if not batter_scores.empty else batter_scores
                spec = BATTER_COLS
            if src is None or src.empty:
                continue
            _shape(src.sort_values("score", ascending=False), spec, top_n) \
                .to_excel(xl, sheet_name=title, index=False)

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
