"""Collection-stage output: one tidy Excel workbook from the day's raw pulls.

This is the hand-off between the collection script and the processing script, per
the requirements doc. It is deliberately NOT the client-facing picks file — it is
the cleaned raw material, with IDs normalised so downstream joins just work.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Savant names the same MLBAM id differently per file. Normalise to `player_id`.
ID_ALIASES = ("id", "player_id", "pitcher", "batter")
NAME_ALIASES = ("name", "player_name", "last_name, first_name")

# Excel caps sheet names at 31 chars
MAX_SHEET = 31


def normalise_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever id/name column this file happens to use."""
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    if "player_id" not in df.columns:
        for alias in ID_ALIASES:
            if alias in cols:
                df = df.rename(columns={cols[alias]: "player_id"})
                break

    if "player_name" not in df.columns:
        for alias in NAME_ALIASES:
            if alias in cols and cols[alias] != "player_id":
                df = df.rename(columns={cols[alias]: "player_name"})
                break

    # Literal "NaN" strings arrive in several Savant exports
    return df.replace({"NaN": pd.NA, "nan": pd.NA})


def _sheet_name(stem: str, used: set[str]) -> str:
    name = stem[:MAX_SHEET]
    i = 1
    while name in used:
        suffix = f"~{i}"
        name = stem[: MAX_SHEET - len(suffix)] + suffix
        i += 1
    used.add(name)
    return name


def build_intermediate(raw_dir, out_path, extra: dict[str, pd.DataFrame] | None = None):
    """Combine every CSV in raw_dir (+ any extra frames) into one workbook.

    Returns a manifest of what went in, which doubles as the collection run log.
    """
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for csv in sorted(raw_dir.glob("*.csv")):
        # raw pitch-level data is far too large for Excel and is not meant for it
        if csv.stem.startswith("statcast_raw"):
            continue
        frames[csv.stem] = normalise_ids(pd.read_csv(csv, low_memory=False))

    for name, df in (extra or {}).items():
        frames[name] = normalise_ids(df)

    manifest = []
    used: set[str] = set()
    with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
        for stem, df in frames.items():
            sheet = _sheet_name(stem, used)
            df.to_excel(xl, sheet_name=sheet, index=False)
            manifest.append({"source": stem, "sheet": sheet,
                             "rows": len(df), "cols": df.shape[1],
                             "has_player_id": "player_id" in df.columns})

    return pd.DataFrame(manifest)
