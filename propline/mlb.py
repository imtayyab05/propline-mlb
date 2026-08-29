"""MLB Stats API — schedule, lineups (projected + confirmed), and bullpen usage.

Free, official, no key required.

Lineup reality: the API's `lineups` hydrate is empty until roughly 2-4 hours before
first pitch. That is a league-wide timing fact, not a limitation of this source. So we
run two modes, exactly as agreed in the requirements doc:

  confirmed  - real posted lineups, available in the afternoon window
  projected  - each team's most recent actual batting order, used as a stand-in
  scratched  - projected to start, absent from the lineup that actually posted

Every row carries a `status` column so a projected pick is never mistaken for a
confirmed one downstream.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

API = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 60
PAUSE = 0.25
LOOKBACK_DAYS = 10  # how far back to search for a team's last started game


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "PropLine-MLB/0.1"})
    return s


def _get(session, path, **params):
    r = session.get(f"{API}{path}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# --- schedule ------------------------------------------------------------------

def get_schedule(game_date, session=None) -> pd.DataFrame:
    """One row per game: teams, venue, start time, probable starters."""
    session = session or _session()
    data = _get(session, "/schedule", sportId=1, date=str(game_date),
                hydrate="probablePitcher,team,venue")

    rows = []
    for dt in data.get("dates", []):
        for g in dt.get("games", []):
            t = g["teams"]
            home, away = t["home"], t["away"]
            hp = home.get("probablePitcher") or {}
            ap = away.get("probablePitcher") or {}
            rows.append({
                "game_pk": g["gamePk"],
                "game_date": str(game_date),
                "game_time_utc": g.get("gameDate"),
                "status": g["status"]["detailedState"],
                "venue": (g.get("venue") or {}).get("name"),
                "home_team": home["team"]["name"],
                "home_team_id": home["team"]["id"],
                "away_team": away["team"]["name"],
                "away_team_id": away["team"]["id"],
                "home_probable_id": hp.get("id"),
                "home_probable": hp.get("fullName"),
                "away_probable_id": ap.get("id"),
                "away_probable": ap.get("fullName"),
            })
    return pd.DataFrame(rows)


# --- lineups -------------------------------------------------------------------

def _lineup_rows(game_pk, game_date, team_id, team_name, opponent_id, side,
                 player_ids, players_blob, status):
    out = []
    for slot, pid in enumerate(player_ids, start=1):
        p = players_blob.get(f"ID{pid}", {})
        out.append({
            "game_pk": game_pk,
            "game_date": game_date,
            "team_id": team_id,
            "team": team_name,
            "opponent_id": opponent_id,
            "home_away": side,
            "batting_order": slot,
            "player_id": pid,
            "player_name": (p.get("person") or {}).get("fullName"),
            "position": (p.get("position") or {}).get("abbreviation"),
            "status": status,
        })
    return out


def get_confirmed_lineups(game_date, session=None) -> pd.DataFrame:
    """Real posted lineups. Empty until ~2-4h before first pitch."""
    session = session or _session()
    data = _get(session, "/schedule", sportId=1, date=str(game_date), hydrate="lineups,team")

    rows = []
    for dt in data.get("dates", []):
        for g in dt.get("games", []):
            lu = g.get("lineups") or {}
            t = g["teams"]
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                players = lu.get(key) or []
                if not players:
                    continue
                other = "away" if side == "home" else "home"
                rows.extend(_lineup_rows(
                    g["gamePk"], str(game_date), t[side]["team"]["id"],
                    t[side]["team"]["name"], t[other]["team"]["id"], side,
                    [p["id"] for p in players],
                    {f"ID{p['id']}": {"person": p, "position": p.get("primaryPosition", {})}
                     for p in players},
                    "confirmed",
                ))
    return pd.DataFrame(rows)


def _starting_order(team_blob) -> list[int]:
    """The nine players who actually STARTED, in batting order.

    The boxscore's top-level `battingOrder` array lists the LAST occupant of each
    lineup slot, not the starter. MLB encodes each player's slot as `slot * 100`,
    with +1, +2... for substitutes who later filled that spot — so a pinch hitter or
    a double-switched reliever shows up as 704 and would otherwise be projected into
    tomorrow's lineup as a hitter. Only entries that are exact multiples of 100 are
    genuine starters.
    """
    starters = []
    for pid_key, p in (team_blob.get("players") or {}).items():
        raw = p.get("battingOrder")
        if raw is None:
            continue
        slot = int(raw)
        if slot % 100 == 0:
            starters.append((slot // 100, p["person"]["id"]))
    return [pid for _, pid in sorted(starters)]


def _recent_final_games(as_of, days, session):
    """All completed games in the window, most recent first."""
    start = (as_of - timedelta(days=days)).isoformat()
    data = _get(session, "/schedule", sportId=1, startDate=start,
                endDate=as_of.isoformat(), hydrate="team")
    games = []
    for dt in data.get("dates", []):
        for g in dt.get("games", []):
            if g["status"]["detailedState"] == "Final":
                games.append({"game_pk": g["gamePk"], "date": dt["date"],
                              "home_id": g["teams"]["home"]["team"]["id"],
                              "away_id": g["teams"]["away"]["team"]["id"]})
    return sorted(games, key=lambda x: x["date"], reverse=True)


def get_projected_lineups(game_date, schedule_df, session=None) -> pd.DataFrame:
    """Each team's most recent actual batting order, used as today's projection."""
    session = session or _session()
    as_of = datetime.strptime(str(game_date), "%Y-%m-%d").date()

    teams = {}
    for _, g in schedule_df.iterrows():
        teams[g["home_team_id"]] = (g["game_pk"], g["home_team"], g["away_team_id"], "home")
        teams[g["away_team_id"]] = (g["game_pk"], g["away_team"], g["home_team_id"], "away")

    recent = _recent_final_games(as_of - timedelta(days=1), LOOKBACK_DAYS, session)

    # most recent completed game per team
    last_game = {}
    for g in recent:
        for tid in (g["home_id"], g["away_id"]):
            if tid in teams and tid not in last_game:
                last_game[tid] = g

    rows = []
    for tid, (game_pk, team_name, opp_id, side) in teams.items():
        src = last_game.get(tid)
        if not src:
            continue
        box = _get(session, f"/game/{src['game_pk']}/boxscore")
        time.sleep(PAUSE)
        src_side = "home" if src["home_id"] == tid else "away"
        blob = box["teams"][src_side]
        order = _starting_order(blob)
        if not order:
            continue
        rows.extend(_lineup_rows(game_pk, str(game_date), tid, team_name, opp_id,
                                 side, order, blob.get("players", {}), "projected"))
    return pd.DataFrame(rows)


# A scratched player's row keeps its original batting slot plus this offset. The
# lineups table is keyed on (game_pk, team_id, batting_order), so a scratched hitter
# reusing slot 3 would collide with whoever actually bats third. Offsetting keeps the
# original slot readable (103 -> was batting 3rd) and sorts them below the real nine.
SCRATCH_SLOT_OFFSET = 100


def _scratched(confirmed: pd.DataFrame, projected: pd.DataFrame) -> pd.DataFrame:
    """Players we projected to start who are absent from the posted lineup.

    Previously these simply disappeared the moment a lineup confirmed, so a hitter who
    was ranked #1 in the morning silently left the board with no explanation. Recording
    them means a scratch is visible rather than inferred, and — more importantly — that
    they can be excluded from picks deliberately rather than by accident.
    """
    if confirmed.empty or projected.empty:
        return pd.DataFrame()

    rows = []
    for team_id, conf_team in confirmed.groupby("team_id"):
        proj_team = projected[projected.team_id == team_id]
        if proj_team.empty:
            continue
        missing = proj_team[~proj_team.player_id.isin(set(conf_team.player_id))].copy()
        if missing.empty:
            continue
        missing["status"] = "scratched"
        missing["batting_order"] = missing["batting_order"] + SCRATCH_SLOT_OFFSET
        rows.append(missing)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def get_lineups(game_date, schedule_df=None, session=None) -> pd.DataFrame:
    """Confirmed lineups where available, projected everywhere else.

    Confirmed always wins for a given team — we never let a projection overwrite a
    real posted lineup, which is the single worst failure mode for a betting tool.
    """
    session = session or _session()
    schedule_df = get_schedule(game_date, session) if schedule_df is None else schedule_df
    if schedule_df.empty:
        return pd.DataFrame()

    confirmed = get_confirmed_lineups(game_date, session)
    have = set(confirmed["team_id"]) if not confirmed.empty else set()

    projected = get_projected_lineups(game_date, schedule_df, session)

    # Anyone we projected for a team that has since posted a real lineup, and who is
    # not in it, has been scratched. Work this out before the projections are dropped.
    scratched = _scratched(confirmed, projected)

    if not projected.empty:
        projected = projected[~projected["team_id"].isin(have)]

    out = pd.concat([df for df in (confirmed, projected, scratched) if not df.empty],
                    ignore_index=True)
    return out.sort_values(["game_pk", "home_away", "batting_order"]).reset_index(drop=True)


def get_player_names(player_ids, session=None) -> dict[int, str]:
    """id -> name for any MLBAM ids, straight from the official people endpoint.

    Savant's leaderboards only cover players who clear minimum thresholds, so they
    miss position players who pitched an inning in a blowout. This does not.
    """
    session = session or _session()
    ids = sorted({int(i) for i in player_ids if pd.notna(i)})
    names: dict[int, str] = {}
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        data = _get(session, "/people", personIds=",".join(map(str, batch)))
        for p in data.get("people", []):
            names[int(p["id"])] = p.get("fullName")
        time.sleep(PAUSE)
    return names


# --- bullpen -------------------------------------------------------------------

def get_bullpen_usage(game_date, days=3, session=None) -> pd.DataFrame:
    """Relief appearances in the last N days -> simple availability flag.

    Deliberately a usage filter, not a fatigue model (per the requirements doc).
    A pitcher who threw yesterday or the day before is flagged likely_unavailable.
    """
    session = session or _session()
    as_of = datetime.strptime(str(game_date), "%Y-%m-%d").date()
    games = _recent_final_games(as_of - timedelta(days=1), days, session)

    rows = []
    for g in games:
        box = _get(session, f"/game/{g['game_pk']}/boxscore")
        time.sleep(PAUSE)
        for side in ("home", "away"):
            blob = box["teams"][side]
            starters = set(blob.get("pitchers", [])[:1])
            for pid in blob.get("pitchers", []):
                p = blob["players"].get(f"ID{pid}", {})
                st = (p.get("stats") or {}).get("pitching") or {}
                rows.append({
                    "team_id": blob["team"]["id"],
                    "team": blob["team"]["name"],
                    "player_id": pid,
                    "player_name": (p.get("person") or {}).get("fullName"),
                    "appearance_date": g["date"],
                    "is_starter": pid in starters,
                    "pitches": st.get("numberOfPitches") or 0,
                })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    relief = df[~df.is_starter]
    agg = (relief.groupby(["team_id", "team", "player_id", "player_name"])
                 .agg(last_appearance=("appearance_date", "max"),
                      appearances=("appearance_date", "count"),
                      pitches=("pitches", "sum"))
                 .reset_index())

    cutoff = (as_of - timedelta(days=2)).isoformat()
    agg["availability"] = agg["last_appearance"].apply(
        lambda d: "likely_unavailable" if d >= cutoff else "available")
    agg["as_of"] = str(game_date)
    return agg.sort_values(["team", "last_appearance"], ascending=[True, False])
