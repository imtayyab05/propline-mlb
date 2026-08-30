"""Ballpark weather at first pitch.

Open-Meteo: free, no API key, no advertised rate limit. That matters here — every other
external service in this project is on a metered budget (see the quota notes), and
weather is needed for every game every day.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not apply a wind multiplier. "Blowing out to centre" is the only wind fact that
moves run scoring much, and deciding whether a given wind is blowing out requires the
stadium's ORIENTATION — the compass bearing from home plate to centre field. The MLB API
does not publish it and this project has no reliable source for it, so applying a
direction-based multiplier would mean inventing thirty bearings and dressing guesswork as
physics.

Wind speed and direction ARE fetched and shown, so the client can apply his own judgement
on a park he knows. Only temperature drives the multiplier, which is the well-established
effect: warmer air is less dense, the ball carries further.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 45

# Roughly 70F is the neutral point; scoring rises with temperature as the ball carries.
# Bounded hard, because temperature is a real but small effect and should never swamp the
# lineup and pitching signals it sits beside.
TEMP_NEUTRAL_F = 70.0
TEMP_PER_DEGREE = 0.0015          # ~1.5% per 10F
TEMP_MULT_MIN, TEMP_MULT_MAX = 0.96, 1.04

# A closed roof means the weather outside is irrelevant. Retractable is treated as closed
# rather than open: whether it is actually shut on the night is not published anywhere we
# can read, and assuming "open" would apply a bogus adjustment to a controlled climate.
INDOOR_ROOFS = {"dome", "retractable", "closed", "indoor"}


def venue_locations(venue_ids, session=None) -> pd.DataFrame:
    """Coordinates and roof type per venue, straight from the MLB API."""
    session = session or requests.Session()
    ids = sorted({int(v) for v in venue_ids if pd.notna(v)})
    if not ids:
        return pd.DataFrame()

    r = session.get(f"{MLB_API}/venues",
                    params={"venueIds": ",".join(map(str, ids)),
                            "hydrate": "location,fieldInfo"}, timeout=TIMEOUT)
    r.raise_for_status()

    rows = []
    for v in r.json().get("venues", []):
        coords = ((v.get("location") or {}).get("defaultCoordinates") or {})
        roof = ((v.get("fieldInfo") or {}).get("roofType") or "")
        rows.append({
            "venue_id": v.get("id"),
            "venue": v.get("name"),
            "lat": coords.get("latitude"),
            "lon": coords.get("longitude"),
            "roof_type": roof,
            "indoor": roof.strip().lower() in INDOOR_ROOFS,
        })
    return pd.DataFrame(rows)


def _venues_by_name(names, session=None) -> pd.DataFrame:
    """Resolve venues by name, for slates collected without venue ids."""
    session = session or requests.Session()
    r = session.get(f"{MLB_API}/venues",
                    params={"sportId": 1, "hydrate": "location,fieldInfo"},
                    timeout=TIMEOUT)
    r.raise_for_status()

    wanted = {str(n).strip().lower() for n in names}
    rows = []
    for v in r.json().get("venues", []):
        if str(v.get("name", "")).strip().lower() not in wanted:
            continue
        coords = ((v.get("location") or {}).get("defaultCoordinates") or {})
        roof = ((v.get("fieldInfo") or {}).get("roofType") or "")
        rows.append({"venue": v.get("name"), "venue_id": v.get("id"),
                     "lat": coords.get("latitude"), "lon": coords.get("longitude"),
                     "roof_type": roof,
                     "indoor": roof.strip().lower() in INDOOR_ROOFS})

    found = {r_["venue"].strip().lower() for r_ in rows}
    missing = wanted - found
    if missing:
        print(f"  WARN  no venue match for: {', '.join(sorted(missing))}")
    return pd.DataFrame(rows)


def _hour_index(times: list[str], target: datetime) -> int | None:
    """Index of the forecast hour closest to first pitch."""
    best, best_gap = None, None
    for i, t in enumerate(times):
        try:
            when = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        gap = abs((when - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best, best_gap = i, gap
    return best


def fetch_weather(schedule: pd.DataFrame, cache_dir, session=None) -> pd.DataFrame:
    """Conditions at first pitch for every game on the slate.

    Cached per slate like the odds are — the forecast for tonight does not change enough
    between the morning and evening runs to be worth re-fetching five times.
    """
    cache = Path(cache_dir) / "weather.json"
    if cache.exists():
        with open(cache, encoding="utf-8") as fh:
            df = pd.DataFrame(json.load(fh))
        print(f"  ok    weather: reusing today's cache ({len(df)} games)")
        return df

    if schedule.empty:
        return pd.DataFrame()

    session = session or requests.Session()

    if "venue_id" in schedule.columns and schedule["venue_id"].notna().any():
        venues = venue_locations(schedule["venue_id"].dropna().unique(), session)
        join_on = "venue_id"
    else:
        # A slate collected before venue_id was carried on the schedule. Fall back to
        # the venue NAME so those days still get weather, and say so out loud —
        # returning an empty frame here once made weather look wired up when every
        # game was silently scoring at a neutral 1.0 multiplier.
        print("  WARN  schedule has no venue_id; matching venues by name instead")
        venues = _venues_by_name(schedule["venue"].dropna().unique(), session)
        join_on = "venue"

    if venues.empty:
        print("  WARN  no venue coordinates resolved — totals scored without weather")
        return pd.DataFrame()

    keep = [join_on] + [c for c in ("lat", "lon", "roof_type", "indoor")
                        if c in venues.columns]
    sched = schedule.merge(venues[keep].drop_duplicates(join_on),
                           on=join_on, how="left", suffixes=("", "_v"))

    rows = []
    for _, g in sched.iterrows():
        row = {"game_pk": g["game_pk"], "venue": g.get("venue"),
               "roof_type": g.get("roof_type"), "indoor": bool(g.get("indoor"))}

        if g.get("indoor") or pd.isna(g.get("lat")):
            # No weather adjustment for a controlled climate, and none possible without
            # coordinates. Neutral rather than missing, so the multiplier still applies.
            row.update({"temp_f": None, "wind_mph": None, "wind_dir_deg": None,
                        "precip_pct": None, "weather_mult": 1.0})
            rows.append(row)
            continue

        try:
            r = session.get(OPEN_METEO, params={
                "latitude": g["lat"], "longitude": g["lon"],
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m,"
                          "precipitation_probability",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "UTC", "forecast_days": 2}, timeout=TIMEOUT)
            r.raise_for_status()
            h = r.json()["hourly"]

            start = pd.to_datetime(g.get("game_time_utc"), utc=True, errors="coerce")
            idx = _hour_index(h["time"], start.to_pydatetime()) if pd.notna(start) else 0
            if idx is None:
                raise ValueError("no matching forecast hour")

            temp = h["temperature_2m"][idx]
            row.update({
                "temp_f": temp,
                "wind_mph": h["wind_speed_10m"][idx],
                "wind_dir_deg": h["wind_direction_10m"][idx],
                "precip_pct": h["precipitation_probability"][idx],
                "weather_mult": temperature_multiplier(temp),
            })
        except Exception as exc:  # noqa: BLE001 — weather must never sink a slate
            print(f"  WARN  weather unavailable for {g.get('venue')}: {exc}")
            row.update({"temp_f": None, "wind_mph": None, "wind_dir_deg": None,
                        "precip_pct": None, "weather_mult": 1.0})
        rows.append(row)

    out = pd.DataFrame(rows)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(out.to_dict("records"), fh)

    warm = out["temp_f"].notna().sum()
    print(f"  ok    weather: {len(out)} games ({warm} outdoor, "
          f"{int(out['indoor'].sum())} indoor)")
    return out


def temperature_multiplier(temp_f) -> float:
    """Warmer air is thinner and the ball carries. Bounded to +/-4%."""
    if temp_f is None or pd.isna(temp_f):
        return 1.0
    mult = 1.0 + (float(temp_f) - TEMP_NEUTRAL_F) * TEMP_PER_DEGREE
    return round(min(max(mult, TEMP_MULT_MIN), TEMP_MULT_MAX), 3)


def wind_description(speed_mph, direction_deg) -> str:
    """Plain-English wind, with no claim about whether it helps or hurts.

    Direction is given as the compass point the wind blows FROM, which is the
    meteorological convention and the one the client will recognise from a forecast.
    """
    if speed_mph is None or pd.isna(speed_mph):
        return ""
    if float(speed_mph) < 5:
        return "calm"
    points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((float(direction_deg or 0) + 22.5) % 360 // 45)
    return f"{round(float(speed_mph))} mph from {points[idx]}"
