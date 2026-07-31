# Baseball Savant — Verified Endpoint Spec

All endpoints below were probed live against the 2026 season on 2026-07-31 and returned
HTTP 200 with headers **matching the client's manually downloaded CSVs exactly**.
Public, no auth, no login. Append `&csv=true` to any leaderboard to get CSV instead of HTML.

Base: `https://baseballsavant.mlb.com`

---

## 1. Season-level leaderboards (the 16 files the client downloads by hand)

| # | Client's file | Path | Key params | Notes |
|---|---|---|---|---|
| 1 | `batter/pitcher stats` | `/leaderboard/custom` | `year`, `type=batter\|pitcher`, `min=q`, `selections=<csv of metrics>` | Column set is chosen via `selections`; must be pinned to match his file |
| 2 | `batter/pitcher expected_stats` | `/leaderboard/expected_statistics` | `year`, `type=batter\|pitcher`, `filterType=bip`, `min=q` | |
| 3 | `league batter/pitcher expected_stats` | `/leaderboard/expected_statistics` | `type=batter-team\|pitcher-team` | Returns exactly 30 rows (team-level) ✔ |
| 4 | `batter/pitcher exit_velocity` | `/leaderboard/statcast` | `year`, `type`, `min=q` | |
| 5 | `batter/pitcher percentile_rankings` | `/leaderboard/percentile-rankings` | `year`, `type` | |
| 6 | `batter/pitcher bat-tracking` | `/leaderboard/bat-tracking` | see §2 | |
| 7 | `bat-tracking-swing-path` | `/leaderboard/bat-tracking/swing-path-attack-angle` | see §2 | Note the nested path |
| 8 | `batter/pitcher batted-ball` | `/leaderboard/batted-ball` | `year`, `type`, `min=q` | **301 redirect — client must follow redirects** |
| 9 | `pitcher pitch_arsenals` | `/leaderboard/pitch-arsenals` | `year`, `min=100`, `type=avg_speed`, `hand=` | |
| 10 | `pitch-arsenal-stats` | `/leaderboard/pitch-arsenal-stats` | `year`, `type=pitcher\|batter`, `min=10`, `pitchType=` | Per-pitch whiff%/put-away% — backbone of the matchup engine |

---

## 2. Date ranges + handedness — CRITICAL FINDING

**Only the bat-tracking family honours date and handedness filters.**

Supported on `/leaderboard/bat-tracking` and `/leaderboard/bat-tracking/swing-path-attack-angle`:

- `dateStart=YYYY-MM-DD` / `dateEnd=YYYY-MM-DD` — **verified working**
  (Caminero: 686 competitive swings season-long vs 60 over the last 10 days)
- `pitchHand=L|R` — batter performance vs LHP/RHP
- `batSide=L|R` — pitcher performance vs LHB/RHB
- `pitchType=FF,SI,SL,...` — this is the manual checkbox panel the client fills in by hand;
  it is just a query param, so **that step automates away completely**
- `minSwings`, `minGroupSwings`, `team`, `gameType`, `count`, `attackZone`, `isHardHit`, `groupBy`

**All other leaderboards silently IGNORE `dateStart`/`dateEnd`.** They return season-to-date
only. Verified: expected_statistics with and without a 10-day window returns a byte-identical
row for James Wood (518 PA both times). No error is raised — it just quietly gives you the
full season, which is the dangerous kind of failure.

### Consequence for L5 / L10

The client explicitly wants L5/L10 alongside season for xwOBA, exit velo, barrel% etc.
Those cannot come from the leaderboards. They must be computed from raw pitch-level data:

```
/statcast_search/csv?all=true&hfSea=2026|&hfGT=R|
    &game_date_gt=YYYY-MM-DD&game_date_lt=YYYY-MM-DD
    &player_type=batter&min_pitches=0&min_results=0&type=details
```

Verified: 2 days of games = 7,704 rows / ~5 MB. Columns include `pitch_type`, `game_date`,
`player_name`, `batter`, `pitcher`, `events`, `stand`, `p_throws`, `launch_speed`,
`estimated_woba_using_speedangle`.

So the pipeline aggregates rolling windows itself from pitch-level data. This is more work
than a leaderboard pull but it is the only correct route, and it has a large upside: once
raw pitch data is in Postgres, **any** split (L5, L10, vs LHP, vs a specific pitch type,
by count) becomes a query rather than another HTTP request.

---

## 2b. Team-level grouping is spelled differently per endpoint

| Endpoint | Correct team value |
|---|---|
| `/leaderboard/expected_statistics` | `type=batter-team` / `pitcher-team` |
| `/leaderboard/bat-tracking/swing-path-attack-angle` | `type=batting-team` / `pitching-team` |

Passing the wrong spelling does **not** error — it quietly returns ~600 player rows
instead of 30 team rows. Row-count assertions in the collector catch this.

---

## 3. ID normalisation required

The ID column is named differently across files and must be unified to `player_id`
(Savant = MLBAM ID) before anything is written to the database:

| File family | ID column |
|---|---|
| bat-tracking, swing-path, batted-ball | `id` |
| expected_stats, exit_velocity, custom stats | `player_id` + `"last_name, first_name"` |
| percentile_rankings | `player_id` + `player_name` |
| pitch_arsenals | `pitcher` |
| statcast_search (raw) | `batter` / `pitcher` |
| team-level files | `team_id` (numeric, e.g. 110 = Orioles) vs `team` abbrev (`BAL`) elsewhere |

Other cleaning needed: numerics arrive quoted as strings, `NaN` appears as a literal string,
`whiffs` is empty in bat-tracking, and `percentile_rankings` is ~60% blank rows for
low-PA players.
