# PropLine MLB

Daily MLB prop-bet automation for Fiverr client **devinmajor**. $500 flat, paid 31 May 2026,
delivery agreed to **end of September 2026**. Full spec: `MLB_Prop_Automation_Requirements (1).docx`.

Replaces the client's manual ~20-minute morning routine (hand-downloading 15-20 Baseball
Savant CSVs, pasting into an AI chat) with an automated pipeline producing ranked prop picks
across seven categories: hits, total bases, strikeouts, home runs, RBIs, runs, game totals.

## Architecture (locked — mirrors the delivered LoL project)

```
collection script -> intermediate Excel -> processing script -> Supabase -> web dashboard
   (GitHub Actions, 2x daily + manual "Update Now" via workflow_dispatch)
```

- Scoring is a **formula-based model**, not an LLM. Groq is used only to write short
  plain-English rationale text for the final ~20-40 pick shortlist, one batched call/day.
- Lineups run **projected (morning) then confirmed (~2-4h pre-first-pitch)**. Every pick
  carries a projected/confirmed tag. Confirmed never gets overwritten by a projection.
- Deferred to v1.1: weather, injuries, Power BI.

## Environment

Python is **not on PATH**. Use the Anaconda interpreter directly:

```
D:/Programming/Anaconda/python.exe
```

Python 3.13.5 — pandas, requests, numpy, openpyxl, sqlalchemy, dotenv present.
`psycopg2` and `supabase` still need installing for the database phase.

Note: Git Bash `/tmp` is not visible to Windows Python. Use the scratchpad dir for temp files.

## Status

**Phase 1 (collection) — COMPLETE.** One command:

```bash
D:/Programming/Anaconda/python.exe scripts/collect.py --date 2026-07-31
```

Produces `data/intermediate/collection_<date>.xlsx` (44 sheets): 20 Savant leaderboards,
windowed L5/L10 splits vs LHP/RHP, rolling per-player aggregates, schedule, lineups,
bullpen availability.

**Phase 2 (processing) — COMPLETE.**

```bash
D:/Programming/Anaconda/python.exe scripts/process.py --date 2026-07-31
```

Produces `data/picks/props_<date>.xlsx` — Summary tab plus one tab per prop category.
Everything: pitcher L5/L10, matchup engine (arsenal-weighted), scoring for all six
player-prop categories, park factors, team + game totals, Groq rationale, client-facing
Excel (9 sheets), and the Supabase write. Verified end to end on the 2026-07-31 slate.

Supabase gotchas hit during setup, both of which fail loudly but confusingly:
- "Automatically expose new tables" is OFF on this project (the safer choice). That also
  means new tables get NO grants, not even for service_role — every write 403s with
  "42501 permission denied" until the grant block at the bottom of db/schema.sql runs.
- Any column pandas widened to float (ids with missing values) is rejected by Postgres
  as `invalid input syntax for type bigint: "701542.0"`. `_ints()` in publish.py casts
  every id column to nullable Int64 before sending. Add new id columns to INT_COLS.

Groq free tier is 12,000 tokens/minute. Firing all prop categories back to back trips it;
`_call()` honours the retry-after and there is a 2s gap between calls.

Park factors are scraped from an embedded `var data` JSON blob — the park-factors page
has no CSV export. Savant covers 29 venues; temporary parks (e.g. the Athletics' Sutter
Health Park) have none and fall back to neutral 100, flagged by `park_matched` so the
default is visible rather than silent.

**Phase 3 (frontend)** — not started.

**Open question — the scoring weights.** `WEIGHTS` in `propline/scoring.py` is a v1 draft
by Claude, NOT the client's rules. It is the only part of the system encoding betting
opinion rather than fact. Get Devin to review before it is treated as settled.

## Savant gotchas — all four cause SILENT wrong data, not errors

Full detail in `docs/savant_endpoints.md`. Every one was found by testing, not documentation.

1. **Date filters are ignored on most leaderboards.** Only the bat-tracking family honours
   `dateStart`/`dateEnd`. Everything else returns season-to-date with no warning. This is why
   L5/L10 is computed from raw pitch data (`propline/rolling.py`) rather than downloaded.
2. **`statcast_search` truncates at 25,000 rows.** A 21-day request silently returned 7 days
   with a partial day at the boundary. Always chunk (`CHUNK_DAYS = 3`); the collector raises
   if any chunk comes back at the cap.
3. **Team grouping is spelled differently per endpoint** — `batter-team` on
   `expected_statistics` but `batting-team` on the swing-path endpoint. The wrong spelling
   returns ~600 player rows instead of 30 team rows without erroring.
4. **`/leaderboard/batted-ball` 301-redirects** and returns nothing unless redirects are followed.

## MLB Stats API gotcha

The boxscore's top-level `battingOrder` array is the **last** occupant of each lineup slot,
not the starter. Each player's own `battingOrder` field encodes `slot * 100`, with `+1/+2/...`
for substitutes. Filtering to `% 100 == 0` is required — otherwise pinch hitters and
double-switched relievers get projected into tomorrow's lineup as hitters. See
`_starting_order()` in `propline/mlb.py`.

## Layout

```
propline/config.py        pull definitions (which endpoints, which params)
propline/savant.py        Savant downloader (chunking, retries, redirects)
propline/mlb.py           schedule, lineups (projected/confirmed), bullpen usage
propline/rolling.py       L5/L10 aggregation from raw pitch data
propline/intermediate.py  ID normalisation + intermediate Excel workbook
scripts/collect.py        Phase 1 entry point
scripts/pull_savant.py    Savant-only pull (subset of collect.py)
docs/savant_endpoints.md  verified endpoint spec + evidence
```
