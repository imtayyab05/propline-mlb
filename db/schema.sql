-- PropLine MLB — database schema
-- Run this ONCE in the Supabase SQL Editor (left sidebar -> SQL Editor -> New query).
-- Safe to re-run: everything is IF NOT EXISTS.

-- ============================================================ core reference

create table if not exists players (
    player_id     bigint primary key,          -- Savant / MLBAM id
    player_name   text,
    team          text,
    bats          text,
    throws        text,
    updated_at    timestamptz default now()
);

-- ============================================================ slate

create table if not exists games (
    game_pk           bigint primary key,
    game_date         date not null,
    game_time_utc     timestamptz,
    status            text,
    venue             text,
    home_team         text,
    home_team_id      int,
    away_team         text,
    away_team_id      int,
    home_probable_id  bigint,
    home_probable     text,
    away_probable_id  bigint,
    away_probable     text,
    updated_at        timestamptz default now()
);
create index if not exists games_date_idx on games (game_date);

create table if not exists lineups (
    game_pk        bigint not null,
    game_date      date not null,
    team_id        int not null,
    team           text,
    opponent_id    int,
    home_away      text,
    batting_order  int not null,
    player_id      bigint not null,
    player_name    text,
    position       text,
    status         text not null,             -- 'projected' | 'confirmed'
    updated_at     timestamptz default now(),
    primary key (game_pk, team_id, batting_order)
);
create index if not exists lineups_date_idx on lineups (game_date);

create table if not exists bullpen_status (
    as_of            date not null,
    team_id          int not null,
    team             text,
    player_id        bigint not null,
    player_name      text,
    last_appearance  date,
    appearances      int,
    pitches          int,
    availability     text,                     -- 'available' | 'likely_unavailable'
    updated_at       timestamptz default now(),
    primary key (as_of, team_id, player_id)
);

-- ============================================================ output

create table if not exists prop_picks (
    slate_date     date not null,
    prop           text not null,              -- hits | total_bases | home_runs | rbis | runs | strikeouts
    subject_id     bigint not null,            -- player_id, or team_id/game_pk for totals
    subject_name   text,
    team           text,
    opponent       text,
    rank           int,
    score          numeric,
    lineup_status  text,                       -- projected | confirmed | n/a
    rationale      text,                       -- Groq-written, ranking is NOT from the LLM
    details        jsonb,                      -- the signals behind the score
    created_at     timestamptz default now(),
    primary key (slate_date, prop, subject_id)
);
create index if not exists prop_picks_date_idx on prop_picks (slate_date, prop, rank);

create table if not exists game_picks (
    slate_date  date not null,
    game_pk     bigint not null,
    prop        text not null,                 -- game_total | team_total
    subject     text,                          -- matchup label or team name
    rank        int,
    score       numeric,
    venue       text,
    park_runs   int,
    park_matched boolean,
    rationale   text,
    details     jsonb,
    created_at  timestamptz default now(),
    primary key (slate_date, prop, game_pk, subject)
);

-- ============================================================ run log

create table if not exists pipeline_runs (
    id            bigserial primary key,
    slate_date    date,
    run_type      text,                        -- morning | afternoon | manual
    stage         text,                        -- collection | processing
    status        text,                        -- ok | failed
    detail        text,
    started_at    timestamptz,
    finished_at   timestamptz default now()
);
create index if not exists pipeline_runs_date_idx on pipeline_runs (slate_date, finished_at desc);

-- ============================================================ security
-- RLS is on for every table. The pipeline writes with the service key, which
-- bypasses RLS entirely. The dashboard reads with the publishable key, so each
-- table needs an explicit read policy — nothing is exposed by accident.

-- NOTE: because "Automatically expose new tables" is disabled on this project (the
-- safer setting), new tables get NO privileges by default — not even for service_role.
-- They must be granted explicitly here, or every write fails with
-- "42501 permission denied". This block is the thing that makes the API usable.

do $$
declare t text;
begin
  foreach t in array array['players','games','lineups','bullpen_status',
                           'prop_picks','game_picks','pipeline_runs']
  loop
    execute format('alter table %I enable row level security', t);

    -- the pipeline writes with the service key (bypasses RLS, still needs grants)
    execute format('grant select, insert, update, delete on %I to service_role', t);

    -- the dashboard reads with the publishable key: read-only, via an explicit policy
    execute format('grant select on %I to anon, authenticated', t);
    execute format('drop policy if exists "public read" on %I', t);
    execute format('create policy "public read" on %I for select to anon, authenticated using (true)', t);
  end loop;
end $$;

-- pipeline_runs.id is a bigserial, so its sequence needs granting too
grant usage, select on all sequences in schema public to service_role;
