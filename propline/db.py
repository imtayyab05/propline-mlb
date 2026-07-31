"""Supabase writes over PostgREST.

Uses plain HTTP rather than the supabase-py client — one less dependency to install
and pin, and all we need is upsert.

The service key is used here and bypasses row-level security. It must never reach the
frontend or a public repo; it is read from .env locally and from GitHub Secrets in CI.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
import requests

BATCH = 500          # rows per request; keeps payloads well under any body limit
TIMEOUT = 120


class SupabaseError(RuntimeError):
    pass


def _creds():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise SupabaseError(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY missing — copy .env.example to .env "
            "and fill it in"
        )
    return url.rstrip("/"), key


def load_env(path: str | Path = ".env") -> None:
    """Load .env without depending on find_dotenv's caller-frame trick."""
    from dotenv import load_dotenv
    p = Path(path)
    if p.exists():
        load_dotenv(p)


def _clean(records: list[dict]) -> list[dict]:
    """JSON cannot carry NaN/NaT/numpy scalars — convert them to null/native types."""
    out = []
    for rec in records:
        row = {}
        for k, v in rec.items():
            if v is None:
                row[k] = None
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[k] = None
            elif isinstance(v, (pd.Timestamp,)):
                row[k] = v.isoformat()
            elif v is pd.NaT:
                row[k] = None
            elif hasattr(v, "item"):          # numpy scalar
                row[k] = v.item()
            elif isinstance(v, (dict, list)):
                row[k] = v
            else:
                row[k] = v
        out.append(row)
    return out


def upsert(table: str, df: pd.DataFrame, on_conflict: str | None = None,
           chunk: int = BATCH) -> int:
    """Insert-or-update a DataFrame into a table. Returns rows sent."""
    if df is None or df.empty:
        return 0

    url, key = _creds()
    endpoint = f"{url}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else {}
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # merge-duplicates makes this an upsert rather than a failing insert
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    records = _clean(df.to_dict(orient="records"))
    sent = 0
    for i in range(0, len(records), chunk):
        batch = records[i:i + chunk]
        r = requests.post(endpoint, params=params, headers=headers,
                          data=json.dumps(batch, default=str), timeout=TIMEOUT)
        if not r.ok:
            raise SupabaseError(f"{table}: {r.status_code} {r.text[:400]}")
        sent += len(batch)
    return sent


def log_run(slate_date, run_type, stage, status, detail=None, started_at=None) -> None:
    """Record a pipeline run — this is what powers the dashboard's 'last updated'."""
    row = pd.DataFrame([{
        "slate_date": str(slate_date), "run_type": run_type, "stage": stage,
        "status": status, "detail": detail,
        "started_at": started_at.isoformat() if started_at else None,
    }])
    try:
        upsert("pipeline_runs", row)
    except SupabaseError as exc:
        # never let run-logging failure take down an otherwise good run
        print(f"  WARN  could not log run: {exc}")


def health_check() -> bool:
    url, key = _creds()
    r = requests.get(f"{url}/rest/v1/", headers={"apikey": key,
                                                 "Authorization": f"Bearer {key}"},
                     timeout=30)
    return r.ok
