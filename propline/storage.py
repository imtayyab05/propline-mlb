"""Upload the generated picks workbook to Supabase Storage.

The requirements doc asks for a download link to the generated Excel file on the
dashboard, not just an on-screen table. The workbook is produced on a GitHub runner
that disappears after the job, so it has to be put somewhere durable — Supabase
Storage, which we already have, rather than adding another service.

The bucket is public: it holds the same picks the dashboard already shows to anyone
with the URL, so a signed-URL flow would add moving parts without protecting anything
that is not already visible.
"""

from __future__ import annotations

from pathlib import Path

import requests

from .db import SupabaseError, _creds

BUCKET = "picks"
TIMEOUT = 120
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def ensure_bucket(name: str = BUCKET) -> None:
    """Create the bucket if it is not already there. Safe to call every run."""
    url, key = _creds()
    headers = {"apikey": key, "Authorization": f"Bearer {key}",
               "Content-Type": "application/json"}

    r = requests.get(f"{url}/storage/v1/bucket/{name}", headers=headers, timeout=TIMEOUT)
    if r.status_code == 200:
        return

    r = requests.post(f"{url}/storage/v1/bucket", headers=headers, timeout=TIMEOUT,
                      json={"id": name, "name": name, "public": True})
    # 409 means someone else created it between our check and our create — fine.
    if not r.ok and r.status_code != 409:
        raise SupabaseError(f"could not create bucket {name}: {r.status_code} {r.text[:200]}")


def upload_workbook(path, slate_date, bucket: str = BUCKET) -> str:
    """Upload the workbook and return its public URL."""
    path = Path(path)
    if not path.exists():
        raise SupabaseError(f"no workbook at {path}")

    url, key = _creds()
    ensure_bucket(bucket)

    object_path = f"props_{slate_date}.xlsx"
    endpoint = f"{url}/storage/v1/object/{bucket}/{object_path}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": XLSX_MIME,
        # overwrite today's file rather than erroring on the second run of a slate
        "x-upsert": "true",
    }

    with open(path, "rb") as fh:
        r = requests.post(endpoint, headers=headers, data=fh, timeout=TIMEOUT)
    if not r.ok:
        raise SupabaseError(f"upload failed: {r.status_code} {r.text[:200]}")

    return f"{url}/storage/v1/object/public/{bucket}/{object_path}"
