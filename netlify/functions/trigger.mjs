// Netlify Functions v2 — powers the dashboard's "Update Now" button.
//
// Triggering a workflow needs a token that can write to the repo. Anything in browser
// JavaScript is public, so that token lives here, where only Netlify sees it.
//
// Passphrase-gated (UPDATE_SECRET): without it this URL is an open button anyone
// could hold down, burning the client's Actions minutes and hammering Baseball Savant
// from a machine he owns.

const GITHUB_API = 'https://api.github.com';
const WORKFLOW = 'daily.yml';

export default async (req) => {
  if (req.method !== 'POST') {
    return Response.json({ error: 'POST only' }, { status: 405 });
  }

  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO;            // e.g. "imtayyab05/propline-mlb"
  const branch = process.env.GITHUB_BRANCH || 'main';
  const secret = process.env.UPDATE_SECRET;

  if (!token || !repo) {
    return Response.json(
      { error: 'GITHUB_TOKEN / GITHUB_REPO are not set in the site environment' },
      { status: 500 }
    );
  }

  if (secret && req.headers.get('x-update-secret') !== secret) {
    return Response.json({ error: 'Wrong or missing passphrase' }, { status: 401 });
  }

  let body = {};
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const inputs = {};
  // This value reaches a shell step in the workflow, so it does not get to be
  // free-form — it must look like a date or it is dropped.
  if (typeof body.slate_date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(body.slate_date)) {
    inputs.slate_date = body.slate_date;
  }
  if (body.window === 'L5' || body.window === 'L10') {
    inputs.window = body.window;
  }

  const r = await fetch(
    `${GITHUB_API}/repos/${repo}/actions/workflows/${WORKFLOW}/dispatches`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ ref: branch, inputs }),
    }
  );

  if (r.status === 204) {
    return Response.json({ ok: true, started: true });
  }

  const detail = await r.text();
  // A 404 here nearly always means the token lacks Actions write permission, rather
  // than the workflow genuinely being missing.
  const hint =
    r.status === 404
      ? 'Workflow not found, or the token lacks "Actions: write" on this repository.'
      : undefined;

  return Response.json(
    { error: `GitHub ${r.status}`, detail: detail.slice(0, 300), hint },
    { status: 502 }
  );
};

export const config = { path: '/api/trigger' };
