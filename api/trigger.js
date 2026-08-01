// Powers the dashboard's "Update Now" button by asking GitHub Actions to run the
// pipeline on demand.
//
// Why a server endpoint rather than calling GitHub from the page: triggering a
// workflow needs a token that can write to the repo. Anything in browser JavaScript
// is public, so that token has to live here, where only the hosting platform sees it.
//
// The endpoint is protected by a shared passphrase (UPDATE_SECRET). Without it, the
// URL would be an open button anyone could hold down, burning Actions minutes and
// hammering Baseball Savant from a machine the client owns.

const GITHUB_API = 'https://api.github.com';
const WORKFLOW = 'daily.yml';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'POST only' });
  }

  const token = process.env.GITHUB_TOKEN;
  const repo = process.env.GITHUB_REPO;          // e.g. "imtayyab05/propline-mlb"
  const branch = process.env.GITHUB_BRANCH || 'main';
  const secret = process.env.UPDATE_SECRET;

  if (!token || !repo) {
    return res.status(500).json({
      error: 'GITHUB_TOKEN / GITHUB_REPO are not set in the hosting environment',
    });
  }

  if (secret) {
    const supplied = req.headers['x-update-secret'];
    if (supplied !== secret) {
      return res.status(401).json({ error: 'Wrong or missing passphrase' });
    }
  }

  const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}')
                                            : (req.body || {});
  const inputs = {};
  // Only pass a date through if it looks like one — this value reaches a shell step
  // in the workflow, so it does not get to be free-form.
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
    return res.status(200).json({ ok: true, started: true });
  }

  const detail = await r.text();
  // 404 here almost always means the token lacks Actions write permission on the repo,
  // rather than the workflow genuinely being missing.
  const hint = r.status === 404
    ? 'Workflow not found, or the token lacks "Actions: write" on this repository.'
    : undefined;
  return res.status(502).json({ error: `GitHub ${r.status}`, detail: detail.slice(0, 300), hint });
}
