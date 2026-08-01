// Netlify Functions v2 — powers the dashboard's "Update Now" button.
//
// Triggering a workflow needs a token that can write to the repo. Anything in browser
// JavaScript is public, so that token lives here, where only Netlify sees it.
//
// Passphrase-gated (UPDATE_SECRET): without it this URL is an open button anyone
// could hold down, burning the client's Actions minutes and hammering Baseball Savant
// from a machine he owns.
//
// The actual dispatch lives in ../lib/github.mjs, shared with the scheduled functions.

import { dispatchWorkflow, cleanInputs } from '../lib/github.mjs';

export default async (req) => {
  if (req.method !== 'POST') {
    return Response.json({ error: 'POST only' }, { status: 405 });
  }

  const secret = process.env.UPDATE_SECRET;
  if (secret && req.headers.get('x-update-secret') !== secret) {
    return Response.json({ error: 'Wrong or missing passphrase' }, { status: 401 });
  }

  let body = {};
  try { body = await req.json(); } catch { body = {}; }

  const res = await dispatchWorkflow(cleanInputs(body));
  if (res.ok) return Response.json({ ok: true, started: true });

  return Response.json(
    { error: `GitHub ${res.status}`, detail: res.detail, hint: res.hint },
    { status: res.status === 500 ? 500 : 502 }
  );
};

export const config = { path: '/api/trigger' };
