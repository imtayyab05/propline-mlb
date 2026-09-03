// The midday slot, 15:37 UTC / 11:37 ET.
//
// Separate from scheduled-slate.mjs only because Netlify allows one schedule per
// function and this one lands at :37, not :07. Day games start around 13:05 ET and
// their lineups post roughly two hours before, so this is the run that turns those
// games from projected to confirmed.

import { dispatchWorkflow } from '../lib/github.mjs';

export default async () => {
  // run_kind both labels the run and tells the workflow to reuse the morning's
  // Savant download rather than re-fetching it.
  const res = await dispatchWorkflow({ run_kind: 'scheduled_midday' });

  if (res.ok) {
    console.log('midday slate dispatched');
    return new Response('dispatched', { status: 202 });
  }

  console.error(`dispatch failed: ${res.status} ${res.detail ?? ''} ${res.hint ?? ''}`);
  return new Response(`dispatch failed: ${res.status}`, { status: 500 });
};

export const config = { schedule: '37 15 * * *' };
