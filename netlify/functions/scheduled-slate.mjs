// Scheduled runs — the four :07 slots.
//
// Why this exists: GitHub's own `schedule:` trigger did not fire once across two
// consecutive slots, while manual dispatch of the same workflow worked every time.
// GitHub documents scheduled events as best-effort and drops them under load, which
// is not good enough for a tool whose entire value is freshness. Netlify's scheduler
// runs this function, which then dispatches the workflow the same way the dashboard
// button does — the path that is known to work.
//
// The workflow keeps its own schedule block as a backstop. If both ever fire, the
// workflow's `concurrency` group queues the second rather than running it twice over
// itself, so the cost is a short duplicate refresh, not corruption.
//
//   13:07 UTC  09:07 ET  full pull   - stats + projected lineups
//   18:07 UTC  14:07 ET  reuse       - late-afternoon games confirmed
//   21:07 UTC  17:07 ET  reuse       - evening games confirmed
//   23:07 UTC  19:07 ET  reuse       - late/west-coast games confirmed
//
// The midday day-game slot is a separate function because it lands at :37.

import { dispatchWorkflow } from '../lib/github.mjs';

export default async () => {
  // Only the first run of the day re-downloads from Savant; the rest reuse it.
  const fullPull = new Date().getUTCHours() === 13;

  const res = await dispatchWorkflow(fullPull ? { full_pull: 'true' } : {});

  if (res.ok) {
    console.log(`slate dispatched (full_pull=${fullPull})`);
    return new Response('dispatched', { status: 202 });
  }

  // Log loudly: a silent failure here is exactly the problem we are solving.
  console.error(`dispatch failed: ${res.status} ${res.detail ?? ''} ${res.hint ?? ''}`);
  return new Response(`dispatch failed: ${res.status}`, { status: 500 });
};

export const config = { schedule: '7 13,18,21,23 * * *' };
