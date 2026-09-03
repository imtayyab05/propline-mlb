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

// Which slot this invocation is. The workflow cannot work this out for itself — it is
// dispatched, not scheduled, so `github.event.schedule` is always empty there — and
// without it every automated run was labelled "manual" on the dashboard.
//
// The morning slot is also the one that re-downloads from Savant; the workflow derives
// that from the label, so there is one source of truth rather than two flags to keep
// in step.
const SLOTS = {
  13: 'scheduled_morning',
  18: 'scheduled_afternoon',
  21: 'scheduled_evening',
  23: 'scheduled_late',
};

export default async () => {
  const hour = new Date().getUTCHours();
  const runKind = SLOTS[hour];

  if (!runKind) {
    // Cron fired outside its declared hours. Dispatch anyway — a labelled-wrong run
    // beats a missed slate — but say so, because it means the schedule has drifted.
    console.warn(`unexpected dispatch hour ${hour}:00 UTC; running unlabelled`);
  }

  const res = await dispatchWorkflow(runKind ? { run_kind: runKind } : {});

  if (res.ok) {
    console.log(`slate dispatched (${runKind ?? 'unlabelled'})`);
    return new Response('dispatched', { status: 202 });
  }

  // Log loudly: a silent failure here is exactly the problem we are solving.
  console.error(`dispatch failed: ${res.status} ${res.detail ?? ''} ${res.hint ?? ''}`);
  return new Response(`dispatch failed: ${res.status}`, { status: 500 });
};

export const config = { schedule: '7 13,18,21,23 * * *' };
