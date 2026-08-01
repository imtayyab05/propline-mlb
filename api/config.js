// Serves the PUBLIC config to the dashboard.
//
// Only the publishable key is exposed here — it is designed to sit in a browser and
// every table is read-only to it via row-level security. The service key and the
// GitHub token never leave the server.
//
// Keeping this in an endpoint rather than hardcoding it means the repo contains no
// project-specific values at all, so the same code deploys against a second Supabase
// project without edits.

export default function handler(req, res) {
  const url = process.env.SUPABASE_URL;
  const anon = process.env.SUPABASE_ANON_KEY;

  if (!url || !anon) {
    return res.status(500).json({
      error: 'SUPABASE_URL / SUPABASE_ANON_KEY are not set in the hosting environment',
    });
  }

  // brief cache: the values never change between deploys
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate');
  res.status(200).json({
    supabaseUrl: url.replace(/\/$/, ''),
    supabaseAnonKey: anon,
    updateEnabled: Boolean(process.env.GITHUB_TOKEN && process.env.GITHUB_REPO),
  });
}
