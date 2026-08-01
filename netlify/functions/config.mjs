// Netlify Functions v2 — serves the PUBLIC config to the dashboard.
//
// Same job as api/config.js (the Vercel version); only the request/response shape
// differs. Keeping both means the project can deploy to either host without edits.
//
// Only the publishable key is exposed here. It is designed to sit in a browser, and
// row-level security makes every table read-only to it. The service key and the
// GitHub token never leave the server.

export default async () => {
  const url = process.env.SUPABASE_URL;
  const anon = process.env.SUPABASE_ANON_KEY;

  if (!url || !anon) {
    return Response.json(
      { error: 'SUPABASE_URL / SUPABASE_ANON_KEY are not set in the site environment' },
      { status: 500 }
    );
  }

  return Response.json(
    {
      supabaseUrl: url.replace(/\/$/, ''),
      supabaseAnonKey: anon,
      updateEnabled: Boolean(process.env.GITHUB_TOKEN && process.env.GITHUB_REPO),
    },
    { headers: { 'Cache-Control': 'public, max-age=0, s-maxage=3600' } }
  );
};

// Serve at /api/config so the dashboard code is identical on both hosts.
export const config = { path: '/api/config' };
