// next.config.mjs
//
// What it does: the Next.js settings. Every page is still built as plain
//   static HTML (all its numbers come from the JSON files in public/data),
//   but the site is no longer a pure static export, because the Pit Wall
//   assistant needs one small server endpoint (app/api/chat) to keep the
//   Anthropic API key secret. Vercel runs that endpoint for free.
// Which files use it: Next.js reads it when you run `npm run dev` or `npm run build`.

/** @type {import('next').NextConfig} */
const nextConfig = {
  trailingSlash: true, // /ranking/ is served from ranking/index.html on any host
};

export default nextConfig;
