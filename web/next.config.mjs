// next.config.mjs
//
// What it does: tells Next.js to build the site as plain static files (HTML,
//   CSS, JavaScript) in web/out. There is no server: every page loads its
//   numbers from the JSON files in public/data, so it can be hosted free.
// Which files use it: Next.js reads it when you run `npm run dev` or `npm run build`.

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true, // /ranking/ is served from ranking/index.html on any host
};

export default nextConfig;
