/** @type {import('next').NextConfig} */
// Static export: `next build` writes a fully static site to web/out/ (no server needed).
// For GitHub Pages under a repo subpath, set NEXT_PUBLIC_BASE_PATH="/Mental-Health".
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || "";

const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
  basePath,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
