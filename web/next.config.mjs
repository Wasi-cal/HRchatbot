// The browser only ever talks to this Next server; /api/* is proxied to the FastAPI backend, so there is no CORS
// and the backend URL is not baked into the client bundle. Deepgram (voice) is the one direct browser connection.
const API_URL = process.env.API_URL || "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};
export default nextConfig;
