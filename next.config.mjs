/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/etl/:path*',
        destination: 'http://localhost:3030/:path*',
      },
      {
        source: '/api/preprocessing/:path*',
        destination: 'http://localhost:3031/:path*',
      },
      {
        source: '/api/eda/:path*',
        destination: 'http://localhost:3035/:path*',
      },
      {
        source: '/api/analysis/:path*',
        destination: 'http://localhost:3040/:path*',
      },
    ];
  },
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Credentials', value: 'true' },
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET,DELETE,PATCH,POST,PUT' },
          { key: 'Access-Control-Allow-Headers', value: 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version' },
        ],
      },
    ];
  },
};

export default nextConfig;
