const environment = {
  // Set NEXT_PUBLIC_API_URL at build time or in your .env.local file.
  // Defaults to localhost for local development.
  apiUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000',
};

export default environment;
