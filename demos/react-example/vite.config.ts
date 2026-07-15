import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// Build React app to a directory that FastAPI will serve
export default defineConfig({
  plugins: [react()],
  build: {
    // Output to a directory FastAPI can serve as static files
    outDir: path.resolve(__dirname, 'dist'),
    emptyDirBeforeWrite: true,
    sourcemap: true,
  },
  // Base path - FastAPI will serve this at /react-app/
  base: '/react-app/',
});
