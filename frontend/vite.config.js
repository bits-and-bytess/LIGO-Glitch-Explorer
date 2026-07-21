import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      // Backend-served images (spectrogram/GradCAM PNGs) come back from
      // the API as bare "/static/..." paths -- these need their own
      // proxy rule since they don't share the "/api" prefix rewritten
      // above. See src/api.js's mediaUrl() for the production-side
      // (non-dev-proxy) equivalent of this.
      '/static': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
