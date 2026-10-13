import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = process.env.TREVIDEON_BACKEND || process.env.EVM_BACKEND || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': backend,
      '/files': backend,
    },
  },
})
