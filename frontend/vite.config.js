import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Проксируем API запросы на backend
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Проксируем WebSocket
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
})
