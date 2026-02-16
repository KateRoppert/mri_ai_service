import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/', 
  server: {
    proxy: {
      // Проксируем API запросы на backend (только для dev режима)
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // Проксируем WebSocket (только для dev режима)
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  }
})