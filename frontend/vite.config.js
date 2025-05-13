import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';
import { visualizer } from 'rollup-plugin-visualizer';
import viteCompression from 'vite-plugin-compression';
import svgr from 'vite-plugin-svgr';
import { VitePWA } from 'vite-plugin-pwa';
import { createHtmlPlugin } from 'vite-plugin-html';
import viteImagemin from 'vite-plugin-imagemin';
import eslint from 'vite-plugin-eslint';
import path from 'path';

export default defineConfig(({ mode }) => {
  const isProduction = mode === 'production';
  
  return {
    // Base public path
    base: '/',
    
    // Define global constants
    define: {
      // Example: __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
      global: 'globalThis',
    },
    
    // Development server configuration
    server: {
      host: true, // Listen on all addresses
      port: 3000,
      open: true, // Automatically open in browser
      cors: true,
      hmr: {
        overlay: true,
      },
      proxy: {
        // Proxy API requests to backend
        '/api': {
          target: process.env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, '/api'),
        },
      },
    },
    
    // Build configuration
    build: {
      target: 'es2020',
      outDir: 'dist',
      assetsDir: 'assets',
      sourcemap: !isProduction,
      minify: isProduction ? 'esbuild' : false,
      
      // Chunk splitting for better caching
      rollupOptions: {
        output: {
          manualChunks: {
            // React and Router
            'react-vendor': ['react', 'react-dom', 'react-router-dom'],
            
            // Charts and visualization
            'charts': ['chart.js', 'react-chartjs-2', 'd3', 'recharts'],
            
            // UI libraries
            'ui-libs': ['@heroicons/react', 'lucide-react', 'framer-motion'],
            
            // Utils and HTTP
            'utils': ['axios', 'date-fns', 'xlsx'],
          },
          
          // Generate readable chunk names
          chunkFileNames: (chunkInfo) => {
            return isProduction
              ? `js/[name]-[hash].js`
              : `js/[name].js`;
          },
          
          // Generate readable asset names
          assetFileNames: (assetInfo) => {
            const extType = assetInfo.name.split('.').at(1);
            if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(extType)) {
              return `img/[name]-[hash][extname]`;
            }
            if (/css/i.test(extType)) {
              return `css/[name]-[hash][extname]`;
            }
            if (/woff|woff2|eot|ttf|otf/i.test(extType)) {
              return `fonts/[name]-[hash][extname]`;
            }
            return `assets/[name]-[hash][extname]`;
          },
        },
      },
      
      // Optimize dependencies
      esbuild: isProduction ? {
        drop: ['console', 'debugger'],
      } : {},
    },
    
    // Preview configuration (for build preview)
    preview: {
      port: 4173,
      open: true,
    },
    
    // Path resolution
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@components': path.resolve(__dirname, './src/components'),
        '@pages': path.resolve(__dirname, './src/pages'),
        '@hooks': path.resolve(__dirname, './src/hooks'),
        '@services': path.resolve(__dirname, './src/services'),
        '@utils': path.resolve(__dirname, './src/utils'),
        '@assets': path.resolve(__dirname, './src/assets'),
      },
    },
    
    // Plugins
    plugins: [
      // React with SWC for faster builds
      react({
        // Include .jsx files
        include: "**/*.jsx",
      }),
      
      // ESLint
      eslint({
        cache: true,
        lintOnStart: true,
        include: ['src/**/*.{js,jsx}'],
      }),
      
      // SVG as React components
      svgr({
        svgrOptions: {
          exportType: 'default',
          ref: true,
          svgo: false,
          titleProp: true,
        },
        include: '**/*.svg',
      }),
      
      // PWA support
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.ico', 'robots.txt', 'apple-touch-icon.png'],
        manifest: {
          name: 'Supply Chain LLM',
          short_name: 'SC-LLM',
          description: 'Intelligent supply chain analytics and optimization platform',
          theme_color: '#1e293b',
          background_color: '#ffffff',
          display: 'standalone',
          orientation: 'portrait',
          start_url: '/',
          icons: [
            {
              src: 'assets/icons/icon-192x192.png',
              sizes: '192x192',
              type: 'image/png',
            },
            {
              src: 'assets/icons/icon-512x512.png',
              sizes: '512x512',
              type: 'image/png',
            },
          ],
        },
        workbox: {
          cleanupOutdatedCaches: true,
          skipWaiting: true,
          clientsClaim: true,
          runtimeCaching: [
            // API calls
            {
              urlPattern: /^https:\/\/api\.supply-chain-llm\.com\//,
              handler: 'NetworkFirst',
              options: {
                cacheName: 'api-cache',
                networkTimeoutSeconds: 10,
                expiration: {
                  maxEntries: 50,
                  maxAgeSeconds: 5 * 60, // 5 minutes
                },
              },
            },
            // Images
            {
              urlPattern: /\.(png|jpg|jpeg|svg|gif)$/,
              handler: 'CacheFirst',
              options: {
                cacheName: 'image-cache',
                expiration: {
                  maxEntries: 100,
                  maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
                },
              },
            },
          ],
        },
      }),
      
      // HTML processing
      createHtmlPlugin({
        minify: isProduction,
        inject: {
          data: {
            title: 'Supply Chain LLM',
            // Inject additional variables as needed
          },
        },
      }),
      
      // Compression for production
      isProduction && viteCompression({
        verbose: true,
        disable: false,
        threshold: 10240,
        algorithm: 'gzip',
        ext: '.gz',
      }),
      
      // Image optimization for production
      isProduction && viteImagemin({
        gifsicle: {
          optimizationLevel: 7,
          interlaced: false,
        },
        optipng: {
          optimizationLevel: 7,
        },
        mozjpeg: {
          quality: 80,
        },
        pngquant: {
          quality: [0.8, 0.9],
          speed: 4,
        },
        svgo: {
          plugins: [
            {
              name: 'removeViewBox',
              active: false,
            },
            {
              name: 'removeEmptyAttrs',
              active: false,
            },
          ],
        },
      }),
      
      // Bundle analyzer (disabled by default)
      process.env.ANALYZE && visualizer({
        open: true,
        filename: 'bundle-analysis.html',
        gzipSize: true,
        brotliSize: true,
      }),
    ].filter(Boolean), // Remove falsy values
    
    // CSS configuration
    css: {
      postcss: './postcss.config.js',
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
      modules: {
        localsConvention: 'camelCase',
      },
      devSourcemap: true,
    },
    
    // Dependency optimization
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        'axios',
        'chart.js',
        'd3',
      ],
      exclude: [],
    },
    
    // Environment variables
    envPrefix: 'VITE_',
    
    // Worker configuration
    worker: {
      format: 'es',
    },
  };
});