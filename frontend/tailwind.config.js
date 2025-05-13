/** @type {import('tailwindcss').Config} */
export default {
    content: [
      "./index.html",
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          // Brand colors
          primary: {
            50: '#f8fafc',
            100: '#f1f5f9',
            200: '#e2e8f0',
            300: '#cbd5e1',
            400: '#94a3b8',
            500: '#64748b',
            600: '#475569',
            700: '#334155',
            800: '#1e293b',
            900: '#0f172a',
            950: '#020617',
          },
          secondary: {
            50: '#f0fdf4',
            100: '#dcfce7',
            200: '#bbf7d0',
            300: '#86efac',
            400: '#4ade80',
            500: '#22c55e',
            600: '#16a34a',
            700: '#15803d',
            800: '#166534',
            900: '#14532d',
            950: '#052e16',
          },
          accent: {
            50: '#fef3c7',
            100: '#fef3c7',
            200: '#fde68a',
            300: '#fcd34d',
            400: '#fbbf24',
            500: '#f59e0b',
            600: '#d97706',
            700: '#b45309',
            800: '#92400e',
            900: '#78350f',
            950: '#451a03',
          },
          // Semantic colors
          success: {
            DEFAULT: '#22c55e',
            light: '#dcfce7',
            dark: '#15803d',
          },
          error: {
            DEFAULT: '#ef4444',
            light: '#fee2e2',
            dark: '#dc2626',
          },
          warning: {
            DEFAULT: '#f59e0b',
            light: '#fef3c7',
            dark: '#d97706',
          },
          info: {
            DEFAULT: '#3b82f6',
            light: '#dbeafe',
            dark: '#2563eb',
          },
        },
        fontFamily: {
          sans: ['Inter', 'system-ui', 'sans-serif'],
          mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        },
        fontSize: {
          'xs': ['0.75rem', { lineHeight: '1rem' }],
          'sm': ['0.875rem', { lineHeight: '1.25rem' }],
          'base': ['1rem', { lineHeight: '1.5rem' }],
          'lg': ['1.125rem', { lineHeight: '1.75rem' }],
          'xl': ['1.25rem', { lineHeight: '1.75rem' }],
          '2xl': ['1.5rem', { lineHeight: '2rem' }],
          '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
          '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
          '5xl': ['3rem', { lineHeight: '1' }],
        },
        spacing: {
          '18': '4.5rem',
          '88': '22rem',
          '100': '25rem',
          '112': '28rem',
          '128': '32rem',
        },
        borderRadius: {
          'none': '0',
          'sm': '0.125rem',
          DEFAULT: '0.25rem',
          'md': '0.375rem',
          'lg': '0.5rem',
          'xl': '0.75rem',
          '2xl': '1rem',
          '3xl': '1.5rem',
          'full': '9999px',
        },
        boxShadow: {
          'xs': '0 0 0 1px rgba(0, 0, 0, 0.05)',
          'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
          DEFAULT: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
          'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
          'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
          'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          'inner': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
          'none': 'none',
        },
        animation: {
          'fade-in': 'fadeIn 0.5s ease-in-out',
          'fade-out': 'fadeOut 0.5s ease-in-out',
          'slide-in-right': 'slideInRight 0.5s ease-out',
          'slide-in-left': 'slideInLeft 0.5s ease-out',
          'slide-in-up': 'slideInUp 0.5s ease-out',
          'slide-in-down': 'slideInDown 0.5s ease-out',
          'scale-in': 'scaleIn 0.3s ease-out',
          'scale-out': 'scaleOut 0.3s ease-out',
          'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        },
        keyframes: {
          fadeIn: {
            '0%': { opacity: '0' },
            '100%': { opacity: '1' },
          },
          fadeOut: {
            '0%': { opacity: '1' },
            '100%': { opacity: '0' },
          },
          slideInRight: {
            '0%': { transform: 'translateX(100%)', opacity: '0' },
            '100%': { transform: 'translateX(0)', opacity: '1' },
          },
          slideInLeft: {
            '0%': { transform: 'translateX(-100%)', opacity: '0' },
            '100%': { transform: 'translateX(0)', opacity: '1' },
          },
          slideInUp: {
            '0%': { transform: 'translateY(100%)', opacity: '0' },
            '100%': { transform: 'translateY(0)', opacity: '1' },
          },
          slideInDown: {
            '0%': { transform: 'translateY(-100%)', opacity: '0' },
            '100%': { transform: 'translateY(0)', opacity: '1' },
          },
          scaleIn: {
            '0%': { transform: 'scale(0.9)', opacity: '0' },
            '100%': { transform: 'scale(1)', opacity: '1' },
          },
          scaleOut: {
            '0%': { transform: 'scale(1)', opacity: '1' },
            '100%': { transform: 'scale(0.9)', opacity: '0' },
          },
        },
        transitionTimingFunction: {
          'bounce-in': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        },
        screens: {
          'xs': '475px',
          '3xl': '1920px',
        },
        zIndex: {
          '60': '60',
          '70': '70',
          '80': '80',
          '90': '90',
          '100': '100',
        },
        maxWidth: {
          '8xl': '90rem',
          '9xl': '100rem',
        },
        container: {
          center: true,
          padding: {
            DEFAULT: '1rem',
            sm: '2rem',
            lg: '4rem',
            xl: '5rem',
            '2xl': '6rem',
          },
        },
        typography: {
          DEFAULT: {
            css: {
              color: 'theme("colors.gray.700")',
              maxWidth: 'none',
              a: {
                color: 'theme("colors.blue.600")',
                textDecoration: 'none',
                '&:hover': {
                  color: 'theme("colors.blue.800")',
                },
              },
              strong: {
                color: 'theme("colors.gray.900")',
              },
              'ol > li::before': {
                color: 'theme("colors.gray.500")',
              },
              'ul > li::before': {
                backgroundColor: 'theme("colors.gray.400")',
              },
              hr: {
                borderColor: 'theme("colors.gray.200")',
              },
              blockquote: {
                color: 'theme("colors.gray.700")',
                borderLeftColor: 'theme("colors.gray.200")',
              },
              h1: {
                color: 'theme("colors.gray.900")',
              },
              h2: {
                color: 'theme("colors.gray.900")',
              },
              h3: {
                color: 'theme("colors.gray.900")',
              },
              h4: {
                color: 'theme("colors.gray.900")',
              },
              thead: {
                borderBottomColor: 'theme("colors.gray.300")',
              },
              'thead th': {
                color: 'theme("colors.gray.900")',
              },
              'tbody tr': {
                borderBottomColor: 'theme("colors.gray.200")',
              },
              code: {
                color: 'theme("colors.gray.900")',
                backgroundColor: 'theme("colors.gray.100")',
                padding: '0.125rem 0.25rem',
                borderRadius: '0.25rem',
                fontWeight: '400',
              },
              'code::before': {
                content: '""',
              },
              'code::after': {
                content: '""',
              },
            },
          },
        },
      },
    },
    plugins: [
      require('@tailwindcss/typography'),
      require('@tailwindcss/forms'),
      require('@tailwindcss/aspect-ratio'),
      require('@tailwindcss/container-queries'),
    ],
  }