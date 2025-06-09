const config = {
  // API Configuration
  api: {
    baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
    timeout: 30000,
    retryAttempts: 3,
  },

  // WebSocket Configuration
  websocket: {
    url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
    reconnectInterval: 5000,
    maxReconnectAttempts: 5,
  },

  // Map Configuration (for logistics dashboard)
  map: {
    defaultCenter: {
      lat: parseFloat(import.meta.env.VITE_MAP_DEFAULT_LAT || '39.8283'),
      lng: parseFloat(import.meta.env.VITE_MAP_DEFAULT_LNG || '-98.5795')
    },
    defaultZoom: parseInt(import.meta.env.VITE_MAP_DEFAULT_ZOOM || '4'),
    tileLayer: {
      url: import.meta.env.VITE_MAP_TILE_URL || 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }
  },

  // Chart Configuration
  charts: {
    colors: {
      primary: ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'],
      secondary: ['#82CA9D', '#FFC658', '#8DD1E1', '#D084D0', '#FFB6C1'],
      heatmap: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
    },
    defaultHeight: 400,
    defaultAnimationDuration: 1000,
  },

  // Date Format Configuration
  dateFormats: {
    display: 'MMM DD, YYYY',
    input: 'YYYY-MM-DD',
    datetime: 'MMM DD, YYYY HH:mm',
    api: 'YYYY-MM-DD',
  },

  // Pagination Configuration
  pagination: {
    defaultPageSize: 20,
    pageSizeOptions: [10, 20, 50, 100],
  },

  // Feature Flags (from environment or backend)
  features: {
    enableMultiTier: import.meta.env.VITE_ENABLE_MULTI_TIER === 'true',
    enableNaturalLanguageQuery: import.meta.env.VITE_ENABLE_NLQ === 'true',
    enableRealtimeUpdates: import.meta.env.VITE_ENABLE_REALTIME === 'true',
    enableAdvancedAnalytics: import.meta.env.VITE_ENABLE_ADVANCED_ANALYTICS === 'true',
  },

  // Analytics Configuration
  analytics: {
    inventory: {
      defaultServiceLevel: 0.95,
      defaultLeadTimeDays: 7,
      forecastMethods: ['auto', 'moving_average', 'exponential_smoothing', 'arima'],
      abcThresholds: {
        a: 0.8,
        b: 0.15,
      }
    },
    logistics: {
      defaultOptimizationObjective: 'minimize_cost',
      vehicleTypes: ['truck', 'van', 'bike'],
      maxVehicles: 50,
    },
    supplier: {
      riskFactors: ['financial', 'operational', 'geopolitical', 'environmental', 'quality'],
      complianceAreas: ['certifications', 'quality_standards', 'environmental', 'labor_practices'],
      performanceMetrics: ['quality', 'delivery', 'cost', 'responsiveness', 'compliance'],
    }
  },

  // Export Configuration
  export: {
    formats: ['csv', 'xlsx', 'pdf'],
    maxRows: 10000,
  },

  // Validation Rules
  validation: {
    password: {
      minLength: 8,
      requireUppercase: true,
      requireLowercase: true,
      requireNumbers: true,
      requireSpecialChars: true,
    },
    query: {
      maxLength: 5000,
      timeout: 60000, // 60 seconds for complex queries
    }
  },

  // Error Messages
  errors: {
    network: 'Network error. Please check your connection and try again.',
    unauthorized: 'Your session has expired. Please log in again.',
    forbidden: 'You do not have permission to perform this action.',
    notFound: 'The requested resource was not found.',
    serverError: 'An unexpected error occurred. Please try again later.',
    validation: 'Please check your input and try again.',
  }
};

// Function to get dynamic configuration from backend
export const loadDynamicConfig = async () => {
  try {
    const response = await fetch(`${config.api.baseURL}/api/config/frontend`);
    if (response.ok) {
      const dynamicConfig = await response.json();
      // Merge dynamic config with static config
      Object.assign(config, dynamicConfig);
    }
  } catch (error) {
    console.warn('Failed to load dynamic configuration:', error);
  }
};

export default config;