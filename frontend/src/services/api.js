// src/services/api.js
import axios from 'axios';

// Use the API URL from env, or fallback to empty string for proxy
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// In development with Vite proxy, we use relative URLs
const useProxy = import.meta.env.DEV && !API_BASE_URL;

// Log the environment for debugging
console.log('API Configuration:', {
  environment: import.meta.env.MODE,
  apiUrl: API_BASE_URL || 'Using Vite Proxy',
  useProxy: useProxy,
  vitePort: 3001,
  backendPort: 8000
});

// Create axios instance with longer timeout for Mixtral
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes for Mixtral (was 30 seconds)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    // Special timeout for complex queries
    if (config.url === '/api/queries/execute' && config.data?.query) {
      const query = config.data.query.toLowerCase();
      
      // Extra long timeout for complex analysis queries
      if (query.includes('analyze') || 
          query.includes('dashboard') || 
          query.includes('comprehensive') ||
          query.split(' ').length > 20) {
        config.timeout = 180000; // 3 minutes for complex queries
        console.log('Using extended timeout for complex query');
      }
    }
    
    // Add timestamp to prevent caching issues
    if (config.method === 'get') {
      config.params = {
        ...config.params,
        _t: Date.now()
      };
    }
    
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Log requests in development
    if (import.meta.env.DEV) {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`, config.data);
    }
    
    return config;
  },
  (error) => {
    console.error('Request setup error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    // Log successful responses in development
    if (import.meta.env.DEV) {
      console.log(`API Response: ${response.config.url}`, response.data);
    }
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Log error details
    if (import.meta.env.DEV) {
      console.error('API Error:', {
        url: error.config?.url,
        method: error.config?.method,
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
        code: error.code
      });
    }
    
    // Handle timeout specifically for Mixtral
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      console.error('Query timeout - Mixtral needs more time');
      const timeoutError = new Error(
        'Query took too long to process. Try breaking it into smaller parts:\n' +
        '• Instead of "analyze everything", try specific questions\n' +
        '• Ask for one chart at a time\n' +
        '• Use simpler language'
      );
      timeoutError.isTimeout = true;
      timeoutError.originalQuery = originalRequest.data?.query;
      return Promise.reject(timeoutError);
    }
    
    // Check if this is a network error (no response)
    if (!error.response && error.code === 'ERR_NETWORK') {
      console.error('Network Error - Backend is not reachable');
      const networkError = new Error(
        'Cannot connect to backend server. Please ensure the server is running on http://localhost:8000'
      );
      networkError.isNetworkError = true;
      return Promise.reject(networkError);
    }
    
    // Handle client closed error specifically
    if (error.message && error.message.includes('client has been closed')) {
      console.error('Connection closed - Attempting to reconnect...');
      // Create a new instance to reset the connection
      api.defaults.adapter = null;
      
      // Retry the request once
      if (!originalRequest._retry) {
        originalRequest._retry = true;
        return api(originalRequest);
      }
    }
    
    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      // Clear auth data
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('user');
      
      // Only redirect to login if we're not already there
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    
    // Extract error message
    let errorMessage = 'An unexpected error occurred';
    
    if (error.response?.data) {
      errorMessage = error.response.data.message || 
                    error.response.data.error || 
                    error.response.data.detail || 
                    errorMessage;
    } else if (error.message) {
      errorMessage = error.message;
    }
    
    // Format the error
    const formattedError = new Error(errorMessage);
    formattedError.status = error.response?.status;
    formattedError.data = error.response?.data;
    formattedError.isNetworkError = !error.response;
    
    return Promise.reject(formattedError);
  }
);

// Health check with retry
const checkBackendHealth = async (retries = 3) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await api.get('/health', {
        timeout: 5000,
        validateStatus: (status) => status < 500
      });
      return response.data;
    } catch (error) {
      console.log(`Health check attempt ${i + 1} failed:`, error.message);
      if (i < retries - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second before retry
      }
    }
  }
  return { status: 'error', message: 'Backend unreachable' };
};

// Enhanced API helper with query-specific methods
const apiHelper = {
  get: (url, params) => api.get(url, { params }),
  post: (url, data) => api.post(url, data),
  put: (url, data) => api.put(url, data),
  patch: (url, data) => api.patch(url, data),
  delete: (url) => api.delete(url),
  
  // Health check
  checkHealth: checkBackendHealth,
  
  // Test connection
  testConnection: async () => {
    try {
      const response = await api.get('/test', { timeout: 5000 });
      return { connected: true, data: response.data };
    } catch (error) {
      return { connected: false, error: error.message };
    }
  },
  
  // Special method for executing queries with Mixtral
  executeQuery: async (query) => {
    try {
      // Show loading message for long queries
      if (query.toLowerCase().includes('analyze') || query.toLowerCase().includes('dashboard')) {
        console.log('🤖 Complex query detected - Mixtral is thinking...');
      }
      
      const response = await api.post('/api/queries/execute', { query });
      return response.data;
    } catch (error) {
      // Provide helpful suggestions on timeout
      if (error.isTimeout) {
        console.log('💡 Suggestion: Try these simpler queries:');
        console.log('  - "show me all products"');
        console.log('  - "create a bar chart of supplier ratings"');
        console.log('  - "which products need reordering?"');
      }
      throw error;
    }
  }
};

// Export both the instance and helper
export { apiHelper };
export default api;