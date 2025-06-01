import axios from 'axios';

// CHANGE THIS: Use empty string to let Vite proxy handle the routing
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken') || sessionStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Add detailed logging
    console.error('API Error Details:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      headers: error.response?.headers,
      message: error.message,
      code: error.code
    });
    
    // Handle token expiration and refresh
    if (error.response && error.response.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        const refreshToken = localStorage.getItem('refreshToken') || sessionStorage.getItem('refreshToken');
        
        if (refreshToken) {
          // CHANGE THIS: Remove API_BASE_URL since we're using relative URLs
          const res = await axios.post('/api/auth/refresh-token', {
            refresh_token: refreshToken,
          });
          
          if (res.data.token) {
            localStorage.setItem('authToken', res.data.token);
            api.defaults.headers.common['Authorization'] = `Bearer ${res.data.token}`;
            return api(originalRequest);
          }
        }
      } catch (refreshError) {
        console.error('Token refresh failed:', refreshError);
        
        localStorage.removeItem('authToken');
        localStorage.removeItem('refreshToken');
        localStorage.removeItem('user');
        window.location.href = '/login';
      }
    }
    
    // Check if this is a network error (no response)
    if (!error.response) {
      console.error('Network Error - Backend might be down or CORS issue');
      const networkError = new Error('Unable to connect to server. Please check if the backend is running.');
      networkError.status = null;
      networkError.data = null;
      return Promise.reject(networkError);
    }
    
    // Extract error message
    let errorMessage = 'An unexpected error occurred';
    
    if (error.response && error.response.data) {
      errorMessage = error.response.data.message || error.response.data.error || errorMessage;
    } else if (error.message) {
      errorMessage = error.message;
    }
    
    // Format the error for consistent handling
    const formattedError = new Error(errorMessage);
    formattedError.status = error.response ? error.response.status : null;
    formattedError.data = error.response ? error.response.data : null;
    formattedError.originalError = error; // Keep original for debugging
    
    return Promise.reject(formattedError);
  }
);

// API helper object with common methods
export const apiHelper = {
  get: (url, params) => api.get(url, { params }),
  post: (url, data) => api.post(url, data),
  put: (url, data) => api.put(url, data),
  patch: (url, data) => api.patch(url, data),
  delete: (url) => api.delete(url),
};

export default api;