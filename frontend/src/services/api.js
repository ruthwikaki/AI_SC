// frontend/src/services/api.js
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  });

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors and token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    // Handle 401 Unauthorized errors
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/api/auth/refresh`, {
            refresh_token: refreshToken,
          });

          const { access_token } = response.data;
          localStorage.setItem('access_token', access_token);

          // Retry the original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return api(originalRequest);
        }
      } catch (refreshError) {
        // Refresh failed, redirect to login
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }

    // Handle other errors
    if (error.response?.status === 403) {
      console.error('Access forbidden:', error.response.data);
    } else if (error.response?.status === 404) {
      console.error('Resource not found:', error.response.data);
    } else if (error.response?.status >= 500) {
      console.error('Server error:', error.response.data);
    }

    return Promise.reject(error);
  }
);

// Analytics API endpoints
export const analytics = {
  // Inventory Optimization
  inventory: {
    getOverview: () => api.get('/api/analytics/inventory/overview'),
    getProducts: () => api.get('/api/analytics/inventory/products'),
    calculateSafetyStock: (data) => api.post('/api/analytics/inventory/safety-stock', data),
    performABCAnalysis: (data) => api.post('/api/analytics/inventory/abc-analysis', data),
    generateForecast: (data) => api.post('/api/analytics/inventory/forecast', data),
    export: (type) => api.get(`/api/analytics/inventory/export/${type}`, { responseType: 'blob' }),
  },

  // Logistics Analytics
  logistics: {
    getOverview: () => api.get('/api/analytics/logistics/overview'),
    getCarriers: () => api.get('/api/analytics/logistics/carriers'),
    getRoutes: () => api.get('/api/analytics/logistics/routes'),
    optimizeRoutes: (data) => api.post('/api/analytics/logistics/optimize-routes', data),
    analyzeCarrierPerformance: (data) => api.post('/api/analytics/logistics/carrier-performance', data),
    analyzeDeliveries: (data) => api.post('/api/analytics/logistics/delivery-analytics', data),
    exportRoutes: (data) => api.post('/api/analytics/logistics/export/routes', data, { responseType: 'blob' }),
  },

  // Supplier Performance
  supplier: {
    getOverview: () => api.get('/api/analytics/supplier/overview'),
    getList: () => api.get('/api/analytics/supplier/list'),
    generateScorecard: (data) => api.post('/api/analytics/supplier/scorecard', data),
    analyzeRisk: (data) => api.post('/api/analytics/supplier/risk-analysis', data),
    checkCompliance: (data) => api.post('/api/analytics/supplier/compliance-check', data),
    exportReport: (supplierId) => api.get(`/api/analytics/supplier/export/${supplierId}`, { responseType: 'blob' }),
  },
};

// Visualization API endpoints
export const visualizations = {
  // Charts
  recommendChart: (data) => api.post('/api/visualizations/recommend', data),
  generateChart: (type, data) => api.post(`/api/visualizations/generate/${type}`, data),
  saveChart: (data) => api.post('/api/visualizations/save', data),
  getSavedCharts: () => api.get('/api/visualizations/saved'),
  deleteChart: (id) => api.delete(`/api/visualizations/charts/${id}`),
  
  // Dashboards
  getDashboards: () => api.get('/api/visualizations/dashboards'),
  getDashboard: (id) => api.get(`/api/visualizations/dashboards/${id}`),
  createDashboard: (data) => api.post('/api/visualizations/dashboards', data),
  updateDashboard: (id, data) => api.put(`/api/visualizations/dashboards/${id}`, data),
  deleteDashboard: (id) => api.delete(`/api/visualizations/dashboards/${id}`),
  exportDashboard: (id) => api.get(`/api/visualizations/dashboards/${id}/export`, { responseType: 'blob' }),
};

// Database/Query API endpoints
export const database = {
  getSchema: () => api.get('/api/database/schema'),
  getTables: () => api.get('/api/database/tables'),
  getTableSchema: (tableName) => api.get(`/api/database/tables/${tableName}`),
  executeQuery: (query) => api.post('/api/database/query', { query }),
  getConnections: () => api.get('/api/database/connections'),
  createConnection: (data) => api.post('/api/database/connections', data),
  testConnection: (id) => api.post(`/api/database/connections/${id}/test`),
  deleteConnection: (id) => api.delete(`/api/database/connections/${id}`),
  getRelationships: () => api.get('/api/database/relationships'),
  syncDatabase: () => api.post('/api/database/sync'),
};

// Natural Language Query API endpoints
export const queries = {
  naturalLanguage: (query) => api.post('/api/queries/natural-language', { query }),
  getHistory: () => api.get('/api/queries/history'),
  getSuggestions: (query) => api.post('/api/queries/suggest', { query }),
  saveQuery: (data) => api.post('/api/queries/save', data),
  getSavedQueries: () => api.get('/api/queries/saved'),
  deleteQuery: (id) => api.delete(`/api/queries/${id}`),
};

// Authentication API endpoints
export const auth = {
  // FIXED: Login needs to use form data for OAuth2
  login: async (credentials) => {
    // Create form data for OAuth2 token endpoint
    const formData = new URLSearchParams();
    formData.append('username', credentials.username);
    formData.append('password', credentials.password);
    
    // Use the api instance but override content-type for this request
    return api.post('/api/auth/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
  },
  register: (userData) => api.post('/api/auth/register', userData),
  logout: () => api.post('/api/auth/logout'),
  getCurrentUser: () => api.get('/api/auth/me'),
  updateProfile: (data) => api.put('/api/auth/profile', data),
  changePassword: (data) => api.post('/api/auth/change-password', data),
  refresh: (refreshToken) => api.post('/api/auth/refresh', { refresh_token: refreshToken }),
};

// Admin API endpoints
export const admin = {
  // User Management
  getUsers: () => api.get('/api/admin/users'),
  getUser: (id) => api.get(`/api/admin/users/${id}`),
  createUser: (data) => api.post('/api/admin/users', data),
  updateUser: (id, data) => api.put(`/api/admin/users/${id}`, data),
  deleteUser: (id) => api.delete(`/api/admin/users/${id}`),
  
  // Client Management
  getClients: () => api.get('/api/admin/clients'),
  getClient: (id) => api.get(`/api/admin/clients/${id}`),
  createClient: (data) => api.post('/api/admin/clients', data),
  updateClient: (id, data) => api.put(`/api/admin/clients/${id}`, data),
  deleteClient: (id) => api.delete(`/api/admin/clients/${id}`),
  
  // Role Management
  getRoles: () => api.get('/api/admin/roles'),
  createRole: (data) => api.post('/api/admin/roles', data),
  updateRole: (id, data) => api.put(`/api/admin/roles/${id}`, data),
  deleteRole: (id) => api.delete(`/api/admin/roles/${id}`),
  
  // System Management
  getAuditLogs: (params) => api.get('/api/admin/audit-logs', { params }),
  getSystemStats: () => api.get('/api/admin/system/stats'),
  getActiveModels: () => api.get('/api/admin/models/active'),
  setActiveModel: (model) => api.post('/api/admin/models/active', { model }),
  triggerDatabaseSync: () => api.post('/api/admin/database/trigger-sync'),
};

// Multi-tier Network API endpoints
export const multiTier = {
  getNetworkVisualization: (params) => api.get('/api/multi-tier/network', { params }),
  analyzeBottlenecks: (data) => api.post('/api/multi-tier/bottlenecks', data),
  calculateRiskPropagation: (data) => api.post('/api/multi-tier/risk-propagation', data),
  getSupplierNetwork: (supplierId) => api.get(`/api/multi-tier/suppliers/${supplierId}/network`),
  runScenarioSimulation: (data) => api.post('/api/multi-tier/simulate', data),
  exportNetworkData: (format) => api.get(`/api/multi-tier/export/${format}`, { responseType: 'blob' }),
};

// WebSocket connection for real-time updates
export const createWebSocketConnection = (endpoint) => {
  const wsUrl = API_BASE_URL.replace('http', 'ws') + endpoint;
  const token = localStorage.getItem('access_token');
  
  return new WebSocket(`${wsUrl}?token=${token}`);
};

export default api;

