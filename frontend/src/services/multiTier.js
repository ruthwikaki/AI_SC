// frontend/src/services/multiTier.js
import api from './api';

export const multiTierService = {
  // Network Visualization
  getNetworkVisualization: async (params = {}) => {
    const response = await api.get('/api/api/multi-tier/network/visualization', { params });
    return response.data;
  },

  // Risk Analysis
  getRiskAnalysis: async (params = {}) => {
    const response = await api.get('/api/api/multi-tier/risk/analysis', { params });
    return response.data;
  },

  // Scenario Simulation
  simulateScenario: async (scenarioData) => {
    const response = await api.post('/api/api/multi-tier/scenario/simulate', scenarioData);
    return response.data;
  },

  // Supplier Tiers
  getSupplierTiers: async (recalculate = false) => {
    const response = await api.get('/api/api/multi-tier/suppliers/tiers', { 
      params: { recalculate } 
    });
    return response.data;
  },

  // Network Metrics
  getNetworkMetrics: async () => {
    const response = await api.get('/api/api/multi-tier/network/metrics');
    return response.data;
  }
};

// frontend/src/services/reports.js
import api from './api';

export const reportsService = {
  // Get report templates
  getTemplates: async (category = null) => {
    const response = await api.get('/api/api/reports/templates', { 
      params: { category } 
    });
    return response.data;
  },

  // Generate report
  generateReport: async (reportData) => {
    const response = await api.post('/api/api/reports/generate', reportData);
    return response.data;
  },

  // List reports
  listReports: async (params = {}) => {
    const response = await api.get('/api/api/reports/list', { params });
    return response.data;
  },

  // Download report
  downloadReport: async (reportId, format = 'pdf') => {
    const response = await api.get(`/api/reports/${reportId}/download`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  // Schedule report
  scheduleReport: async (scheduleData) => {
    const response = await api.post('/api/api/reports/schedule', scheduleData);
    return response.data;
  },

  // Get scheduled reports
  getScheduledReports: async () => {
    const response = await api.get('/api/api/reports/scheduled');
    return response.data;
  }
};

// frontend/src/services/settings.js
import api from './api';

export const settingsService = {
  // User Preferences
  getPreferences: async () => {
    const response = await api.get('/api/api/settings/preferences');
    return response.data;
  },

  updatePreferences: async (preferences) => {
    const response = await api.put('/api/api/settings/preferences', { preferences });
    return response.data;
  },

  // Notifications
  getNotificationSettings: async () => {
    const response = await api.get('/api/api/settings/notifications');
    return response.data;
  },

  updateNotificationSettings: async (settings) => {
    const response = await api.put('/api/api/settings/notifications', settings);
    return response.data;
  },

  // System Settings (admin)
  getSystemSettings: async () => {
    const response = await api.get('/api/api/settings/system');
    return response.data;
  },

  updateSystemSetting: async (key, value) => {
    const response = await api.put(`/api/settings/system/${key}`, value);
    return response.data;
  },

  // Dashboard Layout
  getDashboardLayout: async () => {
    const response = await api.get('/api/api/settings/dashboard-layout');
    return response.data;
  },

  updateDashboardLayout: async (layout) => {
    const response = await api.put('/api/api/settings/dashboard-layout', layout);
    return response.data;
  }
};

// frontend/src/services/dashboards.js
import api from './api';

export const dashboardsService = {
  // Dashboard CRUD
  getDashboards: async (shared = null) => {
    const response = await api.get('/api/api/dashboards', { 
      params: { shared } 
    });
    return response.data;
  },

  createDashboard: async (dashboardData) => {
    const response = await api.post('/api/api/dashboards', dashboardData);
    return response.data;
  },

  getDashboardDetails: async (dashboardId) => {
    const response = await api.get(`/api/dashboards/${dashboardId}`);
    return response.data;
  },

  updateDashboard: async (dashboardId, updateData) => {
    const response = await api.put(`/api/dashboards/${dashboardId}`, updateData);
    return response.data;
  },

  deleteDashboard: async (dashboardId) => {
    const response = await api.delete(`/api/dashboards/${dashboardId}`);
    return response.data;
  },

  // Widget Management
  addWidget: async (dashboardId, widgetData) => {
    const response = await api.post(`/api/dashboards/${dashboardId}/widgets`, widgetData);
    return response.data;
  },

  updateWidget: async (dashboardId, widgetId, updateData) => {
    const response = await api.put(`/api/dashboards/${dashboardId}/widgets/${widgetId}`, updateData);
    return response.data;
  },

  removeWidget: async (dashboardId, widgetId) => {
    const response = await api.delete(`/api/dashboards/${dashboardId}/widgets/${widgetId}`);
    return response.data;
  },

  // Widget Types
  getWidgetTypes: async () => {
    const response = await api.get('/api/api/dashboards/widget-types');
    return response.data;
  }
};

// frontend/src/services/suggestions.js
import api from './api';

export const suggestionsService = {
  // Query Suggestions
  getQuerySuggestions: async (partialQuery, limit = 10) => {
    const response = await api.get('/api/api/suggestions/queries', {
      params: { partial_query: partialQuery, limit }
    });
    return response.data;
  },

  // Query Templates
  getQueryTemplates: async (category = null) => {
    const response = await api.get('/api/api/suggestions/templates', {
      params: { category }
    });
    return response.data;
  },

  // Autocomplete
  getAutocomplete: async (field, value = '', limit = 10) => {
    const response = await api.get('/api/api/suggestions/autocomplete', {
      params: { field, value, limit }
    });
    return response.data;
  },

  // Related Queries
  getRelatedQueries: async (queryId = null, queryText = null) => {
    const response = await api.get('/api/api/suggestions/related', {
      params: { query_id: queryId, query_text: queryText }
    });
    return response.data;
  }
};

// frontend/src/services/export.js
import api from './api';

export const exportService = {
  // Create Export
  createExport: async (exportData) => {
    const response = await api.post('/api/api/export/create', exportData);
    return response.data;
  },

  // Get Export Jobs
  getExportJobs: async (params = {}) => {
    const response = await api.get('/api/api/export/jobs', { params });
    return response.data;
  },

  // Download Export
  downloadExport: async (exportId) => {
    const response = await api.get(`/api/export/download/${exportId}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  // Quick Export
  quickExport: async (exportData, format = 'csv') => {
    const response = await api.post('/api/api/export/quick-export', exportData, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  // Get Export Templates
  getExportTemplates: async (category = null) => {
    const response = await api.get('/api/api/export/templates', {
      params: { category }
    });
    return response.data;
  },

  // Delete Export Job
  deleteExportJob: async (exportId) => {
    const response = await api.delete(`/api/export/jobs/${exportId}`);
    return response.data;
  }
};

// frontend/src/services/analytics.js (enhanced)
import api from './api';

export const analyticsService = {
  // Existing methods...
  
  // Dashboard Analytics
  getInventoryOverview: async (params = {}) => {
    const response = await api.get('/api/api/analytics/dashboard/inventory/overview', { params });
    return response.data;
  },

  getStockLevelsByCategory: async (timeRange = 7) => {
    const response = await api.get('/api/api/analytics/dashboard/inventory/stock-levels', {
      params: { time_range: timeRange }
    });
    return response.data;
  },

  getDeliveryPerformance: async (params = {}) => {
    const response = await api.get('/api/api/analytics/dashboard/logistics/delivery-performance', { params });
    return response.data;
  },

  getRouteEfficiency: async (region = null) => {
    const response = await api.get('/api/api/analytics/dashboard/logistics/route-efficiency', {
      params: { region }
    });
    return response.data;
  },

  getSupplierPerformanceOverview: async (topN = 10) => {
    const response = await api.get('/api/api/analytics/dashboard/supplier/performance-overview', {
      params: { top_n: topN }
    });
    return response.data;
  },

  getComplianceStatus: async () => {
    const response = await api.get('/api/api/analytics/dashboard/supplier/compliance-status');
    return response.data;
  },

  getRealtimeKPIs: async () => {
    const response = await api.get('/api/api/analytics/dashboard/kpi/realtime');
    return response.data;
  }
};
