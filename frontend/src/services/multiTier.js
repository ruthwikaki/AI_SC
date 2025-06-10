// frontend/src/services/multiTier.js
import api from './api';

export const multiTierService = {
  // Network Visualization
  getNetworkVisualization: async (params = {}) => {
    const response = await api.get('/multi-tier/network/visualization', { params });
    return response.data;
  },

  // Risk Analysis
  getRiskAnalysis: async (params = {}) => {
    const response = await api.get('/multi-tier/risk/analysis', { params });
    return response.data;
  },

  // Scenario Simulation
  simulateScenario: async (scenarioData) => {
    const response = await api.post('/multi-tier/scenario/simulate', scenarioData);
    return response.data;
  },

  // Supplier Tiers
  getSupplierTiers: async (recalculate = false) => {
    const response = await api.get('/multi-tier/suppliers/tiers', { 
      params: { recalculate } 
    });
    return response.data;
  },

  // Network Metrics
  getNetworkMetrics: async () => {
    const response = await api.get('/multi-tier/network/metrics');
    return response.data;
  }
};

// frontend/src/services/reports.js
import api from './api';

export const reportsService = {
  // Get report templates
  getTemplates: async (category = null) => {
    const response = await api.get('/reports/templates', { 
      params: { category } 
    });
    return response.data;
  },

  // Generate report
  generateReport: async (reportData) => {
    const response = await api.post('/reports/generate', reportData);
    return response.data;
  },

  // List reports
  listReports: async (params = {}) => {
    const response = await api.get('/reports/list', { params });
    return response.data;
  },

  // Download report
  downloadReport: async (reportId, format = 'pdf') => {
    const response = await api.get(`/reports/${reportId}/download`, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  // Schedule report
  scheduleReport: async (scheduleData) => {
    const response = await api.post('/reports/schedule', scheduleData);
    return response.data;
  },

  // Get scheduled reports
  getScheduledReports: async () => {
    const response = await api.get('/reports/scheduled');
    return response.data;
  }
};

// frontend/src/services/settings.js
import api from './api';

export const settingsService = {
  // User Preferences
  getPreferences: async () => {
    const response = await api.get('/settings/preferences');
    return response.data;
  },

  updatePreferences: async (preferences) => {
    const response = await api.put('/settings/preferences', { preferences });
    return response.data;
  },

  // Notifications
  getNotificationSettings: async () => {
    const response = await api.get('/settings/notifications');
    return response.data;
  },

  updateNotificationSettings: async (settings) => {
    const response = await api.put('/settings/notifications', settings);
    return response.data;
  },

  // System Settings (admin)
  getSystemSettings: async () => {
    const response = await api.get('/settings/system');
    return response.data;
  },

  updateSystemSetting: async (key, value) => {
    const response = await api.put(`/settings/system/${key}`, value);
    return response.data;
  },

  // Dashboard Layout
  getDashboardLayout: async () => {
    const response = await api.get('/settings/dashboard-layout');
    return response.data;
  },

  updateDashboardLayout: async (layout) => {
    const response = await api.put('/settings/dashboard-layout', layout);
    return response.data;
  }
};

// frontend/src/services/dashboards.js
import api from './api';

export const dashboardsService = {
  // Dashboard CRUD
  getDashboards: async (shared = null) => {
    const response = await api.get('/dashboards', { 
      params: { shared } 
    });
    return response.data;
  },

  createDashboard: async (dashboardData) => {
    const response = await api.post('/dashboards', dashboardData);
    return response.data;
  },

  getDashboardDetails: async (dashboardId) => {
    const response = await api.get(`/dashboards/${dashboardId}`);
    return response.data;
  },

  updateDashboard: async (dashboardId, updateData) => {
    const response = await api.put(`/dashboards/${dashboardId}`, updateData);
    return response.data;
  },

  deleteDashboard: async (dashboardId) => {
    const response = await api.delete(`/dashboards/${dashboardId}`);
    return response.data;
  },

  // Widget Management
  addWidget: async (dashboardId, widgetData) => {
    const response = await api.post(`/dashboards/${dashboardId}/widgets`, widgetData);
    return response.data;
  },

  updateWidget: async (dashboardId, widgetId, updateData) => {
    const response = await api.put(`/dashboards/${dashboardId}/widgets/${widgetId}`, updateData);
    return response.data;
  },

  removeWidget: async (dashboardId, widgetId) => {
    const response = await api.delete(`/dashboards/${dashboardId}/widgets/${widgetId}`);
    return response.data;
  },

  // Widget Types
  getWidgetTypes: async () => {
    const response = await api.get('/dashboards/widget-types');
    return response.data;
  }
};

// frontend/src/services/suggestions.js
import api from './api';

export const suggestionsService = {
  // Query Suggestions
  getQuerySuggestions: async (partialQuery, limit = 10) => {
    const response = await api.get('/suggestions/queries', {
      params: { partial_query: partialQuery, limit }
    });
    return response.data;
  },

  // Query Templates
  getQueryTemplates: async (category = null) => {
    const response = await api.get('/suggestions/templates', {
      params: { category }
    });
    return response.data;
  },

  // Autocomplete
  getAutocomplete: async (field, value = '', limit = 10) => {
    const response = await api.get('/suggestions/autocomplete', {
      params: { field, value, limit }
    });
    return response.data;
  },

  // Related Queries
  getRelatedQueries: async (queryId = null, queryText = null) => {
    const response = await api.get('/suggestions/related', {
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
    const response = await api.post('/export/create', exportData);
    return response.data;
  },

  // Get Export Jobs
  getExportJobs: async (params = {}) => {
    const response = await api.get('/export/jobs', { params });
    return response.data;
  },

  // Download Export
  downloadExport: async (exportId) => {
    const response = await api.get(`/export/download/${exportId}`, {
      responseType: 'blob'
    });
    return response.data;
  },

  // Quick Export
  quickExport: async (exportData, format = 'csv') => {
    const response = await api.post('/export/quick-export', exportData, {
      params: { format },
      responseType: 'blob'
    });
    return response.data;
  },

  // Get Export Templates
  getExportTemplates: async (category = null) => {
    const response = await api.get('/export/templates', {
      params: { category }
    });
    return response.data;
  },

  // Delete Export Job
  deleteExportJob: async (exportId) => {
    const response = await api.delete(`/export/jobs/${exportId}`);
    return response.data;
  }
};

// frontend/src/services/analytics.js (enhanced)
import api from './api';

export const analyticsService = {
  // Existing methods...
  
  // Dashboard Analytics
  getInventoryOverview: async (params = {}) => {
    const response = await api.get('/analytics/dashboard/inventory/overview', { params });
    return response.data;
  },

  getStockLevelsByCategory: async (timeRange = 7) => {
    const response = await api.get('/analytics/dashboard/inventory/stock-levels', {
      params: { time_range: timeRange }
    });
    return response.data;
  },

  getDeliveryPerformance: async (params = {}) => {
    const response = await api.get('/analytics/dashboard/logistics/delivery-performance', { params });
    return response.data;
  },

  getRouteEfficiency: async (region = null) => {
    const response = await api.get('/analytics/dashboard/logistics/route-efficiency', {
      params: { region }
    });
    return response.data;
  },

  getSupplierPerformanceOverview: async (topN = 10) => {
    const response = await api.get('/analytics/dashboard/supplier/performance-overview', {
      params: { top_n: topN }
    });
    return response.data;
  },

  getComplianceStatus: async () => {
    const response = await api.get('/analytics/dashboard/supplier/compliance-status');
    return response.data;
  },

  getRealtimeKPIs: async () => {
    const response = await api.get('/analytics/dashboard/kpi/realtime');
    return response.data;
  }
};