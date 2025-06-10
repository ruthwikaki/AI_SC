// frontend/src/services/dashboards.js
import api from './api';

export const dashboardsService = {
  // Dashboard CRUD
  getDashboards: async (shared = null) => {
    const response = await api.get('/api/dashboards', { 
      params: { shared } 
    });
    return response.data;
  },

  createDashboard: async (dashboardData) => {
    const response = await api.post('/api/dashboards', dashboardData);
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
    const response = await api.get('/api/dashboards/widget-types');
    return response.data;
  }
};