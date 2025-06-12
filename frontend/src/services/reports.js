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
