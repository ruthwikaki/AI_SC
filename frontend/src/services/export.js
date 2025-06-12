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
