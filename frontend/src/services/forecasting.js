// frontend/src/services/forecasting.js
import api from './api';

// Time frame options
export const TIME_FRAMES = {
  LAST_WEEK: 'last_week',
  LAST_MONTH: 'last_month',
  LAST_QUARTER: 'last_quarter',
  LAST_YEAR: 'last_year',
  YEAR_TO_DATE: 'year_to_date',
  CUSTOM: 'custom'
};

// Fetch forecast methods dynamically from backend
export const getForecastMethods = async () => {
  try {
    const response = await api.get('/api/reference/forecast-methods');
    return response.data;
  } catch (error) {
    console.error('Error fetching forecast methods:', error);
    return [];
  }
};

// Get warehouses from backend
export const getWarehouses = async () => {
  try {
    const response = await api.get('/api/reference/warehouses');
    return response.data;
  } catch (error) {
    console.error('Error fetching warehouses:', error);
    return [];
  }
};

// Get regions from backend
export const getRegions = async () => {
  try {
    const response = await api.get('/api/reference/regions');
    return response.data;
  } catch (error) {
    console.error('Error fetching regions:', error);
    return [];
  }
};

// Get product categories with statistics
export const getProductCategories = async () => {
  try {
    const response = await api.get('/api/reference/product-categories');
    return response.data;
  } catch (error) {
    console.error('Error fetching product categories:', error);
    return [];
  }
};

// Main forecast function
export const runForecast = async (params) => {
  try {
    const response = await api.post('/api/analytics/inventory/forecast', {
      request_parameters: {
        method: params.method,
        time_frame: params.time_frame,
        forecast_periods: params.forecast_periods,
        period_type: params.period_type || 'month',
        confidence_level: params.confidence_level || 0.95,
        include_anomaly_detection: params.include_anomaly_detection || true,
        include_insights: params.include_insights || true,
        filters: {
          product_category: params.product_category,
          warehouse_id: params.warehouse_id,
          region: params.region,
          supplier_id: params.supplier_id,
        }
      }
    });
    return response.data;
  } catch (error) {
    console.error('Error running forecast:', error);
    throw error;
  }
};

// Get forecast data with filters
export const getForecastData = async (params = {}) => {
  try {
    const response = await api.get('/api/analytics/inventory/forecast', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching forecast data:', error);
    throw error;
  }
};

// Compare multiple forecast models
export const compareModels = async (models, params) => {
  try {
    const promises = models.map(model => 
      runForecast({ ...params, method: model })
    );
    
    const results = await Promise.all(promises);
    
    return {
      models: models.map((model, index) => ({
        id: model,
        results: results[index]
      }))
    };
  } catch (error) {
    console.error('Error comparing models:', error);
    throw error;
  }
};

// Get historical forecast performance
export const getForecastPerformance = async (params = {}) => {
  try {
    const response = await api.get('/api/analytics/forecast/performance', { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching forecast performance:', error);
    throw error;
  }
};

// Save forecast configuration
export const saveForecastConfig = async (config) => {
  try {
    const response = await api.post('/api/analytics/dashboard/preferences', {
      preference_type: 'forecast_config',
      preferences: config
    });
    return response.data;
  } catch (error) {
    console.error('Error saving forecast config:', error);
    throw error;
  }
};

// Get forecast configuration
export const getForecastConfig = async () => {
  try {
    const response = await api.get('/api/analytics/dashboard/preferences', {
      params: { preference_type: 'forecast_config' }
    });
    return response.data.preferences || {};
  } catch (error) {
    console.error('Error fetching forecast config:', error);
    return {};
  }
};

// Export forecast data
export const exportForecast = async (data, format = 'csv') => {
  try {
    const response = await api.post('/api/analytics/export', {
      data,
      format,
      type: 'forecast'
    }, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `forecast_${new Date().toISOString()}.${format}`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    
    return true;
  } catch (error) {
    console.error('Error exporting forecast:', error);
    throw error;
  }
};

// Get product-level forecast data
export const getProductForecast = async (productId, params = {}) => {
  try {
    const response = await api.get(`/api/analytics/products/${productId}/forecast`, { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching product forecast:', error);
    throw error;
  }
};

// Batch forecast for multiple products
export const batchProductForecast = async (productIds, params = {}) => {
  try {
    const response = await api.post('/api/analytics/forecast/batch', {
      product_ids: productIds,
      ...params
    });
    return response.data;
  } catch (error) {
    console.error('Error running batch forecast:', error);
    throw error;
  }
};