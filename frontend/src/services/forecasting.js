// frontend/src/services/forecasting.js
import api from './api';

// Forecast Methods (matching your backend enum)
export const FORECAST_METHODS = {
  MOVING_AVERAGE: 'moving_average',
  EXPONENTIAL_SMOOTHING: 'exponential_smoothing',
  ARIMA: 'arima',
  SARIMA: 'sarima',
  PROPHET: 'prophet',
  LSTM: 'lstm',
};

// Time Frames (matching your backend enum)
export const TIME_FRAMES = {
  LAST_WEEK: 'last_week',
  LAST_MONTH: 'last_month',
  LAST_QUARTER: 'last_quarter',
  LAST_YEAR: 'last_year',
  CUSTOM: 'custom',
  YEAR_TO_DATE: 'year_to_date',
};

// Get forecast data using your existing endpoint
export const getForecastData = async (params = {}) => {
  const defaultParams = {
    forecast_periods: 12,
    period_type: 'month',
    method: FORECAST_METHODS.EXPONENTIAL_SMOOTHING,
    include_confidence_intervals: true,
    confidence_level: 0.95,
    time_frame: TIME_FRAMES.LAST_MONTH,
  };

  const response = await api.post('/api/analytics/inventory/forecast', {
    ...defaultParams,
    ...params,
  });
  
  return response.data;
};

// Get available forecast models (we'll define them client-side since your backend uses an enum)
export const getAvailableModels = async () => {
  // Return the models your backend supports
  return [
    {
      id: FORECAST_METHODS.MOVING_AVERAGE,
      name: 'Moving Average',
      description: 'Simple moving average forecast',
      parameters: {
        window: { type: 'number', default: 3, min: 1, max: 12 },
      },
    },
    {
      id: FORECAST_METHODS.EXPONENTIAL_SMOOTHING,
      name: 'Exponential Smoothing',
      description: 'Exponential smoothing with configurable alpha',
      parameters: {
        alpha: { type: 'number', default: 0.3, min: 0.01, max: 0.99, step: 0.01 },
      },
    },
    {
      id: FORECAST_METHODS.ARIMA,
      name: 'ARIMA',
      description: 'AutoRegressive Integrated Moving Average',
      parameters: {
        p: { type: 'number', default: 1, min: 0, max: 5 },
        d: { type: 'number', default: 1, min: 0, max: 2 },
        q: { type: 'number', default: 1, min: 0, max: 5 },
      },
    },
    {
      id: FORECAST_METHODS.SARIMA,
      name: 'Seasonal ARIMA',
      description: 'ARIMA with seasonal components',
      parameters: {
        p: { type: 'number', default: 1, min: 0, max: 5 },
        d: { type: 'number', default: 1, min: 0, max: 2 },
        q: { type: 'number', default: 1, min: 0, max: 5 },
        seasonal_period: { type: 'number', default: 12, min: 1, max: 52 },
      },
    },
    {
      id: FORECAST_METHODS.PROPHET,
      name: 'Prophet',
      description: 'Facebook Prophet for time series forecasting',
      parameters: {
        changepoint_prior_scale: { type: 'number', default: 0.05, min: 0.001, max: 0.5, step: 0.001 },
        seasonality_mode: { type: 'select', options: ['additive', 'multiplicative'], default: 'additive' },
      },
    },
    {
      id: FORECAST_METHODS.LSTM,
      name: 'LSTM Neural Network',
      description: 'Long Short-Term Memory neural network',
      parameters: {
        sequence_length: { type: 'number', default: 30, min: 10, max: 100 },
        epochs: { type: 'number', default: 100, min: 50, max: 500 },
      },
    },
  ];
};

// Get time series data for a specific product or category
export const getTimeSeriesData = async (params = {}) => {
  // Use your existing inventory interface to get historical data
  // This might need to be adjusted based on your actual endpoints
  const response = await api.get('/api/analytics/inventory/products', {
    params: {
      ...params,
      include_history: true,
    },
  });
  
  return response.data;
};

// Get forecast configuration (stored in user preferences)
export const getForecastConfig = async () => {
  const response = await api.get('/api/analytics/dashboard/preferences');
  return response.data.preferences.forecast_config || {};
};

// Save forecast configuration
export const saveForecastConfig = async (config) => {
  const currentPrefs = await api.get('/api/analytics/dashboard/preferences');
  const response = await api.post('/api/analytics/dashboard/preferences', {
    ...currentPrefs.data.preferences,
    forecast_config: config,
  });
  return response.data;
};

// Run forecast with specific parameters
export const runForecast = async (params) => {
  const response = await api.post('/api/analytics/inventory/forecast', params);
  return response.data;
};

// Compare multiple forecast models
export const compareModels = async (models, params) => {
  // Run forecasts for each model in parallel
  const forecastPromises = models.map(model => 
    runForecast({
      ...params,
      method: model,
    })
  );
  
  const results = await Promise.all(forecastPromises);
  
  // Combine results for comparison
  return {
    models: models.map((model, index) => ({
      id: model,
      name: getModelName(model),
      results: results[index],
    })),
    comparison_date: new Date().toISOString(),
  };
};

// Export forecast results
export const exportForecast = async (forecastData, format = 'csv') => {
  // Create export data
  const exportData = {
    forecast_data: forecastData,
    format: format,
  };
  
  // Use your existing export endpoint
  const response = await api.post('/api/analytics/inventory/export/forecast', exportData, {
    responseType: 'blob',
  });
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `forecast_${new Date().toISOString()}.${format}`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  
  return response.data;
};

// Get specific forecast types (demand, inventory, sales)
export const getDemandForecast = async (params = {}) => {
  return runForecast({
    ...params,
    forecast_type: 'demand',
  });
};

export const getInventoryForecast = async (params = {}) => {
  return runForecast({
    ...params,
    forecast_type: 'inventory',
  });
};

export const getSalesForecast = async (params = {}) => {
  return runForecast({
    ...params,
    forecast_type: 'sales',
  });
};

// Helper function to get model display name
const getModelName = (modelId) => {
  const modelMap = {
    [FORECAST_METHODS.MOVING_AVERAGE]: 'Moving Average',
    [FORECAST_METHODS.EXPONENTIAL_SMOOTHING]: 'Exponential Smoothing',
    [FORECAST_METHODS.ARIMA]: 'ARIMA',
    [FORECAST_METHODS.SARIMA]: 'Seasonal ARIMA',
    [FORECAST_METHODS.PROPHET]: 'Prophet',
    [FORECAST_METHODS.LSTM]: 'LSTM Neural Network',
  };
  return modelMap[modelId] || modelId;
};

// Get forecast accuracy metrics
export const getForecastAccuracy = async (forecastId) => {
  // This would need a backend endpoint to calculate accuracy
  // For now, we'll calculate it client-side if we have the data
  const forecast = await getForecastData({ forecast_id: forecastId });
  
  if (forecast.historical_data && forecast.results) {
    // Calculate metrics using the helper functions
    const actual = forecast.historical_data.map(d => d.value);
    const predicted = forecast.results.forecast.map(f => f.value);
    
    return {
      mape: calculateMAPE(actual.slice(-predicted.length), predicted),
      mae: calculateMAE(actual.slice(-predicted.length), predicted),
      rmse: calculateRMSE(actual.slice(-predicted.length), predicted),
    };
  }
  
  return null;
};

// Import the helper functions from forecastHelpers
import { calculateMAPE, calculateMAE, calculateRMSE } from '../utils/forecasting/forecastHelpers';