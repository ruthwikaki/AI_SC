// Forecast store for state management
// This is a simple example - you can use Redux, Zustand, or Context API

import { create } from 'zustand';

const useForecastStore = create((set) => ({
  // State
  forecasts: {},
  selectedModel: null,
  config: {},
  isLoading: false,
  error: null,
  
  // Actions
  setForecasts: (forecasts) => set({ forecasts }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  setConfig: (config) => set({ config }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  
  // Thunks
  loadForecasts: async (type) => {
    set({ isLoading: true, error: null });
    try {
      // API call to load forecasts
      const response = await fetch(`/api/analytics/forecast/${type}`);
      const data = await response.json();
      set({ forecasts: data, isLoading: false });
    } catch (error) {
      set({ error: error.message, isLoading: false });
    }
  },
  
  clearError: () => set({ error: null })
}));

export default useForecastStore;
