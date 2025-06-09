import { useState, useCallback } from 'react';
import { getForecastData, runForecast as runForecastAPI } from '../../services/forecasting';

export const useForecast = (type = 'all') => {
  const [forecasts, setForecasts] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchForecasts = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getForecastData({ forecast_type: type });
      setForecasts(data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }, [type]);

  const runForecast = useCallback(async (params) => {
    try {
      setLoading(true);
      setError(null);
      const data = await runForecastAPI(params);
      return data;
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshForecasts = async () => {
    await fetchForecasts();
  };

  return { forecasts, loading, error, refreshForecasts, runForecast };
};
