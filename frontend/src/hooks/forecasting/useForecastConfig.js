import { useState, useEffect } from 'react';
import { getForecastConfig, saveForecastConfig } from '../../services/forecasting';

export const useForecastConfig = () => {
  const [config, setConfig] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        setLoading(true);
        const data = await getForecastConfig();
        setConfig(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  const updateConfig = async (newConfig) => {
    try {
      await saveForecastConfig(newConfig);
      setConfig(newConfig);
    } catch (err) {
      setError(err.message);
    }
  };

  return { config, updateConfig, loading, error };
};
