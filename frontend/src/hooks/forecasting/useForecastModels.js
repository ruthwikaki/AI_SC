import { useState, useEffect } from 'react';
import { getForecastMethods } from '../../services/forecasting';

export const useForecastModels = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);
        const data = await getForecastMethods();
        setModels(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  return { models, loading, error };
};

