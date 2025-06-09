import { useState, useEffect } from 'react';
import { getTimeSeriesData } from '../../services/forecasting';

export const useTimeSeriesData = (dataType, dateRange) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!dataType) return;
      
      try {
        setLoading(true);
        const result = await getTimeSeriesData(dataType, dateRange);
        setData(result);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [dataType, dateRange]);

  return { data, loading, error };
};
