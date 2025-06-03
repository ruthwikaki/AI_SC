import { useState, useCallback } from 'react';
import api from '../services/api';

export const useQuery = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);

  const executeQuery = useCallback(async (query) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await api.post('/queries/execute', { query });
      return response.data;
    } catch (err) {
      setError(err.message || 'Failed to execute query');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadSuggestions = useCallback(async () => {
    try {
      // For now, return empty array since the endpoint doesn't exist
      setSuggestions([]);
      return [];
    } catch (err) {
      console.error('Error loading suggestions:', err);
      setSuggestions([]);
      return [];
    }
  }, []);

  const getRecentQueries = useCallback(async () => {
    try {
      // Return empty array since endpoint doesn't exist
      return [];
    } catch (err) {
      console.error('Error fetching recent queries:', err);
      return [];
    }
  }, []);

  return {
    executeQuery,
    loadSuggestions,
    getRecentQueries,
    suggestions,
    isLoading,
    error
  };
};
