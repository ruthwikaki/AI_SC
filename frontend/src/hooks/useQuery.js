import { useState, useCallback } from 'react';
import queryService from '../services/query';

const useQuery = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [savedQueries, setSavedQueries] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  
  // Execute a natural language query
  const executeQuery = useCallback(async (queryText, options = {}) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Execute the query via the API
      const result = await queryService.executeQuery(queryText, options);
      
      // Add to history
      const queryWithResult = {
        id: Date.now(),
        text: queryText,
        timestamp: new Date().toISOString(),
        result
      };
      
      setHistory(prev => [queryWithResult, ...prev]);
      
      return result;
    } catch (err) {
      console.error('Query execution error:', err);
      setError(err.message || 'Failed to execute query');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Get query by ID (from history or saved queries)
  const getQueryById = useCallback(async (queryId) => {
    // Try from local state first
    let query = history.find(q => q.id.toString() === queryId.toString());
    
    if (query) {
      return query;
    }
    
    query = savedQueries.find(q => q.id.toString() === queryId.toString());
    
    if (query) {
      return query;
    }
    
    // If not found locally, fetch from API
    try {
      setIsLoading(true);
      const fetchedQuery = await queryService.getQueryById(queryId);
      return fetchedQuery;
    } catch (err) {
      console.error('Error fetching query:', err);
      setError(err.message || 'Failed to fetch query');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [history, savedQueries]);
  
  // Save a query
  const saveQuery = useCallback(async (queryText, queryResult, queryName = '') => {
    try {
      setIsLoading(true);
      setError(null);
      
      const savedQuery = await queryService.saveQuery({
        text: queryText,
        name: queryName || queryText.substring(0, 30),
        result: queryResult
      });
      
      setSavedQueries(prev => [savedQuery, ...prev]);
      
      return savedQuery;
    } catch (err) {
      console.error('Error saving query:', err);
      setError(err.message || 'Failed to save query');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Delete a saved query
  const deleteQuery = useCallback(async (queryId) => {
    try {
      setIsLoading(true);
      setError(null);
      
      await queryService.deleteQuery(queryId);
      
      // Remove from saved queries list
      setSavedQueries(prev => prev.filter(q => q.id !== queryId));
      
      return true;
    } catch (err) {
      console.error('Error deleting query:', err);
      setError(err.message || 'Failed to delete query');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Get recent queries
  const getRecentQueries = useCallback(async () => {
    try {
      setIsLoading(true);
      
      // Use local history if available
      if (history.length > 0) {
        return history;
      }
      
      // Otherwise fetch from API
      const recentQueries = await queryService.getRecentQueries();
      setHistory(recentQueries);
      
      return recentQueries;
    } catch (err) {
      console.error('Error fetching recent queries:', err);
      setError(err.message || 'Failed to fetch recent queries');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [history]);
  
  // Get saved queries
  const getSavedQueries = useCallback(async () => {
    try {
      setIsLoading(true);
      
      // Use local saved queries if available
      if (savedQueries.length > 0) {
        return savedQueries;
      }
      
      // Otherwise fetch from API
      const queries = await queryService.getSavedQueries();
      setSavedQueries(queries);
      
      return queries;
    } catch (err) {
      console.error('Error fetching saved queries:', err);
      setError(err.message || 'Failed to fetch saved queries');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [savedQueries]);
  
  // Get query suggestions
  const getQuerySuggestions = useCallback(async (category = 'all') => {
    try {
      setIsLoading(true);
      
      const fetchedSuggestions = await queryService.getQuerySuggestions(category);
      setSuggestions(fetchedSuggestions);
      
      return fetchedSuggestions;
    } catch (err) {
      console.error('Error fetching query suggestions:', err);
      setError(err.message || 'Failed to fetch query suggestions');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Generate chart from query result
  const generateChart = useCallback(async (queryResult, chartType = 'auto') => {
    try {
      setIsLoading(true);
      
      const chartData = await queryService.generateChart(queryResult, chartType);
      
      return chartData;
    } catch (err) {
      console.error('Error generating chart:', err);
      setError(err.message || 'Failed to generate chart');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);
  
  // Clear history
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);
  
  return {
    isLoading,
    error,
    history,
    savedQueries,
    suggestions,
    executeQuery,
    getQueryById,
    saveQuery,
    deleteQuery,
    getRecentQueries,
    getSavedQueries,
    getQuerySuggestions,
    generateChart,
    clearHistory
  };
};

export default useQuery;