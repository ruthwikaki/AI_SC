import { useState, useCallback, useEffect } from 'react';
import api from '../services/api';

export const useQuery = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [savedQueries, setSavedQueries] = useState([]);
  const [history, setHistory] = useState([]);

  // Load saved queries on mount
  useEffect(() => {
    loadSavedQueries();
    loadSuggestions();
  }, []);

  const executeQuery = useCallback(async (query) => {
    console.log('executeQuery called with:', query);
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Making API call to /api/queries/execute');
      
      // Show loading message for complex queries
      const isComplexQuery = query.toLowerCase().includes('analyze') || 
                           query.toLowerCase().includes('dashboard') ||
                           query.toLowerCase().includes('comprehensive');
      
      if (isComplexQuery) {
        console.log('🤖 Complex query detected - Mixtral is thinking... This may take up to 3 minutes.');
      }
      
      const response = await api.post('/api/queries/execute', { query });
      console.log('API response:', response.data);
      
      // Add to history
      setHistory(prev => [{
        query: query,
        timestamp: new Date().toISOString(),
        result: response.data
      }, ...prev].slice(0, 10)); // Keep last 10
      
      return response.data;
    } catch (err) {
      console.error('API call failed:', err);
      
      // Handle timeout errors specially
      if (err.code === 'ECONNABORTED' && err.message.includes('timeout')) {
        const timeoutError = {
          message: 'Query took too long to process',
          suggestions: [
            'Try breaking your query into smaller parts',
            'Ask for one specific thing at a time',
            'Use simpler language'
          ],
          examples: [
            'show me all products',
            'create a bar chart of supplier ratings',
            'which products need reordering?'
          ]
        };
        
        setError(timeoutError);
        
        // Log helpful suggestions
        console.log('⏱️ Query timeout. Try these simpler queries:');
        timeoutError.examples.forEach(ex => console.log(`  - "${ex}"`));
      } else {
        setError(err.message || 'Failed to execute query');
      }
      
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadSavedQueries = useCallback(async () => {
    try {
      const response = await api.get('/api/queries/saved');
      setSavedQueries(response.data.queries || []);
      return response.data.queries || [];
    } catch (err) {
      console.error('Error loading saved queries:', err);
      setSavedQueries([]);
      return [];
    }
  }, []);

  const saveQuery = useCallback(async (queryData) => {
    try {
      const response = await api.post('/api/queries/save', queryData);
      // Reload saved queries after saving
      await loadSavedQueries();
      return response.data;
    } catch (err) {
      console.error('Error saving query:', err);
      throw err;
    }
  }, [loadSavedQueries]);

  const getQueryById = useCallback(async (queryId) => {
    try {
      const response = await api.get(`/api/queries/${queryId}`);
      return response.data;
    } catch (err) {
      console.error('Error fetching query:', err);
      throw err;
    }
  }, []);

  const getSavedQueries = useCallback(async () => {
    return await loadSavedQueries();
  }, [loadSavedQueries]);

  const loadSuggestions = useCallback(async () => {
    try {
      const response = await api.get('/api/queries/suggestions');
      let suggestionsList = response.data.suggestions || [];
      
      // Transform string array to object array with text property
      if (Array.isArray(suggestionsList) && typeof suggestionsList[0] === 'string') {
        suggestionsList = suggestionsList.map(text => {
          // Determine category based on keywords
          let category = 'general';
          const lowerText = text.toLowerCase();
          
          if (lowerText.includes('inventory') || lowerText.includes('stock') || lowerText.includes('product')) {
            category = 'inventory';
          } else if (lowerText.includes('supplier') || lowerText.includes('vendor')) {
            category = 'suppliers';
          } else if (lowerText.includes('order') || lowerText.includes('delivery')) {
            category = 'orders';
          } else if (lowerText.includes('analytics') || lowerText.includes('report') || lowerText.includes('kpi')) {
            category = 'analytics';
          }
          
          return {
            text: text,
            category: category,
            isRecent: false
          };
        });
      }
      
      setSuggestions(suggestionsList);
      return suggestionsList;
    } catch (err) {
      console.error('Error loading suggestions:', err);
      // Fallback suggestions in the correct format
      const fallbackSuggestions = [
        { text: "Show me all products", category: "inventory", isRecent: false },
        { text: "Show top suppliers by rating", category: "suppliers", isRecent: false },
        { text: "Show current inventory levels", category: "inventory", isRecent: false },
        { text: "List all recent orders", category: "orders", isRecent: false },
        { text: "What is the total inventory value?", category: "analytics", isRecent: false },
        { text: "Show products below reorder point", category: "inventory", isRecent: false },
        { text: "Which suppliers have rating above 4?", category: "suppliers", isRecent: false }
      ];
      setSuggestions(fallbackSuggestions);
      return fallbackSuggestions;
    }
  }, []);

  const getRecentQueries = useCallback(async () => {
    try {
      const response = await api.get('/api/queries');
      return response.data.queries || [];
    } catch (err) {
      console.error('Error fetching recent queries:', err);
      return [];
    }
  }, []);

  // Helper to check if error is timeout
  const isTimeoutError = useCallback((error) => {
    return error?.code === 'ECONNABORTED' || 
           error?.message?.includes('timeout') ||
           error?.suggestions?.length > 0;
  }, []);

  return {
    executeQuery,
    loadSuggestions,
    getRecentQueries,
    getQueryById,
    getSavedQueries,
    saveQuery,
    savedQueries,
    history,
    suggestions,
    isLoading,
    error,
    isTimeoutError
  };
};