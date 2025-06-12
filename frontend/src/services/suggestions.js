// frontend/src/services/suggestions.js
import api from './api';

export const suggestionsService = {
  // Query Suggestions
  getQuerySuggestions: async (partialQuery, limit = 10) => {
    const response = await api.get('/api/api/suggestions/queries', {
      params: { partial_query: partialQuery, limit }
    });
    return response.data;
  },

  // Query Templates
  getQueryTemplates: async (category = null) => {
    const response = await api.get('/api/api/suggestions/templates', {
      params: { category }
    });
    return response.data;
  },

  // Autocomplete
  getAutocomplete: async (field, value = '', limit = 10) => {
    const response = await api.get('/api/api/suggestions/autocomplete', {
      params: { field, value, limit }
    });
    return response.data;
  },

  // Related Queries
  getRelatedQueries: async (queryId = null, queryText = null) => {
    const response = await api.get('/api/api/suggestions/related', {
      params: { query_id: queryId, query_text: queryText }
    });
    return response.data;
  }
};
