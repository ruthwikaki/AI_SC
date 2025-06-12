import api from './api';

const queryService = {
  /**
   * Execute a natural language query
   * @param {string} queryText - The natural language query text
   * @param {Object} options - Query options
   * @returns {Promise<Object>} Query result data
   */
  executeQuery: async (queryText, options = {}) => {
    try {
      const response = await api.post('/api/queries/natural-language', {
        query: queryText,
        ...options,
      });
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get a query by ID
   * @param {string|number} queryId - Query ID
   * @returns {Promise<Object>} Query data
   */
  getQueryById: async (queryId) => {
    try {
      const response = await api.get(`/queries/${queryId}`);
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Save a query
   * @param {Object} queryData - Query data to save
   * @returns {Promise<Object>} Saved query data
   */
  saveQuery: async (queryData) => {
    try {
      const response = await api.post('/api/queries/save', queryData);
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Delete a saved query
   * @param {string|number} queryId - Query ID to delete
   * @returns {Promise<Object>} Success response
   */
  deleteQuery: async (queryId) => {
    try {
      const response = await api.delete(`/queries/${queryId}`);
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get recent queries
   * @param {number} limit - Maximum number of queries to retrieve
   * @returns {Promise<Array>} Array of recent queries
   */
  getRecentQueries: async (limit = 10) => {
    try {
      const response = await api.get('/api/queries/recent', {
        params: { limit },
      });
      
      return response.queries || [];
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get saved queries
   * @returns {Promise<Array>} Array of saved queries
   */
  getSavedQueries: async () => {
    try {
      const response = await api.get('/api/queries/saved');
      return response.queries || [];
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get query suggestions
   * @param {string} category - Category for suggestions (optional)
   * @returns {Promise<Array>} Array of query suggestions
   */
  getQuerySuggestions: async (category = 'all') => {
    try {
      const response = await api.get('/api/queries/suggestions', {
        params: { category },
      });
      
      return response.suggestions || [];
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Generate a chart from query result
   * @param {Object} queryResult - Query result data
   * @param {string} chartType - Type of chart to generate (or 'auto')
   * @returns {Promise<Object>} Chart data
   */
  generateChart: async (queryResult, chartType = 'auto') => {
    try {
      const response = await api.post('/api/visualizations/generate', {
        data: queryResult,
        chart_type: chartType,
      });
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Analyze a database schema
   * @returns {Promise<Object>} Schema analysis data
   */
  analyzeSchema: async () => {
    try {
      const response = await api.get('/api/database/analyze');
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get a list of database tables
   * @returns {Promise<Array>} Array of tables
   */
  getTables: async () => {
    try {
      const response = await api.get('/api/database/tables');
      return response.tables || [];
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get the schema for a specific table
   * @param {string} tableName - Name of the table
   * @returns {Promise<Object>} Table schema data
   */
  getTableSchema: async (tableName) => {
    try {
      const response = await api.get(`/database/tables/${tableName}/schema`);
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Get paginated data from a table
   * @param {string} tableName - Name of the table
   * @param {Object} options - Query options (page, limit, sort, filter)
   * @returns {Promise<Object>} Table data with pagination info
   */
  getTableData: async (tableName, options = {}) => {
    try {
      const response = await api.get(`/database/tables/${tableName}/data`, {
        params: options,
      });
      
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Execute a SQL query directly
   * @param {string} sqlQuery - SQL query to execute
   * @returns {Promise<Object>} Query result data
   */
  executeSqlQuery: async (sqlQuery) => {
    try {
      const response = await api.post('/api/queries/sql', {
        query: sqlQuery,
      });
      
      return response;
    } catch (error) {
      throw error;
    }
  },
};

export default queryService;

