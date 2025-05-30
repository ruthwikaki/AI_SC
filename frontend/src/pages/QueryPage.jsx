import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { useQuery } from '../hooks/useQuery';
import Loading from '../components/common/Loading';
import QueryInput from '../components/query/QueryInput';
import QueryResult from '../components/query/QueryResult';
import QuerySuggestions from '../components/query/QuerySuggestions';

const QueryPage = () => {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const { 
    executeQuery, 
    getQueryById, 
    getSavedQueries, 
    saveQuery,
    savedQueries: hookSavedQueries,
    history 
  } = useQuery();
  const navigate = useNavigate();
  const { queryId } = useParams();
  
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSavedQueries, setShowSavedQueries] = useState(false);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
      return;
    }

    // Load specific query if queryId is provided
    const loadQuery = async () => {
      if (queryId) {
        try {
          setLoading(true);
          const queryData = await getQueryById(queryId);
          if (queryData) {
            setQuery(queryData.text);
            setResult(queryData.result);
          }
        } catch (err) {
          console.error('Error loading query:', err);
          setError('Failed to load the specified query.');
        } finally {
          setLoading(false);
        }
      }
    };

    if (isAuthenticated) {
      loadQuery();
    }
  }, [isAuthenticated, authLoading, navigate, queryId, getQueryById]);

  const handleQuerySubmit = async (queryText) => {
    setQuery(queryText);
    setLoading(true);
    setError(null);
    
    try {
      const result = await executeQuery(queryText);
      setResult(result);
    } catch (err) {
      console.error('Error executing query:', err);
      setError('Failed to execute query. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveQuery = async () => {
    if (!query || !result) return;
    
    try {
      await saveQuery({
        text: query,
        result: result,
        name: `Query - ${new Date().toLocaleString()}`
      });
      // The hook will update its internal savedQueries state
    } catch (err) {
      console.error('Error saving query:', err);
      setError('Failed to save query.');
    }
  };

  const handleSavedQuerySelect = (selectedQuery) => {
    setQuery(selectedQuery.text);
    handleQuerySubmit(selectedQuery.text);
    setShowSavedQueries(false);
  };

  if (authLoading) {
    return <Loading fullScreen={true} message="Authenticating..." />;
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-800">Supply Chain Query</h1>
        <div className="relative">
          <button
            className="bg-white hover:bg-gray-50 text-gray-700 font-medium py-2 px-4 border border-gray-300 rounded-lg shadow-sm flex items-center"
            onClick={() => setShowSavedQueries(!showSavedQueries)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
            </svg>
            Saved Queries ({hookSavedQueries.length})
          </button>
          
          {showSavedQueries && (
            <div className="absolute right-0 mt-2 w-64 bg-white border border-gray-300 rounded-lg shadow-lg z-10 max-h-96 overflow-y-auto">
              <ul className="py-1">
                {hookSavedQueries.length > 0 ? (
                  hookSavedQueries.map(q => (
                    <li
                      key={q.id}
                      className="px-4 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                      onClick={() => handleSavedQuerySelect(q)}
                    >
                      <div className="font-medium">{q.name || 'Unnamed Query'}</div>
                      <div className="text-gray-500 text-xs truncate">{q.text}</div>
                    </li>
                  ))
                ) : (
                  <li className="px-4 py-2 text-gray-500 text-sm">No saved queries found</li>
                )}
              </ul>
            </div>
          )}
        </div>
      </div>
      
      {/* Query Input Section */}
      <div className="bg-white rounded-lg shadow-sm mb-6">
        <div className="p-5">
          <QueryInput
            initialValue={query}
            onSubmit={handleQuerySubmit}
            placeholder="Ask a question about your supply chain data..."
          />
          
          {/* Error Message */}
          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
              <div className="flex">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {error}
              </div>
            </div>
          )}
          
          {/* Loading Indicator */}
          {loading && (
            <div className="mt-4">
              <Loading size="medium" message="Processing your query..." />
            </div>
          )}
        </div>
      </div>
      
      {/* Query Suggestions */}
      {!result && !loading && (
        <div className="bg-white rounded-lg shadow-sm p-5 mb-6">
          <h2 className="text-lg font-medium mb-4">Suggested Queries</h2>
          <QuerySuggestions onSelect={handleQuerySubmit} />
        </div>
      )}
      
      {/* Query Results */}
      {result && !loading && (
        <QueryResult
          result={result}
          query={query}
          onSave={handleSaveQuery}
        />
      )}
      
      {/* Query History */}
      {history.length > 0 && (
        <div className="mt-8 bg-white rounded-lg shadow-sm p-5">
          <h2 className="text-lg font-medium mb-4">Recent Queries</h2>
          <div className="space-y-2">
            {history.slice(0, 5).map((item, index) => (
              <div
                key={index}
                className="p-3 bg-gray-50 rounded cursor-pointer hover:bg-gray-100"
                onClick={() => handleQuerySubmit(item.query)}
              >
                <div className="text-sm font-medium">{item.query}</div>
                <div className="text-xs text-gray-500">{new Date(item.timestamp).toLocaleString()}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryPage;