import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from '../../visualization/ChartViewer';
import Loading from '../../common/Loading';

const QueryResult = ({ result, isLoading, error }) => {
  const [activeTab, setActiveTab] = useState('visualized');
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const [saveAsName, setSaveAsName] = useState('');
  const navigate = useNavigate();

  // Reset active tab when new results come in
  useEffect(() => {
    if (result) {
      // Choose best default tab based on result type
      if (result.chart) {
        setActiveTab('visualized');
      } else if (result.sql) {
        setActiveTab('query');
      } else {
        setActiveTab('data');
      }
    }
  }, [result]);

  if (isLoading) {
    return <Loading type="card" height="h-96" message="Analyzing your query..." />;
  }

  if (error) {
    return (
      <div className="bg-white shadow rounded-lg p-4">
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error processing query</h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error.message || 'An unexpected error occurred. Please try again.'}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  const handleSaveQuery = () => {
    // Logic to save query would go here
    console.log('Saving query as:', saveAsName);
    setSaveModalOpen(false);
    // Show success notification
  };

  const handleAddToDashboard = () => {
    // Navigate to dashboard creation/edit with this visualization
    if (result.chartId) {
      navigate(`/dashboard/edit?addChart=${result.chartId}`);
    }
  };

  return (
    <div className="bg-white shadow rounded-lg overflow-hidden">
      {/* Explanation of the result */}
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <h3 className="text-lg font-medium text-gray-900">Query Results</h3>
        <p className="mt-1 text-sm text-gray-600">
          {result.explanation || 'Here are the results of your query.'}
        </p>
      </div>

      {/* Tab navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex -mb-px">
          {result.chart && (
            <button
              onClick={() => setActiveTab('visualized')}
              className={`py-4 px-6 text-sm font-medium ${
                activeTab === 'visualized'
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Visualization
            </button>
          )}
          <button
            onClick={() => setActiveTab('data')}
            className={`py-4 px-6 text-sm font-medium ${
              activeTab === 'data'
                ? 'border-b-2 border-indigo-500 text-indigo-600'
                : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Data
          </button>
          {result.sql && (
            <button
              onClick={() => setActiveTab('query')}
              className={`py-4 px-6 text-sm font-medium ${
                activeTab === 'query'
                  ? 'border-b-2 border-indigo-500 text-indigo-600'
                  : 'border-b-2 border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              SQL Query
            </button>
          )}
        </nav>
      </div>

      {/* Tab content */}
      <div className="p-4">
        {activeTab === 'visualized' && result.chart && (
          <div className="h-96">
            <ChartViewer 
              chartData={result.chart}
              fullHeight
            />
          </div>
        )}

        {activeTab === 'data' && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {result.data && result.data.length > 0 && Object.keys(result.data[0]).map((column, index) => (
                    <th 
                      key={index}
                      scope="col" 
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {result.data && result.data.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {Object.values(row).map((cell, cellIndex) => (
                      <td key={cellIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {cell !== null && cell !== undefined ? cell.toString() : 'N/A'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {(!result.data || result.data.length === 0) && (
              <div className="text-center py-8 text-gray-500">
                No data available for this query.
              </div>
            )}
          </div>
        )}

        {activeTab === 'query' && result.sql && (
          <div className="bg-gray-50 p-4 rounded-md">
            <pre className="text-sm text-gray-800 overflow-auto">{result.sql}</pre>
            <div className="mt-4 text-sm text-gray-500">
              <strong>Execution time:</strong> {result.executionTime || 'N/A'}
            </div>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="px-4 py-3 bg-gray-50 text-right sm:px-6 border-t border-gray-200">
        <button
          type="button"
          onClick={() => setSaveModalOpen(true)}
          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 mr-3"
        >
          Save Query
        </button>
        {result.chart && (
          <button
            type="button"
            onClick={handleAddToDashboard}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Add to Dashboard
          </button>
        )}
      </div>

      {/* Save modal */}
      {saveModalOpen && (
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Save Query</h3>
            <div className="mb-4">
              <label htmlFor="queryName" className="block text-sm font-medium text-gray-700 mb-1">
                Query Name
              </label>
              <input
                type="text"
                id="queryName"
                className="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                placeholder="Enter a name for this query"
              />
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => setSaveModalOpen(false)}
                className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveQuery}
                disabled={!saveAsName.trim()}
                className={`inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                  !saveAsName.trim() ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QueryResult;