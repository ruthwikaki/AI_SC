// src/components/query/QueryResult.jsx
import React, { useState } from 'react';
import PropTypes from 'prop-types';

const QueryResult = ({ result, query, onSave }) => {
  const [activeTab, setActiveTab] = useState('data');

  if (!result) {
    return null;
  }

  // FIX: Check for data in result.data (from backend) OR result.results (legacy)
  const data = result.data || result.results || [];
  const hasData = data && data.length > 0;
  const hasVisualization = result.visualization;
  const hasAnalysis = result.analysis;
  const hasResponse = result.response;
  const hasError = result.error;

  // Auto-select appropriate tab
  React.useEffect(() => {
    if (hasVisualization) {
      setActiveTab('visualization');
    } else if (hasAnalysis) {
      setActiveTab('analysis');
    } else if (hasData) {
      setActiveTab('data');
    } else if (hasResponse) {
      setActiveTab('response');
    }
  }, [result]);

  const renderTabs = () => {
    const tabs = [];
    
    if (hasData) {
      tabs.push(
        <button
          key="data"
          className={`px-4 py-2 font-medium ${
            activeTab === 'data'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setActiveTab('data')}
        >
          Data ({result.row_count || data.length} rows)
        </button>
      );
    }
    
    if (hasVisualization) {
      tabs.push(
        <button
          key="visualization"
          className={`px-4 py-2 font-medium ${
            activeTab === 'visualization'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setActiveTab('visualization')}
        >
          📊 Visualization
        </button>
      );
    }
    
    if (hasAnalysis) {
      tabs.push(
        <button
          key="analysis"
          className={`px-4 py-2 font-medium ${
            activeTab === 'analysis'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setActiveTab('analysis')}
        >
          🔍 Analysis
        </button>
      );
    }
    
    if (hasResponse) {
      tabs.push(
        <button
          key="response"
          className={`px-4 py-2 font-medium ${
            activeTab === 'response'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setActiveTab('response')}
        >
          💬 AI Response
        </button>
      );
    }
    
    if (result.sql) {
      tabs.push(
        <button
          key="sql"
          className={`px-4 py-2 font-medium ${
            activeTab === 'sql'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-800'
          }`}
          onClick={() => setActiveTab('sql')}
        >
          SQL Query
        </button>
      );
    }
    
    return tabs;
  };

  const renderContent = () => {
    if (hasError) {
      return (
        <div className="p-6">
          <div className="bg-red-50 border border-red-200 rounded-md p-4">
            <h3 className="text-red-800 font-medium mb-2">Error</h3>
            <p className="text-red-700">{result.error}</p>
            {result.sql && (
              <div className="mt-4">
                <p className="text-sm text-red-600 font-medium">Generated SQL:</p>
                <pre className="mt-1 text-xs bg-red-100 p-2 rounded overflow-x-auto">
                  {result.sql}
                </pre>
              </div>
            )}
          </div>
        </div>
      );
    }

    switch (activeTab) {
      case 'data':
        if (!hasData) {
          return (
            <div className="p-6">
              <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                <p className="text-yellow-700">No data returned. The query executed successfully but returned 0 rows.</p>
              </div>
            </div>
          );
        }
        
        // Get columns from result.columns if available, otherwise from first data row
        const columns = result.columns || (data[0] ? Object.keys(data[0]) : []);
        
        return (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {columns.map((header) => (
                    <th
                      key={header}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.map((row, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    {columns.map((col) => (
                      <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {row[col] === null ? 'NULL' : String(row[col])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'visualization':
        if (!hasVisualization) return null;
        return (
          <div className="p-6">
            <img 
              src={`data:image/png;base64,${result.visualization}`} 
              alt="Data visualization"
              className="max-w-full h-auto mx-auto"
            />
          </div>
        );

      case 'analysis':
        if (!hasAnalysis) return null;
        return (
          <div className="p-6">
            <div className="prose max-w-none">
              <h3 className="text-lg font-medium mb-4">AI Analysis</h3>
              <div className="whitespace-pre-wrap text-gray-700">
                {result.analysis}
              </div>
            </div>
          </div>
        );

      case 'response':
        if (!hasResponse) return null;
        return (
          <div className="p-6">
            <div className="prose max-w-none">
              <div className="whitespace-pre-wrap text-gray-700">
                {result.response}
              </div>
            </div>
          </div>
        );

      case 'sql':
        if (!result.sql) return null;
        return (
          <div className="p-6">
            <pre className="bg-gray-100 p-4 rounded-md overflow-x-auto">
              <code className="text-sm">{result.sql}</code>
            </pre>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm">
      <div className="border-b border-gray-200">
        <div className="flex items-center justify-between px-6 py-3">
          <h2 className="text-lg font-medium text-gray-900">
            {result.intent?.main_intent === 'visualization' && '📊 '}
            {result.intent?.main_intent === 'analysis' && '🔍 '}
            {result.intent?.main_intent === 'prediction' && '🔮 '}
            Query Results
          </h2>
          {onSave && (
            <button
              onClick={onSave}
              className="inline-flex items-center px-3 py-1.5 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              Save Query
            </button>
          )}
        </div>
        <div className="flex space-x-1 px-6">
          {renderTabs()}
        </div>
      </div>
      
      {renderContent()}
    </div>
  );
};

QueryResult.propTypes = {
  result: PropTypes.object,
  query: PropTypes.string,
  onSave: PropTypes.func,
};

export default QueryResult;