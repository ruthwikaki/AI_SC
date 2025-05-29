import React, { useState, useEffect, useRef } from 'react';
import { useQuery } from '../../hooks/useQuery';

const QueryInput = ({ onQueryComplete }) => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  const textareaRef = useRef(null);
  const { executeQuery, suggestions, loadSuggestions } = useQuery();

  useEffect(() => {
    // Auto-resize textarea based on content
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }

    // Load suggestions when component mounts
    loadSuggestions();
  }, [query, loadSuggestions]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const results = await executeQuery(query);
      if (onQueryComplete) {
        onQueryComplete(results);
      }
    } catch (error) {
      console.error("Error executing query:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Cmd/Ctrl + Enter
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  const exampleQueries = [
    "Which suppliers have the highest risk score?",
    "Show me inventory levels for products with less than 30 days of supply",
    "Calculate the optimal safety stock for our top 10 selling products",
    "What was our on-time delivery rate last quarter?",
    "Which routes had the most delivery delays last month?",
    "Identify potential bottlenecks in our tier 2 suppliers"
  ];

  const applyExample = (example) => {
    setQuery(example);
    setShowExamples(false);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-4 md:p-6">
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-1">
            Ask a supply chain question
          </label>
          <div className="relative">
            <textarea
              ref={textareaRef}
              id="query"
              name="query"
              rows={1}
              className="block w-full px-4 py-3 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 resize-none"
              placeholder="Ask something like 'Show me inventory levels for products with less than 30 days of supply'"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
          </div>
          <div className="mt-1 flex justify-between items-center text-xs text-gray-500">
            <button
              type="button"
              className="text-indigo-600 hover:text-indigo-800"
              onClick={() => setShowExamples(!showExamples)}
            >
              {showExamples ? 'Hide examples' : 'Show examples'}
            </button>
            <span>Press Cmd/Ctrl + Enter to submit</span>
          </div>
        </div>

        {showExamples && (
          <div className="mb-4 bg-gray-50 p-3 rounded-md">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Example queries:</h4>
            <div className="space-y-2">
              {exampleQueries.map((example, index) => (
                <button
                  key={index}
                  type="button"
                  className="block w-full text-left text-sm px-3 py-2 rounded-md hover:bg-gray-100 text-gray-700"
                  onClick={() => applyExample(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-end">
          <button
            type="submit"
            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
              isLoading ? 'opacity-75 cursor-not-allowed' : ''
            }`}
            disabled={isLoading || !query.trim()}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </>
            ) : (
              'Submit'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default QueryInput;