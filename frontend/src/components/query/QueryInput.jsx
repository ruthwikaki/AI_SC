import React, { useState, useEffect, useRef } from 'react';

const QueryInput = ({ initialValue = '', onSubmit, placeholder }) => {
  const [query, setQuery] = useState(initialValue);
  const [showExamples, setShowExamples] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    setQuery(initialValue);
  }, [initialValue]);

  useEffect(() => {
    // Auto-resize textarea based on content
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px";
    }
  }, [query]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !onSubmit) return;
    
    onSubmit(query.trim());
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
              placeholder={placeholder || "Ask something like 'Show me inventory levels for products with less than 30 days of supply'"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
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
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            disabled={!query.trim()}
          >
            Submit
          </button>
        </div>
      </form>
    </div>
  );
};

export default QueryInput;