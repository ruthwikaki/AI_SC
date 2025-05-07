import React, { useState, useEffect, useRef } from 'react';
import { 
  FaCopy, 
  FaCheck, 
  FaThumbsUp, 
  FaThumbsDown, 
  FaChartBar, 
  FaTable, 
  FaDownload,
  FaStar,
  FaExclamationCircle,
  FaRegLightbulb,
  FaRedo
} from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

const QueryResult = ({ 
  result, 
  query,
  isLoading, 
  error,
  modelConfig,
  onRegenerateResponse,
  onFollowupQuery
}) => {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [showReferences, setShowReferences] = useState(false);
  const resultRef = useRef(null);
  
  // Reset copied status after 2 seconds
  useEffect(() => {
    let timeout;
    if (copied) {
      timeout = setTimeout(() => setCopied(false), 2000);
    }
    return () => clearTimeout(timeout);
  }, [copied]);
  
  // Scroll to result when it loads
  useEffect(() => {
    if (result && !isLoading && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result, isLoading]);
  
  // Handle copy to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(result.text);
    setCopied(true);
  };
  
  // Handle feedback submission
  const handleFeedback = (isPositive) => {
    setFeedback(isPositive);
    // Here you could also send feedback to your backend
    console.log(`User gave ${isPositive ? 'positive' : 'negative'} feedback for query: ${query}`);
  };
  
  // No result and not loading
  if (!result && !isLoading && !error) {
    return null;
  }
  
  // Error state
  if (error) {
    return (
      <div className="mt-8 p-6 bg-red-50 border border-red-200 rounded-lg" ref={resultRef}>
        <div className="flex items-center mb-4">
          <FaExclamationCircle className="text-red-500 mr-2" size={24} />
          <h3 className="text-xl text-red-700 font-medium">Error Processing Query</h3>
        </div>
        <p className="text-red-700 mb-4">{error.message || "An error occurred while processing your query. Please try again."}</p>
        <button
          onClick={() => onRegenerateResponse(query)}
          className="inline-flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
        >
          <FaRedo className="mr-2" /> Try Again
        </button>
      </div>
    );
  }
  
  // Loading state is handled by parent component
  if (isLoading) {
    return null;
  }
  
  // Extract insights, sources, metrics from result
  const hasInsights = result?.insights && result.insights.length > 0;
  const hasSources = result?.sources && result.sources.length > 0;
  const hasMetrics = result?.metrics && Object.keys(result.metrics).length > 0;
  
  // Generate suggested follow-up queries based on the result
  const suggestedFollowups = result?.suggestedFollowups || [
    "What are the potential risks in this scenario?",
    "How can we optimize this process further?",
    "What cost savings opportunities exist here?",
    "How does this compare to industry benchmarks?"
  ];
  
  return (
    <div 
      className="mt-8 p-6 bg-white border border-gray-200 rounded-lg shadow-sm"
      ref={resultRef}
    >
      {/* Result header with model info and timestamp */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 pb-4 border-b border-gray-200">
        <div>
          <h2 className="text-xl font-semibold text-gray-800">Analysis Results</h2>
          <div className="flex items-center mt-1 text-sm text-gray-500">
            <span>Model: {modelConfig.model}</span>
            <span className="mx-2">•</span>
            <span>Processed in {result.processingTime || "0.8"}s</span>
          </div>
        </div>
        <div className="flex mt-2 sm:mt-0">
          <button 
            onClick={handleCopy}
            className="flex items-center px-3 py-1 text-sm text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-md mr-2"
          >
            {copied ? <FaCheck className="mr-1" /> : <FaCopy className="mr-1" />}
            {copied ? "Copied!" : "Copy"}
          </button>
          <button 
            onClick={() => onRegenerateResponse(query)}
            className="flex items-center px-3 py-1 text-sm text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-md"
          >
            <FaRedo className="mr-1" />
            Regenerate
          </button>
        </div>
      </div>
      
      {/* Main response content */}
      <div className="mb-6">
        <div className="prose max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({node, inline, className, children, ...props}) {
                const match = /language-(\w+)/.exec(className || '')
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={materialLight}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                ) : (
                  <code className={className} {...props}>
                    {children}
                  </code>
                )
              }
            }}
          >
            {result.text}
          </ReactMarkdown>
        </div>
      </div>
      
      {/* Key insights section */}
      {hasInsights && (
        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
          <h3 className="text-lg font-medium text-blue-800 mb-3">Key Insights</h3>
          <ul className="space-y-2">
            {result.insights.map((insight, index) => (
              <li key={index} className="flex items-start">
                <FaRegLightbulb className="text-blue-500 mt-1 mr-2 flex-shrink-0" />
                <span className="text-blue-700">{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Metrics visualization section */}
      {hasMetrics && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-medium text-gray-800">Key Metrics</h3>
            <FaChartBar className="text-gray-500" />
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {Object.entries(result.metrics || {}).map(([key, value], index) => (
              <div key={index} className="bg-white p-3 rounded-md border border-gray-200">
                <div className="text-sm text-gray-500">{key}</div>
                <div className="text-xl font-semibold text-gray-800">{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Data sources section */}
      {hasSources && (
        <div className="mt-6">
          <button
            onClick={() => setShowReferences(!showReferences)}
            className="flex items-center text-gray-600 hover:text-gray-800"
          >
            <FaTable className="mr-2" />
            <span className="text-sm font-medium">
              {showReferences ? "Hide Data Sources" : "Show Data Sources"} ({result.sources.length})
            </span>
          </button>
          
          {showReferences && (
            <div className="mt-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-100">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Updated</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {result.sources.map((source, index) => (
                      <tr key={index}>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-blue-600 hover:text-blue-800">
                          <a href={source.url || "#"} target="_blank" rel="noopener noreferrer">
                            {source.name}
                          </a>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-600">{source.type}</td>
                        <td className="px-3 py-2 whitespace-nowrap text-sm text-gray-600">{source.lastUpdated}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-3 flex justify-end">
                <button className="flex items-center text-sm text-blue-600 hover:text-blue-800">
                  <FaDownload className="mr-1" size={12} />
                  Export Sources
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Suggested follow-up queries */}
      <div className="mt-8 pt-6 border-t border-gray-200">
        <h3 className="text-md font-medium text-gray-700 mb-3">Follow-up Questions</h3>
        <div className="flex flex-wrap gap-2">
          {suggestedFollowups.map((followup, index) => (
            <button
              key={index}
              onClick={() => onFollowupQuery(followup)}
              className="inline-flex items-center px-3 py-1.5 bg-blue-50 hover:bg-blue-100 border border-blue-200 rounded-full text-sm text-blue-700 transition-colors"
            >
              <FaPlus className="mr-2 text-blue-500" size={10} />
              {followup}
            </button>
          ))}
        </div>
      </div>
      
      {/* Feedback section */}
      <div className="mt-6 pt-4 border-t border-gray-200 flex justify-between items-center">
        <div className="text-sm text-gray-500">
          Was this response helpful?
        </div>
        <div className="flex items-center">
          <button
            onClick={() => handleFeedback(true)}
            className={`mr-2 flex items-center px-3 py-1 rounded-md text-sm ${
              feedback === true 
                ? 'bg-green-100 text-green-700 border border-green-200' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
            }`}
          >
            <FaThumbsUp className="mr-1" size={12} />
            Helpful
          </button>
          
          <button
            onClick={() => handleFeedback(false)}
            className={`flex items-center px-3 py-1 rounded-md text-sm ${
              feedback === false 
                ? 'bg-red-100 text-red-700 border border-red-200' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 border border-gray-200'
            }`}
          >
            <FaThumbsDown className="mr-1" size={12} />
            Not Helpful
          </button>
        </div>
      </div>
    </div>
  );
};

export default QueryResult;