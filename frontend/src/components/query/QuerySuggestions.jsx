import React, { useState, useEffect } from 'react';
import { useQuery } from '../../hooks/useQuery';

const QuerySuggestions = ({ onSuggestionSelect }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [categories, setCategories] = useState([]);
  const [activeCategory, setActiveCategory] = useState('recent');
  const { loadSuggestions } = useQuery();

  useEffect(() => {
    const fetchSuggestions = async () => {
      setIsLoading(true);
      try {
        const data = await loadSuggestions();
        setSuggestions(data);
        
        // Extract unique categories
        const uniqueCategories = ['recent', ...new Set(data.map(item => item.category))];
        setCategories(uniqueCategories);
      } catch (err) {
        console.error('Error loading suggestions:', err);
        setError('Unable to load query suggestions. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };

    fetchSuggestions();
  }, [loadSuggestions]);

  const handleSuggestionClick = (suggestion) => {
    if (onSuggestionSelect) {
      onSuggestionSelect(suggestion.text);
    }
  };

  const filteredSuggestions = activeCategory === 'recent'
    ? suggestions.filter(s => s.isRecent)
    : suggestions.filter(s => s.category === activeCategory);

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-4 animate-pulse">
        <div className="h-5 bg-gray-200 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-8 bg-gray-200 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-4">
        <div className="text-red-500 text-sm">{error}</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">Suggested Queries</h3>
      
      <div className="flex overflow-x-auto mb-4 pb-2 -mx-1">
        {categories.map(category => (
          <button
            key={category}
            className={`px-3 py-1 rounded-full text-xs font-medium mx-1 whitespace-nowrap ${
              activeCategory === category
                ? 'bg-indigo-100 text-indigo-800'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
            onClick={() => setActiveCategory(category)}
          >
            {category === 'recent' ? 'Recent' : category.charAt(0).toUpperCase() + category.slice(1)}
          </button>
        ))}
      </div>
      
      <div className="space-y-2">
        {filteredSuggestions.length > 0 ? (
          filteredSuggestions.map((suggestion, index) => (
            <button
              key={index}
              className="w-full text-left p-2 rounded-md hover:bg-gray-50 text-gray-700 text-sm flex items-center"
              onClick={() => handleSuggestionClick(suggestion)}
            >
              <span className="flex-shrink-0 mr-2 text-gray-400">
                {suggestion.isRecent ? (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
                  </svg>
                )}
              </span>
              {suggestion.text}
            </button>
          ))
        ) : (
          <div className="text-gray-500 text-sm p-2">No suggestions available for this category.</div>
        )}
      </div>
    </div>
  );
};

export default QuerySuggestions;