// src/components/query/QuerySuggestions.jsx
import React, { useState, useEffect } from 'react';
import { useQuery } from '../../hooks/useQuery';

// Add these console.logs to your existing component:
const QuerySuggestions = ({ onSuggestionSelect }) => {
  const [suggestions, setSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeCategory, setActiveCategory] = useState('all');
  const { loadSuggestions } = useQuery();

  useEffect(() => {
    console.log('QuerySuggestions component mounted'); // ADD THIS
    
    const fetchSuggestions = async () => {
      setIsLoading(true);
      try {
        console.log('Calling loadSuggestions...'); // ADD THIS
        const loadedSuggestions = await loadSuggestions();
        console.log('Loaded suggestions:', loadedSuggestions); // ADD THIS
        setSuggestions(loadedSuggestions || []);
      } catch (error) {
        console.error('Error loading suggestions:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSuggestions();
  }, [loadSuggestions]);

  console.log('Current state - suggestions:', suggestions, 'isLoading:', isLoading); // ADD THIS

  // Rest of your component remains the same...
  const handleSuggestionClick = (suggestion) => {
    if (onSuggestionSelect) {
      // Handle both string and object suggestions
      const queryText = typeof suggestion === 'string' ? suggestion : suggestion.text;
      onSuggestionSelect(queryText);
    }
  };

  // Get unique categories
  const categories = ['all', ...new Set(suggestions.map(s => s.category || 'general').filter(c => c !== 'general'))];

  // Filter suggestions by category
  const filteredSuggestions = activeCategory === 'all' 
    ? suggestions 
    : suggestions.filter(s => (s.category || 'general') === activeCategory);

  if (isLoading) {
    return <div className="text-gray-500 text-sm">Loading suggestions...</div>;
  }

  return (
    <div>
      {/* Category filters if available */}
      {categories.length > 1 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {categories.map(category => (
            <button
              key={category}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                activeCategory === category
                  ? 'bg-blue-100 text-blue-800'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              onClick={() => setActiveCategory(category)}
            >
              {category === 'all' ? 'All' : category.charAt(0).toUpperCase() + category.slice(1)}
            </button>
          ))}
        </div>
      )}

      {/* Suggestions grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {filteredSuggestions.map((suggestion, index) => {
          const text = typeof suggestion === 'string' ? suggestion : suggestion.text;
          const category = typeof suggestion === 'object' ? suggestion.category : 'general';
          
          return (
            <button
              key={index}
              className="group relative p-3 bg-white border border-gray-200 rounded-lg hover:border-blue-300 hover:shadow-sm transition-all duration-150 text-left"
              onClick={() => handleSuggestionClick(suggestion)}
            >
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 mt-0.5">
                  <svg className="h-5 w-5 text-gray-400 group-hover:text-blue-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-700 group-hover:text-gray-900">
                    {text}
                  </p>
                  {category && category !== 'general' && (
                    <p className="text-xs text-gray-500 mt-1">
                      {category}
                    </p>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default QuerySuggestions;