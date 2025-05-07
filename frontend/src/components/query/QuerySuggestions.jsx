import React from 'react';
import { FaRegLightbulb, FaHistory, FaRegClock, FaPlus, FaStar } from 'react-icons/fa';

const QuerySuggestions = ({ onSelectQuery, recentQueries = [] }) => {
  // Predefined suggestions for supply chain queries
  const suggestions = [
    {
      category: 'Inventory Optimization',
      queries: [
        "What's the optimal inventory level for product SKU-12345 based on historical demand?",
        "Identify inventory items at risk of stockout in the next 30 days",
        "Analyze slow-moving inventory and recommend disposition strategies",
        "Calculate safety stock levels for seasonal products"
      ]
    },
    {
      category: 'Demand Forecasting',
      queries: [
        "Forecast demand for next quarter based on historical sales data",
        "How will the upcoming holiday season affect demand for our top 10 products?",
        "Analyze impact of recent marketing campaigns on product demand",
        "Predict demand shifts based on competitor pricing changes"
      ]
    },
    {
      category: 'Logistics Optimization',
      queries: [
        "What's the most cost-effective shipping method for orders to the East Coast?",
        "Identify potential bottlenecks in the current distribution network",
        "Calculate optimal route for multi-stop deliveries in the Midwest region",
        "Analyze impact of fuel price increases on total logistics costs"
      ]
    },
    {
      category: 'Supplier Analysis',
      queries: [
        "Rank suppliers based on delivery performance over the last 6 months",
        "Identify suppliers with quality issues and recommend action plans",
        "Analyze cost trends for raw materials from our top 5 suppliers",
        "Calculate risk scores for critical component suppliers"
      ]
    }
  ];

  // Popular queries
  const popularQueries = [
    "How can we reduce lead times for international shipments?",
    "What's causing the highest percentage of returns in our e-commerce channel?",
    "Identify opportunities to reduce transportation costs",
    "Analyze warehouse capacity utilization across all distribution centers"
  ];

  return (
    <div className="w-full max-w-4xl mx-auto mt-6">
      {/* Recent queries section - only show if there are recent queries */}
      {recentQueries.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center text-gray-700 mb-2">
            <FaHistory className="mr-2" />
            <h3 className="font-medium">Recent Queries</h3>
          </div>
          
          <div className="flex flex-wrap gap-2">
            {recentQueries.slice(0, 4).map((query, idx) => (
              <button
                key={`recent-${idx}`}
                onClick={() => onSelectQuery(query)}
                className="inline-flex items-center px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-full text-sm text-gray-700 transition-colors"
              >
                <FaRegClock className="mr-2 text-gray-500" size={12} />
                <span className="truncate max-w-xs">{query}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Popular queries section */}
      <div className="mb-6">
        <div className="flex items-center text-gray-700 mb-2">
          <FaStar className="mr-2 text-yellow-500" />
          <h3 className="font-medium">Popular Queries</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {popularQueries.map((query, idx) => (
            <button
              key={`popular-${idx}`}
              onClick={() => onSelectQuery(query)}
              className="text-left px-3 py-2 bg-yellow-50 hover:bg-yellow-100 border border-yellow-200 rounded-md text-sm text-gray-800 transition-colors"
            >
              {query}
            </button>
          ))}
        </div>
      </div>
      
      {/* Category-based suggestions */}
      <div>
        <div className="flex items-center text-gray-700 mb-2">
          <FaRegLightbulb className="mr-2 text-blue-500" />
          <h3 className="font-medium">Suggested Queries</h3>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {suggestions.map((category) => (
            <div 
              key={category.category}
              className="bg-white border border-gray-200 rounded-lg overflow-hidden"
            >
              <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                <h4 className="font-medium text-gray-700">{category.category}</h4>
              </div>
              
              <div className="p-2">
                {category.queries.map((query, idx) => (
                  <button
                    key={`${category.category}-${idx}`}
                    onClick={() => onSelectQuery(query)}
                    className="w-full text-left px-3 py-2 hover:bg-blue-50 rounded-md text-sm text-gray-700 transition-colors flex items-center"
                  >
                    <FaPlus className="mr-2 text-blue-400" size={10} />
                    <span>{query}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuerySuggestions;