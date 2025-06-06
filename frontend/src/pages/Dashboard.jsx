// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  CubeIcon, 
  ChartBarIcon, 
  TruckIcon, 
  UsersIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChevronRightIcon,
  BellIcon,
  ClipboardDocumentCheckIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../hooks/useAuth';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [metrics, setMetrics] = useState({
    inventoryValue: 2457000,
    inventoryChange: 2.4,
    orderFillRate: 94.7,
    orderFillChange: -0.8,
    onTimeDelivery: 92.3,
    deliveryChange: 1.2,
    supplierPerformance: 87.2,
    supplierChange: 0.5
  });

  const [suggestions] = useState([
    { id: 1, text: "show me all products", category: "inventory" },
    { id: 2, text: "create a bar chart of supplier ratings", category: "suppliers" },
    { id: 3, text: "analyze order trends for last month", category: "orders" },
    { id: 4, text: "which products are below reorder point", category: "inventory" },
    { id: 5, text: "show supplier performance metrics", category: "suppliers" },
    { id: 6, text: "visualize inventory levels by category", category: "inventory" },
    { id: 7, text: "create a pie chart of products by category", category: "products" }
  ]);

  const [selectedCategory, setSelectedCategory] = useState('All');

  const MetricCard = ({ title, value, change, icon: Icon, format = 'number' }) => {
    const isPositive = change > 0;
    const formatValue = () => {
      if (format === 'currency') {
        return `$${(value / 1000000).toFixed(3)}M`;
      } else if (format === 'percentage') {
        return `${value}%`;
      }
      return value.toLocaleString();
    };

    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="bg-gray-50 p-3 rounded-lg mr-4">
              <Icon className="h-6 w-6 text-gray-600" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-600">{title}</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">{formatValue()}</p>
              <div className="flex items-center mt-1">
                {isPositive ? (
                  <ArrowTrendingUpIcon className="h-4 w-4 text-green-500 mr-1" />
                ) : (
                  <ArrowTrendingDownIcon className="h-4 w-4 text-red-500 mr-1" />
                )}
                <span className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                  {isPositive ? '+' : ''}{Math.abs(change)}% from last month
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const handleSuggestionClick = (suggestion) => {
    navigate('/query', { state: { query: suggestion.text } });
  };

  const categories = ['All', 'Inventory', 'Suppliers'];
  const filteredSuggestions = selectedCategory === 'All' 
    ? suggestions 
    : suggestions.filter(s => s.category.toLowerCase() === selectedCategory.toLowerCase());

  return (
    <div>
      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">
          Welcome back, {user?.name || 'User'}
        </h1>
        <p className="text-gray-600 mt-1">
          Here's what's happening across your supply chain today
        </p>
      </div>

      {/* KPI Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <MetricCard
          title="Inventory Value"
          value={metrics.inventoryValue}
          change={metrics.inventoryChange}
          icon={CubeIcon}
          format="currency"
        />
        <MetricCard
          title="Order Fill Rate"
          value={metrics.orderFillRate}
          change={metrics.orderFillChange}
          icon={ClipboardDocumentCheckIcon}
          format="percentage"
        />
        <MetricCard
          title="On-Time Delivery"
          value={metrics.onTimeDelivery}
          change={metrics.deliveryChange}
          icon={TruckIcon}
          format="percentage"
        />
        <MetricCard
          title="Supplier Performance"
          value={metrics.supplierPerformance}
          change={metrics.supplierChange}
          icon={UsersIcon}
          format="percentage"
        />
      </div>

      {/* Ask a Question Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Ask a question</h2>
        
        <div className="mb-6">
          <p className="text-sm text-gray-600 mb-3">Ask a supply chain question</p>
          <button
            onClick={() => navigate('/query')}
            className="w-full text-left px-4 py-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors flex items-center justify-between group border border-gray-200"
          >
            <span className="text-gray-500">Ask about your supply chain...</span>
            <ChevronRightIcon className="h-5 w-5 text-gray-400 group-hover:text-gray-600 transition-colors" />
          </button>
        </div>

        {/* Category Tabs */}
        <div className="mb-4">
          <p className="text-sm font-medium text-gray-700 mb-3">Suggested questions:</p>
          <div className="flex space-x-2 mb-4">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-3 py-1 text-sm rounded-full transition-colors ${
                  selectedCategory === category
                    ? 'bg-blue-100 text-blue-700'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>

        {/* Suggested Questions */}
        <div className="space-y-2">
          {filteredSuggestions.map((suggestion) => (
            <button
              key={suggestion.id}
              onClick={() => handleSuggestionClick(suggestion)}
              className="w-full text-left px-4 py-3 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors flex items-center justify-between group"
            >
              <div className="flex items-center space-x-3">
                <ChevronRightIcon className="h-4 w-4 text-gray-400" />
                <span>{suggestion.text}</span>
              </div>
              <span className="text-xs text-gray-500 capitalize">{suggestion.category}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;