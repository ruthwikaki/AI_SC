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
  ClipboardDocumentCheckIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../hooks/useAuth';
import api from '../services/api';

const Dashboard = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [overview, setOverview] = useState(null);
  const [recentOrders, setRecentOrders] = useState([]);
  const [inventoryAlerts, setInventoryAlerts] = useState([]);
  const [supplierMetrics, setSupplierMetrics] = useState(null);
  const [logisticsSummary, setLogisticsSummary] = useState(null);

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

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch all dashboard data in parallel
      const [
        overviewRes,
        ordersRes,
        alertsRes,
        supplierRes,
        logisticsRes
      ] = await Promise.all([
        api.get('/api/dashboard/overview').catch(err => ({ error: err })),
        api.get('/api/dashboard/recent-orders').catch(err => ({ error: err })),
        api.get('/api/dashboard/inventory-alerts').catch(err => ({ error: err })),
        api.get('/api/dashboard/supplier-metrics').catch(err => ({ error: err })),
        api.get('/api/dashboard/logistics-summary').catch(err => ({ error: err }))
      ]);

      // Set data from successful responses
      if (!overviewRes.error) setOverview(overviewRes.data);
      if (!ordersRes.error) setRecentOrders(ordersRes.data.orders || []);
      if (!alertsRes.error) setInventoryAlerts(alertsRes.data.alerts || []);
      if (!supplierRes.error) setSupplierMetrics(supplierRes.data);
      if (!logisticsRes.error) setLogisticsSummary(logisticsRes.data);

    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ title, value, change, icon: Icon, format = 'number' }) => {
    const isPositive = change > 0;
    const formatValue = () => {
      if (format === 'currency') {
        return `$${(value / 1000000).toFixed(3)}M`;
      } else if (format === 'percentage') {
        return `${value}%`;
      }
      return value?.toLocaleString() || '0';
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
              {change !== undefined && (
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
              )}
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

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
        <div className="flex">
          <ExclamationTriangleIcon className="h-5 w-5 mr-2" />
          <p>{error}</p>
        </div>
      </div>
    );
  }

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
          value={overview?.total_inventory_value || 0}
          change={2.4} // You can calculate this from historical data
          icon={CubeIcon}
          format="currency"
        />
        <MetricCard
          title="Order Fill Rate"
          value={94.7} // This could come from analytics endpoint
          change={-0.8}
          icon={ClipboardDocumentCheckIcon}
          format="percentage"
        />
        <MetricCard
          title="On-Time Delivery"
          value={overview?.on_time_delivery_rate || 0}
          change={1.2}
          icon={TruckIcon}
          format="percentage"
        />
        <MetricCard
          title="Active Suppliers"
          value={overview?.active_suppliers || 0}
          change={0.5}
          icon={UsersIcon}
          format="number"
        />
      </div>

      {/* Quick Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        {/* Recent Orders */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Orders</h3>
          {recentOrders.length > 0 ? (
            <div className="space-y-3">
              {recentOrders.slice(0, 3).map((order) => (
                <div key={order.id} className="flex justify-between items-center">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{order.id}</p>
                    <p className="text-xs text-gray-500">{order.customer}</p>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full ${
                    order.status === 'Shipped' ? 'bg-green-100 text-green-800' :
                    order.status === 'Processing' ? 'bg-blue-100 text-blue-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {order.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No recent orders</p>
          )}
        </div>

        {/* Inventory Alerts */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Inventory Alerts</h3>
          {inventoryAlerts.length > 0 ? (
            <div className="space-y-3">
              {inventoryAlerts.slice(0, 3).map((alert) => (
                <div key={alert.id} className="flex items-start">
                  <ExclamationTriangleIcon className={`h-5 w-5 mr-2 flex-shrink-0 ${
                    alert.severity === 'high' ? 'text-red-500' :
                    alert.severity === 'medium' ? 'text-yellow-500' :
                    'text-blue-500'
                  }`} />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{alert.product}</p>
                    <p className="text-xs text-gray-500">
                      {alert.type === 'low_stock' ? `Stock: ${alert.current_stock}` : alert.type}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">No active alerts</p>
          )}
        </div>

        {/* Logistics Summary */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Logistics Status</h3>
          {logisticsSummary?.summary ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">In Transit</span>
                <span className="text-sm font-medium">{logisticsSummary.summary.in_transit}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Delivered Today</span>
                <span className="text-sm font-medium">{logisticsSummary.summary.delivered}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">On-Time Rate</span>
                <span className="text-sm font-medium">{logisticsSummary.summary.on_time_rate}%</span>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-500">No logistics data</p>
          )}
        </div>
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