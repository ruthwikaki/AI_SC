import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from '../visualization/ChartViewer';
import Loading from '../common/Loading';
import {
  ArrowPathIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  PlusCircleIcon
} from '@heroicons/react/24/outline';

const InventoryDashboard = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [inventoryData, setInventoryData] = useState(null);
  const [dateRange, setDateRange] = useState('30d'); // Default to last 30 days
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    fetchInventoryData();
    // Get available categories for filtering
    fetchCategories();
  }, [dateRange, selectedCategories]);

  const fetchInventoryData = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/analytics/inventory', { params: { dateRange, categories: selectedCategories } });
      
      // Simulated API response
      const mockData = getMockInventoryData(dateRange, selectedCategories);
      setInventoryData(mockData);
    } catch (error) {
      console.error('Error fetching inventory data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCategories = async () => {
    try {
      // Simulated categories
      setCategories(['Raw Materials', 'Work in Progress', 'Finished Goods', 'MRO Supplies']);
    } catch (error) {
      console.error('Error fetching categories:', error);
    }
  };

  const handleDateRangeChange = (range) => {
    setDateRange(range);
  };

  const toggleCategoryFilter = (category) => {
    setSelectedCategories(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category) 
        : [...prev, category]
    );
  };

  const handleExportDashboard = () => {
    // Implementation for exporting the entire dashboard
    alert('Exporting dashboard...');
  };

  const navigateToABCAnalysis = () => {
    navigate('/analytics/inventory/abc-analysis');
  };

  const navigateToSafetyStock = () => {
    navigate('/analytics/inventory/safety-stock');
  };

  const navigateToForecast = () => {
    navigate('/analytics/inventory/forecast');
  };

  // Mock data generator function
  const getMockInventoryData = (range, categories) => {
    // Generate appropriate mock data based on selected date range and categories
    return {
      summary: {
        totalItems: 1245,
        totalValue: 3782000,
        lowStockItems: 28,
        excessStockItems: 35
      },
      stockLevelTrend: {
        type: 'line',
        title: 'Inventory Levels Over Time',
        data: [
          { date: '2023-01-01', value: 3500000 },
          { date: '2023-02-01', value: 3650000 },
          { date: '2023-03-01', value: 3720000 },
          { date: '2023-04-01', value: 3800000 },
          { date: '2023-05-01', value: 3782000 }
        ],
        config: {
          xKey: 'date',
          yKey: 'value',
          curve: 'curveMonotoneX',
          showArea: true,
          valueFormatter: (value) => `$${(value / 1000000).toFixed(2)}M`
        }
      },
      inventoryByCategory: {
        type: 'pie',
        title: 'Inventory Distribution by Category',
        data: [
          { name: 'Raw Materials', value: 1250000 },
          { name: 'Work in Progress', value: 850000 },
          { name: 'Finished Goods', value: 1420000 },
          { name: 'MRO Supplies', value: 262000 }
        ],
        config: {
          nameKey: 'name',
          valueKey: 'value',
          innerRadius: 0.6,
          valueFormatter: (value) => `$${(value / 1000).toFixed(0)}K`
        }
      },
      topLowStockItems: {
        type: 'bar',
        title: 'Top 10 Low Stock Items',
        data: [
          { name: 'Product A', value: 15 },
          { name: 'Product B', value: 22 },
          { name: 'Product C', value: 28 },
          { name: 'Product D', value: 30 },
          { name: 'Product E', value: 35 }
        ],
        config: {
          xKey: 'name',
          yKey: 'value',
          horizontal: true,
          color: '#ef4444'
        }
      },
      abcAnalysis: {
        type: 'bar',
        title: 'ABC Analysis',
        data: [
          { category: 'A (High Value)', items: 187, value: 2725000 },
          { category: 'B (Medium Value)', items: 458, value: 853000 },
          { category: 'C (Low Value)', items: 600, value: 204000 }
        ],
        config: {
          xKey: 'category',
          yKey: 'value',
          showValues: true,
          valueFormatter: (value) => `$${(value / 1000).toFixed(0)}K`
        }
      }
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading inventory analytics..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full">
      {/* Dashboard Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">Inventory Analytics</h1>
          <p className="text-sm text-gray-500 mt-1">Monitor and optimize your inventory levels</p>
        </div>
        
        <div className="flex items-center space-x-3 mt-3 sm:mt-0">
          <div className="relative">
            <button
              onClick={() => setFilterOpen(!filterOpen)}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <FunnelIcon className="-ml-1 mr-2 h-4 w-4" />
              Filter
            </button>
            
            {filterOpen && (
              <div className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                <div className="py-1 px-3">
                  <div className="mb-2">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Date Range</h4>
                    <div className="mt-1 space-y-1">
                      {['7d', '30d', '90d', '1y'].map(range => (
                        <label key={range} className="flex items-center">
                          <input
                            type="radio"
                            name="dateRange"
                            value={range}
                            checked={dateRange === range}
                            onChange={() => handleDateRangeChange(range)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">
                            {range === '7d' ? 'Last 7 days' : 
                             range === '30d' ? 'Last 30 days' : 
                             range === '90d' ? 'Last 90 days' : 'Last year'}
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Categories</h4>
                    <div className="mt-1 space-y-1">
                      {categories.map(category => (
                        <label key={category} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedCategories.includes(category)}
                            onChange={() => toggleCategoryFilter(category)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{category}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-3 flex justify-end">
                    <button
                      onClick={() => setFilterOpen(false)}
                      className="px-4 py-1.5 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700"
                    >
                      Apply
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <button
            onClick={fetchInventoryData}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
            Refresh
          </button>
          
          <button
            onClick={handleExportDashboard}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowDownTrayIcon className="-ml-1 mr-2 h-4 w-4" />
            Export
          </button>
        </div>
      </div>
      
      {/* Dashboard Content */}
      <div className="container mx-auto px-4 py-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Total Inventory Items</p>
                <p className="text-2xl font-bold text-gray-900">{inventoryData.summary.totalItems.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-indigo-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Total Inventory Value</p>
                <p className="text-2xl font-bold text-gray-900">${(inventoryData.summary.totalValue / 1000000).toFixed(2)}M</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Low Stock Items</p>
                <p className="text-2xl font-bold text-gray-900">{inventoryData.summary.lowStockItems}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-red-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Excess Stock Items</p>
                <p className="text-2xl font-bold text-gray-900">{inventoryData.summary.excessStockItems}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-yellow-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
        </div>
        
        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartViewer chartData={inventoryData.stockLevelTrend} />
          <ChartViewer chartData={inventoryData.inventoryByCategory} />
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartViewer chartData={inventoryData.topLowStockItems} />
          <ChartViewer chartData={inventoryData.abcAnalysis} />
        </div>
        
        {/* Advanced Analytics Section */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-800">Advanced Inventory Analysis</h2>
          </div>
          <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div 
              onClick={navigateToABCAnalysis}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
            >
              <div className="flex items-center mb-3">
                <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                </div>
                <h3 className="text-md font-medium text-gray-800">ABC Analysis</h3>
              </div>
              <p className="text-sm text-gray-600">Categorize inventory by value to optimize management strategies.</p>
              <div className="mt-3 flex items-center text-indigo-600 text-sm font-medium">
                <span>Run analysis</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
            
            <div 
              onClick={navigateToSafetyStock}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
            >
              <div className="flex items-center mb-3">
                <div className="h-10 w-10 rounded-full bg-green-100 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <h3 className="text-md font-medium text-gray-800">Safety Stock Calculator</h3>
              </div>
              <p className="text-sm text-gray-600">Calculate optimal safety stock levels to prevent stockouts.</p>
              <div className="mt-3 flex items-center text-indigo-600 text-sm font-medium">
                <span>Calculate</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
            
            <div 
              onClick={navigateToForecast}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
            >
              <div className="flex items-center mb-3">
                <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <h3 className="text-md font-medium text-gray-800">Demand Forecast</h3>
              </div>
              <p className="text-sm text-gray-600">Predict future demand to optimize purchasing and production.</p>
              <div className="mt-3 flex items-center text-indigo-600 text-sm font-medium">
                <span>Generate forecast</span>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InventoryDashboard;

