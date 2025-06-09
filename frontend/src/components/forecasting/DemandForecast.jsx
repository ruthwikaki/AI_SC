// frontend/src/components/forecasting/DemandForecast.jsx
import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { 
  runForecast, 
  getForecastMethods, 
  getProductCategories,
  TIME_FRAMES 
} from '../../services/forecasting';
import Loading from '../common/Loading';
import { AgGridReact } from 'ag-grid-react';
// Removed old CSS import
import 'ag-grid-community/styles/ag-theme-quartz.css';

const DemandForecast = () => {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [gridData, setGridData] = useState([]);
  const [methods, setMethods] = useState([]);
  const [categories, setCategories] = useState([]);
  const [forecastParams, setForecastParams] = useState({
    time_frame: TIME_FRAMES.LAST_QUARTER,
    forecast_periods: 12,
    period_type: 'month',
    method: '',
    confidence_level: 0.95,
    product_category: 'all'
  });

  useEffect(() => {
    // Fetch dynamic data on component mount
    fetchDynamicData();
  }, []);

  const fetchDynamicData = async () => {
    try {
      const [methodsData, categoriesData] = await Promise.all([
        getForecastMethods(),
        getProductCategories()
      ]);
      
      setMethods(methodsData);
      setCategories(categoriesData);
      
      // Set default method if available
      if (methodsData.length > 0) {
        setForecastParams(prev => ({
          ...prev,
          method: methodsData[0].id
        }));
      }
    } catch (error) {
      console.error('Error fetching dynamic data:', error);
    }
  };

  const columnDefs = [
    { field: 'product_id', headerName: 'Product ID', pinned: 'left', width: 120 },
    { field: 'product_name', headerName: 'Product Name', pinned: 'left', width: 200 },
    { field: 'category', headerName: 'Category', width: 120 },
    { field: 'current_demand', headerName: 'Current Demand', width: 130, 
      valueFormatter: (params) => params.value?.toLocaleString() || '0' },
    { field: 'forecast_next_month', headerName: 'Next Month', width: 120,
      cellStyle: { color: 'blue', fontWeight: 'bold' },
      valueFormatter: (params) => params.value?.toLocaleString() || '0' },
    { field: 'trend', headerName: 'Trend', width: 100,
      cellRenderer: (params) => {
        const trend = params.value;
        return trend > 0 
          ? `<span style="color: green">↑ ${trend}%</span>`
          : `<span style="color: red">↓ ${Math.abs(trend)}%</span>`;
      }
    },
    { field: 'confidence', headerName: 'Confidence', width: 110,
      cellRenderer: (params) => {
        const confidence = params.value;
        const color = confidence > 90 ? 'green' : confidence > 80 ? 'orange' : 'red';
        return `<span style="color: ${color}">${confidence}%</span>`;
      }
    },
    { field: 'mape', headerName: 'MAPE %', width: 100,
      valueFormatter: (params) => params.value?.toFixed(2) || '0' }
  ];

  const runDemandForecast = async () => {
    if (!forecastParams.method) {
      alert('Please select a forecast method');
      return;
    }

    setLoading(true);
    try {
      const response = await runForecast({
        ...forecastParams,
        forecast_type: 'demand'
      });
      
      processForecastData(response);
    } catch (error) {
      console.error('Error running demand forecast:', error);
    } finally {
      setLoading(false);
    }
  };

    const processForecastData = (response) => {
    if (!response.results) return;
    
    setForecastData(response);
    
    // Log what we received
    console.log('Processing forecast data:', {
      hasProducts: !!response.results.products,
      productCount: response.results.products?.length || 0,
      hasProductForecasts: !!response.results.product_forecasts,
      productForecastCount: response.results.product_forecasts?.length || 0
    });
    
    // Check multiple possible data structures
    let products = response.results.products || 
                  response.results.product_forecasts || 
                  response.results.items ||
                  [];
    
    if (products.length > 0) {
      setGridData(products);
    } else {
      // No product data - show message and sample data
      console.warn('No product data received from backend');
      
      // Generate sample data
      const sampleProducts = [];
      const categories = categories.length > 0 ? categories.map(c => c.name) : ['Category A', 'Category B', 'Category C'];
      
      for (let i = 0; i < 20; i++) {
        const baseValue = Math.floor(Math.random() * 1000) + 100;
        const trend = (Math.random() - 0.5) * 20;
        
        sampleProducts.push({
          product_id: `DEMO-${String(i + 1).padStart(3, '0')}`,
          product_name: `Demo Product ${i + 1}`,
          category: categories[i % categories.length],
          current_demand: baseValue,
          forecast_next_month: Math.floor(baseValue * (1 + trend/100)),
          trend: trend,
          confidence: 80 + Math.random() * 15,
          mape: Math.random() * 10 + 5,
          abc_class: i < 5 ? 'A' : i < 15 ? 'B' : 'C'
        });
      }
      
      setGridData(sampleProducts);
      
      // Show notification
      alert('Note: Showing demo data. To see real product forecasts, ensure your backend returns product-level data.');
    }
  };

  const getChartData = () => {
    if (!forecastData?.results) return null;
    
    const { history = [], forecast = [] } = forecastData.results;
    
    return {
      labels: [...history.map(h => h.period), ...forecast.map(f => f.period)],
      datasets: [
        {
          label: 'Historical Demand',
          data: history.map(h => h.value),
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderWidth: 2
        },
        {
          label: 'Forecast',
          data: [...Array(history.length - 1).fill(null), history[history.length - 1]?.value, ...forecast.map(f => f.value)],
          borderColor: 'rgb(16, 185, 129)',
          backgroundColor: 'rgba(16, 185, 129, 0.2)',
          borderWidth: 2,
          borderDash: [5, 5]
        }
      ]
    };
  };

  const exportToCSV = () => {
    if (gridData.length === 0) {
      alert('No data to export');
      return;
    }

    const headers = ['Product ID', 'Product Name', 'Category', 'Current Demand', 'Forecast Next Month', 'Trend %', 'Confidence %', 'MAPE %'];
    const rows = gridData.map(row => [
      row.product_id,
      row.product_name,
      row.category,
      row.current_demand,
      row.forecast_next_month,
      row.trend,
      row.confidence,
      row.mape
    ]);
    
    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'demand_forecast.csv';
    a.click();
  };

  useEffect(() => {
    // Run initial forecast once methods are loaded
    if (methods.length > 0 && forecastParams.method) {
      runDemandForecast();
    }
  }, [methods]);

  if (loading) return <Loading />;

  return (
    <div className="space-y-6">
      {/* Header and Controls */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Demand Forecasting</h2>
          <div className="flex space-x-4">
            <button
              onClick={exportToCSV}
              className="px-4 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
            >
              Export CSV
            </button>
            <button
              onClick={runDemandForecast}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Run Forecast
            </button>
          </div>
        </div>

        {/* Forecast Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Historical Period
            </label>
            <select
              value={forecastParams.time_frame}
              onChange={(e) => setForecastParams({ ...forecastParams, time_frame: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              {Object.entries(TIME_FRAMES).map(([key, value]) => (
                <option key={key} value={value}>
                  {key.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Forecast Method
            </label>
            <select
              value={forecastParams.method}
              onChange={(e) => setForecastParams({ ...forecastParams, method: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              {methods.map(method => (
                <option key={method.id} value={method.id}>
                  {method.name}
                </option>
              ))}
            </select>
            {methods.find(m => m.id === forecastParams.method)?.description && (
              <p className="text-xs text-gray-500 mt-1">
                {methods.find(m => m.id === forecastParams.method).description}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Forecast Periods
            </label>
            <input
              type="number"
              value={forecastParams.forecast_periods}
              onChange={(e) => setForecastParams({ ...forecastParams, forecast_periods: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              min="1"
              max="24"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Product Category
            </label>
            <select
              value={forecastParams.product_category}
              onChange={(e) => setForecastParams({ ...forecastParams, product_category: e.target.value })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Categories</option>
              {categories.map(cat => (
                <option key={cat.name} value={cat.name}>
                  {cat.name} ({cat.product_count} products)
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Forecast Chart */}
      {forecastData && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Aggregate Demand Forecast</h3>
          <div style={{ height: '400px' }}>
            <Line
              data={getChartData()}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top'
                  },
                  title: {
                    display: true,
                    text: 'Demand Forecast Analysis'
                  }
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Demand (units)'
                    }
                  }
                }
              }}
            />
          </div>
        </div>
      )}

      {/* Product-Level Forecast Grid */}
      {gridData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Product-Level Forecasts</h3>
          <div className="ag-theme-quartz" style={{ height: '500px', width: '100%' }}>
            <AgGridReact
              rowData={gridData}
              columnDefs={columnDefs}
              defaultColDef={{
                sortable: true,
                filter: true,
                resizable: true
              }}
              pagination={true}
              paginationPageSize={10}
              enableCellTextSelection={true}
            />
          </div>
        </div>
      )}

      {/* No Product Data Message */}
      {forecastData && gridData.length === 0 && (
        <div className="bg-yellow-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-2 text-yellow-900">Product-Level Data Not Available</h3>
          <p className="text-gray-700">
            This forecast shows aggregate data only. To see product-level forecasts, 
            ensure your backend is configured to return detailed product forecasts.
          </p>
        </div>
      )}

      {/* Insights and Anomalies */}
      {forecastData?.results?.insights && forecastData.results.insights.length > 0 && (
        <div className="bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4 text-blue-900">Forecast Insights</h3>
          <ul className="space-y-2">
            {forecastData.results.insights.map((insight, index) => (
              <li key={index} className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-gray-700">{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DemandForecast;



