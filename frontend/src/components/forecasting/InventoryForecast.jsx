// frontend/src/components/forecasting/InventoryForecast.jsx
import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import { 
  runForecast, 
  getForecastMethods, 
  getWarehouses,
  TIME_FRAMES 
} from '../../services/forecasting';
import Loading from '../common/Loading';
import { AgGridReact } from 'ag-grid-react';

import 'ag-grid-community/styles/ag-theme-quartz.css';

const InventoryForecast = () => {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [warehouses, setWarehouses] = useState([]);
  const [methods, setMethods] = useState([]);
  const [selectedWarehouse, setSelectedWarehouse] = useState('all');
  const [forecastParams, setForecastParams] = useState({
    time_frame: TIME_FRAMES.LAST_QUARTER,
    forecast_periods: 12,
    period_type: 'month',
    method: '',
    confidence_level: 0.95,
    include_safety_stock: true
  });

  useEffect(() => {
    fetchDynamicData();
  }, []);

  const fetchDynamicData = async () => {
    try {
      const [warehouseData, methodsData] = await Promise.all([
        getWarehouses(),
        getForecastMethods()
      ]);
      
      setWarehouses(warehouseData);
      setMethods(methodsData);
      
      // Set default method
      if (methodsData.length > 0) {
        setForecastParams(prev => ({
          ...prev,
          method: methodsData.find(m => m.id === 'arima')?.id || methodsData[0].id
        }));
      }
    } catch (error) {
      console.error('Error fetching dynamic data:', error);
    }
  };

  const runInventoryForecast = async () => {
    if (!forecastParams.method) {
      alert('Please select a forecast method');
      return;
    }

    setLoading(true);
    try {
      const response = await runForecast({
        ...forecastParams,
        forecast_type: 'inventory',
        warehouse_id: selectedWarehouse !== 'all' ? selectedWarehouse : null
      });
      
      setForecastData(processInventoryForecast(response));
    } catch (error) {
      console.error('Error running inventory forecast:', error);
    } finally {
      setLoading(false);
    }
  };

  const processInventoryForecast = (response) => {
    if (!response.results) return null;
    
    const { forecast, history, insights } = response.results;
    
    return {
      chartData: {
        labels: [...(history?.map(h => h.period) || []), ...(forecast?.map(f => f.period) || [])],
        datasets: [
          {
            label: 'Current Inventory',
            data: history?.map(h => h.value) || [],
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.2)'
          },
          {
            label: 'Projected Inventory',
            data: [...Array(history?.length || 0).fill(null), ...(forecast?.map(f => f.value) || [])],
            borderColor: 'rgb(16, 185, 129)',
            backgroundColor: 'rgba(16, 185, 129, 0.2)',
            borderDash: [5, 5]
          },
          {
            label: 'Safety Stock',
            data: Array((history?.length || 0) + (forecast?.length || 0)).fill(response.results.safety_stock_level || 0),
            borderColor: 'rgb(239, 68, 68)',
            borderDash: [2, 2],
            fill: false
          }
        ]
      },
      insights: insights || [],
      stockoutRisk: response.results.stockout_risk || [],
      recommendations: response.results.recommendations || [],
      safetyStockLevel: response.results.safety_stock_level || 0
    };
  };

  useEffect(() => {
    // Run initial forecast once data is loaded
    if (methods.length > 0 && forecastParams.method) {
      runInventoryForecast();
    }
  }, [methods]);

  if (loading) return <Loading />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Inventory Forecasting</h2>
          <button
            onClick={runInventoryForecast}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Update Forecast
          </button>
        </div>

        {/* Dynamic Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Warehouse
            </label>
            <select
              value={selectedWarehouse}
              onChange={(e) => setSelectedWarehouse(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="all">All Warehouses</option>
              {warehouses.map(wh => (
                <option key={wh.id} value={wh.id}>
                  {wh.name} - {wh.location}
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
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              {methods.map(method => (
                <option key={method.id} value={method.id}>
                  {method.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Include Safety Stock
            </label>
            <select
              value={forecastParams.include_safety_stock}
              onChange={(e) => setForecastParams({ ...forecastParams, include_safety_stock: e.target.value === 'true' })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>
      </div>

      {/* Forecast Chart */}
      {forecastData && (
        <>
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Inventory Level Forecast</h3>
            <div style={{ height: '400px' }}>
              <Line
                data={forecastData.chartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'top' },
                    title: { display: true, text: 'Inventory Levels Over Time' }
                  },
                  scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Units' } }
                  }
                }}
              />
            </div>
          </div>

          {/* Stockout Risk Analysis */}
          {forecastData.stockoutRisk.length > 0 && (
            <div className="bg-red-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-red-900">Stockout Risk Alert</h3>
              <div className="space-y-3">
                {forecastData.stockoutRisk.map((risk, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-white rounded">
                    <div>
                      <p className="font-medium">{risk.product_name}</p>
                      <p className="text-sm text-gray-600">Expected stockout: {risk.expected_date}</p>
                    </div>
                    <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm">
                      {risk.risk_level}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {forecastData.recommendations.length > 0 && (
            <div className="bg-blue-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4 text-blue-900">AI Recommendations</h3>
              <ul className="space-y-2">
                {forecastData.recommendations.map((rec, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-blue-600 mr-2">•</span>
                    <span className="text-gray-700">{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default InventoryForecast;


