import React, { useState } from 'react';
import { Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import ForecastDashboard from '../components/forecasting/ForecastDashboard';
import ForecastSettings from '../components/forecasting/ForecastSettings';
import ForecastResults from '../components/forecasting/ForecastResults';
import ForecastComparison from '../components/forecasting/ForecastComparison';
import SeasonalityAnalysis from '../components/forecasting/SeasonalityAnalysis';
import { useForecast } from '../hooks/forecasting/useForecast';
import { FORECAST_METHODS, TIME_FRAMES } from '../services/forecasting';

const Forecasting = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [forecastResults, setForecastResults] = useState(null);
  const { runForecast, loading } = useForecast();

  const handleRunForecast = async (type, params = {}) => {
    try {
      const results = await runForecast({
        ...params,
        forecast_type: type,
      });
      setForecastResults(results);
      setActiveTab(type);
    } catch (error) {
      console.error('Error running forecast:', error);
    }
  };

  const tabs = [
    { id: 'dashboard', name: 'Dashboard', path: '/forecasting' },
    { id: 'demand', name: 'Demand Forecast', path: '/forecasting/demand' },
    { id: 'inventory', name: 'Inventory Forecast', path: '/forecasting/inventory' },
    { id: 'sales', name: 'Sales Forecast', path: '/forecasting/sales' },
    { id: 'comparison', name: 'Model Comparison', path: '/forecasting/comparison' },
    { id: 'settings', name: 'Settings', path: '/forecasting/settings' },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">Forecasting Engine</h1>
          </div>
          
          {/* Tab Navigation */}
          <nav className="flex space-x-8" aria-label="Tabs">
            {tabs.map((tab) => (
              <NavLink
                key={tab.id}
                to={tab.path}
                className={({ isActive }) =>
                  `${
                    isActive
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`
                }
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.name}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<ForecastDashboard />} />
          <Route path="/demand" element={
            <div>
              <div className="mb-6 flex justify-between items-center">
                <h2 className="text-2xl font-bold">Demand Forecasting</h2>
                <button
                  onClick={() => handleRunForecast('demand')}
                  disabled={loading}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? 'Running...' : 'Run Forecast'}
                </button>
              </div>
              {forecastResults && activeTab === 'demand' && (
                <ForecastResults results={forecastResults} />
              )}
            </div>
          } />
          <Route path="/inventory" element={
            <div>
              <div className="mb-6 flex justify-between items-center">
                <h2 className="text-2xl font-bold">Inventory Forecasting</h2>
                <button
                  onClick={() => handleRunForecast('inventory')}
                  disabled={loading}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? 'Running...' : 'Run Forecast'}
                </button>
              </div>
              {forecastResults && activeTab === 'inventory' && (
                <ForecastResults results={forecastResults} />
              )}
            </div>
          } />
          <Route path="/sales" element={
            <div>
              <div className="mb-6 flex justify-between items-center">
                <h2 className="text-2xl font-bold">Sales Forecasting</h2>
                <button
                  onClick={() => handleRunForecast('sales')}
                  disabled={loading}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {loading ? 'Running...' : 'Run Forecast'}
                </button>
              </div>
              {forecastResults && activeTab === 'sales' && (
                <ForecastResults results={forecastResults} />
              )}
            </div>
          } />
          <Route path="/comparison" element={<ForecastComparison models={[]} />} />
          <Route path="/settings" element={<ForecastSettings />} />
        </Routes>
      </div>
    </div>
  );
};

export default Forecasting;
