// frontend/src/components/forecasting/SalesForecast.jsx
import React, { useState, useEffect } from 'react';
import { Area, Bar } from 'react-chartjs-2';
import { 
  runForecast, 
  getForecastMethods, 
  getRegions,
  TIME_FRAMES 
} from '../../services/forecasting';
import Loading from '../common/Loading';

const SalesForecast = () => {
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState(null);
  const [regions, setRegions] = useState([]);
  const [methods, setMethods] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState('all');
  const [forecastParams, setForecastParams] = useState({
    time_frame: TIME_FRAMES.LAST_YEAR,
    forecast_periods: 12,
    period_type: 'month',
    method: '',
    confidence_level: 0.95,
    include_seasonality: true,
  });

  useEffect(() => {
    fetchDynamicData();
  }, []);

  const fetchDynamicData = async () => {
    try {
      const [regionsData, methodsData] = await Promise.all([
        getRegions(),
        getForecastMethods()
      ]);
      
      setRegions(regionsData);
      setMethods(methodsData);
      
      // Set default method (prefer Prophet or SARIMA for sales)
      if (methodsData.length > 0) {
        const preferredMethod = methodsData.find(m => m.id === 'prophet') || 
                               methodsData.find(m => m.id === 'sarima') || 
                               methodsData[0];
        setForecastParams(prev => ({
          ...prev,
          method: preferredMethod.id
        }));
      }
    } catch (error) {
      console.error('Error fetching dynamic data:', error);
    }
  };

  const runSalesForecast = async () => {
    if (!forecastParams.method) {
      alert('Please select a forecast method');
      return;
    }

    setLoading(true);
    try {
      const response = await runForecast({
        ...forecastParams,
        forecast_type: 'sales',
        region: selectedRegion !== 'all' ? selectedRegion : null,
      });
      
      setForecastData(processSalesForecast(response));
    } catch (error) {
      console.error('Error running sales forecast:', error);
    } finally {
      setLoading(false);
    }
  };

  const processSalesForecast = (response) => {
    if (!response.results) return null;
    
    const { forecast = [], history = [], insights = [] } = response.results;
    
    // Calculate key metrics from the data
    const totalRevenue = forecast.reduce((sum, f) => sum + (f.revenue || f.value || 0), 0);
    const avgRevenue = totalRevenue / (forecast.length || 1);
    const growthRate = history.length > 0 && forecast.length > 0
      ? ((forecast[0].value - history[history.length - 1].value) / history[history.length - 1].value) * 100
      : 0;
    
    return {
      revenue: response.results.revenue_forecast || {},
      units: response.results.units_forecast || {},
      seasonality: response.results.seasonality_pattern || {},
      topProducts: response.results.top_products || [],
      insights: insights,
      metrics: {
        totalRevenue,
        avgRevenue,
        growthRate,
        forecastAccuracy: response.results.metrics?.mape || 0
      },
      chartData: {
        labels: forecast.map(f => f.period),
        revenue: forecast.map(f => f.revenue || f.value || 0),
        units: forecast.map(f => f.units || 0)
      }
    };
  };

  useEffect(() => {
    if (methods.length > 0 && forecastParams.method) {
      runSalesForecast();
    }
  }, [methods]);

  if (loading) return <Loading />;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Sales Forecasting</h2>
          <button
            onClick={runSalesForecast}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Update Forecast
          </button>
        </div>

        {/* Dynamic Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Region
            </label>
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="all">All Regions</option>
              {regions.map(region => (
                <option key={region.id} value={region.id}>
                  {region.name}
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
            {methods.find(m => m.id === forecastParams.method)?.best_for && (
              <p className="text-xs text-gray-500 mt-1">
                Best for: {methods.find(m => m.id === forecastParams.method).best_for}
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Include Seasonality
            </label>
            <select
              value={forecastParams.include_seasonality}
              onChange={(e) => setForecastParams({ ...forecastParams, include_seasonality: e.target.value === 'true' })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </div>
        </div>

        {/* Key Metrics */}
        {forecastData && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-green-900">Projected Revenue</h4>
              <p className="text-2xl font-bold text-green-600">
                ${(forecastData.metrics.totalRevenue / 1000000).toFixed(2)}M
              </p>
              <p className="text-sm text-green-700 mt-1">
                {forecastData.metrics.growthRate > 0 ? '+' : ''}{forecastData.metrics.growthRate.toFixed(1)}% YoY
              </p>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-blue-900">Avg Monthly Revenue</h4>
              <p className="text-2xl font-bold text-blue-600">
                ${(forecastData.metrics.avgRevenue / 1000).toFixed(0)}K
              </p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-purple-900">Forecast Accuracy</h4>
              <p className="text-2xl font-bold text-purple-600">
                {(100 - forecastData.metrics.forecastAccuracy).toFixed(1)}%
              </p>
            </div>
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h4 className="text-sm font-medium text-yellow-900">Selected Region</h4>
              <p className="text-2xl font-bold text-yellow-600">
                {selectedRegion === 'all' ? 'Global' : regions.find(r => r.id === selectedRegion)?.name || selectedRegion}
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Forecast Results */}
      {forecastData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Revenue Forecast */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Revenue Forecast</h3>
            <Area
              data={{
                labels: forecastData.chartData.labels,
                datasets: [{
                  label: 'Projected Revenue',
                  data: forecastData.chartData.revenue,
                  borderColor: 'rgb(16, 185, 129)',
                  backgroundColor: 'rgba(16, 185, 129, 0.2)',
                }],
              }}
              options={{
                responsive: true,
                plugins: {
                  legend: { display: false },
                },
              }}
            />
          </div>

          {/* Top Products or Insights */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Forecast Insights</h3>
            {forecastData.insights.length > 0 ? (
              <ul className="space-y-3">
                {forecastData.insights.map((insight, index) => (
                  <li key={index} className="flex items-start">
                    <span className="text-blue-600 mr-2">•</span>
                    <span className="text-gray-700">{insight}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-gray-500">No specific insights available for this forecast.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SalesForecast;
