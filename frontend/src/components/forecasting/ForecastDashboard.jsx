// frontend/src/components/forecasting/ForecastDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Tab } from '@headlessui/react';
import { useForecast } from '../../hooks/forecasting/useForecast';
import ForecastSummaryCard from './widgets/ForecastSummaryCard';
import TimeSeriesChart from './TimeSeriesChart';
import ForecastDataGrid from './ForecastDataGrid';
import Loading from '../common/Loading';
import { 
  ChartBarIcon, 
  TableCellsIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon 
} from '@heroicons/react/24/outline';

function classNames(...classes) {
  return classes.filter(Boolean).join(' ');
}

const ForecastDashboard = () => {
  const { forecasts, loading, error, refreshForecasts, runForecast } = useForecast();
  const [selectedView, setSelectedView] = useState(0);
  const [isRunningForecast, setIsRunningForecast] = useState(false);
  const [forecastData, setForecastData] = useState(null);

  useEffect(() => {
    // Load initial forecast data
    loadForecastData();
  }, []);

  const loadForecastData = async () => {
    try {
      setIsRunningForecast(true);
      const data = await runForecast({
        forecast_periods: 12,
        period_type: 'month',
        include_products: true, // Request product-level data
      });
      setForecastData(data);
    } catch (err) {
      console.error('Error loading forecast data:', err);
    } finally {
      setIsRunningForecast(false);
    }
  };

  const handleDataChange = (change) => {
    console.log('Data changed:', change);
    // Handle manual forecast adjustments here
  };

  if (loading || isRunningForecast) return <Loading />;
  if (error) return <div className="text-red-500">Error: {error}</div>;

  const tabs = [
    { name: 'Data Grid', icon: TableCellsIcon },
    { name: 'Charts', icon: ChartBarIcon },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Forecast Dashboard</h2>
            <p className="text-sm text-gray-500 mt-1">
              Manage and analyze your inventory forecasts
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={loadForecastData}
              className="px-4 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50 flex items-center"
            >
              <ArrowPathIcon className="w-4 h-4 mr-2" />
              Refresh
            </button>
            <button
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
            >
              <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
              Export All
            </button>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="px-6 py-4 bg-gray-50">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <ForecastSummaryCard
            title="Total Products"
            value={forecastData?.results?.products?.length || 0}
            trend={{ direction: 'up', percentage: 5.2 }}
          />
          <ForecastSummaryCard
            title="Avg. Forecast Accuracy"
            value={`${forecastData?.results?.average_accuracy || 94.5}%`}
            trend={{ direction: 'up', percentage: 2.1 }}
          />
          <ForecastSummaryCard
            title="Stock Value Forecast"
            value={`$${((forecastData?.results?.total_forecast_value || 1250000) / 1000).toFixed(0)}K`}
            trend={{ direction: 'up', percentage: 8.3 }}
          />
          <ForecastSummaryCard
            title="Low Stock Alerts"
            value={forecastData?.results?.low_stock_count || 12}
            trend={{ direction: 'down', percentage: 15.0 }}
          />
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Tab.Group selectedIndex={selectedView} onChange={setSelectedView}>
          <Tab.List className="flex space-x-1 bg-white border-b px-6">
            {tabs.map((tab) => (
              <Tab
                key={tab.name}
                className={({ selected }) =>
                  classNames(
                    'flex items-center px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-all',
                    selected
                      ? 'text-blue-600 border-blue-600'
                      : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                  )
                }
              >
                <tab.icon className="w-4 h-4 mr-2" />
                {tab.name}
              </Tab>
            ))}
          </Tab.List>

          <Tab.Panels className="flex-1 overflow-hidden">
            {/* Data Grid View */}
            <Tab.Panel className="h-full">
              <ForecastDataGrid 
                forecastData={forecastData}
                onDataChange={handleDataChange}
              />
            </Tab.Panel>

            {/* Charts View */}
            <Tab.Panel className="h-full overflow-auto p-6">
              <div className="space-y-6">
                {/* Time Series Chart */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium mb-4">Aggregate Forecast Trend</h3>
                  <TimeSeriesChart data={forecastData} />
                </div>

                {/* Category Breakdown */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-medium mb-4">Forecast by Category</h3>
                    {/* Add category chart here */}
                  </div>
                  
                  <div className="bg-white rounded-lg shadow p-6">
                    <h3 className="text-lg font-medium mb-4">ABC Analysis Distribution</h3>
                    {/* Add ABC chart here */}
                  </div>
                </div>
              </div>
            </Tab.Panel>
          </Tab.Panels>
        </Tab.Group>
      </div>
    </div>
  );
};

export default ForecastDashboard;
