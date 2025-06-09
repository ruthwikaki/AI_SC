import React, { useState, useEffect } from 'react';
import { useForecast } from '../../hooks/forecasting/useForecast';
import ForecastSummaryCard from './widgets/ForecastSummaryCard';
import TimeSeriesChart from './TimeSeriesChart';
import Loading from '../common/Loading';

const ForecastDashboard = () => {
  const { forecasts, loading, error } = useForecast();
  const [selectedForecast, setSelectedForecast] = useState(null);

  if (loading) return <Loading />;
  if (error) return <div className="text-red-500">Error: {error}</div>;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Forecast Dashboard</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Forecast summary cards */}
        <ForecastSummaryCard
          title="Demand Forecast"
          value={forecasts?.demand?.summary}
          trend={forecasts?.demand?.trend}
          onClick={() => setSelectedForecast(forecasts?.demand)}
        />
        <ForecastSummaryCard
          title="Inventory Forecast"
          value={forecasts?.inventory?.summary}
          trend={forecasts?.inventory?.trend}
          onClick={() => setSelectedForecast(forecasts?.inventory)}
        />
        <ForecastSummaryCard
          title="Sales Forecast"
          value={forecasts?.sales?.summary}
          trend={forecasts?.sales?.trend}
          onClick={() => setSelectedForecast(forecasts?.sales)}
        />
      </div>
      
      {/* Time series chart */}
      {selectedForecast && (
        <div className="mt-8">
          <TimeSeriesChart data={selectedForecast} />
        </div>
      )}
    </div>
  );
};

export default ForecastDashboard;
