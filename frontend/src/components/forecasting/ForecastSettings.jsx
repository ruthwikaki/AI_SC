import React, { useState } from 'react';
import ModelSelector from './ModelSelector';
import { useForecastConfig } from '../../hooks/forecasting/useForecastConfig';
import { TIME_FRAMES } from '../../services/forecasting';

const ForecastSettings = () => {
  const { config, updateConfig } = useForecastConfig();
  const [forecastPeriod, setForecastPeriod] = useState(config.forecast_periods || 12);
  const [periodType, setPeriodType] = useState(config.period_type || 'month');
  const [confidence, setConfidence] = useState(config.confidence_level || 0.95);
  const [timeFrame, setTimeFrame] = useState(config.time_frame || TIME_FRAMES.LAST_MONTH);
  const [selectedModel, setSelectedModel] = useState(config.method || 'exponential_smoothing');

  const handleSave = () => {
    updateConfig({
      forecast_periods: forecastPeriod,
      period_type: periodType,
      confidence_level: confidence,
      time_frame: timeFrame,
      method: selectedModel,
    });
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Forecast Settings</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Historical Data Period
          </label>
          <select
            value={timeFrame}
            onChange={(e) => setTimeFrame(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value={TIME_FRAMES.LAST_WEEK}>Last Week</option>
            <option value={TIME_FRAMES.LAST_MONTH}>Last Month</option>
            <option value={TIME_FRAMES.LAST_QUARTER}>Last Quarter</option>
            <option value={TIME_FRAMES.LAST_YEAR}>Last Year</option>
            <option value={TIME_FRAMES.YEAR_TO_DATE}>Year to Date</option>
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Forecast Periods
            </label>
            <input
              type="number"
              value={forecastPeriod}
              onChange={(e) => setForecastPeriod(parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
              min="1"
              max="52"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Period Type
            </label>
            <select
              value={periodType}
              onChange={(e) => setPeriodType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="day">Daily</option>
              <option value="week">Weekly</option>
              <option value="month">Monthly</option>
            </select>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Confidence Interval
          </label>
          <select
            value={confidence}
            onChange={(e) => setConfidence(parseFloat(e.target.value))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="0.90">90%</option>
            <option value="0.95">95%</option>
            <option value="0.99">99%</option>
          </select>
        </div>
        
        <ModelSelector onModelSelect={setSelectedModel} selectedModel={selectedModel} />
        
        <button
          onClick={handleSave}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
};

export default ForecastSettings;
