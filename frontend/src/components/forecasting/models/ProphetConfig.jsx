import React, { useState } from 'react';

const ProphetConfig = ({ onConfigChange }) => {
  const [config, setConfig] = useState({
    changepoint_prior_scale: 0.05,
    seasonality_prior_scale: 10,
    holidays_prior_scale: 10,
    seasonality_mode: 'additive',
    yearly_seasonality: 'auto',
    weekly_seasonality: 'auto',
    daily_seasonality: 'auto',
  });

  const handleChange = (field, value) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">Prophet Configuration</h4>
      
      <div>
        <label className="block text-sm text-gray-600">Changepoint Prior Scale</label>
        <input
          type="number"
          value={config.changepoint_prior_scale}
          onChange={(e) => handleChange('changepoint_prior_scale', parseFloat(e.target.value))}
          className="w-full px-2 py-1 border rounded"
          min="0.001"
          max="0.5"
          step="0.001"
        />
      </div>
      
      <div>
        <label className="block text-sm text-gray-600">Seasonality Mode</label>
        <select
          value={config.seasonality_mode}
          onChange={(e) => handleChange('seasonality_mode', e.target.value)}
          className="w-full px-2 py-1 border rounded"
        >
          <option value="additive">Additive</option>
          <option value="multiplicative">Multiplicative</option>
        </select>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm text-gray-600">Yearly</label>
          <select
            value={config.yearly_seasonality}
            onChange={(e) => handleChange('yearly_seasonality', e.target.value)}
            className="w-full px-2 py-1 border rounded"
          >
            <option value="auto">Auto</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Weekly</label>
          <select
            value={config.weekly_seasonality}
            onChange={(e) => handleChange('weekly_seasonality', e.target.value)}
            className="w-full px-2 py-1 border rounded"
          >
            <option value="auto">Auto</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Daily</label>
          <select
            value={config.daily_seasonality}
            onChange={(e) => handleChange('daily_seasonality', e.target.value)}
            className="w-full px-2 py-1 border rounded"
          >
            <option value="auto">Auto</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
      </div>
    </div>
  );
};

export default ProphetConfig;
