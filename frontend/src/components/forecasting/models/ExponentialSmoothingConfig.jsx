import React, { useState } from 'react';

const ExponentialSmoothingConfig = ({ onConfigChange }) => {
  const [config, setConfig] = useState({
    method: 'simple',
    alpha: 0.3,
    beta: 0.1,
    gamma: 0.1,
    seasonalPeriods: 12,
  });

  const handleChange = (field, value) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">Exponential Smoothing Configuration</h4>
      
      <div>
        <label className="block text-sm text-gray-600">Method</label>
        <select
          value={config.method}
          onChange={(e) => handleChange('method', e.target.value)}
          className="w-full px-2 py-1 border rounded"
        >
          <option value="simple">Simple</option>
          <option value="double">Double (Holt)</option>
          <option value="triple">Triple (Holt-Winters)</option>
        </select>
      </div>
      
      <div>
        <label className="block text-sm text-gray-600">Alpha (Level)</label>
        <input
          type="number"
          value={config.alpha}
          onChange={(e) => handleChange('alpha', parseFloat(e.target.value))}
          className="w-full px-2 py-1 border rounded"
          min="0"
          max="1"
          step="0.1"
        />
      </div>
      
      {['double', 'triple'].includes(config.method) && (
        <div>
          <label className="block text-sm text-gray-600">Beta (Trend)</label>
          <input
            type="number"
            value={config.beta}
            onChange={(e) => handleChange('beta', parseFloat(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="0"
            max="1"
            step="0.1"
          />
        </div>
      )}
      
      {config.method === 'triple' && (
        <>
          <div>
            <label className="block text-sm text-gray-600">Gamma (Seasonal)</label>
            <input
              type="number"
              value={config.gamma}
              onChange={(e) => handleChange('gamma', parseFloat(e.target.value))}
              className="w-full px-2 py-1 border rounded"
              min="0"
              max="1"
              step="0.1"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-600">Seasonal Periods</label>
            <input
              type="number"
              value={config.seasonalPeriods}
              onChange={(e) => handleChange('seasonalPeriods', parseInt(e.target.value))}
              className="w-full px-2 py-1 border rounded"
              min="1"
            />
          </div>
        </>
      )}
    </div>
  );
};

export default ExponentialSmoothingConfig;
