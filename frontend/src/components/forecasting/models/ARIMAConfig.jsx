import React, { useState } from 'react';

const ARIMAConfig = ({ onConfigChange }) => {
  const [config, setConfig] = useState({
    p: 1, // autoregressive order
    d: 1, // degree of differencing
    q: 1, // moving average order
    seasonal: false,
    seasonalPeriod: 12,
  });

  const handleChange = (field, value) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">ARIMA Configuration</h4>
      
      <div className="grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm text-gray-600">p (AR order)</label>
          <input
            type="number"
            value={config.p}
            onChange={(e) => handleChange('p', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="0"
            max="5"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">d (Differencing)</label>
          <input
            type="number"
            value={config.d}
            onChange={(e) => handleChange('d', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="0"
            max="2"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">q (MA order)</label>
          <input
            type="number"
            value={config.q}
            onChange={(e) => handleChange('q', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="0"
            max="5"
          />
        </div>
      </div>
      
      <div>
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={config.seasonal}
            onChange={(e) => handleChange('seasonal', e.target.checked)}
            className="mr-2"
          />
          <span className="text-sm">Enable seasonal ARIMA</span>
        </label>
      </div>
      
      {config.seasonal && (
        <div>
          <label className="block text-sm text-gray-600">Seasonal Period</label>
          <input
            type="number"
            value={config.seasonalPeriod}
            onChange={(e) => handleChange('seasonalPeriod', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
      )}
    </div>
  );
};

export default ARIMAConfig;
