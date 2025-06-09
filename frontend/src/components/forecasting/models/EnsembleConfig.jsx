import React, { useState } from 'react';

const EnsembleConfig = ({ onConfigChange }) => {
  const [config, setConfig] = useState({
    models: [],
    weights: {},
    combination_method: 'weighted_average',
  });

  const availableModels = [
    { id: 'arima', name: 'ARIMA' },
    { id: 'exponential', name: 'Exponential Smoothing' },
    { id: 'prophet', name: 'Prophet' },
    { id: 'lstm', name: 'LSTM' },
  ];

  const toggleModel = (modelId) => {
    const newModels = config.models.includes(modelId)
      ? config.models.filter(id => id !== modelId)
      : [...config.models, modelId];
    
    const newConfig = { ...config, models: newModels };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  const handleWeightChange = (modelId, weight) => {
    const newWeights = { ...config.weights, [modelId]: parseFloat(weight) };
    const newConfig = { ...config, weights: newWeights };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">Ensemble Configuration</h4>
      
      <div>
        <label className="block text-sm text-gray-600 mb-2">Select Models</label>
        {availableModels.map((model) => (
          <label key={model.id} className="flex items-center mb-2">
            <input
              type="checkbox"
              checked={config.models.includes(model.id)}
              onChange={() => toggleModel(model.id)}
              className="mr-2"
            />
            <span>{model.name}</span>
          </label>
        ))}
      </div>
      
      <div>
        <label className="block text-sm text-gray-600">Combination Method</label>
        <select
          value={config.combination_method}
          onChange={(e) => {
            const newConfig = { ...config, combination_method: e.target.value };
            setConfig(newConfig);
            onConfigChange?.(newConfig);
          }}
          className="w-full px-2 py-1 border rounded"
        >
          <option value="simple_average">Simple Average</option>
          <option value="weighted_average">Weighted Average</option>
          <option value="median">Median</option>
        </select>
      </div>
      
      {config.combination_method === 'weighted_average' && config.models.length > 0 && (
        <div>
          <label className="block text-sm text-gray-600 mb-2">Model Weights</label>
          {config.models.map((modelId) => {
            const model = availableModels.find(m => m.id === modelId);
            return (
              <div key={modelId} className="flex items-center mb-2">
                <span className="w-32">{model.name}:</span>
                <input
                  type="number"
                  value={config.weights[modelId] || 1}
                  onChange={(e) => handleWeightChange(modelId, e.target.value)}
                  className="px-2 py-1 border rounded"
                  min="0"
                  step="0.1"
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default EnsembleConfig;
