import React, { useState, useEffect } from 'react';
import { useForecastModels } from '../../hooks/forecasting/useForecastModels';

const ModelSelector = ({ onModelSelect, selectedModel }) => {
  const { models, loading } = useForecastModels();
  const [currentModel, setCurrentModel] = useState(selectedModel || '');

  useEffect(() => {
    if (selectedModel) {
      setCurrentModel(selectedModel);
    }
  }, [selectedModel]);

  const handleModelChange = (modelId) => {
    setCurrentModel(modelId);
    onModelSelect?.(modelId);
  };

  if (loading) return <div>Loading models...</div>;

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Forecast Model
      </label>
      <select
        value={currentModel}
        onChange={(e) => handleModelChange(e.target.value)}
        className="w-full px-3 py-2 border border-gray-300 rounded-md"
      >
        <option value="">Select a model</option>
        {models.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name} - {model.description}
          </option>
        ))}
      </select>
    </div>
  );
};

export default ModelSelector;
