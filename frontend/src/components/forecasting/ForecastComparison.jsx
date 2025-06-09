import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';

const ForecastComparison = ({ models }) => {
  const [selectedModels, setSelectedModels] = useState([]);

  const toggleModel = (modelId) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  const getChartData = () => {
    const datasets = selectedModels.map((modelId) => {
      const model = models.find(m => m.id === modelId);
      return {
        label: model.name,
        data: model.forecast,
        borderColor: model.color,
        backgroundColor: `${model.color}20`,
      };
    });

    return {
      labels: models[0]?.dates || [],
      datasets,
    };
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Model Comparison</h3>
      
      <div className="mb-4 space-y-2">
        {models.map((model) => (
          <label key={model.id} className="flex items-center">
            <input
              type="checkbox"
              checked={selectedModels.includes(model.id)}
              onChange={() => toggleModel(model.id)}
              className="mr-2"
            />
            <span>{model.name}</span>
          </label>
        ))}
      </div>
      
      {selectedModels.length > 0 && (
        <Line data={getChartData()} />
      )}
    </div>
  );
};

export default ForecastComparison;
