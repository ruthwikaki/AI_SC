import React, { useState } from 'react';

const LSTMConfig = ({ onConfigChange }) => {
  const [config, setConfig] = useState({
    sequence_length: 30,
    lstm_units: 50,
    dense_units: 25,
    dropout_rate: 0.2,
    epochs: 100,
    batch_size: 32,
  });

  const handleChange = (field, value) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange?.(newConfig);
  };

  return (
    <div className="space-y-4">
      <h4 className="font-medium">LSTM Configuration</h4>
      
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm text-gray-600">Sequence Length</label>
          <input
            type="number"
            value={config.sequence_length}
            onChange={(e) => handleChange('sequence_length', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">LSTM Units</label>
          <input
            type="number"
            value={config.lstm_units}
            onChange={(e) => handleChange('lstm_units', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Dense Units</label>
          <input
            type="number"
            value={config.dense_units}
            onChange={(e) => handleChange('dense_units', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Dropout Rate</label>
          <input
            type="number"
            value={config.dropout_rate}
            onChange={(e) => handleChange('dropout_rate', parseFloat(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="0"
            max="0.5"
            step="0.1"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Epochs</label>
          <input
            type="number"
            value={config.epochs}
            onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-600">Batch Size</label>
          <input
            type="number"
            value={config.batch_size}
            onChange={(e) => handleChange('batch_size', parseInt(e.target.value))}
            className="w-full px-2 py-1 border rounded"
            min="1"
          />
        </div>
      </div>
    </div>
  );
};

export default LSTMConfig;
