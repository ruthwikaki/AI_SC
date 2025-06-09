import React from 'react';

const AccuracyMetrics = ({ metrics }) => {
  if (!metrics) return null;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <div className="bg-gray-50 p-4 rounded">
        <h4 className="text-sm font-medium text-gray-600">MAPE</h4>
        <p className="text-2xl font-bold">{metrics.mape?.toFixed(2)}%</p>
      </div>
      <div className="bg-gray-50 p-4 rounded">
        <h4 className="text-sm font-medium text-gray-600">MAE</h4>
        <p className="text-2xl font-bold">{metrics.mae?.toFixed(2)}</p>
      </div>
      <div className="bg-gray-50 p-4 rounded">
        <h4 className="text-sm font-medium text-gray-600">RMSE</h4>
        <p className="text-2xl font-bold">{metrics.rmse?.toFixed(2)}</p>
      </div>
      <div className="bg-gray-50 p-4 rounded">
        <h4 className="text-sm font-medium text-gray-600">R²</h4>
        <p className="text-2xl font-bold">{metrics.r2?.toFixed(3)}</p>
      </div>
    </div>
  );
};

export default AccuracyMetrics;
