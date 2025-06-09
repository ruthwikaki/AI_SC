import React from 'react';
import TrendIndicator from './TrendIndicator';

const ForecastSummaryCard = ({ title, value, trend, onClick }) => {
  return (
    <div
      className="bg-white rounded-lg shadow p-6 cursor-pointer hover:shadow-lg transition-shadow"
      onClick={onClick}
    >
      <h3 className="text-sm font-medium text-gray-600">{title}</h3>
      <div className="mt-2 flex items-baseline">
        <p className="text-2xl font-semibold text-gray-900">{value || '-'}</p>
        {trend && (
          <div className="ml-2">
            <TrendIndicator trend={trend} />
          </div>
        )}
      </div>
    </div>
  );
};

export default ForecastSummaryCard;
