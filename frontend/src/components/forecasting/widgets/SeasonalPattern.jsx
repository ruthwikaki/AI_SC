import React from 'react';

const SeasonalPattern = ({ pattern }) => {
  if (!pattern) return null;
  
  const getPatternDescription = () => {
    switch (pattern.type) {
      case 'weekly':
        return `Weekly pattern detected with peak on ${pattern.peak}`;
      case 'monthly':
        return `Monthly pattern detected with peak around day ${pattern.peak}`;
      case 'yearly':
        return `Yearly pattern detected with peak in ${pattern.peak}`;
      default:
        return 'No clear seasonal pattern detected';
    }
  };
  
  return (
    <div className="p-4 bg-yellow-50 rounded">
      <h4 className="text-sm font-medium text-yellow-900">Seasonal Pattern</h4>
      <p className="mt-1 text-sm text-yellow-700">{getPatternDescription()}</p>
      {pattern.strength && (
        <p className="mt-1 text-sm text-yellow-600">
          Strength: {(pattern.strength * 100).toFixed(1)}%
        </p>
      )}
    </div>
  );
};

export default SeasonalPattern;
