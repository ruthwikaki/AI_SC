import React from 'react';

const TrendIndicator = ({ trend }) => {
  const { direction, percentage } = trend || {};
  
  const getColor = () => {
    if (direction === 'up') return 'text-green-600';
    if (direction === 'down') return 'text-red-600';
    return 'text-gray-600';
  };
  
  const getIcon = () => {
    if (direction === 'up') return '↑';
    if (direction === 'down') return '↓';
    return '→';
  };
  
  return (
    <span className={`flex items-center ${getColor()}`}>
      <span className="text-lg">{getIcon()}</span>
      <span className="ml-1 text-sm font-medium">{percentage}%</span>
    </span>
  );
};

export default TrendIndicator;
