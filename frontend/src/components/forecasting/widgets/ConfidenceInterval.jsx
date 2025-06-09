import React from 'react';

const ConfidenceInterval = ({ data }) => {
  if (!data) return null;
  
  return (
    <div className="mt-4 p-4 bg-blue-50 rounded">
      <h4 className="text-sm font-medium text-blue-900">
        Confidence Interval ({data.level}%)
      </h4>
      <div className="mt-2 text-sm text-blue-700">
        <p>Upper Bound: {data.upper}</p>
        <p>Lower Bound: {data.lower}</p>
      </div>
    </div>
  );
};

export default ConfidenceInterval;
