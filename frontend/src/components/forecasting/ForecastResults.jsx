import React from 'react';
import AccuracyMetrics from './AccuracyMetrics';
import ForecastTable from './widgets/ForecastTable';
import ForecastExport from './ForecastExport';

const ForecastResults = ({ results }) => {
  if (!results) return null;

  // Extract data from your backend response format
  const forecastData = results.results?.forecast || [];
  const history = results.results?.history || [];
  const methodology = results.results?.methodology || '';
  const insights = results.results?.insights || [];
  const anomalies = results.results?.anomalies || [];

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Forecast Results</h3>
        
        {/* Methodology */}
        {methodology && (
          <div className="mb-4 p-3 bg-blue-50 rounded">
            <p className="text-sm text-blue-800">
              <span className="font-medium">Method:</span> {methodology}
            </p>
          </div>
        )}
        
        {/* Insights */}
        {insights.length > 0 && (
          <div className="mb-6">
            <h4 className="text-md font-medium mb-2">Key Insights</h4>
            <ul className="list-disc list-inside space-y-1">
              {insights.map((insight, index) => (
                <li key={index} className="text-sm text-gray-700">{insight}</li>
              ))}
            </ul>
          </div>
        )}
        
        {/* Anomalies */}
        {anomalies.length > 0 && (
          <div className="mb-6 p-3 bg-yellow-50 rounded">
            <h4 className="text-md font-medium mb-2 text-yellow-800">Anomalies Detected</h4>
            <ul className="space-y-1">
              {anomalies.map((anomaly, index) => (
                <li key={index} className="text-sm text-yellow-700">
                  {anomaly.description}
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {/* Forecast data table */}
        <div className="mt-6">
          <ForecastTable data={forecastData} />
        </div>
        
        {/* Export options */}
        <div className="mt-4">
          <ForecastExport data={results} />
        </div>
      </div>
    </div>
  );
};

export default ForecastResults;
