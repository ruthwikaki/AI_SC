// frontend/src/components/forecasting/TimeSeriesChart.jsx
import React from 'react';
import { Line } from 'react-chartjs-2';

const TimeSeriesChart = ({ data, title = 'Time Series Forecast' }) => {
  // Check if we have data
  if (!data || (!data.history && !data.results)) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <p className="text-gray-500 text-center">No forecast data available</p>
      </div>
    );
  }

  // Extract data from the backend response format
  const history = data.results?.history || data.history || [];
  const forecast = data.results?.forecast || data.forecast || [];
  
  // Create chart data
  const chartData = {
    labels: [
      ...history.map(h => h.period || `H-${history.length - history.indexOf(h)}`),
      ...forecast.map(f => f.period || `F+${forecast.indexOf(f) + 1}`)
    ],
    datasets: [
      {
        label: 'Historical Data',
        data: history.map(h => h.value || 0),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
        fill: true,
      },
      {
        label: 'Forecast',
        data: [
          ...Array(history.length - 1).fill(null),
          history[history.length - 1]?.value || null,
          ...forecast.map(f => f.value || 0)
        ],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],
        tension: 0.1,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: title,
      },
      tooltip: {
        mode: 'index',
        intersect: false,
      },
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time Period',
        },
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Value',
        },
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div style={{ height: '400px' }}>
        <Line data={chartData} options={options} />
      </div>
      
      {/* Display insights if available */}
      {data.results?.insights && data.results.insights.length > 0 && (
        <div className="mt-4 p-4 bg-blue-50 rounded">
          <h4 className="font-semibold text-blue-900 mb-2">Insights</h4>
          <ul className="space-y-1">
            {data.results.insights.map((insight, idx) => (
              <li key={idx} className="text-sm text-blue-800">• {insight}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesChart;
