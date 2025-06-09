import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import ConfidenceInterval from './widgets/ConfidenceInterval';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const TimeSeriesChart = ({ data }) => {
  if (!data || (!data.history && !data.results)) return null;

  // Extract data from the backend response format
  const history = data.results?.history || data.history || [];
  const forecast = data.results?.forecast || data.forecast || [];
  
  // Combine dates and values
  const historicalDates = history.map(h => h.period);
  const historicalValues = history.map(h => h.value);
  const forecastDates = forecast.map(f => f.period);
  const forecastValues = forecast.map(f => f.value);
  const lowerBounds = forecast.map(f => f.lower_bound);
  const upperBounds = forecast.map(f => f.upper_bound);

  const chartData = {
    labels: [...historicalDates, ...forecastDates],
    datasets: [
      {
        label: 'Historical',
        data: [...historicalValues, ...Array(forecastValues.length).fill(null)],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
      },
      {
        label: 'Forecast',
        data: [...Array(historicalValues.length - 1).fill(null), historicalValues[historicalValues.length - 1], ...forecastValues],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderDash: [5, 5],
        tension: 0.1,
      },
      {
        label: 'Upper Bound',
        data: [...Array(historicalValues.length).fill(null), ...upperBounds],
        borderColor: 'rgba(255, 159, 64, 0.5)',
        backgroundColor: 'transparent',
        borderDash: [2, 2],
        pointRadius: 0,
        fill: false,
      },
      {
        label: 'Lower Bound',
        data: [...Array(historicalValues.length).fill(null), ...lowerBounds],
        borderColor: 'rgba(255, 159, 64, 0.5)',
        backgroundColor: 'rgba(255, 159, 64, 0.1)',
        borderDash: [2, 2],
        pointRadius: 0,
        fill: '-1', // Fill between this line and the previous one
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Time Series Forecast',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(2);
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: 'Value',
        },
      },
    },
  };

  // Calculate average confidence interval
  const avgConfidenceInterval = forecast.length > 0
    ? forecast.reduce((acc, f) => acc + (f.upper_bound - f.lower_bound), 0) / forecast.length
    : 0;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <Line data={chartData} options={options} />
      {forecast.length > 0 && (
        <ConfidenceInterval 
          data={{
            level: (data.request_parameters?.confidence_level || 0.95) * 100,
            upper: avgConfidenceInterval / 2,
            lower: -avgConfidenceInterval / 2,
          }} 
        />
      )}
    </div>
  );
};

export default TimeSeriesChart;
