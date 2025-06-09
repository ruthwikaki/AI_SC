import React from 'react';
import { Bar } from 'react-chartjs-2';

const SeasonalityAnalysis = ({ seasonalData }) => {
  if (!seasonalData) return null;

  const chartData = {
    labels: seasonalData.periods,
    datasets: [
      {
        label: 'Seasonal Factor',
        data: seasonalData.factors,
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
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
        text: 'Seasonality Analysis',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <Bar data={chartData} options={options} />
      <div className="mt-4">
        <p className="text-sm text-gray-600">
          Seasonal patterns detected: {seasonalData.pattern}
        </p>
      </div>
    </div>
  );
};

export default SeasonalityAnalysis;
