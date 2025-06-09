// Chart configuration for forecasting visualizations

export const getTimeSeriesChartConfig = (theme = 'light') => {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: (context) => {
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
        grid: {
          display: false
        },
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      },
      y: {
        grid: {
          color: theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };
};

export const getChartColors = () => {
  return {
    primary: 'rgb(59, 130, 246)', // blue-500
    secondary: 'rgb(16, 185, 129)', // green-500
    danger: 'rgb(239, 68, 68)', // red-500
    warning: 'rgb(245, 158, 11)', // amber-500
    info: 'rgb(14, 165, 233)', // sky-500
    purple: 'rgb(139, 92, 246)', // violet-500
  };
};

export const getForecastChartData = (historical, forecast, confidence) => {
  const colors = getChartColors();
  
  const datasets = [
    {
      label: 'Historical',
      data: historical.values,
      borderColor: colors.primary,
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 2,
      pointHoverRadius: 4
    },
    {
      label: 'Forecast',
      data: [...Array(historical.values.length - 1).fill(null), 
             historical.values[historical.values.length - 1],
             ...forecast.values],
      borderColor: colors.secondary,
      backgroundColor: 'transparent',
      borderWidth: 2,
      borderDash: [5, 5],
      pointRadius: 2,
      pointHoverRadius: 4
    }
  ];
  
  if (confidence) {
    datasets.push(
      {
        label: 'Upper Bound',
        data: [...Array(historical.values.length).fill(null), ...confidence.upper],
        borderColor: colors.warning,
        backgroundColor: 'transparent',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0
      },
      {
        label: 'Lower Bound',
        data: [...Array(historical.values.length).fill(null), ...confidence.lower],
        borderColor: colors.warning,
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 0,
        fill: '-1'
      }
    );
  }
  
  return {
    labels: [...historical.dates, ...forecast.dates],
    datasets
  };
};
