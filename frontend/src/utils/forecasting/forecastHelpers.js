// Forecast helper utilities

export const calculateMAPE = (actual, forecast) => {
  if (!actual || !forecast || actual.length !== forecast.length) return null;
  
  let sum = 0;
  let count = 0;
  
  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== 0) {
      sum += Math.abs((actual[i] - forecast[i]) / actual[i]);
      count++;
    }
  }
  
  return count > 0 ? (sum / count) * 100 : null;
};

export const calculateMAE = (actual, forecast) => {
  if (!actual || !forecast || actual.length !== forecast.length) return null;
  
  const sum = actual.reduce((acc, val, i) => {
    return acc + Math.abs(val - forecast[i]);
  }, 0);
  
  return sum / actual.length;
};

export const calculateRMSE = (actual, forecast) => {
  if (!actual || !forecast || actual.length !== forecast.length) return null;
  
  const sum = actual.reduce((acc, val, i) => {
    return acc + Math.pow(val - forecast[i], 2);
  }, 0);
  
  return Math.sqrt(sum / actual.length);
};

export const formatForecastPeriod = (period, unit = 'days') => {
  if (unit === 'days') {
    if (period === 1) return '1 day';
    if (period < 7) return `${period} days`;
    if (period < 30) return `${Math.round(period / 7)} weeks`;
    if (period < 365) return `${Math.round(period / 30)} months`;
    return `${Math.round(period / 365)} years`;
  }
  // Add other unit conversions as needed
  return `${period} ${unit}`;
};
