// Metrics calculation utilities

export const calculateR2 = (actual, forecast) => {
  if (!actual || !forecast || actual.length !== forecast.length) return null;
  
  const meanActual = actual.reduce((a, b) => a + b, 0) / actual.length;
  
  const ssTotal = actual.reduce((sum, val) => {
    return sum + Math.pow(val - meanActual, 2);
  }, 0);
  
  const ssResidual = actual.reduce((sum, val, i) => {
    return sum + Math.pow(val - forecast[i], 2);
  }, 0);
  
  return 1 - (ssResidual / ssTotal);
};

export const calculateConfidenceInterval = (forecast, stdDev, confidence = 0.95) => {
  const zScore = getZScore(confidence);
  
  return forecast.map((value, i) => ({
    forecast: value,
    lower: value - zScore * stdDev[i],
    upper: value + zScore * stdDev[i]
  }));
};

const getZScore = (confidence) => {
  const zScores = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576
  };
  return zScores[confidence] || 1.96;
};

export const calculateForecastAccuracy = (actual, forecast) => {
  return {
    mape: calculateMAPE(actual, forecast),
    mae: calculateMAE(actual, forecast),
    rmse: calculateRMSE(actual, forecast),
    r2: calculateR2(actual, forecast)
  };
};

// Import the individual metric functions from forecastHelpers
import { calculateMAPE, calculateMAE, calculateRMSE } from './forecastHelpers';
