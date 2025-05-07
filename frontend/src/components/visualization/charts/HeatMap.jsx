import React, { useState, useEffect } from 'react';
import { FaCog } from 'react-icons/fa';

const HeatMap = ({
  data = [],
  title = "Heat Map",
  xLabels = [],
  yLabels = [],
  colorRange = ['#e6f7ff', '#1890ff'],
  height = 400,
  cellSize = 30,
  showValues = true,
  config = {}
}) => {
  const [chartData, setChartData] = useState([]);
  const [minValue, setMinValue] = useState(0);
  const [maxValue, setMaxValue] = useState(100);
  
  // Process and format the data when it changes
  useEffect(() => {
    if (data && data.length > 0) {
      // Find min and max values for color scaling
      let min = Infinity;
      let max = -Infinity;
      
      data.forEach(row => {
        row.forEach(value => {
          if (value < min) min = value;
          if (value > max) max = value;
        });
      });
      
      setMinValue(min);
      setMaxValue(max);
      setChartData(data);
    }
  }, [data]);
  
  // If no data, show placeholder message
  if (!chartData || chartData.length === 0) {
    return (
      <div className="border border-gray-200 rounded-lg p-6 flex flex-col items-center justify-center bg-gray-50 h-80">
        <p className="text-gray-500 mb-4">No data available for heat map</p>
        <FaCog className="animate-spin text-gray-400 h-8 w-8" />
      </div>
    );
  }
  
  // Function to interpolate colors
  const getColorForValue = (value) => {
    const ratio = (value - minValue) / (maxValue - minValue) || 0;
    
    // RGB values for start and end colors
    const startColor = hexToRgb(colorRange[0]);
    const endColor = hexToRgb(colorRange[1]);
    
    // Interpolate between the colors
    const r = Math.round(startColor.r + (endColor.r - startColor.r) * ratio);
    const g = Math.round(startColor.g + (endColor.g - startColor.g) * ratio);
    const b = Math.round(startColor.b + (endColor.b - startColor.b) * ratio);
    
    return `rgb(${r}, ${g}, ${b})`;
  };
  
  // Helper function to convert hex to RGB
  const hexToRgb = (hex) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  };
  
  // Format values if formatter is provided
  const formatValue = (value) => {
    return config.valueFormatter ? config.valueFormatter(value) : value;
  };
  
  // Calculate cell size based on container
  const calculateCellSize = () => {
    if (cellSize) return cellSize;
    
    // Default cell size calculation could be more sophisticated
    return 30;
  };
  
  const cellSizeValue = calculateCellSize();
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      {title && <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>}
      
      <div className="overflow-x-auto">
        <div 
          className="relative" 
          style={{ 
            height: `${height}px`,
            minWidth: `${(xLabels.length + 1) * cellSizeValue}px`
          }}
        >
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 flex flex-col">
            <div style={{ height: `${cellSizeValue}px` }}></div> {/* Empty space for corner */}
            {yLabels.map((label, index) => (
              <div 
                key={`y-label-${index}`}
                className="flex items-center justify-end text-xs text-gray-600 font-medium pr-2"
                style={{ height: `${cellSizeValue}px` }}
              >
                {label}
              </div>
            ))}
          </div>
          
          {/* X-axis labels */}
          <div className="absolute top-0 left-0 flex">
            <div style={{ width: `${cellSizeValue}px` }}></div> {/* Empty space for corner */}
            {xLabels.map((label, index) => (
              <div 
                key={`x-label-${index}`}
                className="flex items-center justify-center text-xs text-gray-600 font-medium"
                style={{ 
                  width: `${cellSizeValue}px`,
                  height: `${cellSizeValue}px`
                }}
              >
                {label}
              </div>
            ))}
          </div>
          
          {/* Heat map cells */}
          <div 
            className="absolute"
            style={{ 
              top: `${cellSizeValue}px`, 
              left: `${cellSizeValue}px` 
            }}
          >
            {chartData.map((row, rowIndex) => (
              <div 
                key={`row-${rowIndex}`}
                className="flex"
              >
                {row.map((value, colIndex) => {
                  const backgroundColor = getColorForValue(value);
                  const textColor = getBrightness(hexToRgb(backgroundColor)) > 128 ? '#000' : '#fff';
                  
                  return (
                    <div 
                      key={`cell-${rowIndex}-${colIndex}`}
                      className="flex items-center justify-center border border-white transition-all hover:z-10 hover:shadow-md group"
                      style={{ 
                        backgroundColor,
                        width: `${cellSizeValue}px`,
                        height: `${cellSizeValue}px`
                      }}
                      title={`${yLabels[rowIndex]}, ${xLabels[colIndex]}: ${formatValue(value)}`}
                    >
                      {showValues && (
                        <span 
                          className="text-xs font-medium group-hover:font-bold"
                          style={{ color: textColor }}
                        >
                          {formatValue(value)}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          
          {/* Legend */}
          <div className="absolute bottom-2 right-2 flex items-center">
            <div className="text-xs text-gray-600 mr-2">
              {formatValue(minValue)}
            </div>
            <div 
              className="h-4 w-24 rounded-sm"
              style={{
                background: `linear-gradient(to right, ${colorRange[0]}, ${colorRange[1]})`
              }}
            ></div>
            <div className="text-xs text-gray-600 ml-2">
              {formatValue(maxValue)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Helper function to determine text color based on background brightness
const getBrightness = ({ r, g, b }) => {
  return (r * 299 + g * 587 + b * 114) / 1000;
};

export default HeatMap;