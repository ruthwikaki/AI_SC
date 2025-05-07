import React, { useState, useEffect } from 'react';
import { BarChart as RechartsBarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList } from 'recharts';
import { FaCog } from 'react-icons/fa';

const BarChart = ({ 
  data = [], 
  title = "Bar Chart", 
  xAxisKey = "name", 
  barKeys = ["value"], 
  barColors = ["#3b82f6"], 
  showValues = false,
  stacked = false,
  horizontal = false,
  height = 400,
  config = {}
}) => {
  const [chartData, setChartData] = useState([]);
  
  // Process and format the data when it changes
  useEffect(() => {
    setChartData(data);
  }, [data]);
  
  // If no data, show placeholder message
  if (!chartData || chartData.length === 0) {
    return (
      <div className="border border-gray-200 rounded-lg p-6 flex flex-col items-center justify-center bg-gray-50 h-80">
        <p className="text-gray-500 mb-4">No data available for chart</p>
        <FaCog className="animate-spin text-gray-400 h-8 w-8" />
      </div>
    );
  }
  
  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded">
          <p className="font-medium text-gray-900">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {`${entry.name}: ${config.valueFormatter ? config.valueFormatter(entry.value) : entry.value}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      {title && <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>}
      
      <ResponsiveContainer width="100%" height={height}>
        <RechartsBarChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          layout={horizontal ? "vertical" : "horizontal"}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          
          {horizontal ? (
            <>
              <XAxis type="number" />
              <YAxis dataKey={xAxisKey} type="category" width={150} />
            </>
          ) : (
            <>
              <XAxis dataKey={xAxisKey} />
              <YAxis />
            </>
          )}
          
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {barKeys.map((key, index) => (
            <Bar 
              key={key}
              dataKey={key} 
              fill={barColors[index % barColors.length]}
              stackId={stacked ? "stack" : null}
            >
              {showValues && (
                <LabelList 
                  dataKey={key} 
                  position={horizontal ? "right" : "top"} 
                  formatter={config.valueFormatter || null}
                  style={{ fill: '#333', fontSize: 12 }}
                />
              )}
            </Bar>
          ))}
        </RechartsBarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BarChart;