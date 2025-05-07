import React, { useState, useEffect } from 'react';
import { 
  LineChart as RechartsLineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  ReferenceLine,
  Brush
} from 'recharts';
import { FaCog } from 'react-icons/fa';

const LineChart = ({
  data = [],
  title = "Line Chart",
  xAxisKey = "name",
  lineKeys = ["value"],
  lineColors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"],
  height = 400,
  showDots = true,
  showBrush = false,
  smoothCurve = false,
  showReference = false,
  referenceValue = 0,
  referenceLabel = "Target",
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
        <RechartsLineChart
          data={chartData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey={xAxisKey} 
            tick={{ fontSize: 12 }}
            tickFormatter={config.xAxisFormatter || null}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            tickFormatter={config.yAxisFormatter || null}
            domain={config.yAxisDomain || ['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Add reference line if enabled */}
          {showReference && (
            <ReferenceLine 
              y={referenceValue} 
              stroke="#ff7300" 
              strokeDasharray="3 3"
              label={{ 
                value: referenceLabel, 
                position: 'right',
                fill: '#ff7300',
                fontSize: 12
              }} 
            />
          )}
          
          {/* Create lines for each key */}
          {lineKeys.map((key, index) => (
            <Line
              key={key}
              type={smoothCurve ? "monotone" : "linear"}
              dataKey={key}
              name={config.seriesNames?.[key] || key}
              stroke={lineColors[index % lineColors.length]}
              strokeWidth={2}
              dot={showDots}
              activeDot={{ r: 8 }}
            />
          ))}
          
          {/* Add brush for date range selection if enabled */}
          {showBrush && (
            <Brush 
              dataKey={xAxisKey} 
              height={30} 
              stroke="#8884d8"
              startIndex={chartData.length > 30 ? chartData.length - 30 : 0}
            />
          )}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LineChart;