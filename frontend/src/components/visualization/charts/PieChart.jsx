import React, { useState, useEffect } from 'react';
import { PieChart as RechartsPieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, Sector } from 'recharts';
import { FaCog } from 'react-icons/fa';

const PieChart = ({
  data = [],
  title = "Pie Chart",
  nameKey = "name",
  valueKey = "value",
  colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#6366f1", "#14b8a6", "#f43f5e", "#64748b"],
  height = 400,
  isDoughnut = false,
  showPercentage = true,
  activeIndex = -1,
  setActiveIndex = null,
  config = {}
}) => {
  const [chartData, setChartData] = useState([]);
  const [activeSegment, setActiveSegment] = useState(-1);
  
  // Process and format the data when it changes
  useEffect(() => {
    setChartData(data);
  }, [data]);
  
  // Manage active segment state
  useEffect(() => {
    if (activeIndex !== undefined && activeIndex !== null) {
      setActiveSegment(activeIndex);
    }
  }, [activeIndex]);
  
  // If no data, show placeholder message
  if (!chartData || chartData.length === 0) {
    return (
      <div className="border border-gray-200 rounded-lg p-6 flex flex-col items-center justify-center bg-gray-50 h-80">
        <p className="text-gray-500 mb-4">No data available for chart</p>
        <FaCog className="animate-spin text-gray-400 h-8 w-8" />
      </div>
    );
  }
  
  // Calculate total for percentage calculations
  const total = chartData.reduce((sum, entry) => sum + entry[valueKey], 0);
  
  // Handle segment click
  const handleSegmentClick = (_, index) => {
    const newIndex = activeSegment === index ? -1 : index;
    setActiveSegment(newIndex);
    if (setActiveIndex) setActiveIndex(newIndex);
  };
  
  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const value = data[valueKey];
      const percentage = (value / total * 100).toFixed(1);
      
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded">
          <p className="font-medium text-gray-900">{data[nameKey]}</p>
          <p className="text-sm text-gray-700">
            {`${config.valueFormatter ? config.valueFormatter(value) : value}`}
            {showPercentage && ` (${percentage}%)`}
          </p>
        </div>
      );
    }
    return null;
  };
  
  // Active shape renderer for hover effect
  const renderActiveShape = (props) => {
    const { cx, cy, innerRadius, outerRadius, startAngle, endAngle, fill, payload } = props;
    const value = payload[valueKey];
    const percentage = (value / total * 100).toFixed(1);
    
    return (
      <g>
        <Sector
          cx={cx}
          cy={cy}
          innerRadius={innerRadius}
          outerRadius={outerRadius + 6}
          startAngle={startAngle}
          endAngle={endAngle}
          fill={fill}
        />
        <Sector
          cx={cx}
          cy={cy}
          startAngle={startAngle}
          endAngle={endAngle}
          innerRadius={outerRadius + 6}
          outerRadius={outerRadius + 10}
          fill={fill}
        />
      </g>
    );
  };
  
  // Custom legend renderer
  const renderCustomizedLegend = (props) => {
    const { payload } = props;
    
    return (
      <ul className="flex flex-wrap justify-center gap-x-6 gap-y-2 mb-4">
        {payload.map((entry, index) => {
          const isActive = index === activeSegment;
          const value = chartData[index][valueKey];
          const percentage = (value / total * 100).toFixed(1);
          
          return (
            <li 
              key={`item-${index}`}
              className={`flex items-center cursor-pointer transition-all ${
                isActive ? 'scale-105 font-medium' : ''
              }`}
              onClick={() => handleSegmentClick(null, index)}
            >
              <div 
                className={`w-3 h-3 mr-2 rounded-sm`} 
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-sm text-gray-700">{entry.value}</span>
              {showPercentage && (
                <span className="text-xs text-gray-500 ml-1">{`(${percentage}%)`}</span>
              )}
            </li>
          );
        })}
      </ul>
    );
  };
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      {title && <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>}
      
      <ResponsiveContainer width="100%" height={height}>
        <RechartsPieChart>
          <Pie
            data={chartData}
            dataKey={valueKey}
            nameKey={nameKey}
            cx="50%"
            cy="50%"
            innerRadius={isDoughnut ? 60 : 0}
            outerRadius={100}
            paddingAngle={2}
            activeIndex={activeSegment}
            activeShape={renderActiveShape}
            onClick={handleSegmentClick}
          >
            {chartData.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={colors[index % colors.length]} 
              />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            content={renderCustomizedLegend}
            verticalAlign="bottom"
            align="center"
          />
        </RechartsPieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PieChart;