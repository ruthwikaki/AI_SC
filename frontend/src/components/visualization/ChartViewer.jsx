import React, { useState, useEffect } from 'react';
import { FaDownload, FaExpandAlt, FaCog, FaChartBar, FaChartLine, FaChartPie, FaTable, FaFilter, FaCalendarAlt } from 'react-icons/fa';

// Import chart components
import BarChart from './charts/BarChart';
import LineChart from './charts/LineChart';
import PieChart from './charts/PieChart';
import HeatMap from './charts/HeatMap';
import SankeyDiagram from './charts/SankeyDiagram';
import NetworkGraph from './charts/NetworkGraph';

const ChartViewer = ({
  chartType = 'bar',
  chartData = null,
  chartConfig = {},
  title = '',
  description = '',
  showControls = true,
  showFilters = false,
  showDateRange = false,
  showExport = true,
  showFullscreen = true,
  height = 400,
  onConfigChange = () => {},
  onExport = () => {},
  className = ''
}) => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [localConfig, setLocalConfig] = useState(chartConfig);
  const [dateRange, setDateRange] = useState({
    start: chartConfig.dateRange?.start || '',
    end: chartConfig.dateRange?.end || ''
  });
  const [filters, setFilters] = useState(chartConfig.filters || {});
  
  // Update local config when props change
  useEffect(() => {
    setLocalConfig(chartConfig);
    setDateRange({
      start: chartConfig.dateRange?.start || '',
      end: chartConfig.dateRange?.end || ''
    });
    setFilters(chartConfig.filters || {});
  }, [chartConfig]);
  
  // Toggle fullscreen mode
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // Apply configuration changes
  const applyConfig = () => {
    const newConfig = {
      ...localConfig,
      dateRange,
      filters
    };
    onConfigChange(newConfig);
    setIsConfigOpen(false);
  };
  
  // Handle date range change
  const handleDateRangeChange = (field, value) => {
    setDateRange(prev => ({
      ...prev,
      [field]: value
    }));
  };
  
  // Handle filter change
  const handleFilterChange = (key, value) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Handle exporting chart
  const handleExport = (format) => {
    setIsLoading(true);
    
    // In a real application, this would trigger a download
    setTimeout(() => {
      onExport(format);
      setIsLoading(false);
    }, 500);
  };
  
  // Render the appropriate chart component based on type
  const renderChart = () => {
    if (!chartData) {
      return (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-md border border-gray-200">
          <div className="text-center">
            <FaChartBar className="text-gray-300 h-12 w-12 mx-auto mb-3" />
            <p className="text-gray-500">No chart data available</p>
          </div>
        </div>
      );
    }
    
    const props = {
      ...localConfig,
      height,
      data: chartData
    };
    
    switch (chartType.toLowerCase()) {
      case 'bar':
        return <BarChart {...props} />;
      case 'line':
        return <LineChart {...props} />;
      case 'pie':
        return <PieChart {...props} />;
      case 'heat':
      case 'heatmap':
        return <HeatMap {...props} />;
      case 'sankey':
        return <SankeyDiagram {...props} />;
      case 'network':
      case 'graph':
        return <NetworkGraph {...props} />;
      default:
        return (
          <div className="flex items-center justify-center h-64 bg-gray-50 rounded-md border border-gray-200">
            <div className="text-center">
              <p className="text-gray-500">Unsupported chart type: {chartType}</p>
            </div>
          </div>
        );
    }
  };
  
  // Get chart type icon
  const getChartIcon = () => {
    switch (chartType.toLowerCase()) {
      case 'bar': return <FaChartBar className="mr-2" />;
      case 'line': return <FaChartLine className="mr-2" />;
      case 'pie': return <FaChartPie className="mr-2" />;
      default: return <FaChartBar className="mr-2" />;
    }
  };
  
  return (
    <div className={`
      bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden
      ${isFullscreen ? 'fixed top-0 left-0 w-full h-full z-50' : ''}
      ${className}
    `}>
      {/* Chart Header */}
      <div className="flex justify-between items-center border-b border-gray-200 px-4 py-3">
        <div className="flex items-center">
          {getChartIcon()}
          <div>
            <h3 className="font-medium text-gray-800">{title}</h3>
            {description && <p className="text-xs text-gray-500 mt-1">{description}</p>}
          </div>
        </div>
        
        {showControls && (
          <div className="flex space-x-2">
            {showFilters && (
              <button 
                onClick={() => setIsConfigOpen(!isConfigOpen)}
                className={`
                  p-1 rounded-md hover:bg-gray-100
                  ${isConfigOpen ? 'text-blue-600 bg-blue-50' : 'text-gray-500'}
                `}
                title="Filters & Settings"
              >
                <FaFilter />
              </button>
            )}
            
            {showExport && (
              <div className="relative group">
                <button 
                  className="p-1 rounded-md hover:bg-gray-100 text-gray-500"
                  title="Export"
                >
                  <FaDownload />
                </button>
                
                <div className="absolute right-0 mt-1 w-36 bg-white shadow-lg rounded-md border border-gray-200 hidden group-hover:block z-10">
                  <div className="py-1">
                    <button 
                      className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 text-gray-700"
                      onClick={() => handleExport('png')}
                      disabled={isLoading}
                    >
                      Export as PNG
                    </button>
                    <button 
                      className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 text-gray-700"
                      onClick={() => handleExport('svg')}
                      disabled={isLoading}
                    >
                      Export as SVG
                    </button>
                    <button 
                      className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 text-gray-700"
                      onClick={() => handleExport('csv')}
                      disabled={isLoading}
                    >
                      Export Data (CSV)
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {showFullscreen && (
              <button 
                onClick={toggleFullscreen}
                className="p-1 rounded-md hover:bg-gray-100 text-gray-500"
                title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
              >
                <FaExpandAlt />
              </button>
            )}
            
            <button 
              onClick={() => setIsConfigOpen(!isConfigOpen)}
              className={`
                p-1 rounded-md hover:bg-gray-100
                ${isConfigOpen ? 'text-blue-600 bg-blue-50' : 'text-gray-500'}
              `}
              title="Chart Settings"
            >
              <FaCog />
            </button>
          </div>
        )}
      </div>
      
      {/* Chart Configuration Panel */}
      {isConfigOpen && (
        <div className="border-b border-gray-200 bg-gray-50 p-4">
          <h4 className="font-medium text-gray-700 mb-3">Chart Configuration</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Basic Chart Settings */}
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Chart Title
                </label>
                <input
                  type="text"
                  value={localConfig.title || ''}
                  onChange={(e) => setLocalConfig(prev => ({ ...prev, title: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  placeholder="Enter chart title"
                />
              </div>
              
              {/* Color Scheme */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Color Scheme
                </label>
                <select
                  value={localConfig.colorScheme || 'default'}
                  onChange={(e) => setLocalConfig(prev => ({ ...prev, colorScheme: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="default">Default</option>
                  <option value="blue">Blues</option>
                  <option value="green">Greens</option>
                  <option value="red">Reds</option>
                  <option value="purple">Purples</option>
                  <option value="categorical">Categorical</option>
                </select>
              </div>
            </div>
            
            {/* Advanced Settings */}
            <div className="space-y-3">
              {/* Date Range */}
              {showDateRange && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    <div className="flex items-center">
                      <FaCalendarAlt className="mr-1 text-gray-500" />
                      Date Range
                    </div>
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="date"
                      value={dateRange.start}
                      onChange={(e) => handleDateRangeChange('start', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    />
                    <input
                      type="date"
                      value={dateRange.end}
                      onChange={(e) => handleDateRangeChange('end', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                    />
                  </div>
                </div>
              )}
              
              {/* Filters */}
              {showFilters && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    <div className="flex items-center">
                      <FaFilter className="mr-1 text-gray-500" />
                      Filters
                    </div>
                  </label>
                  
                  {Object.keys(filters).length === 0 ? (
                    <p className="text-sm text-gray-500">No filters available</p>
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(filters).map(([key, value]) => (
                        <div key={key} className="flex items-center">
                          <span className="text-sm text-gray-700 mr-2 w-1/3">{key}:</span>
                          <input
                            type="text"
                            value={value}
                            onChange={(e) => handleFilterChange(key, e.target.value)}
                            className="flex-1 px-3 py-1 border border-gray-300 rounded-md text-sm"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          
          <div className="mt-4 flex justify-end">
            <button
              onClick={() => setIsConfigOpen(false)}
              className="px-4 py-2 border border-gray-300 rounded-md mr-2 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={applyConfig}
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
            >
              Apply Changes
            </button>
          </div>
        </div>
      )}
      
      {/* Chart Content */}
      <div className="p-4" style={{ height: isFullscreen ? 'calc(100vh - 120px)' : height }}>
        {renderChart()}
      </div>
    </div>
  );
};

export default ChartViewer;