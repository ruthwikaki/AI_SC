import React, { useState, useEffect, useCallback } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { FaPlus, FaTrash, FaEdit, FaCog, FaChartBar, FaChartLine, FaChartPie, 
         FaThermometerHalf, FaProjectDiagram, FaTable, FaSave, FaClone } from 'react-icons/fa';
import ChartViewer from './ChartViewer';

// Mock data for available visualizations
const AVAILABLE_CHARTS = [
  { id: 'bar-chart', name: 'Bar Chart', type: 'bar', icon: <FaChartBar /> },
  { id: 'line-chart', name: 'Line Chart', type: 'line', icon: <FaChartLine /> },
  { id: 'pie-chart', name: 'Pie Chart', type: 'pie', icon: <FaChartPie /> },
  { id: 'heat-map', name: 'Heat Map', type: 'heat', icon: <FaThermometerHalf /> },
  { id: 'sankey-diagram', name: 'Sankey Diagram', type: 'sankey', icon: <FaProjectDiagram /> },
  { id: 'network-graph', name: 'Network Graph', type: 'network', icon: <FaProjectDiagram /> },
  { id: 'data-table', name: 'Data Table', type: 'table', icon: <FaTable /> }
];

// Default layout configurations
const DEFAULT_LAYOUTS = {
  '1x1': [{ i: 'chart-1', x: 0, y: 0, w: 12, h: 6 }],
  '2x2': [
    { i: 'chart-1', x: 0, y: 0, w: 6, h: 6 },
    { i: 'chart-2', x: 6, y: 0, w: 6, h: 6 },
    { i: 'chart-3', x: 0, y: 6, w: 6, h: 6 },
    { i: 'chart-4', x: 6, y: 6, w: 6, h: 6 }
  ],
  '2x1': [
    { i: 'chart-1', x: 0, y: 0, w: 6, h: 12 },
    { i: 'chart-2', x: 6, y: 0, w: 6, h: 12 }
  ],
  '3x1': [
    { i: 'chart-1', x: 0, y: 0, w: 4, h: 12 },
    { i: 'chart-2', x: 4, y: 0, w: 4, h: 12 },
    { i: 'chart-3', x: 8, y: 0, w: 4, h: 12 }
  ]
};

const DashboardBuilder = ({
  initialCharts = [],
  onSave = () => {},
  availableDataSets = [],
  className = ''
}) => {
  const [charts, setCharts] = useState(initialCharts);
  const [layout, setLayout] = useState('custom');
  const [editingChart, setEditingChart] = useState(null);
  const [isChartSettingsOpen, setIsChartSettingsOpen] = useState(false);
  const [dashboardTitle, setDashboardTitle] = useState('New Dashboard');
  const [dashboardDescription, setDashboardDescription] = useState('');
  const [nextChartId, setNextChartId] = useState(
    initialCharts.length > 0 
      ? Math.max(...initialCharts.map(c => parseInt(c.id.replace('chart-', '')))) + 1 
      : 1
  );

  // Update charts when initialCharts changes
  useEffect(() => {
    if (initialCharts && initialCharts.length > 0) {
      setCharts(initialCharts);
      
      // Update nextChartId
      const maxId = Math.max(...initialCharts.map(c => {
        const idMatch = c.id.match(/chart-(\d+)/);
        return idMatch ? parseInt(idMatch[1]) : 0;
      }));
      setNextChartId(maxId + 1);
    }
  }, [initialCharts]);

  // Apply a preset layout
  const applyLayout = (layoutName) => {
    if (layoutName === 'custom') return;
    
    const layoutConfig = DEFAULT_LAYOUTS[layoutName];
    if (!layoutConfig) return;
    
    // Create new charts based on the layout
    const newCharts = layoutConfig.map((item, index) => {
      // Reuse existing chart if available
      if (charts[index]) {
        return {
          ...charts[index],
          id: item.i,
          gridPos: {
            x: item.x,
            y: item.y,
            w: item.w,
            h: item.h
          }
        };
      }
      
      // Create a new chart
      return {
        id: item.i,
        title: `Chart ${index + 1}`,
        type: 'bar',
        dataSet: availableDataSets.length > 0 ? availableDataSets[0].id : null,
        config: {},
        gridPos: {
          x: item.x,
          y: item.y,
          w: item.w,
          h: item.h
        }
      };
    });
    
    setCharts(newCharts);
    setLayout(layoutName);
  };

  // Handle drag and drop reordering
  const onDragEnd = (result) => {
    if (!result.destination) return;
    
    const items = Array.from(charts);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    
    setCharts(items);
  };

  // Add a new chart
  const addChart = (chartType) => {
    const newChart = {
      id: `chart-${nextChartId}`,
      title: `Chart ${nextChartId}`,
      type: chartType,
      dataSet: availableDataSets.length > 0 ? availableDataSets[0].id : null,
      config: {},
      gridPos: {
        x: 0,
        y: charts.length > 0 ? Math.max(...charts.map(c => c.gridPos.y + c.gridPos.h)) : 0,
        w: 12,
        h: 6
      }
    };
    
    setCharts([...charts, newChart]);
    setNextChartId(nextChartId + 1);
    setLayout('custom'); // Switch to custom layout when adding charts manually
  };

  // Remove a chart
  const removeChart = (chartId) => {
    setCharts(charts.filter(chart => chart.id !== chartId));
  };

  // Open chart settings
  const openChartSettings = (chart) => {
    setEditingChart(chart);
    setIsChartSettingsOpen(true);
  };

  // Save chart settings
  const saveChartSettings = (updatedChart) => {
    setCharts(charts.map(chart => 
      chart.id === updatedChart.id ? updatedChart : chart
    ));
    setIsChartSettingsOpen(false);
    setEditingChart(null);
  };

  // Save dashboard
  const saveDashboard = () => {
    const dashboard = {
      title: dashboardTitle,
      description: dashboardDescription,
      charts: charts,
      layout: layout
    };
    
    onSave(dashboard);
  };

  // Mock function to get chart data based on dataset
  const getChartData = useCallback((dataSetId, chartType) => {
    // In a real app, this would fetch data from an API based on the dataSetId
    // For this example, return some mock data based on the chart type
    
    switch (chartType) {
      case 'bar':
        return {
          labels: ['Category A', 'Category B', 'Category C', 'Category D'],
          datasets: [{
            data: [65, 59, 80, 81],
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
          }]
        };
      case 'line':
        return {
          labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
          datasets: [{
            label: 'Data Series 1',
            data: [12, 19, 3, 5, 2, 3],
            borderColor: '#FF6384',
            fill: false
          }]
        };
      case 'pie':
        return {
          labels: ['Red', 'Blue', 'Yellow'],
          datasets: [{
            data: [300, 50, 100],
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56']
          }]
        };
      // Add cases for other chart types
      default:
        return {
          labels: ['Sample A', 'Sample B'],
          datasets: [{
            data: [50, 50],
            backgroundColor: ['#FF6384', '#36A2EB']
          }]
        };
    }
  }, []);

  // Duplicate a chart
  const duplicateChart = (chart) => {
    const newChart = {
      ...chart,
      id: `chart-${nextChartId}`,
      title: `${chart.title} (Copy)`,
      gridPos: {
        ...chart.gridPos,
        y: chart.gridPos.y + chart.gridPos.h
      }
    };
    
    setCharts([...charts, newChart]);
    setNextChartId(nextChartId + 1);
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Dashboard Header */}
      <div className="border-b border-gray-200 px-6 py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="mb-4 md:mb-0">
            <input
              type="text"
              value={dashboardTitle}
              onChange={(e) => setDashboardTitle(e.target.value)}
              className="text-2xl font-semibold text-gray-800 border-none focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-md px-2 py-1"
              placeholder="Dashboard Title"
            />
            <input
              type="text"
              value={dashboardDescription}
              onChange={(e) => setDashboardDescription(e.target.value)}
              className="text-sm text-gray-500 mt-1 border-none focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-md px-2 py-1 w-full md:w-96"
              placeholder="Add a description..."
            />
          </div>
          
          <div className="flex items-center space-x-3">
            <div>
              <label className="text-sm text-gray-600 mr-2">Layout:</label>
              <select
                value={layout}
                onChange={(e) => applyLayout(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-md text-sm"
              >
                <option value="custom">Custom</option>
                <option value="1x1">1x1 (Single)</option>
                <option value="2x2">2x2 (Four Panel)</option>
                <option value="2x1">2x1 (Two Column)</option>
                <option value="3x1">3x1 (Three Column)</option>
              </select>
            </div>
            
            <button
              onClick={saveDashboard}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md"
            >
              <FaSave className="mr-2" /> Save Dashboard
            </button>
          </div>
        </div>
      </div>
      
      {/* Chart Toolbar */}
      <div className="border-b border-gray-200 px-6 py-2 bg-gray-50">
        <div className="flex items-center space-x-1 overflow-x-auto pb-2">
          <span className="text-sm text-gray-600 mr-2">Add Chart:</span>
          
          {AVAILABLE_CHARTS.map(chart => (
            <button
              key={chart.id}
              onClick={() => addChart(chart.type)}
              className="flex items-center px-3 py-1 bg-white border border-gray-300 rounded-md text-sm hover:bg-gray-100"
              title={`Add ${chart.name}`}
            >
              <span className="mr-1">{chart.icon}</span>
              <span className="hidden sm:inline">{chart.name}</span>
            </button>
          ))}
        </div>
      </div>
      
      {/* Dashboard Content */}
      <div className="p-6">
        {charts.length === 0 ? (
          <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed border-gray-300">
            <FaChartBar className="mx-auto h-12 w-12 text-gray-300 mb-4" />
            <h3 className="text-lg font-medium text-gray-900">No charts yet</h3>
            <p className="mt-1 text-sm text-gray-500">
              Get started by adding a chart or selecting a layout template.
            </p>
            <div className="mt-6">
              <button
                onClick={() => addChart('bar')}
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                <FaPlus className="mr-2" /> Add Bar Chart
              </button>
            </div>
          </div>
        ) : (
          <DragDropContext onDragEnd={onDragEnd}>
            <Droppable droppableId="charts">
              {(provided) => (
                <div
                  {...provided.droppableProps}
                  ref={provided.innerRef}
                  className="space-y-6"
                >
                  {charts.map((chart, index) => (
                    <Draggable key={chart.id} draggableId={chart.id} index={index}>
                      {(provided) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          className="relative"
                        >
                          {/* Chart Controls */}
                          <div
                            {...provided.dragHandleProps}
                            className="absolute top-0 right-0 bg-white border border-gray-200 rounded-bl-md rounded-tr-md z-10 flex shadow-sm"
                          >
                            <button 
                              onClick={() => openChartSettings(chart)}
                              className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50"
                              title="Edit Chart"
                            >
                              <FaEdit size={14} />
                            </button>
                            <button 
                              onClick={() => duplicateChart(chart)}
                              className="p-2 text-gray-500 hover:text-green-600 hover:bg-green-50"
                              title="Duplicate Chart"
                            >
                              <FaClone size={14} />
                            </button>
                            <button 
                              onClick={() => removeChart(chart.id)}
                              className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50"
                              title="Remove Chart"
                            >
                              <FaTrash size={14} />
                            </button>
                          </div>
                          
                          {/* Chart Component */}
                          <ChartViewer
                            chartType={chart.type}
                            chartData={getChartData(chart.dataSet, chart.type)}
                            chartConfig={chart.config}
                            title={chart.title}
                            description={chart.description}
                            height={chart.gridPos.h * 70} // Approximate height based on grid
                            showControls={false}
                          />
                        </div>
                      )}
                    </Draggable>
                  ))}
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        )}
      </div>
      
      {/* Chart Settings Modal */}
      {isChartSettingsOpen && editingChart && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] overflow-y-auto">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">Edit Chart: {editingChart.title}</h3>
            </div>
            
            <div className="px-6 py-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Basic Settings */}
                <div>
                  <h4 className="font-medium text-gray-800 mb-3">Chart Settings</h4>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Chart Title
                      </label>
                      <input
                        type="text"
                        value={editingChart.title}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          title: e.target.value
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Description
                      </label>
                      <textarea
                        value={editingChart.description || ''}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          description: e.target.value
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                        rows={3}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Chart Type
                      </label>
                      <select
                        value={editingChart.type}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          type: e.target.value
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        {AVAILABLE_CHARTS.map(chart => (
                          <option key={chart.id} value={chart.type}>
                            {chart.name}
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Data Source
                      </label>
                      <select
                        value={editingChart.dataSet}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          dataSet: e.target.value
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        {availableDataSets.length === 0 ? (
                          <option value="">No data sources available</option>
                        ) : (
                          availableDataSets.map(ds => (
                            <option key={ds.id} value={ds.id}>
                              {ds.name}
                            </option>
                          ))
                        )}
                      </select>
                    </div>
                  </div>
                </div>
                
                {/* Advanced Settings */}
                <div>
                  <h4 className="font-medium text-gray-800 mb-3">Display Options</h4>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Size
                      </label>
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-xs text-gray-500">Width</label>
                          <select
                            value={editingChart.gridPos.w}
                            onChange={(e) => setEditingChart({
                              ...editingChart,
                              gridPos: {
                                ...editingChart.gridPos,
                                w: parseInt(e.target.value)
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                          >
                            {[1, 2, 3, 4, 6, 8, 12].map(w => (
                              <option key={w} value={w}>
                                {w} / 12 columns
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="block text-xs text-gray-500">Height</label>
                          <select
                            value={editingChart.gridPos.h}
                            onChange={(e) => setEditingChart({
                              ...editingChart,
                              gridPos: {
                                ...editingChart.gridPos,
                                h: parseInt(e.target.value)
                              }
                            })}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                          >
                            {[2, 3, 4, 6, 8, 10, 12].map(h => (
                              <option key={h} value={h}>
                                {h} units ({h * 70}px)
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </div>
                    
                    {/* Additional settings based on chart type */}
                    {editingChart.type === 'bar' && (
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Bar Orientation
                        </label>
                        <select
                          value={editingChart.config?.orientation || 'vertical'}
                          onChange={(e) => setEditingChart({
                            ...editingChart,
                            config: {
                              ...editingChart.config,
                              orientation: e.target.value
                            }
                          })}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                        >
                          <option value="vertical">Vertical</option>
                          <option value="horizontal">Horizontal</option>
                        </select>
                      </div>
                    )}
                    
                    {/* Color scheme */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Color Scheme
                      </label>
                      <select
                        value={editingChart.config?.colorScheme || 'default'}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          config: {
                            ...editingChart.config,
                            colorScheme: e.target.value
                          }
                        })}
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
                    
                    {/* Legend position */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Legend Position
                      </label>
                      <select
                        value={editingChart.config?.legendPosition || 'top'}
                        onChange={(e) => setEditingChart({
                          ...editingChart,
                          config: {
                            ...editingChart.config,
                            legendPosition: e.target.value
                          }
                        })}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                      >
                        <option value="top">Top</option>
                        <option value="right">Right</option>
                        <option value="bottom">Bottom</option>
                        <option value="left">Left</option>
                        <option value="none">None (hidden)</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end">
              <button
                onClick={() => setIsChartSettingsOpen(false)}
                className="px-4 py-2 border border-gray-300 rounded-md text-sm mr-2"
              >
                Cancel
              </button>
              <button
                onClick={() => saveChartSettings(editingChart)}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm"
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DashboardBuilder;