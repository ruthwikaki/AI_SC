import React, { useState, useMemo } from 'react';
import BarChart from './charts/BarChart';
import LineChart from './charts/LineChart';
import PieChart from './charts/PieChart';
import HeatMap from './charts/HeatMap';
import SankeyDiagram from './charts/SankeyDiagram';
import NetworkGraph from './charts/NetworkGraph';
import { 
  DownloadIcon, 
  RefreshIcon, 
  PencilIcon, 
  PlusIcon,
  DotsVerticalIcon
} from '@heroicons/react/outline';

const ChartViewer = ({ 
  chartData, 
  onExport, 
  onRefresh, 
  onEdit,
  fullHeight = false,
  showControls = true
}) => {
  const [menuOpen, setMenuOpen] = useState(false);
  
  const renderChart = useMemo(() => {
    if (!chartData) return null;
    
    const { type, data, config = {} } = chartData;
    
    const height = fullHeight ? '100%' : 400;
    
    switch (type) {
      case 'bar':
        return <BarChart data={data} config={config} height={height} />;
      case 'line':
        return <LineChart data={data} config={config} height={height} />;
      case 'pie':
        return <PieChart data={data} config={config} height={height} />;
      case 'heatmap':
        return <HeatMap data={data} config={config} height={height} />;
      case 'sankey':
        return <SankeyDiagram data={data} config={config} height={height} />;
      case 'network':
        return <NetworkGraph data={data} config={config} height={height} />;
      default:
        return (
          <div className="flex items-center justify-center h-full">
            <div className="text-gray-500">Unsupported chart type: {type}</div>
          </div>
        );
    }
  }, [chartData, fullHeight]);

  const handleExport = (format) => {
    if (onExport) {
      onExport(format);
    }
    setMenuOpen(false);
  };

  if (!chartData) {
    return (
      <div className="bg-white shadow rounded-lg p-4 flex items-center justify-center h-64">
        <div className="text-gray-500">No chart data available</div>
      </div>
    );
  }

  return (
    <div className={`bg-white shadow rounded-lg overflow-hidden ${fullHeight ? 'h-full' : ''}`}>
      {/* Chart header */}
      <div className="border-b border-gray-200 px-4 py-3 flex justify-between items-center">
        <h3 className="text-md font-medium text-gray-700">
          {chartData.title || 'Chart'}
          {chartData.description && (
            <span className="ml-2 text-xs text-gray-500">{chartData.description}</span>
          )}
        </h3>
        
        {showControls && (
          <div className="flex space-x-2 items-center">
            {onRefresh && (
              <button 
                onClick={onRefresh}
                className="p-1 rounded-full text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none"
              >
                <RefreshIcon className="h-5 w-5" />
              </button>
            )}
            
            {onEdit && (
              <button 
                onClick={onEdit}
                className="p-1 rounded-full text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none"
              >
                <PencilIcon className="h-5 w-5" />
              </button>
            )}
            
            <div className="relative">
              <button 
                onClick={() => setMenuOpen(!menuOpen)}
                className="p-1 rounded-full text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none"
             >
               <DotsVerticalIcon className="h-5 w-5" />
             </button>
             
             {menuOpen && (
               <div className="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                 <div className="py-1" role="menu" aria-orientation="vertical">
                   <button
                     onClick={() => handleExport('png')}
                     className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                     role="menuitem"
                   >
                     Export as PNG
                   </button>
                   <button
                     onClick={() => handleExport('svg')}
                     className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                     role="menuitem"
                   >
                     Export as SVG
                   </button>
                   <button
                     onClick={() => handleExport('csv')}
                     className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                     role="menuitem"
                   >
                     Export data as CSV
                   </button>
                   <button
                     onClick={() => handleExport('excel')}
                     className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                     role="menuitem"
                   >
                     Export data as Excel
                   </button>
                 </div>
               </div>
             )}
           </div>
         </div>
       )}
     </div>
     
     {/* Chart content */}
     <div className={`p-4 ${fullHeight ? 'h-[calc(100%-3.5rem)]' : ''}`}>
       {renderChart}
     </div>
   </div>
 );
};

export default ChartViewer;