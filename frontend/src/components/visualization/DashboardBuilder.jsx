import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from './ChartViewer';
import {
  ViewGridAddIcon,
  ArrowDownOnSquareIcon,
  PlusCircleIcon,
  XMarkIcon,
  MenuAlt2Icon,
  ShareIcon
} from '@heroicons/react/24/outline';

// React Grid Layout
import { Responsive, WidthProvider } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

const ResponsiveGridLayout = WidthProvider(Responsive);

const DashboardBuilder = ({
  dashboardId,
  initialLayout = [],
  initialCharts = [],
  onSave,
  onAddChart,
  isEditing = false
}) => {
  const navigate = useNavigate();
  const [layouts, setLayouts] = useState({ lg: initialLayout });
  const [charts, setCharts] = useState(initialCharts);
  const [isEditMode, setIsEditMode] = useState(isEditing);
  const [dashboardName, setDashboardName] = useState('New Dashboard');
  const [isShareModalOpen, setIsShareModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Update internal state when props change
  useEffect(() => {
    if (initialLayout.length > 0) {
      setLayouts({ lg: initialLayout });
    }
    if (initialCharts.length > 0) {
      setCharts(initialCharts);
    }
  }, [initialLayout, initialCharts]);

  const handleLayoutChange = (currentLayout, allLayouts) => {
    setLayouts(allLayouts);
  };

  const handleSaveDashboard = async () => {
    if (!dashboardName.trim()) {
      alert('Please provide a dashboard name');
      return;
    }

    setIsSaving(true);
    try {
      if (onSave) {
        await onSave({
          id: dashboardId,
          name: dashboardName,
          layout: layouts.lg,
          charts: charts.map(chart => chart.id)
        });
      }
      setIsEditMode(false);
    } catch (error) {
      console.error('Error saving dashboard:', error);
      alert('Failed to save dashboard. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  const handleRemoveChart = (chartId) => {
    setCharts(charts.filter(chart => chart.id !== chartId));
    setLayouts({
      ...layouts,
      lg: layouts.lg.filter(item => item.i !== chartId.toString())
    });
  };

  const handleAddNewChart = () => {
    if (onAddChart) {
      onAddChart();
    }
  };

  const generateShareLink = () => {
    // This would typically generate a unique sharing link
    // For now, we'll just return the current URL
    return window.location.href;
  };

  const copyLinkToClipboard = () => {
    const link = generateShareLink();
    navigator.clipboard.writeText(link);
    // Show temporary success message
    setShareSuccess(true);
    setTimeout(() => setShareSuccess(false), 3000);
  };

  const [shareSuccess, setShareSuccess] = useState(false);

  return (
    <div className="h-full flex flex-col">
      {/* Dashboard header */}
      <div className="bg-white shadow-sm border-b border-gray-200 px-4 py-3 flex justify-between items-center">
        <div className="flex items-center">
          {isEditMode ? (
            <input
              type="text"
              value={dashboardName}
              onChange={(e) => setDashboardName(e.target.value)}
              className="border-gray-300 focus:ring-indigo-500 focus:border-indigo-500 block w-56 shadow-sm sm:text-sm border rounded-md px-3 py-1"
              placeholder="Dashboard name"
            />
          ) : (
            <h1 className="text-lg font-semibold text-gray-800">{dashboardName}</h1>
          )}
        </div>
        
        <div className="flex space-x-2">
          {!isEditMode && (
            <>
              <button
                onClick={() => setIsShareModalOpen(true)}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                <ShareIcon className="-ml-1 mr-2 h-4 w-4" />
                Share
              </button>
              
              <button
                onClick={() => setIsEditMode(true)}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                <MenuAlt2Icon className="-ml-1 mr-2 h-4 w-4" />
                Edit
              </button>
            </>
          )}
          
          {isEditMode && (
            <>
              <button
                onClick={handleAddNewChart}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                <PlusCircleIcon className="-ml-1 mr-2 h-4 w-4" />
                Add Chart
              </button>
              
              <button
                onClick={handleSaveDashboard}
                disabled={isSaving}
                className={`inline-flex items-center px-3 py-1.5 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                  isSaving ? 'opacity-75 cursor-not-allowed' : ''
                }`}
              >
                {isSaving ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Saving...
                  </>
                ) : (
                  <>
                    <ArrowDownOnSquareIcon className="-ml-1 mr-2 h-4 w-4" />
                    Save Dashboard
                  </>
                )}
              </button>
              
              <button
                onClick={() => setIsEditMode(false)}
                className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {/* Dashboard content */}
      <div className="flex-1 overflow-auto bg-gray-50 p-4">
        {charts.length === 0 ? (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 flex flex-col items-center justify-center h-64">
            <ViewGridAddIcon className="h-12 w-12 text-gray-400 mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Add charts to your dashboard</h3>
            <p className="text-sm text-gray-500 mb-4 text-center max-w-md">
              Create a custom dashboard by adding charts from your analytics or query results.
            </p>
            <button
              onClick={handleAddNewChart}
              className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <PlusCircleIcon className="-ml-1 mr-2 h-5 w-5" />
              Add Your First Chart
            </button>
          </div>
        ) : (
          <ResponsiveGridLayout
            className="layout"
            layouts={layouts}
            onLayoutChange={handleLayoutChange}
            isDraggable={isEditMode}
            isResizable={isEditMode}
            breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
            cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
            rowHeight={100}
            margin={[16, 16]}
            containerPadding={[0, 0]}
          >
            {charts.map(chart => (
              <div key={chart.id.toString()} className="bg-white rounded-lg shadow-sm overflow-hidden">
                {isEditMode && (
                  <button
                    onClick={() => handleRemoveChart(chart.id)}
                    className="absolute top-2 right-2 z-10 p-1 rounded-full bg-red-100 text-red-500 hover:bg-red-200"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                )}
                <ChartViewer 
                  chartData={chart} 
                  fullHeight={true}
                  showControls={!isEditMode}
                  onEdit={() => {
                    // Navigate to chart edit page
                    navigate(`/charts/${chart.id}/edit`);
                  }} 
                />
              </div>
            ))}
          </ResponsiveGridLayout>
        )}
      </div>

      {/* Share Modal */}
      {isShareModalOpen && (
        <div className="fixed z-10 inset-0 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" aria-hidden="true">
              <div className="absolute inset-0 bg-gray-500 opacity-75" onClick={() => setIsShareModalOpen(false)}></div>
            </div>

            <div className="inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
              <div className="bg-white px-4 pt-5 pb-4 sm:p-6 sm:pb-4">
                <div className="sm:flex sm:items-start">
                  <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left w-full">
                    <h3 className="text-lg leading-6 font-medium text-gray-900" id="modal-title">
                      Share Dashboard
                    </h3>
                    <div className="mt-4">
                      <p className="text-sm text-gray-500 mb-2">
                        Share this link with others who should have access to this dashboard:
                      </p>
                      <div className="mt-1 flex rounded-md shadow-sm">
                        <div className="relative flex items-stretch flex-grow">
                          <input
                            type="text"
                            className="focus:ring-indigo-500 focus:border-indigo-500 block w-full rounded-none rounded-l-md sm:text-sm border-gray-300"
                            value={generateShareLink()}
                            readOnly
                          />
                        </div>
                        <button
                          type="button"
                          onClick={copyLinkToClipboard}
                          className="inline-flex items-center px-3 py-2 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 text-gray-500 sm:text-sm hover:bg-gray-100"
                        >
                          Copy
                        </button>
                      </div>
                      {shareSuccess && (
                        <p className="mt-2 text-sm text-green-600">
                          Link copied to clipboard!
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 px-4 py-3 sm:px-6 sm:flex sm:flex-row-reverse">
                <button
                  type="button"
                  className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:ml-3 sm:w-auto sm:text-sm"
                  onClick={() => setIsShareModalOpen(false)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DashboardBuilder;
