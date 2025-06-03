import React, { useState } from 'react';
import { 
  DocumentTextIcon, 
  CalendarIcon, 
  ArrowDownTrayIcon,
  PlusIcon,
  ChartBarIcon,
  TruckIcon,
  CubeIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

const Reports = () => {
  const [selectedReport, setSelectedReport] = useState(null);
  const [dateRange, setDateRange] = useState('last30days');

  // Mock report templates
  const reportTemplates = [
    {
      id: 1,
      name: 'Inventory Status Report',
      description: 'Current inventory levels, stockouts, and overstock analysis',
      icon: CubeIcon,
      lastGenerated: '2025-05-28',
      frequency: 'Daily',
      status: 'ready'
    },
    {
      id: 2,
      name: 'Supplier Performance Report',
      description: 'Supplier delivery performance, quality metrics, and risk assessment',
      icon: TruckIcon,
      lastGenerated: '2025-05-25',
      frequency: 'Weekly',
      status: 'ready'
    },
    {
      id: 3,
      name: 'Supply Chain Analytics',
      description: 'Comprehensive analytics including trends, forecasts, and KPIs',
      icon: ChartBarIcon,
      lastGenerated: '2025-05-20',
      frequency: 'Monthly',
      status: 'ready'
    },
    {
      id: 4,
      name: 'Order Fulfillment Report',
      description: 'Order processing times, fulfillment rates, and delivery performance',
      icon: ClockIcon,
      lastGenerated: '2025-05-30',
      frequency: 'Daily',
      status: 'processing'
    }
  ];

  // Mock recent reports
  const recentReports = [
    {
      id: 101,
      name: 'May 2025 Inventory Report',
      type: 'Inventory Status Report',
      generatedDate: '2025-05-31',
      size: '2.4 MB',
      format: 'PDF'
    },
    {
      id: 102,
      name: 'Q2 Supplier Performance',
      type: 'Supplier Performance Report',
      generatedDate: '2025-05-30',
      size: '1.8 MB',
      format: 'PDF'
    },
    {
      id: 103,
      name: 'Weekly Analytics - W22',
      type: 'Supply Chain Analytics',
      generatedDate: '2025-05-29',
      size: '3.1 MB',
      format: 'XLSX'
    }
  ];

  const handleGenerateReport = (reportId) => {
    setSelectedReport(reportId);
    // In a real app, this would trigger report generation
    console.log('Generating report:', reportId, 'with date range:', dateRange);
  };

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900">Reports</h1>
        <p className="mt-2 text-gray-600">
          Generate and download supply chain reports
        </p>
      </div>

      {/* Date Range Selector */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow">
        <label htmlFor="dateRange" className="block text-sm font-medium text-gray-700 mb-2">
          Report Date Range
        </label>
        <select
          id="dateRange"
          name="dateRange"
          value={dateRange}
          onChange={(e) => setDateRange(e.target.value)}
          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
        >
          <option value="last7days">Last 7 Days</option>
          <option value="last30days">Last 30 Days</option>
          <option value="lastQuarter">Last Quarter</option>
          <option value="yearToDate">Year to Date</option>
          <option value="custom">Custom Range</option>
        </select>
      </div>

      {/* Report Templates */}
      <div className="mb-8">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Available Report Templates</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {reportTemplates.map((report) => (
            <div
              key={report.id}
              className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow"
            >
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <report.icon className="h-10 w-10 text-indigo-600" />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <h3 className="text-lg font-medium text-gray-900 truncate">
                      {report.name}
                    </h3>
                    <p className="mt-1 text-sm text-gray-500">
                      {report.description}
                    </p>
                    <div className="mt-2 flex items-center text-sm text-gray-500">
                      <CalendarIcon className="flex-shrink-0 mr-1.5 h-4 w-4 text-gray-400" />
                      {report.frequency} • Last: {new Date(report.lastGenerated).toLocaleDateString()}
                    </div>
                  </div>
                </div>
                <div className="mt-4">
                  <button
                    onClick={() => handleGenerateReport(report.id)}
                    disabled={report.status === 'processing'}
                    className={`w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm ${
                      report.status === 'processing'
                        ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                        : 'text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                    }`}
                  >
                    {report.status === 'processing' ? (
                      <>
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <PlusIcon className="-ml-1 mr-2 h-4 w-4" />
                        Generate Report
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Reports */}
      <div>
        <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Reports</h2>
        <div className="bg-white shadow overflow-hidden sm:rounded-md">
          <ul className="divide-y divide-gray-200">
            {recentReports.map((report) => (
              <li key={report.id}>
                <div className="px-4 py-4 sm:px-6 hover:bg-gray-50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <DocumentTextIcon className="flex-shrink-0 h-6 w-6 text-gray-400" />
                      <div className="ml-4">
                        <p className="text-sm font-medium text-gray-900">{report.name}</p>
                        <p className="text-sm text-gray-500">{report.type}</p>
                      </div>
                    </div>
                    <div className="flex items-center">
                      <div className="mr-4 text-right">
                        <p className="text-sm text-gray-900">{report.size}</p>
                        <p className="text-sm text-gray-500">
                          {new Date(report.generatedDate).toLocaleDateString()}
                        </p>
                      </div>
                      <button className="inline-flex items-center p-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                        <ArrowDownTrayIcon className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Coming Soon Notice */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-md p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">
              Advanced Features Coming Soon
            </h3>
            <div className="mt-2 text-sm text-blue-700">
              <p>
                We're working on additional reporting features including:
              </p>
              <ul className="list-disc list-inside mt-1">
                <li>Custom report builder</li>
                <li>Scheduled report generation</li>
                <li>Email delivery of reports</li>
                <li>Interactive dashboards</li>
                <li>Real-time data integration</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Reports;