import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from '../visualization/ChartViewer';
import Loading from '../common/Loading';
import {
  ArrowPathIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  ArrowTopRightOnSquareIcon
} from '@heroicons/react/24/outline';

const SupplierDashboard = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [supplierData, setSupplierData] = useState(null);
  const [timeFrame, setTimeFrame] = useState('last90days');
  const [supplierCategory, setSupplierCategory] = useState('all');
  const [filterOpen, setFilterOpen] = useState(false);

  useEffect(() => {
    fetchSupplierData();
  }, [timeFrame, supplierCategory]);

  const fetchSupplierData = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/analytics/supplier', { params: { timeFrame, category: supplierCategory } });
      
      // Simulated API response with mock data
      const mockData = getMockSupplierData(timeFrame, supplierCategory);
      setSupplierData(mockData);
    } catch (error) {
      console.error('Error fetching supplier data:', error);
    } finally {
      setTimeout(() => setIsLoading(false), 800); // Simulated loading delay
    }
  };

  const handleViewSupplierDetails = (supplierId) => {
    navigate(`/suppliers/${supplierId}`);
  };

  const handleExportDashboard = () => {
    // Implementation for exporting the entire dashboard
    alert('Exporting supplier dashboard...');
  };

  // Mock data generator function
  const getMockSupplierData = (timeFrame, category) => {
    return {
      summary: {
        totalSuppliers: 87,
        activeSuppliers: 81,
        onTimeDelivery: 93.4,
        qualityScore: 92.7,
        riskScore: 12.3
      },
      performanceTrend: {
        type: 'line',
        title: 'Supplier Performance Trend',
        data: [
          { date: '2023-01-01', onTime: 92.1, quality: 91.5, responsiveness: 89.2 },
          { date: '2023-02-01', onTime: 92.5, quality: 92.0, responsiveness: 90.1 },
          { date: '2023-03-01', onTime: 93.0, quality: 92.2, responsiveness: 90.5 },
          { date: '2023-04-01', onTime: 93.2, quality: 92.5, responsiveness: 91.0 },
          { date: '2023-05-01', onTime: 93.4, quality: 92.7, responsiveness: 91.2 }
        ],
        config: {
          xKey: 'date',
          multiSeries: true,
          series: [
            { name: 'On-Time Delivery', dataKey: 'onTime', color: '#4f46e5' },
            { name: 'Quality Score', dataKey: 'quality', color: '#10b981' },
            { name: 'Responsiveness', dataKey: 'responsiveness', color: '#f59e0b' }
          ],
          valueFormatter: (value) => `${value}%`
        }
      },
      suppliersByCategory: {
        type: 'pie',
        title: 'Suppliers by Category',
        data: [
          { name: 'Raw Materials', value: 32 },
          { name: 'Components', value: 28 },
          { name: 'Packaging', value: 15 },
          { name: 'Services', value: 12 }
        ],
        config: {
          nameKey: 'name',
          valueKey: 'value',
          showPercentages: true
        }
      },
      topSuppliers: {
        type: 'bar',
        title: 'Top 5 Suppliers by Performance Score',
        data: [
          { name: 'Supplier A', score: 97.8 },
          { name: 'Supplier B', score: 96.5 },
          { name: 'Supplier C', score: 95.9 },
          { name: 'Supplier D', score: 95.2 },
          { name: 'Supplier E', score: 94.8 }
        ],
        config: {
          xKey: 'name',
          yKey: 'score',
          horizontal: true,
          color: '#10b981',
          valueFormatter: (value) => `${value}%`
        }
      },
      supplierRisk: {
        type: 'heatmap',
        title: 'Supplier Risk Analysis',
        data: [
          { x: 'Low', y: 'Financial', value: 15 },
          { x: 'Medium', y: 'Financial', value: 8 },
          { x: 'High', y: 'Financial', value: 4 },
          { x: 'Low', y: 'Operational', value: 12 },
          { x: 'Medium', y: 'Operational', value: 6 },
          { x: 'High', y: 'Operational', value: 2 },
          { x: 'Low', y: 'Compliance', value: 18 },
          { x: 'Medium', y: 'Compliance', value: 5 },
          { x: 'High', y: 'Compliance', value: 1 },
          { x: 'Low', y: 'Geographical', value: 10 },
          { x: 'Medium', y: 'Geographical', value: 7 },
          { x: 'High', y: 'Geographical', value: 3 }
        ],
        config: {
          xKey: 'x',
          yKey: 'y',
          valueKey: 'value',
          colorScheme: 'interpolateReds'
        }
      },
      supplierList: [
        { id: 1, name: 'Acme Supplies', category: 'Raw Materials', onTimeDelivery: 97.8, qualityScore: 96.5, riskScore: 'Low' },
        { id: 2, name: 'Globex Components', category: 'Components', onTimeDelivery: 93.2, qualityScore: 95.1, riskScore: 'Low' },
        { id: 3, name: 'Quality Packaging Co', category: 'Packaging', onTimeDelivery: 94.5, qualityScore: 91.8, riskScore: 'Medium' },
        { id: 4, name: 'TechPro Solutions', category: 'Components', onTimeDelivery: 96.7, qualityScore: 94.3, riskScore: 'Low' },
        { id: 5, name: 'Global Logistics', category: 'Services', onTimeDelivery: 89.5, qualityScore: 88.2, riskScore: 'Medium' }
      ]
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading supplier analytics..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full">
      {/* Dashboard Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">Supplier Analytics</h1>
          <p className="text-sm text-gray-500 mt-1">Track and evaluate supplier performance</p>
        </div>
        
        <div className="flex items-center space-x-3 mt-3 sm:mt-0">
          <div className="relative">
            <button
              onClick={() => setFilterOpen(!filterOpen)}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <FunnelIcon className="-ml-1 mr-2 h-4 w-4" />
              Filter
            </button>
            
            {filterOpen && (
              <div className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                <div className="py-1 px-3">
                  <div className="mb-2">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Time Frame</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'last30days', label: 'Last 30 Days' },
                        { id: 'last90days', label: 'Last 90 Days' },
                        { id: 'last6months', label: 'Last 6 Months' },
                        { id: 'last12months', label: 'Last 12 Months' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="timeFrame"
                            value={option.id}
                            checked={timeFrame === option.id}
                            onChange={() => setTimeFrame(option.id)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{option.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Supplier Category</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'all', label: 'All Categories' },
                        { id: 'rawMaterials', label: 'Raw Materials' },
                        { id: 'components', label: 'Components' },
                        { id: 'packaging', label: 'Packaging' },
                        { id: 'services', label: 'Services' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="supplierCategory"
                            value={option.id}
                            checked={supplierCategory === option.id}
                            onChange={() => setSupplierCategory(option.id)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{option.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-3 flex justify-end">
                    <button
                      onClick={() => setFilterOpen(false)}
                      className="px-4 py-1.5 bg-indigo-600 text-white text-sm rounded hover:bg-indigo-700"
                    >
                      Apply
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <button
            onClick={fetchSupplierData}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
            Refresh
          </button>
          
          <button
            onClick={handleExportDashboard}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowDownTrayIcon className="-ml-1 mr-2 h-4 w-4" />
            Export
          </button>
        </div>
      </div>
      
      {/* Dashboard Content */}
      <div className="container mx-auto px-4 py-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-5 mb-6">
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Total Suppliers</p>
                <p className="text-2xl font-bold text-gray-900">{supplierData.summary.totalSuppliers}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-indigo-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Active Suppliers</p>
                <p className="text-2xl font-bold text-gray-900">{supplierData.summary.activeSuppliers}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round(supplierData.summary.activeSuppliers / supplierData.summary.totalSuppliers * 100)}% of total
                </p>
              </div>
              <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">On-Time Delivery</p>
                <p className="text-2xl font-bold text-gray-900">{supplierData.summary.onTimeDelivery}%</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Quality Score</p>
                <p className="text-2xl font-bold text-gray-900">{supplierData.summary.qualityScore}%</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-yellow-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Risk Score</p>
                <p className="text-2xl font-bold text-gray-900">{supplierData.summary.riskScore}%</p>
                <p className="text-xs text-gray-500 mt-1">
                  {supplierData.summary.riskScore < 15 ? 'Low Risk' : supplierData.summary.riskScore < 30 ? 'Medium Risk' : 'High Risk'}
                </p>
              </div>
              <div className="h-12 w-12 rounded-full bg-red-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>
          </div>
        </div>
        
        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartViewer chartData={supplierData.performanceTrend} />
          <ChartViewer chartData={supplierData.suppliersByCategory} />
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ChartViewer chartData={supplierData.topSuppliers} />
          <ChartViewer chartData={supplierData.supplierRisk} />
        </div>
        
        {/* Top Suppliers Table */}
        <div className="bg-white shadow rounded-lg mb-6">
          <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
            <h2 className="text-lg font-medium text-gray-800">Top Suppliers</h2>
            <a href="/suppliers" className="text-sm text-indigo-600 hover:text-indigo-900">View all suppliers</a>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Supplier Name
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    On-Time Delivery
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Quality Score
                  </th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk Rating
                  </th>
                  <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {supplierData.supplierList.map((supplier) => (
                  <tr key={supplier.id}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{supplier.name}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-500">{supplier.category}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`text-sm ${supplier.onTimeDelivery >= 95 ? 'text-green-600' : supplier.onTimeDelivery >= 90 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {supplier.onTimeDelivery}%
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`text-sm ${supplier.qualityScore >= 95 ? 'text-green-600' : supplier.qualityScore >= 90 ? 'text-yellow-600' : 'text-red-600'}`}>
                        {supplier.qualityScore}%
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                        ${supplier.riskScore === 'Low' ? 'bg-green-100 text-green-800' : 
                          supplier.riskScore === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                          'bg-red-100 text-red-800'}`}>
                        {supplier.riskScore}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => handleViewSupplierDetails(supplier.id)}
                        className="text-indigo-600 hover:text-indigo-900 inline-flex items-center"
                      >
                        View Details
                        <ArrowTopRightOnSquareIcon className="ml-1 h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SupplierDashboard;

