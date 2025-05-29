import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from '../visualization/ChartViewer';
import Loading from '../common/Loading';
import {
  ArrowPathIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  MapIcon,
  TruckIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

const LogisticsDashboard = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [logisticsData, setLogisticsData] = useState(null);
  const [timeFrame, setTimeFrame] = useState('lastMonth');
  const [region, setRegion] = useState('all');
  const [transportMode, setTransportMode] = useState('all');
  const [filterOpen, setFilterOpen] = useState(false);

  useEffect(() => {
    fetchLogisticsData();
  }, [timeFrame, region, transportMode]);

  const fetchLogisticsData = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/analytics/logistics', { params: { timeFrame, region, transportMode } });
      
      // Simulated API response with mock data
      const mockData = getMockLogisticsData(timeFrame, region, transportMode);
      
      // Simulate API delay
      setTimeout(() => {
        setLogisticsData(mockData);
        setIsLoading(false);
      }, 800);
    } catch (error) {
      console.error('Error fetching logistics data:', error);
      setIsLoading(false);
    }
  };

  const handleViewRouteOptimization = () => {
    navigate('/logistics/route-optimization');
  };

  const handleViewCarrierAnalysis = () => {
    navigate('/logistics/carrier-analysis');
  };

  const handleExportDashboard = () => {
    // Implementation for exporting the entire dashboard
    alert('Exporting logistics dashboard...');
  };

  // Mock data generator function
  const getMockLogisticsData = (timeFrame, region, transportMode) => {
    return {
      summary: {
        totalShipments: 1254,
        onTimeDelivery: 94.7,
        averageTransitTime: 3.2,
        freightCost: 453000,
        carbonEmissions: 285
      },
      deliveryPerformance: {
        type: 'line',
        title: 'Delivery Performance Over Time',
        data: [
          { date: '2023-01-01', onTime: 93.2, delay: 6.8 },
          { date: '2023-02-01', onTime: 93.8, delay: 6.2 },
          { date: '2023-03-01', onTime: 94.3, delay: 5.7 },
          { date: '2023-04-01', onTime: 94.5, delay: 5.5 },
          { date: '2023-05-01', onTime: 94.7, delay: 5.3 }
        ],
        config: {
          xKey: 'date',
          multiSeries: true,
          series: [
            { name: 'On Time', dataKey: 'onTime', color: '#10b981' },
            { name: 'Delayed', dataKey: 'delay', color: '#ef4444' }
          ],
          valueFormatter: (value) => `${value}%`,
          showArea: true
        }
      },
      transportModes: {
        type: 'pie',
        title: 'Shipments by Transport Mode',
        data: [
          { name: 'Truck', value: 850 },
          { name: 'Rail', value: 210 },
          { name: 'Air', value: 120 },
          { name: 'Ocean', value: 74 }
        ],
        config: {
          nameKey: 'name',
          valueKey: 'value',
          innerRadius: 0.5,
          showPercentages: true
        }
      },
      costByRegion: {
        type: 'bar',
        title: 'Freight Cost by Region',
        data: [
          { region: 'North America', cost: 180000 },
          { region: 'Europe', cost: 125000 },
          { region: 'Asia Pacific', cost: 98000 },
          { region: 'Latin America', cost: 50000 }
        ],
        config: {
          xKey: 'region',
          yKey: 'cost',
          color: '#6366f1',
          valueFormatter: (value) => `$${(value / 1000).toFixed(0)}K`
        }
      },
      delayReasons: {
        type: 'bar',
        title: 'Delay Reasons Distribution',
        data: [
          { reason: 'Weather', count: 32 },
          { reason: 'Customs', count: 28 },
          { reason: 'Traffic', count: 25 },
          { reason: 'Mechanical', count: 18 },
          { reason: 'Documentation', count: 15 },
          { reason: 'Other', count: 10 }
        ],
        config: {
          xKey: 'reason',
          yKey: 'count',
          horizontal: true,
          color: '#f59e0b'
        }
      },
      routeNetwork: {
        type: 'network',
        title: 'Shipping Network',
        data: {
          nodes: [
            { id: '1', name: 'Chicago', value: 350, group: 'Distribution Center' },
            { id: '2', name: 'New York', value: 280, group: 'Distribution Center' },
            { id: '3', name: 'Atlanta', value: 210, group: 'Distribution Center' },
            { id: '4', name: 'Dallas', value: 180, group: 'Distribution Center' },
            { id: '5', name: 'Los Angeles', value: 190, group: 'Distribution Center' },
            { id: '6', name: 'Seattle', value: 140, group: 'Distribution Center' },
            { id: '7', name: 'Detroit', value: 90, group: 'Customer' },
            { id: '8', name: 'Phoenix', value: 110, group: 'Customer' },
            { id: '9', name: 'Miami', value: 95, group: 'Customer' },
            { id: '10', name: 'Boston', value: 85, group: 'Customer' }
          ],
          links: [
            { source: '1', target: '7', value: 80 },
            { source: '1', target: '10', value: 60 },
            { source: '2', target: '10', value: 70 },
            { source: '3', target: '9', value: 90 },
            { source: '4', target: '8', value: 75 },
            { source: '5', target: '8', value: 95 },
            { source: '6', target: '7', value: 40 }
          ]
        },
        config: {
          forceStrength: -200,
          nodeSize: 'value',
          colorScheme: 'schemeCategory10'
        }
      }
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading logistics analytics..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full">
      {/* Dashboard Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">Logistics Analytics</h1>
          <p className="text-sm text-gray-500 mt-1">Optimize transportation performance and costs</p>
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
                        { id: 'lastWeek', label: 'Last Week' },
                        { id: 'lastMonth', label: 'Last Month' },
                        { id: 'lastQuarter', label: 'Last Quarter' },
                        { id: 'lastYear', label: 'Last Year' }
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
                  
                  <div className="mb-2">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Region</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'all', label: 'All Regions' },
                        { id: 'northAmerica', label: 'North America' },
                        { id: 'europe', label: 'Europe' },
                        { id: 'asia', label: 'Asia Pacific' },
                        { id: 'latinAmerica', label: 'Latin America' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="region"
                            value={option.id}
                            checked={region === option.id}
                            onChange={() => setRegion(option.id)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{option.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Transport Mode</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'all', label: 'All Modes' },
                        { id: 'truck', label: 'Truck' },
                        { id: 'rail', label: 'Rail' },
                        { id: 'air', label: 'Air' },
                        { id: 'ocean', label: 'Ocean' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="transportMode"
                            value={option.id}
                            checked={transportMode === option.id}
                            onChange={() => setTransportMode(option.id)}
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
            onClick={fetchLogisticsData}
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
                <p className="text-sm font-medium text-gray-500">Total Shipments</p>
                <p className="text-2xl font-bold text-gray-900">{logisticsData.summary.totalShipments.toLocaleString()}</p>
              </div>
              <div className="h-12 w-12 rounded-full bg-indigo-100 flex items-center justify-center">
                <TruckIcon className="h-6 w-6 text-indigo-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
              <p className="text-sm font-medium text-gray-500">On-Time Delivery</p>
               <p className="text-2xl font-bold text-gray-900">{logisticsData.summary.onTimeDelivery}%</p>
             </div>
             <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
               <ClockIcon className="h-6 w-6 text-green-600" />
             </div>
           </div>
         </div>
         
         <div className="bg-white rounded-lg shadow p-5">
           <div className="flex justify-between">
             <div>
               <p className="text-sm font-medium text-gray-500">Avg. Transit Time</p>
               <p className="text-2xl font-bold text-gray-900">{logisticsData.summary.averageTransitTime} days</p>
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
               <p className="text-sm font-medium text-gray-500">Freight Cost</p>
               <p className="text-2xl font-bold text-gray-900">${(logisticsData.summary.freightCost / 1000).toFixed(0)}K</p>
             </div>
             <div className="h-12 w-12 rounded-full bg-purple-100 flex items-center justify-center">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
               </svg>
             </div>
           </div>
         </div>
         
         <div className="bg-white rounded-lg shadow p-5">
           <div className="flex justify-between">
             <div>
               <p className="text-sm font-medium text-gray-500">Carbon Emissions</p>
               <p className="text-2xl font-bold text-gray-900">{logisticsData.summary.carbonEmissions} tons</p>
             </div>
             <div className="h-12 w-12 rounded-full bg-emerald-100 flex items-center justify-center">
               <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
               </svg>
             </div>
           </div>
         </div>
       </div>
       
       {/* Charts */}
       <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
         <ChartViewer chartData={logisticsData.deliveryPerformance} />
         <ChartViewer chartData={logisticsData.transportModes} />
       </div>
       
       <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
         <ChartViewer chartData={logisticsData.costByRegion} />
         <ChartViewer chartData={logisticsData.delayReasons} />
       </div>
       
       {/* Network Visualization */}
       <div className="mb-6">
         <ChartViewer chartData={logisticsData.routeNetwork} />
       </div>
       
       {/* Advanced Analysis Tools */}
       <div className="bg-white rounded-lg shadow mb-6">
         <div className="px-6 py-4 border-b border-gray-200">
           <h2 className="text-lg font-medium text-gray-800">Advanced Logistics Tools</h2>
         </div>
         
         <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
           <div 
             onClick={handleViewRouteOptimization}
             className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
           >
             <div className="flex items-center mb-3">
               <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center mr-3">
                 <MapIcon className="h-5 w-5 text-indigo-600" />
               </div>
               <h3 className="text-md font-medium text-gray-800">Route Optimization</h3>
             </div>
             <p className="text-sm text-gray-600">
               Optimize delivery routes to minimize distance, time, and costs while meeting all constraints.
             </p>
             <div className="mt-3 flex items-center text-indigo-600 text-sm font-medium">
               <span>Optimize routes</span>
               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
               </svg>
             </div>
           </div>
           
           <div 
             onClick={handleViewCarrierAnalysis}
             className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 cursor-pointer transition"
           >
             <div className="flex items-center mb-3">
               <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                 <TruckIcon className="h-5 w-5 text-blue-600" />
               </div>
               <h3 className="text-md font-medium text-gray-800">Carrier Performance Analysis</h3>
             </div>
             <p className="text-sm text-gray-600">
               Compare and analyze carrier performance across multiple metrics to optimize your carrier selection.
             </p>
             <div className="mt-3 flex items-center text-indigo-600 text-sm font-medium">
               <span>Analyze carriers</span>
               <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
               </svg>
             </div>
           </div>
         </div>
       </div>
     </div>
   </div>
 );
};

export default LogisticsDashboard;

