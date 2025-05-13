import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ChartViewer from '../visualization/ChartViewer';
import HeatMap from '../visualization/charts/HeatMap';
import Loading from '../common/Loading';
import {
  RefreshIcon,
  FilterIcon,
  DownloadIcon,
  ExclamationCircleIcon,
  ArrowSmRightIcon
} from '@heroicons/react/outline';

const RiskVisualizer = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [riskData, setRiskData] = useState(null);
  const [riskType, setRiskType] = useState('overall');
  const [riskThreshold, setRiskThreshold] = useState('medium');
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedSupplier, setSelectedSupplier] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    fetchRiskData();
  }, [riskType, riskThreshold]);

  const fetchRiskData = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/multi-tier/risk', { params: { type: riskType, threshold: riskThreshold } });
      
      // Simulated API response with mock data
      const mockData = getMockRiskData(riskType, riskThreshold);
      
      // Simulate API delay
      setTimeout(() => {
        setRiskData(mockData);
        setIsLoading(false);
      }, 800);
    } catch (error) {
      console.error('Error fetching risk data:', error);
      setIsLoading(false);
    }
  };

  const handleCellClick = (cell) => {
    // Find the supplier that corresponds to this cell
    const supplier = riskData.suppliers.find(s => 
      s.name === cell.y && s.riskFactors[cell.x.toLowerCase()] === cell.value
    );
    
    if (supplier) {
      setSelectedSupplier(supplier);
      setShowDetails(true);
    }
  };

  const handleExportRisk = () => {
    // In a real implementation, this would call an API to export the risk data
    alert('Exporting risk analysis...');
  };

  const handleViewMitigationPlan = (supplier) => {
    // In a real implementation, this would navigate to a mitigation plan page
    navigate(`/multi-tier/risk/mitigation/${supplier.id}`);
  };

  // Mock data generator function
  const getMockRiskData = (type, threshold) => {
    const riskFactors = ['Financial', 'Operational', 'Geographic', 'Compliance', 'Reputational'];
    
    // Generate 10 suppliers with risk profiles
    const suppliers = Array(10).fill().map((_, index) => {
      const name = `Supplier ${String.fromCharCode(65 + index)}`; // A, B, C, ...
      
      // Generate random risk factors
      const factors = {};
      riskFactors.forEach(factor => {
        factors[factor.toLowerCase()] = Math.floor(Math.random() * 100) + 1;
      });
      
      // Calculate overall risk based on the highest risk factor
      const overallRisk = Math.max(...Object.values(factors));
      
      return {
        id: `s-${index + 1}`,
        name,
        tier: Math.floor(Math.random() * 3) + 1, // Random tier 1-3
        category: ['Electronics', 'Raw Materials', 'Chemicals', 'Packaging', 'Mechanical Parts'][Math.floor(Math.random() * 5)],
        riskFactors: factors,
        overallRisk,
        riskLevel: overallRisk >= 70 ? 'high' : overallRisk >= 40 ? 'medium' : 'low',
        alerts: Math.floor(Math.random() * 5) // 0-4 random alerts
      };
    });
    
    // Filter suppliers based on threshold
    const thresholdValue = threshold === 'high' ? 70 : threshold === 'medium' ? 40 : 0;
    const filteredSuppliers = suppliers.filter(s => s.overallRisk >= thresholdValue);
    
    // Create heatmap data
    const heatmapData = [];
    filteredSuppliers.forEach(supplier => {
      riskFactors.forEach(factor => {
        heatmapData.push({
          x: factor,
          y: supplier.name,
          value: supplier.riskFactors[factor.toLowerCase()]
        });
      });
    });
    
    // Create risk distribution data
    const riskDistribution = [
      { category: 'Low Risk', count: suppliers.filter(s => s.riskLevel === 'low').length },
      { category: 'Medium Risk', count: suppliers.filter(s => s.riskLevel === 'medium').length },
      { category: 'High Risk', count: suppliers.filter(s => s.riskLevel === 'high').length }
    ];
    
    // Create risk by tier data
    const riskByTier = [
      { tier: 'Tier 1', low: 0, medium: 0, high: 0 },
      { tier: 'Tier 2', low: 0, medium: 0, high: 0 },
      { tier: 'Tier 3', low: 0, medium: 0, high: 0 }
    ];
    
    suppliers.forEach(supplier => {
      const tierIndex = supplier.tier - 1;
      riskByTier[tierIndex][supplier.riskLevel]++;
    });
    
    // Create risk by category data
    const categories = [...new Set(suppliers.map(s => s.category))];
    const riskByCategory = categories.map(category => {
      const suppliersInCategory = suppliers.filter(s => s.category === category);
      return {
        category,
        avgRisk: Math.round(suppliersInCategory.reduce((sum, s) => sum + s.overallRisk, 0) / suppliersInCategory.length)
      };
    });
    
    return {
      suppliers: filteredSuppliers,
      summary: {
        totalSuppliers: suppliers.length,
        highRiskCount: suppliers.filter(s => s.riskLevel === 'high').length,
        mediumRiskCount: suppliers.filter(s => s.riskLevel === 'medium').length,
        lowRiskCount: suppliers.filter(s => s.riskLevel === 'low').length,
        alertsCount: suppliers.reduce((sum, s) => sum + s.alerts, 0)
      },
      heatmapData,
      riskDistribution,
      riskByTier,
      riskByCategory
    };
  };

  // Format the heatmap data for the HeatMap component
  const formatHeatmapData = (data) => {
    if (!data) return null;
    
    return {
      type: 'heatmap',
      title: 'Supplier Risk Heatmap',
      data: data.heatmapData,
      config: {
        xKey: 'x',
        yKey: 'y',
        valueKey: 'value',
        colorScheme: 'interpolateReds',
        showValues: true,
        valueFormatter: (value) => `${value}`
      }
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading risk visualization..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full">
      {/* Risk Visualizer Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">Supply Chain Risk Visualization</h1>
          <p className="text-sm text-gray-500 mt-1">Identify and analyze risks across your supply chain</p>
        </div>
        
        <div className="flex items-center space-x-3 mt-3 sm:mt-0">
          <div className="relative">
            <button
              onClick={() => setFilterOpen(!filterOpen)}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <FilterIcon className="-ml-1 mr-2 h-4 w-4" />
              Filter
            </button>
            
            {filterOpen && (
              <div className="origin-top-right absolute right-0 mt-2 w-56 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                <div className="py-1 px-3">
                  <div className="mb-3">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Risk Type</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'overall', label: 'Overall Risk' },
                        { id: 'financial', label: 'Financial Risk' },
                        { id: 'operational', label: 'Operational Risk' },
                        { id: 'geographic', label: 'Geographic Risk' },
                        { id: 'compliance', label: 'Compliance Risk' },
                        { id: 'reputational', label: 'Reputational Risk' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="riskType"
                            value={option.id}
                            checked={riskType === option.id}
                            onChange={() => setRiskType(option.id)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{option.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Risk Threshold</h4>
                    <div className="mt-1 space-y-1">
                      {[
                        { id: 'all', label: 'Show All' },
                        { id: 'medium', label: 'Medium & High Risk' },
                        { id: 'high', label: 'High Risk Only' }
                      ].map(option => (
                        <label key={option.id} className="flex items-center">
                          <input
                            type="radio"
                            name="riskThreshold"
                            value={option.id}
                            checked={riskThreshold === option.id}
                            onChange={() => setRiskThreshold(option.id)}
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
            onClick={fetchRiskData}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <RefreshIcon className="-ml-1 mr-2 h-4 w-4" />
            Refresh
          </button>
          
          <button
            onClick={handleExportRisk}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <DownloadIcon className="-ml-1 mr-2 h-4 w-4" />
            Export
          </button>
        </div>
      </div>
      
      {/* Risk Visualizer Content */}
      <div className="container mx-auto px-4 py-6">
        {/* Risk Summary */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-5 mb-6">
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Total Suppliers</p>
                <p className="text-2xl font-bold text-gray-900">{riskData.summary.totalSuppliers}</p>
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
                <p className="text-sm font-medium text-gray-500">High Risk Suppliers</p>
                <p className="text-2xl font-bold text-red-600">{riskData.summary.highRiskCount}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round(riskData.summary.highRiskCount / riskData.summary.totalSuppliers * 100)}% of total
                </p>
              </div>
              <div className="h-12 w-12 rounded-full bg-red-100 flex items-center justify-center">
                <ExclamationCircleIcon className="h-6 w-6 text-red-600" />
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Medium Risk Suppliers</p>
                <p className="text-2xl font-bold text-yellow-600">{riskData.summary.mediumRiskCount}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round(riskData.summary.mediumRiskCount / riskData.summary.totalSuppliers * 100)}% of total
                </p>
              </div>
              <div className="h-12 w-12 rounded-full bg-yellow-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-5">
            <div className="flex justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500">Low Risk Suppliers</p>
                <p className="text-2xl font-bold text-green-600">{riskData.summary.lowRiskCount}</p>
                <p className="text-xs text-gray-500 mt-1">
                  {Math.round(riskData.summary.lowRiskCount / riskData.summary.totalSuppliers * 100)}% of total
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
                <p className="text-sm font-medium text-gray-500">Total Alerts</p>
                <p className="text-2xl font-bold text-purple-600">{riskData.summary.alertsCount}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Last 30 days
                </p>
              </div>
              <div className="h-12 w-12 rounded-full bg-purple-100 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
            </div>
          </div>
        </div>
        
        {/* Risk Heatmap */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6">
          <h2 className="text-lg font-medium text-gray-800 mb-4">Supplier Risk Heatmap</h2>
           <div className="w-full h-96">
             <HeatMap 
               data={riskData.heatmapData} 
               config={{
                 xKey: 'x',
                 yKey: 'y',
                 valueKey: 'value',
                 colorScheme: 'interpolateReds',
                 showValues: true,
                 valueFormatter: (value) => `${value}`
               }}
               height={350}
               onCellClick={handleCellClick}
             />
           </div>
         </div>
       </div>
       
       {/* Risk Charts */}
       <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
         {/* Risk Distribution Chart */}
         <ChartViewer 
           chartData={{
             type: 'pie',
             title: 'Risk Distribution',
             data: riskData.riskDistribution,
             config: {
               nameKey: 'category',
               valueKey: 'count',
               innerRadius: 0.6,
               colorScheme: 'schemeReds'
             }
           }}
         />
         
         {/* Risk by Category Chart */}
         <ChartViewer 
           chartData={{
             type: 'bar',
             title: 'Average Risk by Category',
             data: riskData.riskByCategory,
             config: {
               xKey: 'category',
               yKey: 'avgRisk',
               color: '#e11d48',
               valueFormatter: (value) => `${value}`
             }
           }}
         />
       </div>
       
       {/* Risk by Tier */}
       <div className="bg-white rounded-lg shadow mb-6">
         <div className="px-6 py-4 border-b border-gray-200">
           <h2 className="text-lg font-medium text-gray-800">Risk by Tier</h2>
         </div>
         <div className="p-6">
           <div className="w-full">
             <div className="flex mb-2">
               <div className="w-1/5 text-sm font-medium text-gray-600 py-2"></div>
               <div className="w-1/5 text-sm font-medium text-gray-600 py-2 text-center">Low Risk</div>
               <div className="w-1/5 text-sm font-medium text-gray-600 py-2 text-center">Medium Risk</div>
               <div className="w-1/5 text-sm font-medium text-gray-600 py-2 text-center">High Risk</div>
               <div className="w-1/5 text-sm font-medium text-gray-600 py-2 text-center">Total</div>
             </div>
             {riskData.riskByTier.map((tier, index) => (
               <div key={index} className={`flex ${index < riskData.riskByTier.length - 1 ? 'border-b border-gray-200' : ''}`}>
                 <div className="w-1/5 text-sm font-medium text-gray-800 py-3">{tier.tier}</div>
                 <div className="w-1/5 py-3 text-center">
                   <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                     {tier.low}
                   </span>
                 </div>
                 <div className="w-1/5 py-3 text-center">
                   <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                     {tier.medium}
                   </span>
                 </div>
                 <div className="w-1/5 py-3 text-center">
                   <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                     {tier.high}
                   </span>
                 </div>
                 <div className="w-1/5 py-3 text-center text-sm font-medium text-gray-800">
                   {tier.low + tier.medium + tier.high}
                 </div>
               </div>
             ))}
           </div>
         </div>
       </div>
       
       {/* High Risk Suppliers */}
       <div className="bg-white rounded-lg shadow mb-6">
         <div className="px-6 py-4 border-b border-gray-200">
           <h2 className="text-lg font-medium text-gray-800">High Risk Suppliers</h2>
         </div>
         <div className="overflow-x-auto">
           <table className="min-w-full divide-y divide-gray-200">
             <thead className="bg-gray-50">
               <tr>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Supplier
                 </th>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Tier
                 </th>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Category
                 </th>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Risk Score
                 </th>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Risk Level
                 </th>
                 <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Alerts
                 </th>
                 <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                   Actions
                 </th>
               </tr>
             </thead>
             <tbody className="bg-white divide-y divide-gray-200">
               {riskData.suppliers
                 .filter(supplier => supplier.riskLevel === 'high')
                 .map((supplier) => (
                 <tr key={supplier.id}>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <div className="text-sm font-medium text-gray-900">{supplier.name}</div>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <div className="text-sm text-gray-500">Tier {supplier.tier}</div>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <div className="text-sm text-gray-500">{supplier.category}</div>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <div className="text-sm font-medium text-gray-900">{supplier.overallRisk}</div>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                       High
                     </span>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap">
                     <div className="text-sm text-gray-500">{supplier.alerts}</div>
                   </td>
                   <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                     <button 
                       onClick={() => {
                         setSelectedSupplier(supplier);
                         setShowDetails(true);
                       }}
                       className="text-indigo-600 hover:text-indigo-900"
                     >
                       View Details
                     </button>
                   </td>
                 </tr>
               ))}
             </tbody>
           </table>
         </div>
       </div>
     </div>
     
     {/* Supplier Risk Detail Modal */}
     {showDetails && selectedSupplier && (
       <div className="fixed inset-0 overflow-y-auto z-50">
         <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
           <div className="fixed inset-0 transition-opacity" aria-hidden="true">
             <div className="absolute inset-0 bg-gray-500 opacity-75" onClick={() => setShowDetails(false)}></div>
           </div>

           <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

           <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl sm:w-full sm:p-6">
             <div className="sm:flex sm:items-start">
               <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-red-100 sm:mx-0 sm:h-10 sm:w-10">
                 <ExclamationCircleIcon className="h-6 w-6 text-red-600" aria-hidden="true" />
               </div>
               <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                 <h3 className="text-lg leading-6 font-medium text-gray-900">{selectedSupplier.name} Risk Profile</h3>
                 <div className="mt-2">
                   <p className="text-sm text-gray-500">
                     This supplier has been identified as high risk. Review the risk breakdown below and create a mitigation plan.
                   </p>
                 </div>
               </div>
             </div>
             
             <div className="mt-6">
               <div className="bg-gray-50 p-4 rounded-lg mb-6">
                 <h4 className="text-sm font-medium text-gray-700 mb-2">Supplier Information</h4>
                 <div className="grid grid-cols-3 gap-4">
                   <div>
                     <p className="text-xs text-gray-500">Tier</p>
                     <p className="text-sm font-medium text-gray-900">Tier {selectedSupplier.tier}</p>
                   </div>
                   <div>
                     <p className="text-xs text-gray-500">Category</p>
                     <p className="text-sm font-medium text-gray-900">{selectedSupplier.category}</p>
                   </div>
                   <div>
                     <p className="text-xs text-gray-500">Risk Level</p>
                     <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                       High Risk ({selectedSupplier.overallRisk}/100)
                     </span>
                   </div>
                 </div>
               </div>
               
               <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Breakdown</h4>
               <div className="space-y-3 mb-6">
                 {Object.entries(selectedSupplier.riskFactors).map(([factor, value]) => (
                   <div key={factor} className="flex items-center">
                     <span className="text-sm font-medium text-gray-700 w-32 flex-shrink-0 capitalize">{factor}</span>
                     <div className="flex-1 h-4 bg-gray-200 rounded-full overflow-hidden">
                       <div 
                         className={`h-4 rounded-full ${
                           value >= 70 ? 'bg-red-500' : 
                           value >= 40 ? 'bg-yellow-500' : 
                           'bg-green-500'
                         }`} 
                         style={{ width: `${value}%` }}
                       ></div>
                     </div>
                     <span className="text-sm text-gray-700 w-12 text-right">{value}/100</span>
                   </div>
                 ))}
               </div>
               
               <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Alerts</h4>
               <div className="bg-red-50 p-4 rounded-lg mb-6">
                 <div className="space-y-3">
                   {Array(selectedSupplier.alerts).fill().map((_, index) => {
                     const alertTypes = [
                       'Missed delivery deadline by 7+ days',
                       'Financial stability rating decreased',
                       'New compliance issue reported',
                       'Political unrest in manufacturing region',
                       'Quality control failures above threshold'
                     ];
                     return (
                       <div key={index} className="flex items-start">
                         <div className="flex-shrink-0">
                           <ExclamationCircleIcon className="h-5 w-5 text-red-400" aria-hidden="true" />
                         </div>
                         <div className="ml-3">
                           <p className="text-sm text-red-700">
                             {alertTypes[index % alertTypes.length]}
                           </p>
                           <p className="text-xs text-red-500 mt-1">
                             {new Date(Date.now() - (index * 1000 * 60 * 60 * 24 * Math.floor(Math.random() * 10))).toLocaleDateString()}
                           </p>
                         </div>
                       </div>
                     );
                   })}
                   {selectedSupplier.alerts === 0 && (
                     <p className="text-sm text-gray-500">No current alerts for this supplier.</p>
                   )}
                 </div>
               </div>
               
               <h4 className="text-sm font-medium text-gray-700 mb-2">Recommended Actions</h4>
               <div className="bg-indigo-50 p-4 rounded-lg mb-6">
                 <div className="space-y-3">
                   <div className="flex items-start">
                     <div className="flex-shrink-0">
                       <ArrowSmRightIcon className="h-5 w-5 text-indigo-400" aria-hidden="true" />
                     </div>
                     <div className="ml-3">
                       <p className="text-sm text-indigo-700">
                         Create risk mitigation plan with supplier
                       </p>
                     </div>
                   </div>
                   <div className="flex items-start">
                     <div className="flex-shrink-0">
                       <ArrowSmRightIcon className="h-5 w-5 text-indigo-400" aria-hidden="true" />
                     </div>
                     <div className="ml-3">
                       <p className="text-sm text-indigo-700">
                         Schedule risk assessment meeting
                       </p>
                     </div>
                   </div>
                   <div className="flex items-start">
                     <div className="flex-shrink-0">
                       <ArrowSmRightIcon className="h-5 w-5 text-indigo-400" aria-hidden="true" />
                     </div>
                     <div className="ml-3">
                       <p className="text-sm text-indigo-700">
                         Identify alternative suppliers for critical components
                       </p>
                     </div>
                   </div>
                 </div>
               </div>
             </div>
             
             <div className="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
               <button
                 type="button"
                 className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-indigo-600 text-base font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:ml-3 sm:w-auto sm:text-sm"
                 onClick={() => handleViewMitigationPlan(selectedSupplier)}
               >
                 Create Mitigation Plan
               </button>
               <button
                 type="button"
                 className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:w-auto sm:text-sm"
                 onClick={() => setShowDetails(false)}
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

export default RiskVisualizer;