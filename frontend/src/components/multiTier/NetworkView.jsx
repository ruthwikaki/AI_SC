import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Loading from '../common/Loading';
import NetworkGraph from '../visualization/charts/NetworkGraph';
import {
  ArrowPathIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  AdjustmentsHorizontalIcon,
  ArrowDownTrayIcon,
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon,
  ListBulletIcon
} from '@heroicons/react/24/outline';

const NetworkView = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [networkData, setNetworkData] = useState(null);
  const [showTier, setShowTier] = useState({
    tier1: true,
    tier2: true,
    tier3: true
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [filterOpen, setFilterOpen] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedSupplier, setSelectedSupplier] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  
  const networkRef = useRef(null);

  useEffect(() => {
    fetchNetworkData();
    // Get available categories for filtering
    fetchCategories();
  }, [showTier, selectedCategories]);

  const fetchNetworkData = async () => {
    setIsLoading(true);
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/multi-tier/network', 
      //   { params: { tiers: Object.entries(showTier).filter(([_, v]) => v).map(([k]) => k), categories: selectedCategories } });
      
      // Simulated API response with mock data
      const mockData = getMockNetworkData(showTier, selectedCategories);
      
      // Simulate API delay
      setTimeout(() => {
        setNetworkData(mockData);
        setIsLoading(false);
      }, 800);
    } catch (error) {
      console.error('Error fetching network data:', error);
      setIsLoading(false);
    }
  };

  const fetchCategories = async () => {
    try {
      // Simulated categories
      setCategories(['Electronics', 'Raw Materials', 'Chemicals', 'Packaging', 'Mechanical Parts']);
    } catch (error) {
      console.error('Error fetching categories:', error);
    }
  };

  const toggleCategory = (category) => {
    setSelectedCategories(prev => 
      prev.includes(category) 
        ? prev.filter(c => c !== category) 
        : [...prev, category]
    );
  };

  const toggleTier = (tier) => {
    setShowTier(prev => ({
      ...prev,
      [tier]: !prev[tier]
    }));
  };

  const handleNodeClick = (node) => {
    setSelectedSupplier(node);
  };

  const handleViewSupplierDetails = () => {
    if (selectedSupplier) {
      navigate(`/suppliers/${selectedSupplier.id}`);
    }
  };

  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev + 0.2, 2.0));
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev - 0.2, 0.5));
  };

  const handleResetView = () => {
    setZoomLevel(1);
    // In a real implementation, you would reset the network view
    // if you're using a library like D3 or react-force-graph
  };

  const handleExportNetwork = () => {
    // In a real implementation, this would call an API to export the network visualization
    alert('Exporting network visualization...');
  };

  // Mock data generator function
  const getMockNetworkData = (tiers, categories) => {
    // Create nodes and links based on the selected tiers and categories
    const nodes = [];
    const links = [];
    
    // Add your company as the central node
    nodes.push({
      id: 'company',
      name: 'Your Company',
      type: 'company',
      tier: 0,
      value: 50,
      risk: 'low'
    });
    
    // Add tier 1 suppliers
    if (tiers.tier1) {
      const tier1Suppliers = [
        { id: 't1-1', name: 'Supplier A', type: 'supplier', category: 'Electronics', tier: 1, value: 35, risk: 'low' },
        { id: 't1-2', name: 'Supplier B', type: 'supplier', category: 'Raw Materials', tier: 1, value: 30, risk: 'low' },
        { id: 't1-3', name: 'Supplier C', type: 'supplier', category: 'Chemicals', tier: 1, value: 25, risk: 'medium' },
        { id: 't1-4', name: 'Supplier D', type: 'supplier', category: 'Packaging', tier: 1, value: 20, risk: 'low' },
        { id: 't1-5', name: 'Supplier E', type: 'supplier', category: 'Mechanical Parts', tier: 1, value: 28, risk: 'low' }
      ];
      
      // Filter by categories if any are selected
      const filteredTier1 = categories.length > 0 
        ? tier1Suppliers.filter(s => categories.includes(s.category))
        : tier1Suppliers;
      
      nodes.push(...filteredTier1);
      
      // Add links from your company to tier 1 suppliers
      filteredTier1.forEach(supplier => {
        links.push({
          source: 'company',
          target: supplier.id,
          value: supplier.value / 2
        });
      });
      
      // Add tier 2 suppliers if tier 1 is selected
      if (tiers.tier2) {
        const tier2Suppliers = [
          { id: 't2-1', name: 'Supplier F', type: 'supplier', category: 'Electronics', tier: 2, value: 18, risk: 'medium' },
          { id: 't2-2', name: 'Supplier G', type: 'supplier', category: 'Raw Materials', tier: 2, value: 15, risk: 'high' },
          { id: 't2-3', name: 'Supplier H', type: 'supplier', category: 'Chemicals', tier: 2, value: 20, risk: 'low' },
          { id: 't2-4', name: 'Supplier I', type: 'supplier', category: 'Raw Materials', tier: 2, value: 16, risk: 'medium' },
          { id: 't2-5', name: 'Supplier J', type: 'supplier', category: 'Mechanical Parts', tier: 2, value: 22, risk: 'low' },
          { id: 't2-6', name: 'Supplier K', type: 'supplier', category: 'Electronics', tier: 2, value: 14, risk: 'low' },
          { id: 't2-7', name: 'Supplier L', type: 'supplier', category: 'Packaging', tier: 2, value: 12, risk: 'medium' }
        ];
        
        // Filter by categories if any are selected
        const filteredTier2 = categories.length > 0 
          ? tier2Suppliers.filter(s => categories.includes(s.category))
          : tier2Suppliers;
        
        nodes.push(...filteredTier2);
        
        // Add links from tier 1 to tier 2 suppliers
        const tier1ToTier2Links = [
          { source: 't1-1', target: 't2-1', value: 15 },
          { source: 't1-1', target: 't2-6', value: 12 },
          { source: 't1-2', target: 't2-2', value: 14 },
          { source: 't1-2', target: 't2-4', value: 13 },
          { source: 't1-3', target: 't2-3', value: 16 },
          { source: 't1-4', target: 't2-7', value: 10 },
          { source: 't1-5', target: 't2-5', value: 18 }
        ];
        
        // Filter links based on filtered tier 1 and tier 2 suppliers
        const filteredTier1Ids = filteredTier1.map(s => s.id);
        const filteredTier2Ids = filteredTier2.map(s => s.id);
        
        const filteredLinks = tier1ToTier2Links.filter(link => 
          filteredTier1Ids.includes(link.source) && filteredTier2Ids.includes(link.target)
        );
        
        links.push(...filteredLinks);
        
        // Add tier 3 suppliers if tier 2 is selected
        if (tiers.tier3) {
          const tier3Suppliers = [
            { id: 't3-1', name: 'Supplier M', type: 'supplier', category: 'Raw Materials', tier: 3, value: 10, risk: 'medium' },
            { id: 't3-2', name: 'Supplier N', type: 'supplier', category: 'Chemicals', tier: 3, value: 8, risk: 'high' },
            { id: 't3-3', name: 'Supplier O', type: 'supplier', category: 'Electronics', tier: 3, value: 12, risk: 'low' },
            { id: 't3-4', name: 'Supplier P', type: 'supplier', category: 'Raw Materials', tier: 3, value: 7, risk: 'high' },
            { id: 't3-5', name: 'Supplier Q', type: 'supplier', category: 'Packaging', tier: 3, value: 9, risk: 'medium' }
          ];
          
          // Filter by categories if any are selected
          const filteredTier3 = categories.length > 0 
            ? tier3Suppliers.filter(s => categories.includes(s.category))
            : tier3Suppliers;
          
          nodes.push(...filteredTier3);
          
          // Add links from tier 2 to tier 3 suppliers
          const tier2ToTier3Links = [
            { source: 't2-1', target: 't3-3', value: 8 },
            { source: 't2-2', target: 't3-1', value: 9 },
            { source: 't2-2', target: 't3-4', value: 6 },
            { source: 't2-3', target: 't3-2', value: 7 },
            { source: 't2-7', target: 't3-5', value: 7 }
          ];
          
          // Filter links based on filtered tier 2 and tier 3 suppliers
          const filteredTier3Ids = filteredTier3.map(s => s.id);
          
          const filteredTier3Links = tier2ToTier3Links.filter(link => 
            filteredTier2Ids.includes(link.source) && filteredTier3Ids.includes(link.target)
          );
          
          links.push(...filteredTier3Links);
        }
      }
    }
    
    // Filter nodes by search term if provided
    const searchFilteredNodes = searchTerm 
      ? nodes.filter(node => node.name.toLowerCase().includes(searchTerm.toLowerCase()))
      : nodes;
    
    // Filter links to include only those connected to our filtered nodes
    const nodeIds = searchFilteredNodes.map(node => node.id);
    const searchFilteredLinks = links.filter(link => 
      nodeIds.includes(link.source) && nodeIds.includes(link.target)
    );
    
    return {
      nodes: searchFilteredNodes,
      links: searchFilteredLinks,
      summary: {
        totalSuppliers: nodes.length - 1, // Exclude your company
        tier1Count: tiers.tier1 ? 5 : 0,
        tier2Count: tiers.tier1 && tiers.tier2 ? 7 : 0,
        tier3Count: tiers.tier1 && tiers.tier2 && tiers.tier3 ? 5 : 0,
        highRiskCount: nodes.filter(node => node.risk === 'high').length,
        mediumRiskCount: nodes.filter(node => node.risk === 'medium').length
      }
    };
  };

  // Format the network data for the NetworkGraph component
  const formatNetworkData = (data) => {
    if (!data) return null;
    
    // Determine node color based on risk and type
    const getNodeColor = (node) => {
      if (node.type === 'company') return '#4f46e5'; // Company is indigo
      
      // Suppliers based on risk
      switch (node.risk) {
        case 'low': return '#10b981'; // Green
        case 'medium': return '#f59e0b'; // Amber
        case 'high': return '#ef4444'; // Red
        default: return '#6b7280'; // Gray
      }
    };
    
    // Create formatted data for NetworkGraph
    return {
      type: 'network',
      title: 'Multi-Tier Supply Chain Network',
      data: {
        nodes: data.nodes.map(node => ({
          id: node.id,
          name: node.name,
          value: node.value,
          group: node.tier === 0 ? 'Company' : `Tier ${node.tier}`,
          category: node.category || '',
          risk: node.risk || 'unknown',
          color: getNodeColor(node)
        })),
        links: data.links
      },
      config: {
        nodeSize: 'value',
        nodeSizeRange: [5, 25],
        linkWidth: 'value',
        linkWidthRange: [1, 5],
        forceStrength: -150,
        distanceMin: 100,
        distanceMax: 300,
        nodeLabels: true
      }
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading supply chain network..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full flex flex-col">
      {/* Network View Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">Supply Chain Network View</h1>
          <p className="text-sm text-gray-500 mt-1">Visualize and analyze your multi-tier supplier network</p>
        </div>
        
        <div className="flex items-center space-x-3 mt-3 sm:mt-0">
          <div className="relative">
            <div className="flex rounded-md shadow-sm">
              <div className="relative flex items-stretch flex-grow focus-within:z-10">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
                </div>
                <input
                  type="text"
                  className="focus:ring-indigo-500 focus:border-indigo-500 block w-full rounded-none rounded-l-md pl-10 sm:text-sm border-gray-300"
                  placeholder="Search suppliers..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
              <button
                type="button"
                className="-ml-px relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-r-md text-gray-700 bg-gray-50 hover:bg-gray-100 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                onClick={() => fetchNetworkData()}
              >
                Search
              </button>
            </div>
          </div>
          
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
                  <div className="mb-3">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Supplier Tiers</h4>
                    <div className="mt-1 space-y-1">
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={showTier.tier1}
                          onChange={() => toggleTier('tier1')}
                          className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">Tier 1 Suppliers</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={showTier.tier2}
                          onChange={() => toggleTier('tier2')}
                          className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">Tier 2 Suppliers</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="checkbox"
                          checked={showTier.tier3}
                          onChange={() => toggleTier('tier3')}
                          className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                        />
                        <span className="ml-2 text-sm text-gray-700">Tier 3 Suppliers</span>
                      </label>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Categories</h4>
                    <div className="mt-1 space-y-1 max-h-40 overflow-y-auto">
                      {categories.map(category => (
                        <label key={category} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={selectedCategories.includes(category)}
                            onChange={() => toggleCategory(category)}
                            className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
                          />
                          <span className="ml-2 text-sm text-gray-700">{category}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                  
                  <div className="mt-3 flex justify-end">
                    <button
                      onClick={() => {
                        fetchNetworkData();
                        setFilterOpen(false);
                      }}
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
            onClick={fetchNetworkData}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
            Refresh
          </button>
          
          <button
            onClick={handleExportNetwork}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <ArrowDownTrayIcon className="-ml-1 mr-2 h-4 w-4" />
            Export
          </button>
        </div>
      </div>
      
      {/* Network View Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Network Visualization */}
        <div className="flex-1 relative overflow-hidden">
          <div 
            ref={networkRef} 
            className="w-full h-full"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            {networkData && (
              <NetworkGraph 
                data={formatNetworkData(networkData).data}
                config={formatNetworkData(networkData).config}
                height="100%"
                onNodeClick={handleNodeClick}
              />
            )}
          </div>
          
          {/* Zoom Controls */}
          <div className="absolute bottom-4 right-4 flex flex-col bg-white rounded-lg shadow">
            <button
              onClick={handleZoomIn}
              className="p-2 hover:bg-gray-100 rounded-t-lg"
              disabled={zoomLevel >= 2.0}
            >
              <MagnifyingGlassPlusIcon className="h-5 w-5 text-gray-700" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 hover:bg-gray-100"
              disabled={zoomLevel <= 0.5}
            >
              <MagnifyingGlassMinusIcon className="h-5 w-5 text-gray-700" />
            </button>
            <button
              onClick={handleResetView}
              className="p-2 hover:bg-gray-100 rounded-b-lg"
            >
              <AdjustmentsHorizontalIcon className="h-5 w-5 text-gray-700" />
            </button>
          </div>
          
          {/* Network Summary */}
          <div className="absolute top-4 left-4 bg-white rounded-lg shadow p-4 max-w-xs">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Network Summary</h3>
            <div className="space-y-1">
              <p className="text-xs text-gray-600">
                <span className="font-medium">Total Suppliers:</span> {networkData?.summary.totalSuppliers}
              </p>
              <p className="text-xs text-gray-600">
                <span className="font-medium">Tier 1:</span> {networkData?.summary.tier1Count}
              </p>
              <p className="text-xs text-gray-600">
                <span className="font-medium">Tier 2:</span> {networkData?.summary.tier2Count}
              </p>
              <p className="text-xs text-gray-600">
                <span className="font-medium">Tier 3:</span> {networkData?.summary.tier3Count}
              </p>
              <div className="pt-1 border-t border-gray-200 mt-1">
                <p className="text-xs text-gray-600">
                  <span className="font-medium text-red-600">High Risk:</span> {networkData?.summary.highRiskCount}
                </p>
                <p className="text-xs text-gray-600">
                  <span className="font-medium text-yellow-600">Medium Risk:</span> {networkData?.summary.mediumRiskCount}
                </p>
              </div>
            </div>
          </div>
        </div>
        
        {/* Supplier Details Panel */}
        {selectedSupplier && (
          <div className="w-80 border-l border-gray-200 bg-white p-4 overflow-y-auto">
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-lg font-medium text-gray-900">{selectedSupplier.name}</h3>
              <button
                onClick={() => setSelectedSupplier(null)}
                className="text-gray-400 hover:text-gray-500"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
            
            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-500">Tier</span>
                <span className="text-sm text-gray-900">{selectedSupplier.group}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-500">Category</span>
                <span className="text-sm text-gray-900">{selectedSupplier.category || 'N/A'}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-500">Risk Level</span>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                  ${selectedSupplier.risk === 'low' ? 'bg-green-100 text-green-800' : 
                    selectedSupplier.risk === 'medium' ? 'bg-yellow-100 text-yellow-800' : 
                    'bg-red-100 text-red-800'}`}>
                  {selectedSupplier.risk === 'low' ? 'Low' : 
                    selectedSupplier.risk === 'medium' ? 'Medium' : 'High'}
                </span>
              </div>
            </div>
            
            <div className="border-t border-gray-200 pt-4 mb-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Key Metrics</h4>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-xs text-gray-500 mb-1">On-Time Delivery</div>
                  <div className="text-base font-semibold text-gray-900">
                    {94 + Math.floor(Math.random() * 6)}%
                  </div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-xs text-gray-500 mb-1">Quality Score</div>
                  <div className="text-base font-semibold text-gray-900">
                    {90 + Math.floor(Math.random() * 10)}%
                  </div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-xs text-gray-500 mb-1">Response Time</div>
                  <div className="text-base font-semibold text-gray-900">
                    {1 + Math.floor(Math.random() * 3)} days
                  </div>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <div className="text-xs text-gray-500 mb-1">Relationship</div>
                  <div className="text-base font-semibold text-gray-900">
                    {2 + Math.floor(Math.random() * 8)} years
                  </div>
                </div>
              </div>
            </div>
            
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Risk Factors</h4>
              <div className="space-y-2">
                <div className="flex items-center">
                  <div className={`h-2 flex-1 rounded-full ${
                    selectedSupplier.risk === 'low' ? 'bg-green-200' : 
                    selectedSupplier.risk === 'medium' ? 'bg-yellow-200' : 
                    'bg-red-200'
                  }`}>
                    <div 
                      className={`h-2 rounded-full ${
                        selectedSupplier.risk === 'low' ? 'bg-green-500' : 
                        selectedSupplier.risk === 'medium' ? 'bg-yellow-500' : 
                        'bg-red-500'
                      }`}
                      style={{ width: `${selectedSupplier.risk === 'low' ? 20 : selectedSupplier.risk === 'medium' ? 50 : 80}%` }}
                    />
                  </div>
                  <span className="ml-2 text-xs text-gray-500 w-20">Financial</span>
                </div>
                <div className="flex items-center">
                  <div className={`h-2 flex-1 rounded-full ${
                    selectedSupplier.risk === 'low' ? 'bg-green-200' : 
                    selectedSupplier.risk === 'medium' ? 'bg-yellow-200' : 
                    'bg-red-200'
                  }`}>
                    <div 
                      className={`h-2 rounded-full ${
                        selectedSupplier.risk === 'low' ? 'bg-green-500' : 
                        selectedSupplier.risk === 'medium' ? 'bg-yellow-500' : 
                        'bg-red-500'
                      }`}
                      style={{ width: `${selectedSupplier.risk === 'low' ? 15 : selectedSupplier.risk === 'medium' ? 45 : 75}%` }}
                    />
                  </div>
                  <span className="ml-2 text-xs text-gray-500 w-20">Operations</span>
                </div>
                <div className="flex items-center">
                  <div className={`h-2 flex-1 rounded-full ${
                    selectedSupplier.risk === 'low' ? 'bg-green-200' : 
                    selectedSupplier.risk === 'medium' ? 'bg-yellow-200' : 
                    'bg-red-200'
                  }`}>
                    <div 
                      className={`h-2 rounded-full ${
                        selectedSupplier.risk === 'low' ? 'bg-green-500' : 
                        selectedSupplier.risk === 'medium' ? 'bg-yellow-500' : 
                        'bg-red-500'
                      }`}
                      style={{ width: `${selectedSupplier.risk === 'low' ? 10 : selectedSupplier.risk === 'medium' ? 40 : 70}%` }}
                    />
                  </div>
                  <span className="ml-2 text-xs text-gray-500 w-20">Compliance</span>
                </div>
                <div className="flex items-center">
                  <div className={`h-2 flex-1 rounded-full ${
                    selectedSupplier.risk === 'low' ? 'bg-green-200' : 
                    selectedSupplier.risk === 'medium' ? 'bg-yellow-200' : 
                    'bg-red-200'
                  }`}>
                    <div 
                      className={`h-2 rounded-full ${
                        selectedSupplier.risk === 'low' ? 'bg-green-500' : 
                        selectedSupplier.risk === 'medium' ? 'bg-yellow-500' : 
                        'bg-red-500'
                      }`}
                      style={{ width: `${selectedSupplier.risk === 'low' ? 25 : selectedSupplier.risk === 'medium' ? 55 : 85}%` }}
                    />
                  </div>
                  <span className="ml-2 text-xs text-gray-500 w-20">Geographic</span>
                </div>
              </div>
            </div>
            
            <div className="flex flex-col space-y-2">
              <button
                onClick={handleViewSupplierDetails}
                className="inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700"
              >
                View Detailed Profile
              </button>
              <button
                className="inline-flex justify-center items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50"
              >
                <ListBulletIcon className="-ml-1 mr-2 h-4 w-4" />
                View Connected Suppliers
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default NetworkView;



