import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import Navbar from '../components/common/Navbar';
import Sidebar from '../components/common/Sidebar';
import Loading from '../components/common/Loading';
import NetworkView from '../components/multiTier/NetworkView';
import RiskVisualizer from '../components/multiTier/RiskVisualizer';
import ScenarioSimulator from '../components/multiTier/ScenarioSimulator';

const MultiTier = () => {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState('network');
  const [loading, setLoading] = useState(true);
  const [networkData, setNetworkData] = useState(null);
  const [riskData, setRiskData] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [filters, setFilters] = useState({
    tier: 'all', // all, tier1, tier2, tier3
    category: 'all', // all, raw, component, packaging
    risk: 'all', // all, high, medium, low
  });

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
      return;
    }

    // Fetch network data
    const fetchNetworkData = async () => {
      try {
        setLoading(true);
        
        // This would be an API call in a real implementation
        // Simulating API delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock network data
        const mockNetworkData = {
          nodes: [
            { id: 'company', label: 'Your Company', type: 'company', tier: 0 },
            // Tier 1 suppliers
            { id: 'supplier1', label: 'ABC Manufacturing', type: 'supplier', tier: 1, category: 'raw', risk: 'low' },
            { id: 'supplier2', label: 'XYZ Components', type: 'supplier', tier: 1, category: 'component', risk: 'medium' },
            { id: 'supplier3', label: 'Global Packaging', type: 'supplier', tier: 1, category: 'packaging', risk: 'low' },
            // Tier 2 suppliers
            { id: 'supplier4', label: 'Metal Works Inc', type: 'supplier', tier: 2, category: 'raw', risk: 'high' },
            { id: 'supplier5', label: 'Polymer Solutions', type: 'supplier', tier: 2, category: 'raw', risk: 'medium' },
            { id: 'supplier6', label: 'Circuit Systems', type: 'supplier', tier: 2, category: 'component', risk: 'low' },
            // Tier 3 suppliers
            { id: 'supplier7', label: 'Basic Materials Co', type: 'supplier', tier: 3, category: 'raw', risk: 'high' },
            { id: 'supplier8', label: 'Mineral Extraction', type: 'supplier', tier: 3, category: 'raw', risk: 'high' }
          ],
          links: [
            // Tier 1 links
            { source: 'supplier1', target: 'company', value: 1.5 },
            { source: 'supplier2', target: 'company', value: 2.3 },
            { source: 'supplier3', target: 'company', value: 1.0 },
            // Tier 2 links
            { source: 'supplier4', target: 'supplier1', value: 1.2 },
            { source: 'supplier5', target: 'supplier1', value: 0.8 },
            { source: 'supplier5', target: 'supplier2', value: 1.1 },
            { source: 'supplier6', target: 'supplier2', value: 1.7 },
            // Tier 3 links
            { source: 'supplier7', target: 'supplier4', value: 0.9 },
            { source: 'supplier8', target: 'supplier4', value: 0.7 },
            { source: 'supplier7', target: 'supplier5', value: 1.1 }
          ]
        };
        
        // Mock risk data
        const mockRiskData = {
          riskDistribution: {
            high: 3,
            medium: 2,
            low: 3
          },
          categories: {
            raw: { high: 3, medium: 1, low: 0 },
            component: { high: 0, medium: 1, low: 1 },
            packaging: { high: 0, medium: 0, low: 1 }
          },
          tiers: {
            tier1: { high: 0, medium: 1, low: 2 },
            tier2: { high: 1, medium: 1, low: 1 },
            tier3: { high: 2, medium: 0, low: 0 }
          },
          highRiskSuppliers: [
            { id: 'supplier4', name: 'Metal Works Inc', tier: 2, category: 'raw', score: 78, factors: ['Geopolitical instability', 'Single sourcing', 'Financial stability'] },
            { id: 'supplier7', name: 'Basic Materials Co', tier: 3, category: 'raw', score: 82, factors: ['Environmental compliance', 'Labor practices', 'Capacity constraints'] },
            { id: 'supplier8', name: 'Mineral Extraction', tier: 3, category: 'raw', score: 85, factors: ['Regulatory issues', 'Environmental impact', 'Political stability'] }
          ]
        };
        
        // Mock scenarios
        const mockScenarios = [
          { id: 1, name: 'Tier 2 Raw Material Disruption', description: 'Disruption at Metal Works Inc affecting raw material supply', supplier: 'supplier4', duration: 30, impactLevel: 'high' },
          { id: 2, name: 'Logistics Disruption', description: 'Port congestion affecting multiple tier 1 suppliers', suppliers: ['supplier1', 'supplier3'], duration: 15, impactLevel: 'medium' },
          { id: 3, name: 'Multiple Tier 3 Disruptions', description: 'Simultaneous disruptions at two tier 3 suppliers', suppliers: ['supplier7', 'supplier8'], duration: 45, impactLevel: 'critical' }
        ];
        
        setNetworkData(mockNetworkData);
        setRiskData(mockRiskData);
        setScenarios(mockScenarios);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching network data:', error);
        setLoading(false);
      }
    };

    if (isAuthenticated) {
      fetchNetworkData();
    }
  }, [isAuthenticated, authLoading, navigate]);

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({ ...prev, [filterType]: value }));
  };

  const applyFilters = (data) => {
    if (!data) return null;
    
    let filteredNodes = [...data.nodes];
    
    // Apply tier filter
    if (filters.tier !== 'all') {
      const tierValue = parseInt(filters.tier.replace('tier', ''));
      filteredNodes = filteredNodes.filter(node => node.tier === tierValue || node.tier === 0); // Keep company node
    }
    
    // Apply category filter
    if (filters.category !== 'all') {
      filteredNodes = filteredNodes.filter(node => node.category === filters.category || node.type === 'company');
    }
    
    // Apply risk filter
    if (filters.risk !== 'all') {
      filteredNodes = filteredNodes.filter(node => node.risk === filters.risk || node.type === 'company');
    }
    
    // Filter links to only include connections between filtered nodes
    const filteredNodeIds = filteredNodes.map(node => node.id);
    const filteredLinks = data.links.filter(link => 
      filteredNodeIds.includes(link.source) && filteredNodeIds.includes(link.target)
    );
    
    return {
      nodes: filteredNodes,
      links: filteredLinks
    };
  };

  if (authLoading) {
    return <Loading type="overlay" message="Authenticating..." />;
  }

  const filteredNetworkData = applyFilters(networkData);

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto">
          <div className="px-6 py-4 bg-white border-b">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between">
              <div className="mb-4 md:mb-0">
                <h1 className="text-2xl font-semibold text-gray-800">Multi-Tier Supply Chain</h1>
                <p className="text-gray-600">
                  {activeTab === 'network' && 'Visualize your end-to-end supply chain network'}
                  {activeTab === 'risk' && 'Analyze risk across your supply chain tiers'}
                  {activeTab === 'scenario' && 'Simulate disruption scenarios to evaluate impact'}
                </p>
              </div>
            </div>
            
            {/* Tab Navigation */}
            <div className="mt-4 border-b border-gray-200">
              <nav className="-mb-px flex">
                <button
                  onClick={() => setActiveTab('network')}
                  className={`py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'network'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Network View
                </button>
                <button
                  onClick={() => setActiveTab('risk')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'risk'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Risk Analysis
                </button>
                <button
                  onClick={() => setActiveTab('scenario')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'scenario'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Scenario Simulation
                </button>
              </nav>
            </div>
          </div>
          
          {/* Filters Bar */}
          {activeTab !== 'scenario' && (
            <div className="px-6 py-3 bg-gray-50">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tier</label>
                  <select
                    value={filters.tier}
                    onChange={(e) => handleFilterChange('tier', e.target.value)}
                    className="block w-32 bg-white border border-gray-300 rounded-md shadow-sm py-1.5 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  >
                    <option value="all">All Tiers</option>
                    <option value="tier1">Tier 1</option>
                    <option value="tier2">Tier 2</option>
                    <option value="tier3">Tier 3</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                  <select
                    value={filters.category}
                    onChange={(e) => handleFilterChange('category', e.target.value)}
                    className="block w-36 bg-white border border-gray-300 rounded-md shadow-sm py-1.5 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  >
                    <option value="all">All Categories</option>
                    <option value="raw">Raw Materials</option>
                    <option value="component">Components</option>
                    <option value="packaging">Packaging</option>
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Risk Level</label>
                  <select
                    value={filters.risk}
                    onChange={(e) => handleFilterChange('risk', e.target.value)}
                    className="block w-32 bg-white border border-gray-300 rounded-md shadow-sm py-1.5 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  >
                    <option value="all">All Risks</option>
                    <option value="high">High Risk</option>
                    <option value="medium">Medium Risk</option>
                    <option value="low">Low Risk</option>
                  </select>
                </div>
              </div>
            </div>
          )}
          
          {/* Tab Content */}
          <div className="p-6">
            {loading ? (
              <Loading type="card" message="Loading supply chain data..." />
            ) : (
              <>
                {activeTab === 'network' && filteredNetworkData && (
                  <NetworkView 
                    data={filteredNetworkData} 
                    onNodeSelect={(nodeId) => console.log('Node selected:', nodeId)}
                  />
                )}
                
                {activeTab === 'risk' && riskData && (
                  <RiskVisualizer 
                    data={riskData} 
                    filters={filters}
                  />
                )}
                
                {activeTab === 'scenario' && networkData && (
                  <ScenarioSimulator 
                    networkData={networkData}
                    scenarios={scenarios}
                    onScenarioCreate={(scenario) => setScenarios([...scenarios, scenario])}
                  />
                )}
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default MultiTier;