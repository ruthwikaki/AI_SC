import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import NetworkGraph from '../visualization/charts/NetworkGraph';
import ChartViewer from '../visualization/ChartViewer';
import Loading from '../common/Loading';
import {
  ArrowPathIcon,
  ArrowDownOnSquareIcon,
  PlusIcon,
  XMarkIcon,
  PlayIcon,
  DocumentTextIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';

const ScenarioSimulator = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [simulationData, setSimulationData] = useState(null);
  const [savedScenarios, setSavedScenarios] = useState([]);
  const [activeScenario, setActiveScenario] = useState(null);
  const [showNewScenarioModal, setShowNewScenarioModal] = useState(false);
  const [newScenarioName, setNewScenarioName] = useState('');
  const [newScenarioDescription, setNewScenarioDescription] = useState('');
  const [disruptions, setDisruptions] = useState([]);
  const [availableDisruptions, setAvailableDisruptions] = useState([]);
  const [selectedDisruption, setSelectedDisruption] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationCompleted, setSimulationCompleted] = useState(false);

  useEffect(() => {
    fetchSavedScenarios();
    fetchAvailableDisruptions();
  }, []);

  useEffect(() => {
    if (activeScenario) {
      fetchScenarioData(activeScenario);
    }
  }, [activeScenario]);

  const fetchSavedScenarios = async () => {
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/multi-tier/scenarios');
      
      // Simulated API response with mock data
      const mockScenarios = [
        {
          id: 's1',
          name: 'Supplier A Disruption',
          description: 'Simulate complete disruption of Supplier A for 30 days',
          created: '2023-04-15',
          disruptions: [
            { id: 'd1', type: 'supplier', name: 'Supplier A', severity: 'complete', duration: 30 }
          ]
        },
        {
          id: 's2',
          name: 'Natural Disaster in Asia',
          description: 'Simulate impact of earthquake affecting multiple suppliers',
          created: '2023-03-22',
          disruptions: [
            { id: 'd2', type: 'region', name: 'South East Asia', severity: 'severe', duration: 45 },
            { id: 'd3', type: 'transportation', name: 'Sea Freight', severity: 'moderate', duration: 60 }
          ]
        },
        {
          id: 's3',
          name: 'Global Pandemic',
          description: 'Long-term disruption across entire supply chain',
          created: '2023-02-10',
          disruptions: [
            { id: 'd4', type: 'global', name: 'All Suppliers', severity: 'moderate', duration: 180 },
            { id: 'd5', type: 'transportation', name: 'All Transport', severity: 'moderate', duration: 180 },
            { id: 'd6', type: 'demand', name: 'Consumer Demand', severity: 'variable', duration: 180 }
          ]
        }
      ];
      
      setSavedScenarios(mockScenarios);
      // Set the first scenario as active by default
      if (mockScenarios.length > 0 && !activeScenario) {
        setActiveScenario(mockScenarios[0]);
        setDisruptions(mockScenarios[0].disruptions);
      }
    } catch (error) {
      console.error('Error fetching saved scenarios:', error);
    }
  };

  const fetchAvailableDisruptions = async () => {
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get('/multi-tier/available-disruptions');
      
      // Simulated API response with mock data
      const mockDisruptions = [
        { type: 'supplier', name: 'Supplier A', description: 'Direct supplier disruption' },
        { type: 'supplier', name: 'Supplier B', description: 'Direct supplier disruption' },
        { type: 'supplier', name: 'Supplier C', description: 'Direct supplier disruption' },
        { type: 'supplier', name: 'Supplier D', description: 'Direct supplier disruption' },
        { type: 'region', name: 'North America', description: 'Regional disruption affecting all suppliers in the area' },
        { type: 'region', name: 'Europe', description: 'Regional disruption affecting all suppliers in the area' },
        { type: 'region', name: 'South East Asia', description: 'Regional disruption affecting all suppliers in the area' },
        { type: 'region', name: 'China', description: 'Regional disruption affecting all suppliers in the area' },
        { type: 'transportation', name: 'Sea Freight', description: 'Disruption to sea transportation' },
        { type: 'transportation', name: 'Air Freight', description: 'Disruption to air transportation' },
        { type: 'transportation', name: 'Rail', description: 'Disruption to rail transportation' },
        { type: 'transportation', name: 'Truck', description: 'Disruption to truck transportation' },
        { type: 'material', name: 'Raw Material X', description: 'Shortage of a specific raw material' },
        { type: 'material', name: 'Raw Material Y', description: 'Shortage of a specific raw material' },
        { type: 'global', name: 'All Suppliers', description: 'Global disruption affecting all suppliers' }
      ];
      
      setAvailableDisruptions(mockDisruptions);
    } catch (error) {
      console.error('Error fetching available disruptions:', error);
    }
  };

  const fetchScenarioData = async (scenario) => {
    setIsLoading(true);
    setSimulationCompleted(false);
    
    try {
      // In a real implementation, this would call your API
      // Example: const response = await api.get(`/multi-tier/scenarios/${scenario.id}`);
      
      // Simulated API response with mock data
      const mockData = getMockScenarioData(scenario);
      
      // Simulate API delay
      setTimeout(() => {
        setSimulationData(mockData);
        setDisruptions(scenario.disruptions);
        setIsLoading(false);
      }, 800);
    } catch (error) {
      console.error('Error fetching scenario data:', error);
      setIsLoading(false);
    }
  };

  const runSimulation = async () => {
    setIsSimulating(true);
    
    try {
      // In a real implementation, this would call your API to run the simulation
      // Example: const response = await api.post(`/multi-tier/scenarios/${activeScenario.id}/run`);
      
      // Simulated API response with mock data for the simulation result
      const mockSimulationResult = getMockSimulationResult(activeScenario);
      
      // Simulate API delay for the simulation
      setTimeout(() => {
        setSimulationData(mockSimulationResult);
        setIsSimulating(false);
        setSimulationCompleted(true);
      }, 3000);
    } catch (error) {
      console.error('Error running simulation:', error);
      setIsSimulating(false);
    }
  };

  const saveScenario = async () => {
    if (!newScenarioName.trim()) {
      return; // Don't save without a name
    }
    
    const newScenario = {
      id: `s${savedScenarios.length + 1}`,
      name: newScenarioName,
      description: newScenarioDescription,
      created: new Date().toISOString().split('T')[0],
      disruptions: disruptions
    };
    
    try {
      // In a real implementation, this would call your API to save the scenario
      // Example: const response = await api.post('/multi-tier/scenarios', newScenario);
      
      // Update the list of saved scenarios
      setSavedScenarios([...savedScenarios, newScenario]);
      setActiveScenario(newScenario);
      setShowNewScenarioModal(false);
      setNewScenarioName('');
      setNewScenarioDescription('');
    } catch (error) {
      console.error('Error saving scenario:', error);
    }
  };

  const addDisruption = () => {
    if (!selectedDisruption) return;
    
    const newDisruption = {
      id: `d${disruptions.length + 1}`,
      type: selectedDisruption.type,
      name: selectedDisruption.name,
      severity: 'moderate', // Default severity
      duration: 30 // Default duration in days
    };
    
    setDisruptions([...disruptions, newDisruption]);
    setSelectedDisruption(null);
  };

  const removeDisruption = (id) => {
    setDisruptions(disruptions.filter(d => d.id !== id));
  };

  const updateDisruptionSeverity = (id, severity) => {
    setDisruptions(disruptions.map(d => 
      d.id === id ? { ...d, severity } : d
    ));
  };

  const updateDisruptionDuration = (id, duration) => {
    setDisruptions(disruptions.map(d => 
      d.id === id ? { ...d, duration: parseInt(duration, 10) } : d
    ));
  };

  const createNewScenario = () => {
    setActiveScenario(null);
    setDisruptions([]);
    setSimulationData(null);
    setSimulationCompleted(false);
    setShowNewScenarioModal(true);
  };

  const startFromScratch = () => {
    setActiveScenario(null);
    setDisruptions([]);
    setSimulationData(null);
    setSimulationCompleted(false);
  };

  // Mock data generator function for scenario data
  const getMockScenarioData = (scenario) => {
    return {
      scenario,
      baselineNetwork: {
        nodes: [
          { id: 'company', name: 'Your Company', group: 'Company', value: 50 },
          { id: 's1', name: 'Supplier A', group: 'Tier 1', value: 35, category: 'Electronics' },
          { id: 's2', name: 'Supplier B', group: 'Tier 1', value: 30, category: 'Raw Materials' },
          { id: 's3', name: 'Supplier C', group: 'Tier 1', value: 25, category: 'Chemicals' },
          { id: 's4', name: 'Supplier D', group: 'Tier 1', value: 20, category: 'Packaging' },
          { id: 's5', name: 'Supplier E', group: 'Tier 1', value: 28, category: 'Mechanical' },
          { id: 's6', name: 'Supplier F', group: 'Tier 2', value: 18, category: 'Electronics' },
          { id: 's7', name: 'Supplier G', group: 'Tier 2', value: 15, category: 'Raw Materials' },
          { id: 's8', name: 'Supplier H', group: 'Tier 2', value: 20, category: 'Chemicals' },
          { id: 's9', name: 'Supplier I', group: 'Tier 2', value: 16, category: 'Raw Materials' },
          { id: 's10', name: 'Supplier J', group: 'Tier 2', value: 22, category: 'Mechanical' }
        ],
        links: [
          { source: 'company', target: 's1', value: 35 },
          { source: 'company', target: 's2', value: 30 },
          { source: 'company', target: 's3', value: 25 },
          { source: 'company', target: 's4', value: 20 },
          { source: 'company', target: 's5', value: 28 },
          { source: 's1', target: 's6', value: 18 },
          { source: 's2', target: 's7', value: 15 },
          { source: 's2', target: 's9', value: 16 },
          { source: 's3', target: 's8', value: 20 },
          { source: 's5', target: 's10', value: 22 }
        ]
      },
      baselineMetrics: {
        inventoryCoverage: 45, // days
        onTimeDelivery: 95, // percentage
        leadTime: 12, // days
        productionCapacity: 100, // percentage
        costImpact: 0 // baseline has no additional cost
      }
    };
  };

  // Mock data generator function for simulation result
  const getMockSimulationResult = (scenario) => {
    const hasSupplierADisruption = scenario.disruptions.some(d => 
      d.name === 'Supplier A' && d.severity === 'complete'
    );
    
    const hasSouthEastAsiaDisruption = scenario.disruptions.some(d => 
      d.name === 'South East Asia'
    );
    
    const hasGlobalDisruption = scenario.disruptions.some(d => 
      d.name === 'All Suppliers'
    );
    
    // Clone the baseline network
    const baselineNetwork = {
      nodes: [
        { id: 'company', name: 'Your Company', group: 'Company', value: 50 },
        { id: 's1', name: 'Supplier A', group: 'Tier 1', value: 35, category: 'Electronics' },
        { id: 's2', name: 'Supplier B', group: 'Tier 1', value: 30, category: 'Raw Materials' },
        { id: 's3', name: 'Supplier C', group: 'Tier 1', value: 25, category: 'Chemicals' },
        { id: 's4', name: 'Supplier D', group: 'Tier 1', value: 20, category: 'Packaging' },
        { id: 's5', name: 'Supplier E', group: 'Tier 1', value: 28, category: 'Mechanical' },
        { id: 's6', name: 'Supplier F', group: 'Tier 2', value: 18, category: 'Electronics' },
        { id: 's7', name: 'Supplier G', group: 'Tier 2', value: 15, category: 'Raw Materials' },
        { id: 's8', name: 'Supplier H', group: 'Tier 2', value: 20, category: 'Chemicals' },
        { id: 's9', name: 'Supplier I', group: 'Tier 2', value: 16, category: 'Raw Materials' },
        { id: 's10', name: 'Supplier J', group: 'Tier 2', value: 22, category: 'Mechanical' }
      ],
      links: [
        { source: 'company', target: 's1', value: 35 },
        { source: 'company', target: 's2', value: 30 },
        { source: 'company', target: 's3', value: 25 },
        { source: 'company', target: 's4', value: 20 },
        { source: 'company', target: 's5', value: 28 },
        { source: 's1', target: 's6', value: 18 },
        { source: 's2', target: 's7', value: 15 },
        { source: 's2', target: 's9', value: 16 },
        { source: 's3', target: 's8', value: 20 },
        { source: 's5', target: 's10', value: 22 }
      ]
    };
    
    // Modify the network based on the scenario
    const simulatedNetwork = {
      nodes: [...baselineNetwork.nodes],
      links: [...baselineNetwork.links]
    };
    
    if (hasSupplierADisruption) {
      // Mark Supplier A as disrupted
      simulatedNetwork.nodes = simulatedNetwork.nodes.map(node => {
        if (node.id === 's1') {
          return { ...node, status: 'disrupted', color: '#ef4444' };
        }
        if (node.id === 'company') {
          return { ...node, status: 'impacted', color: '#f59e0b' };
        }
        if (node.id === 's6') {
          return { ...node, status: 'impacted', color: '#f59e0b' };
        }
        return node;
      });
      
      // Reduce the value of links to/from disrupted suppliers
      simulatedNetwork.links = simulatedNetwork.links.map(link => {
        if (link.source === 'company' && link.target === 's1') {
          return { ...link, value: link.value * 0.2, status: 'disrupted', color: '#ef4444' };
        }
        if (link.source === 's1' && link.target === 's6') {
          return { ...link, value: link.value * 0.2, status: 'disrupted', color: '#ef4444' };
        }
        return link;
      });
    }
    
    if (hasSouthEastAsiaDisruption) {
      // Assuming suppliers B, G, and I are in South East Asia
      simulatedNetwork.nodes = simulatedNetwork.nodes.map(node => {
        if (['s2', 's7', 's9'].includes(node.id)) {
          return { ...node, status: 'disrupted', color: '#ef4444' };
        }
        if (['company', 's1', 's3', 's4', 's5'].includes(node.id)) {
          return { ...node, status: 'impacted', color: '#f59e0b' };
        }
        return node;
      });
      
      // Reduce the value of links to/from disrupted suppliers
      simulatedNetwork.links = simulatedNetwork.links.map(link => {
        if (
          (link.source === 'company' && link.target === 's2') ||
          (link.source === 's2' && link.target === 's7') ||
          (link.source === 's2' && link.target === 's9')
        ) {
          return { ...link, value: link.value * 0.4, status: 'disrupted', color: '#ef4444' };
        }
        return link;
      });
    }
    
    if (hasGlobalDisruption) {
      // All suppliers are affected to some degree
      simulatedNetwork.nodes = simulatedNetwork.nodes.map(node => {
        if (node.id === 'company') {
          return { ...node, status: 'severely impacted', color: '#b91c1c' };
        }
        return { ...node, status: 'disrupted', color: '#ef4444' };
      });
      
      // All links are affected
      simulatedNetwork.links = simulatedNetwork.links.map(link => {
        return { ...link, value: link.value * 0.6, status: 'disrupted', color: '#ef4444' };
      });
    }
    
    // Calculate the impact on metrics based on the scenario
    let inventoryCoverage = 45; // baseline 45 days
    let onTimeDelivery = 95; // baseline 95%
    let leadTime = 12; // baseline 12 days
    let productionCapacity = 100; // baseline 100%
    let costImpact = 0; // baseline $0
    
    if (hasSupplierADisruption) {
      inventoryCoverage -= 15;
      onTimeDelivery -= 12;
      leadTime += 5;
      productionCapacity -= 15;
      costImpact += 120000;
    }
    
    if (hasSouthEastAsiaDisruption) {
      inventoryCoverage -= 18;
      onTimeDelivery -= 15;
      leadTime += 8;
      productionCapacity -= 20;
      costImpact += 250000;
    }
    
    if (hasGlobalDisruption) {
      inventoryCoverage -= 30;
      onTimeDelivery -= 25;
      leadTime += 15;
      productionCapacity -= 40;
      costImpact += 500000;
    }
    
    // Ensure no negative values
    inventoryCoverage = Math.max(0, inventoryCoverage);
    onTimeDelivery = Math.max(0, onTimeDelivery);
    productionCapacity = Math.max(0, productionCapacity);
    
    const timeSeriesData = generateTimeSeriesData(scenario);
    
    return {
      scenario,
      baselineNetwork,
      simulatedNetwork,
      baselineMetrics: {
        inventoryCoverage: 45,
        onTimeDelivery: 95,
        leadTime: 12,
        productionCapacity: 100,
        costImpact: 0
      },
      simulatedMetrics: {
        inventoryCoverage,
        onTimeDelivery,
        leadTime,
        productionCapacity,
        costImpact
      },
      timeSeriesData
    };
  };

  // Helper function to generate time series data for the simulation
  const generateTimeSeriesData = (scenario) => {
    const days = 180; // 6 months of data
    const data = [];
    
    // Determine the overall severity of the scenario
    const overallSeverity = scenario.disruptions.reduce((severity, disruption) => {
      const disruptionSeverity = 
        disruption.severity === 'complete' ? 1.0 :
        disruption.severity === 'severe' ? 0.8 :
        disruption.severity === 'moderate' ? 0.5 :
        disruption.severity === 'mild' ? 0.3 : 0.1;
      
      return Math.max(severity, disruptionSeverity);
    }, 0);
    
    // Determine the longest disruption duration
    const longestDuration = scenario.disruptions.reduce((max, disruption) => 
      Math.max(max, disruption.duration || 30), 30);
    
    // Generate daily data points
    for (let day = 0; day < days; day++) {
      // Calculate the impact factor, which decreases over time after the longest disruption
      const impactFactor = day <= longestDuration ? 
        overallSeverity : 
        overallSeverity * Math.max(0, 1 - (day - longestDuration) / 90);
      
      // Create baseline values with some random variation
      const randomFactor = () => 1 + (Math.random() * 0.1 - 0.05);
      
      const baselineInventory = 45 * randomFactor();
      const baselineDelivery = 95 * randomFactor();
      const baselineLeadTime = 12 * randomFactor();
      const baselineCapacity = 100 * randomFactor();
      
      // Apply the impact to create the simulated values
      const simulatedInventory = Math.max(0, baselineInventory * (1 - impactFactor * 0.7));
      const simulatedDelivery = Math.max(0, baselineDelivery * (1 - impactFactor * 0.3));
      const simulatedLeadTime = baselineLeadTime * (1 + impactFactor * 0.8);
      const simulatedCapacity = Math.max(0, baselineCapacity * (1 - impactFactor * 0.5));
      
      data.push({
        day,
        baselineInventory,
        simulatedInventory,
        baselineDelivery,
        simulatedDelivery,
        baselineLeadTime,
        simulatedLeadTime,
        baselineCapacity,
        simulatedCapacity
      });
    }
    
    return data;
  };

  // Format network data for visualization
  const formatNetworkData = (network) => {
    if (!network) return null;
    
    return {
      type: 'network',
      data: network,
      config: {
        nodeSize: 'value',
        nodeSizeRange: [5, 25],
        forceStrength: -150,
        distanceMin: 100,
        distanceMax: 300,
        nodeLabels: true
      }
    };
  };

  if (isLoading) {
    return <Loading type="card" message="Loading scenario data..." />;
  }

  return (
    <div className="bg-gray-50 min-h-full">
      {/* Scenario Simulator Header */}
      <div className="bg-white shadow-sm px-4 py-4 flex flex-wrap justify-between items-center">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">What-If Scenario Simulator</h1>
          <p className="text-sm text-gray-500 mt-1">
            {activeScenario 
              ? `Scenario: ${activeScenario.name}`
              : 'Create a new scenario or select an existing one'}
          </p>
        </div>
        
        <div className="flex items-center space-x-3 mt-3 sm:mt-0">
          <div className="relative">
            <select 
              className="block w-48 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              value={activeScenario ? activeScenario.id : ''}
              onChange={(e) => {
                const selected = savedScenarios.find(s => s.id === e.target.value);
                if (selected) {
                  setActiveScenario(selected);
                }
              }}
            >
              <option value="">Select a scenario</option>
              {savedScenarios.map(scenario => (
                <option key={scenario.id} value={scenario.id}>
                  {scenario.name}
                </option>
              ))}
            </select>
          </div>
          
          <button
            onClick={createNewScenario}
            className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
          >
            <PlusIcon className="-ml-1 mr-2 h-4 w-4" />
            New Scenario
          </button>
          
          {activeScenario && (
            <button
              onClick={startFromScratch}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <ArrowPathIcon className="-ml-1 mr-2 h-4 w-4" />
              Start New
            </button>
          )}
          
          {activeScenario && !simulationCompleted && (
            <button
              onClick={runSimulation}
              disabled={isSimulating}
              className={`inline-flex items-center px-3 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 ${
                isSimulating ? 'opacity-75 cursor-not-allowed' : ''
              }`}
            >
              {isSimulating ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Simulating...
                </>
              ) : (
                <>
                  <PlayIcon className="-ml-1 mr-2 h-4 w-4" />
                  Run Simulation
                </>
              )}
            </button>
          )}
          
          {simulationCompleted && (
            <button
              onClick={() => navigate('/reports/create', { state: { simulationData } })}
              className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              <DocumentTextIcon className="-ml-1 mr-2 h-4 w-4" />
              Generate Report
            </button>
          )}
        </div>
      </div>
      
      {/* Scenario Content */}
      <div className="container mx-auto px-4 py-6">
        {/* If no active scenario, show list of saved scenarios */}
        {!activeScenario && savedScenarios.length > 0 && (
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-800">Saved Scenarios</h2>
            </div>
            <div className="overflow-hidden">
              <ul className="divide-y divide-gray-200">
                {savedScenarios.map((scenario) => (
                  <li key={scenario.id} className="px-6 py-4 hover:bg-gray-50">
                    <div className="flex items-center justify-between">
                      <div className="flex-1 min-w-0">
                        <h3 className="text-sm font-medium text-gray-900 truncate">{scenario.name}</h3>
                        <p className="text-sm text-gray-500">{scenario.description}</p>
                        <div className="mt-2 flex items-center text-xs text-gray-500">
                          <span>Created: {scenario.created}</span>
                          <span className="mx-1">â€¢</span>
                          <span>{scenario.disruptions.length} disruption{scenario.disruptions.length !== 1 ? 's' : ''}</span>
                        </div>
                      </div>
                      <div>
                        <button
                          onClick={() => setActiveScenario(scenario)}
                          className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
                        >
                          Select
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
        
        {/* Configure Disruptions */}
        {activeScenario && !simulationCompleted && (
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-800">Configure Disruptions</h2>
            </div>
            <div className="p-6">
              <div className="mb-4">
                <label htmlFor="disruption" className="block text-sm font-medium text-gray-700">
                  Add Disruption
                </label>
                <div className="mt-1 flex rounded-md shadow-sm">
                  <div className="relative flex items-stretch flex-grow focus-within:z-10">
                    <select
                      id="disruption"
                      className="focus:ring-indigo-500 focus:border-indigo-500 block w-full rounded-none rounded-l-md sm:text-sm border-gray-300"
                      value={selectedDisruption ? `${selectedDisruption.type}-${selectedDisruption.name}` : ''}
                      onChange={(e) => {
                        if (e.target.value) {
                          const [type, name] = e.target.value.split('-');
                          const disruption = availableDisruptions.find(d => d.type === type && d.name === name);
                          setSelectedDisruption(disruption);
                        } else {
                          setSelectedDisruption(null);
                        }
                      }}
                    >
                      <option value="">Select a disruption</option>
                      <optgroup label="Supplier Disruptions">
                        {availableDisruptions
                          .filter(d => d.type === 'supplier')
                          .map(d => (
                            <option key={`${d.type}-${d.name}`} value={`${d.type}-${d.name}`}>
                              {d.name}
                            </option>
                          ))
                        }
                      </optgroup>
                      <optgroup label="Regional Disruptions">
                        {availableDisruptions
                          .filter(d => d.type === 'region')
                          .map(d => (
                            <option key={`${d.type}-${d.name}`} value={`${d.type}-${d.name}`}>
                              {d.name}
                            </option>
                          ))
                        }
                      </optgroup>
                      <optgroup label="Transportation Disruptions">
                        {availableDisruptions
                          .filter(d => d.type === 'transportation')
                          .map(d => (
                            <option key={`${d.type}-${d.name}`} value={`${d.type}-${d.name}`}>
                              {d.name}
                            </option>
                          ))
                        }
                      </optgroup>
                      <optgroup label="Material Disruptions">
                        {availableDisruptions
                          .filter(d => d.type === 'material')
                          .map(d => (
                            <option key={`${d.type}-${d.name}`} value={`${d.type}-${d.name}`}>
                              {d.name}
                            </option>
                          ))
                        }
                      </optgroup>
                      <optgroup label="Global Disruptions">
                        {availableDisruptions
                          .filter(d => d.type === 'global')
                          .map(d => (
                            <option key={`${d.type}-${d.name}`} value={`${d.type}-${d.name}`}>
                              {d.name}
                            </option>
                          ))
                        }
                      </optgroup>
                    </select>
                  </div>
                  <button
                    type="button"
                    onClick={addDisruption}
                    disabled={!selectedDisruption}
                    className={`-ml-px relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-r-md text-gray-700 bg-gray-50 hover:bg-gray-100 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 ${
                      !selectedDisruption ? 'opacity-50 cursor-not-allowed' : ''
                    }`}
                  >
                    Add
                  </button>
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Configured Disruptions</h3>
                {disruptions.length === 0 ? (
                  <p className="text-sm text-gray-500">No disruptions configured yet.</p>
                ) : (
                  <ul className="space-y-3">
                    {disruptions.map((disruption) => (
                      <li key={disruption.id} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex justify-between items-start">
                          <div>
                            <h4 className="text-sm font-medium text-gray-700">{disruption.name}</h4>
                            <p className="text-xs text-gray-500 capitalize">{disruption.type} Disruption</p>
                          </div>
                          <button
                            onClick={() => removeDisruption(disruption.id)}
                            className="text-gray-400 hover:text-gray-500"
                          >
                            <XMarkIcon className="h-5 w-5" />
                          </button>
                        </div>
                        <div className="mt-3 grid grid-cols-2 gap-3">
                          <div>
                            <label htmlFor={`severity-${disruption.id}`} className="block text-xs font-medium text-gray-500">
                              Severity
                            </label>
                            <select
                              id={`severity-${disruption.id}`}
                              className="mt-1 block w-full py-1.5 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                              value={disruption.severity}
                              onChange={(e) => updateDisruptionSeverity(disruption.id, e.target.value)}
                            >
                              <option value="mild">Mild (10-30% impact)</option>
                              <option value="moderate">Moderate (30-60% impact)</option>
                              <option value="severe">Severe (60-90% impact)</option>
                              <option value="complete">Complete (90-100% impact)</option>
                            </select>
                          </div>
                          <div>
                            <label htmlFor={`duration-${disruption.id}`} className="block text-xs font-medium text-gray-500">
                              Duration (days)
                            </label>
                            <input
                              type="number"
                              id={`duration-${disruption.id}`}
                              className="mt-1 block w-full py-1.5 px-3 border border-gray-300 bg-white rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                              value={disruption.duration}
                              min="1"
                              max="365"
                              onChange={(e) => updateDisruptionDuration(disruption.id, e.target.value)}
                            />
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        )}
        
        {/* Baseline Network */}
        {simulationData && !simulationCompleted && (
          <div className="bg-white rounded-lg shadow mb-6">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-medium text-gray-800">Baseline Supply Chain Network</h2>
            </div>
            <div className="p-6 h-96">
              <NetworkGraph 
                data={formatNetworkData(simulationData.baselineNetwork).data}
                config={formatNetworkData(simulationData.baselineNetwork).config}
                height="100%"
              />
            </div>
          </div>
        )}
        
        {/* Simulation Results */}
        {simulationCompleted && (
          <>
            {/* Networks Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-800">Baseline Network</h2>
                </div>
                <div className="p-6 h-96">
                  <NetworkGraph 
                    data={formatNetworkData(simulationData.baselineNetwork).data}
                    config={formatNetworkData(simulationData.baselineNetwork).config}
                    height="100%"
                  />
                </div>
              </div>
              
              <div className="bg-white rounded-lg shadow">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-800">Disrupted Network</h2>
                </div>
                <div className="p-6 h-96">
                  <NetworkGraph 
                    data={formatNetworkData(simulationData.simulatedNetwork).data}
                    config={formatNetworkData(simulationData.simulatedNetwork).config}
                    height="100%"
                  />
                </div>
              </div>
            </div>
            
            {/* Key Metrics Comparison */}
            <div className="bg-white rounded-lg shadow mb-6">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-medium text-gray-800">Impact on Key Metrics</h2>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                  {/* Inventory Coverage */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Inventory Coverage</h3>
                    <div className="flex items-baseline">
                      <span className="text-2xl font-bold text-gray-900">
                        {simulationData.simulatedMetrics.inventoryCoverage}
                      </span>
                      <span className="ml-1 text-sm text-gray-600">days</span>
                      <span className={`ml-2 text-sm ${
                        simulationData.simulatedMetrics.inventoryCoverage < simulationData.baselineMetrics.inventoryCoverage
                          ? 'text-red-600'
                          : 'text-green-600'
                      }`}>
                        {simulationData.simulatedMetrics.inventoryCoverage < simulationData.baselineMetrics.inventoryCoverage
                          ? `â†“ ${Math.round((simulationData.baselineMetrics.inventoryCoverage - simulationData.simulatedMetrics.inventoryCoverage) / simulationData.baselineMetrics.inventoryCoverage * 100)}%`
                          : `â†‘ ${Math.round((simulationData.simulatedMetrics.inventoryCoverage - simulationData.baselineMetrics.inventoryCoverage) / simulationData.baselineMetrics.inventoryCoverage * 100)}%`
                        }
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Baseline: {simulationData.baselineMetrics.inventoryCoverage} days</div>
                  </div>
                  
                  {/* On-Time Delivery */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">On-Time Delivery</h3>
                    <div className="flex items-baseline">
                      <span className="text-2xl font-bold text-gray-900">
                        {simulationData.simulatedMetrics.onTimeDelivery}
                      </span>
                      <span className="ml-1 text-sm text-gray-600">%</span>
                      <span className={`ml-2 text-sm ${
                        simulationData.simulatedMetrics.onTimeDelivery < simulationData.baselineMetrics.onTimeDelivery
                          ? 'text-red-600'
                          : 'text-green-600'
                      }`}>
                        {simulationData.simulatedMetrics.onTimeDelivery < simulationData.baselineMetrics.onTimeDelivery
                          ? `â†“ ${Math.round(simulationData.baselineMetrics.onTimeDelivery - simulationData.simulatedMetrics.onTimeDelivery)}pp`
                          : `â†‘ ${Math.round(simulationData.simulatedMetrics.onTimeDelivery - simulationData.baselineMetrics.onTimeDelivery)}pp`
                        }
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Baseline: {simulationData.baselineMetrics.onTimeDelivery}%</div>
                  </div>
                  
                  {/* Lead Time */}
                  {/* Lead Time */}
                 <div className="bg-gray-50 p-4 rounded-lg">
                   <h3 className="text-sm font-medium text-gray-700 mb-2">Lead Time</h3>
                   <div className="flex items-baseline">
                     <span className="text-2xl font-bold text-gray-900">
                       {simulationData.simulatedMetrics.leadTime}
                     </span>
                     <span className="ml-1 text-sm text-gray-600">days</span>
                     <span className={`ml-2 text-sm ${
                       simulationData.simulatedMetrics.leadTime > simulationData.baselineMetrics.leadTime
                         ? 'text-red-600'
                         : 'text-green-600'
                     }`}>
                       {simulationData.simulatedMetrics.leadTime > simulationData.baselineMetrics.leadTime
                         ? `â†‘ ${Math.round((simulationData.simulatedMetrics.leadTime - simulationData.baselineMetrics.leadTime) / simulationData.baselineMetrics.leadTime * 100)}%`
                         : `â†“ ${Math.round((simulationData.baselineMetrics.leadTime - simulationData.simulatedMetrics.leadTime) / simulationData.baselineMetrics.leadTime * 100)}%`
                       }
                     </span>
                   </div>
                   <div className="text-xs text-gray-500 mt-1">Baseline: {simulationData.baselineMetrics.leadTime} days</div>
                 </div>
                 
                 {/* Production Capacity */}
                 <div className="bg-gray-50 p-4 rounded-lg">
                   <h3 className="text-sm font-medium text-gray-700 mb-2">Production Capacity</h3>
                   <div className="flex items-baseline">
                     <span className="text-2xl font-bold text-gray-900">
                       {simulationData.simulatedMetrics.productionCapacity}
                     </span>
                     <span className="ml-1 text-sm text-gray-600">%</span>
                     <span className={`ml-2 text-sm ${
                       simulationData.simulatedMetrics.productionCapacity < simulationData.baselineMetrics.productionCapacity
                         ? 'text-red-600'
                         : 'text-green-600'
                     }`}>
                       {simulationData.simulatedMetrics.productionCapacity < simulationData.baselineMetrics.productionCapacity
                         ? `â†“ ${Math.round(simulationData.baselineMetrics.productionCapacity - simulationData.simulatedMetrics.productionCapacity)}pp`
                         : `â†‘ ${Math.round(simulationData.simulatedMetrics.productionCapacity - simulationData.baselineMetrics.productionCapacity)}pp`
                       }
                     </span>
                   </div>
                   <div className="text-xs text-gray-500 mt-1">Baseline: {simulationData.baselineMetrics.productionCapacity}%</div>
                 </div>
                 
                 {/* Cost Impact */}
                 <div className="bg-gray-50 p-4 rounded-lg">
                   <h3 className="text-sm font-medium text-gray-700 mb-2">Additional Cost</h3>
                   <div className="flex items-baseline">
                     <span className="text-2xl font-bold text-gray-900">
                       ${(simulationData.simulatedMetrics.costImpact / 1000).toFixed(0)}K
                     </span>
                     <span className={`ml-2 text-sm ${
                       simulationData.simulatedMetrics.costImpact > 0
                         ? 'text-red-600'
                         : 'text-green-600'
                     }`}>
                       {simulationData.simulatedMetrics.costImpact > 0
                         ? 'â†‘'
                         : 'â†“'
                       }
                     </span>
                   </div>
                   <div className="text-xs text-gray-500 mt-1">Baseline: ${simulationData.baselineMetrics.costImpact}</div>
                 </div>
               </div>
             </div>
           </div>
           
           {/* Time Series Charts */}
           <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
             {/* Inventory Coverage Over Time */}
             <ChartViewer 
               chartData={{
                 type: 'line',
                 title: 'Inventory Coverage Over Time',
                 data: simulationData.timeSeriesData,
                 config: {
                   xKey: 'day',
                   multiSeries: true,
                   series: [
                     { name: 'Baseline', dataKey: 'baselineInventory', color: '#6366f1' },
                     { name: 'Simulated', dataKey: 'simulatedInventory', color: '#ef4444' }
                   ],
                   valueFormatter: (value) => `${value.toFixed(1)} days`,
                   xAxisFormatter: (value) => `Day ${value}`
                 }
               }}
             />
             
             {/* On-Time Delivery Over Time */}
             <ChartViewer 
               chartData={{
                 type: 'line',
                 title: 'On-Time Delivery Over Time',
                 data: simulationData.timeSeriesData,
                 config: {
                   xKey: 'day',
                   multiSeries: true,
                   series: [
                     { name: 'Baseline', dataKey: 'baselineDelivery', color: '#6366f1' },
                     { name: 'Simulated', dataKey: 'simulatedDelivery', color: '#ef4444' }
                   ],
                   valueFormatter: (value) => `${value.toFixed(1)}%`,
                   xAxisFormatter: (value) => `Day ${value}`
                 }
               }}
             />
             
             {/* Lead Time Over Time */}
             <ChartViewer 
               chartData={{
                 type: 'line',
                 title: 'Lead Time Over Time',
                 data: simulationData.timeSeriesData,
                 config: {
                   xKey: 'day',
                   multiSeries: true,
                   series: [
                     { name: 'Baseline', dataKey: 'baselineLeadTime', color: '#6366f1' },
                     { name: 'Simulated', dataKey: 'simulatedLeadTime', color: '#ef4444' }
                   ],
                   valueFormatter: (value) => `${value.toFixed(1)} days`,
                   xAxisFormatter: (value) => `Day ${value}`
                 }
               }}
             />
             
             {/* Production Capacity Over Time */}
             <ChartViewer 
               chartData={{
                 type: 'line',
                 title: 'Production Capacity Over Time',
                 data: simulationData.timeSeriesData,
                 config: {
                   xKey: 'day',
                   multiSeries: true,
                   series: [
                     { name: 'Baseline', dataKey: 'baselineCapacity', color: '#6366f1' },
                     { name: 'Simulated', dataKey: 'simulatedCapacity', color: '#ef4444' }
                   ],
                   valueFormatter: (value) => `${value.toFixed(1)}%`,
                   xAxisFormatter: (value) => `Day ${value}`
                 }
               }}
             />
           </div>
           
           {/* Recommendations */}
           <div className="bg-white rounded-lg shadow mb-6">
             <div className="px-6 py-4 border-b border-gray-200">
               <h2 className="text-lg font-medium text-gray-800">Mitigation Recommendations</h2>
             </div>
             <div className="p-6">
               <div className="bg-indigo-50 p-4 rounded-lg mb-4">
                 <h3 className="text-sm font-medium text-indigo-800 mb-2">Simulation Summary</h3>
                 <p className="text-sm text-indigo-700">
                   This simulation shows that the {activeScenario.name} scenario would cause significant disruption to your supply chain, 
                   with impacts on inventory coverage, delivery performance, lead times, and production capacity. Consider implementing 
                   the following mitigation strategies to reduce these impacts.
                 </p>
               </div>
               
               <div className="space-y-4">
                 <div className="border-l-4 border-indigo-500 pl-4">
                   <h3 className="text-md font-medium text-gray-800">Short-term Mitigations</h3>
                   <ul className="mt-2 space-y-1 text-sm text-gray-600 list-disc list-inside">
                     <li>Increase safety stock levels for critical components from disrupted suppliers</li>
                     <li>Activate backup suppliers for the most critical materials</li>
                     <li>Implement expedited shipping options for critical deliveries</li>
                     <li>Adjust production schedules to prioritize high-margin products</li>
                   </ul>
                 </div>
                 
                 <div className="border-l-4 border-indigo-500 pl-4">
                   <h3 className="text-md font-medium text-gray-800">Medium-term Mitigations</h3>
                   <ul className="mt-2 space-y-1 text-sm text-gray-600 list-disc list-inside">
                     <li>Qualify additional suppliers for key components</li>
                     <li>Review and update inventory policies for critical materials</li>
                     <li>Develop alternate logistics routes and transportation modes</li>
                     <li>Train staff on disruption response procedures</li>
                   </ul>
                 </div>
                 
                 <div className="border-l-4 border-indigo-500 pl-4">
                   <h3 className="text-md font-medium text-gray-800">Long-term Mitigations</h3>
                   <ul className="mt-2 space-y-1 text-sm text-gray-600 list-disc list-inside">
                     <li>Redesign products to use more readily available components</li>
                     <li>Diversify supplier base across geographic regions</li>
                     <li>Implement multi-tier supply chain visibility solutions</li>
                     <li>Develop more robust supply chain risk management capabilities</li>
                   </ul>
                 </div>
               </div>
             </div>
           </div>
         </>
       )}
     </div>
     
     {/* New Scenario Modal */}
     {showNewScenarioModal && (
       <div className="fixed inset-0 overflow-y-auto z-50">
         <div className="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
           <div className="fixed inset-0 transition-opacity" aria-hidden="true">
             <div className="absolute inset-0 bg-gray-500 opacity-75" onClick={() => setShowNewScenarioModal(false)}></div>
           </div>

           <span className="hidden sm:inline-block sm:align-middle sm:h-screen" aria-hidden="true">&#8203;</span>

           <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
             <div className="sm:flex sm:items-start">
               <div className="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-indigo-100 sm:mx-0 sm:h-10 sm:w-10">
                 <Cog6ToothIcon className="h-6 w-6 text-indigo-600" aria-hidden="true" />
               </div>
               <div className="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
                 <h3 className="text-lg leading-6 font-medium text-gray-900">New Scenario</h3>
                 <div className="mt-2">
                   <p className="text-sm text-gray-500">
                     Create a new what-if scenario to simulate the impact of disruptions on your supply chain.
                   </p>
                 </div>
               </div>
             </div>
             
             <div className="mt-6 space-y-4">
               <div>
                 <label htmlFor="scenarioName" className="block text-sm font-medium text-gray-700">
                   Scenario Name
                 </label>
                 <input
                   type="text"
                   id="scenarioName"
                   className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                   placeholder="E.g. 'Supplier A Disruption'"
                   value={newScenarioName}
                   onChange={(e) => setNewScenarioName(e.target.value)}
                 />
               </div>
               
               <div>
                 <label htmlFor="scenarioDescription" className="block text-sm font-medium text-gray-700">
                   Scenario Description
                 </label>
                 <textarea
                   id="scenarioDescription"
                   rows={3}
                   className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                   placeholder="Describe the disruption scenario you want to simulate"
                   value={newScenarioDescription}
                   onChange={(e) => setNewScenarioDescription(e.target.value)}
                 />
               </div>
             </div>
             
             <div className="mt-5 sm:mt-6 sm:grid sm:grid-cols-2 sm:gap-3 sm:grid-flow-row-dense">
               <button
                 type="button"
                 className="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-indigo-600 text-base font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:col-start-2 sm:text-sm"
                 onClick={saveScenario}
                 disabled={!newScenarioName.trim()}
               >
                 <ArrowDownOnSquareIcon className="-ml-1 mr-2 h-5 w-5" />
                 Create
               </button>
               <button
                 type="button"
                 className="mt-3 w-full inline-flex justify-center rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 sm:mt-0 sm:col-start-1 sm:text-sm"
                 onClick={() => setShowNewScenarioModal(false)}
               >
                 Cancel
               </button>
             </div>
           </div>
         </div>
       </div>
     )}
   </div>
 );
};

export default ScenarioSimulator;

