import React, { useState, useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { FaCog, FaSearchPlus, FaSearchMinus, FaExpand } from 'react-icons/fa';

const NetworkGraph = ({
  data = { nodes: [], links: [] },
  title = "Network Graph",
  height = 500,
  nodeSize = 20,
  directed = false,
  colorScheme = d3.schemeCategory10,
  config = {}
}) => {
  const svgRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [simulation, setSimulation] = useState(null);
  
  // Initialize the simulation and render graph on component mount or data change
  useEffect(() => {
    if (!data || !data.nodes || !data.links || data.nodes.length === 0) {
      setIsLoading(false);
      return;
    }
    
    // Clear previous graph
    d3.select(svgRef.current).selectAll("*").remove();
    
    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;
    
    // Process data to ensure it has required properties
    const nodes = data.nodes.map((node, index) => ({
      ...node,
      id: node.id || `node-${index}`,
      group: node.group || 0,
      radius: node.size ? nodeSize * node.size : nodeSize
    }));
    
    // Ensure links reference nodes by id
    const links = data.links.map((link, index) => ({
      ...link,
      id: link.id || `link-${index}`,
      source: link.source,
      target: link.target,
      value: link.value || 1
    }));
    
    // Create D3 force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(link => 100 / (link.value || 1)))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => d.radius + 5));
    
    // Create SVG elements
    const svg = d3.select(svgRef.current);
    
    // Add zoom functionality
    const zoom = d3.zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
        setZoomLevel(event.transform.k);
      });
    
    svg.call(zoom);
    
    // Create container for zoomable content
    const g = svg.append("g");
    
    // Add arrow marker for directed graph
    if (directed) {
      svg.append("defs").append("marker")
        .attr("id", "arrowhead")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 25)
        .attr("refY", 0)
        .attr("orient", "auto")
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#999");
    }
    
    // Add links
    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter()
      .append("line")
      .attr("stroke-width", d => Math.max(1, Math.sqrt(d.value)))
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .attr("class", "link")
      .attr("marker-end", directed ? "url(#arrowhead)" : null);
    
    // Create node groups
    const node = g.append("g")
      .selectAll(".node")
      .data(nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .on("click", (event, d) => {
        setSelectedNode(selectedNode === d.id ? null : d.id);
      })
      .call(d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded));
    
    // Add circles for nodes
    node.append("circle")
      .attr("r", d => d.radius)
      .attr("fill", d => colorScheme[d.group % colorScheme.length])
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5);
    
    // Add labels
    node.append("text")
      .attr("dy", 4)
      .attr("text-anchor", "middle")
      .text(d => d.name || d.id)
      .style("font-size", "10px")
      .style("pointer-events", "none")
      .style("fill", d => {
        const color = d3.rgb(colorScheme[d.group % colorScheme.length]);
        return color.r * 0.299 + color.g * 0.587 + color.b * 0.114 > 150 ? "#000" : "#fff";
      });
    
    // Add title for tooltip
    node.append("title")
      .text(d => d.name || d.id);
    
    // Update positions on each tick
    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);
      
      node
        .attr("transform", d => `translate(${d.x},${d.y})`);
    });
    
    // Drag functions
    function dragStarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragEnded(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      if (!config.fixNodesOnDrag) {
        d.fx = null;
        d.fy = null;
      }
    }
    
    // Store simulation for controls
    setSimulation(simulation);
    setIsLoading(false);
    
    // Cleanup function
    return () => {
      simulation.stop();
    };
  }, [data, nodeSize, directed, colorScheme, config.fixNodesOnDrag]);
  
  // Handle node selection effect
  useEffect(() => {
    if (!svgRef.current) return;
    
    const svg = d3.select(svgRef.current);
    
    svg.selectAll(".node circle")
      .attr("stroke-width", d => d.id === selectedNode ? 3 : 1.5)
      .attr("stroke", d => d.id === selectedNode ? "#000" : "#fff");
    
    svg.selectAll(".link")
      .attr("stroke-opacity", d => {
        if (!selectedNode) return 0.6;
        return (d.source.id === selectedNode || d.target.id === selectedNode) ? 1 : 0.2;
      })
      .attr("stroke", d => {
        if (!selectedNode) return "#999";
        return (d.source.id === selectedNode || d.target.id === selectedNode) ? "#666" : "#ccc";
      });
    
  }, [selectedNode]);
  
  // Handle zoom controls
  const handleZoomIn = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(d3.zoom().scaleBy, 1.3);
  };
  
  const handleZoomOut = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(d3.zoom().scaleBy, 0.7);
  };
  
  const handleReset = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().call(
      d3.zoom().transform,
      d3.zoomIdentity
    );
  };
  
  // If no data, show placeholder message
  if (!data || !data.nodes || !data.links || data.nodes.length === 0) {
    return (
      <div className="border border-gray-200 rounded-lg p-6 flex flex-col items-center justify-center bg-gray-50 h-80">
        <p className="text-gray-500 mb-4">No data available for network graph</p>
        <FaCog className="animate-spin text-gray-400 h-8 w-8" />
      </div>
    );
  }
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      {title && <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>}
      
      {/* Controls */}
      <div className="flex justify-end mb-2">
        <div className="flex space-x-2">
          <button 
            onClick={handleZoomIn}
            className="p-1 rounded hover:bg-gray-200 text-gray-700"
            title="Zoom In"
          >
            <FaSearchPlus />
          </button>
          <button 
            onClick={handleZoomOut}
            className="p-1 rounded hover:bg-gray-200 text-gray-700"
            title="Zoom Out"
          >
            <FaSearchMinus />
          </button>
          <button 
            onClick={handleReset}
            className="p-1 rounded hover:bg-gray-200 text-gray-700"
            title="Reset View"
          >
            <FaExpand />
          </button>
        </div>
      </div>
      
      {/* SVG Container */}
      <div 
        className="relative w-full overflow-hidden border border-gray-100 rounded-lg"
        style={{ height: `${height}px` }}
      >
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-50">
            <FaCog className="animate-spin text-gray-400 h-8 w-8" />
          </div>
        ) : (
          <svg 
            ref={svgRef} 
            width="100%" 
            height="100%"
            className="cursor-move"
          ></svg>
        )}
        
        {/* Zoom level indicator */}
        <div className="absolute bottom-2 right-2 bg-white bg-opacity-70 px-2 py-1 rounded text-xs text-gray-600">
          {Math.round(zoomLevel * 100)}%
        </div>
      </div>
      
      {/* Node info panel - show when node is selected */}
      {selectedNode && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <h4 className="font-medium text-gray-800 mb-2">
            {data.nodes.find(n => n.id === selectedNode)?.name || selectedNode}
          </h4>
          
          {data.nodes.find(n => n.id === selectedNode)?.description && (
            <p className="text-sm text-gray-600 mb-2">
              {data.nodes.find(n => n.id === selectedNode)?.description}
            </p>
          )}
          
          <div className="text-xs text-gray-500">
            <p>Connections: {data.links.filter(l => l.source.id === selectedNode || l.target.id === selectedNode).length}</p>
            {config.showDetails && (
              <>
                <p className="mt-1">Group: {data.nodes.find(n => n.id === selectedNode)?.group || 'None'}</p>
                {data.nodes.find(n => n.id === selectedNode)?.attributes && (
                  <div className="mt-1">
                    <p className="font-medium">Attributes:</p>
                    <ul className="mt-1 pl-3">
                      {Object.entries(data.nodes.find(n => n.id === selectedNode)?.attributes || {}).map(([key, value]) => (
                        <li key={key}>{key}: {value}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
      
      {/* Legend */}
      {config.showLegend && data.nodes.some(n => n.group !== undefined) && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-sm font-medium text-gray-700 mb-2">Legend</p>
          <div className="flex flex-wrap gap-3">
            {Array.from(new Set(data.nodes.map(n => n.group))).map(group => (
              <div key={group} className="flex items-center">
                <div
                  className="w-3 h-3 mr-1 rounded-full"
                  style={{ backgroundColor: colorScheme[group % colorScheme.length] }}
                ></div>
                <span className="text-xs text-gray-700">
                  {config.groupLabels?.[group] || `Group ${group}`}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default NetworkGraph;