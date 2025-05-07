import React, { useState, useEffect } from 'react';
import { Sankey, Tooltip, ResponsiveContainer, Rectangle } from 'recharts';
import { FaCog } from 'react-icons/fa';

const SankeyDiagram = ({
  data = { nodes: [], links: [] },
  title = "Sankey Diagram",
  height = 500, 
  nodeWidth = 20,
  nodePadding = 50,
  colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#6366f1", "#14b8a6"],
  config = {}
}) => {
  const [chartData, setChartData] = useState({ nodes: [], links: [] });
  const [hoveredNode, setHoveredNode] = useState(null);
  const [hoveredLink, setHoveredLink] = useState(null);
  
  // Process and format the data when it changes
  useEffect(() => {
    if (data && data.nodes && data.links) {
      // Ensure nodes have colors assigned
      const nodesWithColors = data.nodes.map((node, index) => ({
        ...node,
        color: node.color || colors[index % colors.length]
      }));
      
      setChartData({
        nodes: nodesWithColors,
        links: data.links
      });
    }
  }, [data, colors]);
  
  // If no data, show placeholder message
  if (!chartData.nodes.length || !chartData.links.length) {
    return (
      <div className="border border-gray-200 rounded-lg p-6 flex flex-col items-center justify-center bg-gray-50 h-80">
        <p className="text-gray-500 mb-4">No data available for Sankey diagram</p>
        <FaCog className="animate-spin text-gray-400 h-8 w-8" />
      </div>
    );
  }
  
  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      
      if (data.payload.source && data.payload.target) {
        // Link tooltip
        const sourceNode = chartData.nodes.find(n => n.nodeId === data.payload.source);
        const targetNode = chartData.nodes.find(n => n.nodeId === data.payload.target);
        
        return (
          <div className="bg-white p-3 border border-gray-200 shadow-md rounded">
            <p className="font-medium text-gray-800">Flow</p>
            <p className="text-sm text-gray-700">
              From: <span className="font-medium">{sourceNode?.name}</span>
            </p>
            <p className="text-sm text-gray-700">
              To: <span className="font-medium">{targetNode?.name}</span>
            </p>
            <p className="text-sm text-gray-700 mt-1">
              Value: <span className="font-medium">
                {config.valueFormatter ? config.valueFormatter(data.payload.value) : data.payload.value}
              </span>
            </p>
          </div>
        );
      } else {
        // Node tooltip
        return (
          <div className="bg-white p-3 border border-gray-200 shadow-md rounded">
            <p className="font-medium text-gray-800">{data.payload.name}</p>
            {data.payload.description && (
              <p className="text-sm text-gray-600 mt-1">{data.payload.description}</p>
            )}
          </div>
        );
      }
    }
    
    return null;
  };
  
  // Custom node renderer with hover effect
  const CustomNode = ({ x, y, width, height, index, payload }) => {
    const isHovered = hoveredNode === payload.nodeId;
    
    return (
      <Rectangle
        x={x}
        y={y}
        width={width}
        height={height}
        fill={payload.color}
        fillOpacity={isHovered ? 0.8 : 0.6}
        onMouseEnter={() => setHoveredNode(payload.nodeId)}
        onMouseLeave={() => setHoveredNode(null)}
        className="cursor-pointer transition-all duration-150"
        style={{
          stroke: isHovered ? '#000' : payload.color,
          strokeWidth: isHovered ? 2 : 0,
          filter: isHovered ? 'drop-shadow(0 4px 3px rgb(0 0 0 / 0.07))' : 'none'
        }}
      />
    );
  };
  
  // Custom link renderer with hover effect
  const CustomLink = ({ sourceX, targetX, sourceY, targetY, sourceControlX, targetControlX, linkWidth, index, payload }) => {
    const isHovered = hoveredLink === `${payload.source}-${payload.target}`;
    
    // Generate unique gradient ID for each link
    const gradientId = `gradient-${payload.source}-${payload.target}`;
    
    // Find source and target nodes to get colors
    const sourceNode = chartData.nodes.find(n => n.nodeId === payload.source);
    const targetNode = chartData.nodes.find(n => n.nodeId === payload.target);
    
    const sourceColor = sourceNode?.color || colors[0];
    const targetColor = targetNode?.color || colors[1];
    
    return (
      <g
        onMouseEnter={() => setHoveredLink(`${payload.source}-${payload.target}`)}
        onMouseLeave={() => setHoveredLink(null)}
        className="cursor-pointer"
      >
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={sourceColor} />
            <stop offset="100%" stopColor={targetColor} />
          </linearGradient>
        </defs>
        <path
          d={`
            M${sourceX},${sourceY}
            C${sourceControlX},${sourceY} ${targetControlX},${targetY} ${targetX},${targetY}
          `}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={linkWidth}
          strokeOpacity={isHovered ? 0.9 : 0.6}
          style={{
            filter: isHovered ? 'drop-shadow(0 4px 3px rgb(0 0 0 / 0.07))' : 'none',
            transition: 'all 150ms ease'
          }}
        />
      </g>
    );
  };
  
  return (
    <div className="border border-gray-200 rounded-lg p-4 bg-white">
      {title && <h3 className="text-lg font-medium text-gray-700 mb-4">{title}</h3>}
      
      <div className="mt-2">
        <ResponsiveContainer width="100%" height={height}>
          <Sankey
            data={chartData}
            nodeWidth={nodeWidth}
            nodePadding={nodePadding}
            margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
            iterations={64}
            link={CustomLink}
            node={CustomNode}
          >
            <Tooltip content={<CustomTooltip />} />
          </Sankey>
        </ResponsiveContainer>
      </div>
      
      {/* Legend */}
      {config.showLegend && (
        <div className="mt-4 flex flex-wrap gap-3 justify-center">
          {chartData.nodes.map((node, index) => (
            <div key={index} className="flex items-center">
              <div
                className="w-3 h-3 mr-1 rounded-sm"
                style={{ backgroundColor: node.color }}
              />
              <span className="text-xs text-gray-700">{node.name}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SankeyDiagram;