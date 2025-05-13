import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { formatValue } from '../../../utils/formatting';

const NetworkGraph = ({ 
  data, 
  config = {}, 
  height = 500,
  onNodeClick,
  onLinkClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);
  const [simulation, setSimulation] = useState(null);

  const {
    title = '',
    nodeSize = 'value', // 'value' or static number
    nodeSizeRange = [5, 20],
    linkWidth = 'value', // 'value' or static number
    linkWidthRange = [1, 5],
    colorScheme = 'schemeCategory10',
    valueFormatter,
    nodeLabels = true,
    margin = { top: 30, right: 30, bottom: 30, left: 30 },
    forceStrength = -100,
    distanceMin = 30,
    distanceMax = 200
  } = config;

  useEffect(() => {
    if (!data || !svgRef.current) return;
    if (!data.nodes || !data.links) {
      console.error('Invalid data format for Network graph. Expected {nodes: [], links: []}');
      return;
    }

    // Clean up previous chart
    d3.select(svgRef.current).selectAll('*').remove();
    
    // Create tooltip if it doesn't exist
    if (!tooltipRef.current) {
      tooltipRef.current = d3.select('body')
        .append('div')
        .attr('class', 'absolute hidden p-2 bg-gray-800 text-white rounded shadow-lg text-xs z-50 pointer-events-none')
        .style('opacity', 0);
    }

    // Stop any existing simulation
    if (simulation) {
      simulation.stop();
    }

    // Setup dimensions
    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Create chart group
    const chart = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add title if provided
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .attr('class', 'text-sm font-semibold')
        .text(title);
    }

    // Make a deep copy of the data
    const graphData = {
      nodes: data.nodes.map(d => ({...d})),
      links: data.links.map(d => ({...d}))
    };

    // Color scale for nodes
    const color = d3.scaleOrdinal(d3[colorScheme] || d3.schemeCategory10);

    // Node size scale (if using value-based sizing)
    let nodeSizeScale;
    if (nodeSize === 'value') {
      const nodeValues = graphData.nodes.map(d => d.value || 1);
      nodeSizeScale = d3.scaleLinear()
        .domain([Math.min(...nodeValues), Math.max(...nodeValues)])
        .range(nodeSizeRange);
    }

    // Link width scale (if using value-based width)
    let linkWidthScale;
    if (linkWidth === 'value') {
      const linkValues = graphData.links.map(d => d.value || 1);
      linkWidthScale = d3.scaleLinear()
        .domain([Math.min(...linkValues), Math.max(...linkValues)])
        .range(linkWidthRange);
    }

    // Create a map for faster lookups
    const nodeById = new Map(graphData.nodes.map(node => [node.id, node]));
    
    // Convert link references from IDs to objects
    const links = graphData.links.map(link => ({
      source: typeof link.source === 'object' ? link.source : nodeById.get(link.source),
      target: typeof link.target === 'object' ? link.target : nodeById.get(link.target),
      value: link.value || 1
    }));

    // Create force simulation
    const sim = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(links)
        .id(d => d.id)
        .distance(d => distanceMin + (d.value ? distanceMax * (1 / d.value) : distanceMax / 2))
      )
      .force('charge', d3.forceManyBody().strength(forceStrength))
      .force('center', d3.forceCenter(chartWidth / 2, chartHeight / 2))
      .force('collision', d3.forceCollide().radius(d => {
        return nodeSize === 'value' 
          ? nodeSizeScale(d.value || 1) + 5 
          : (typeof nodeSize === 'number' ? nodeSize : 10) + 5;
      }));

    // Save simulation to state for cleanup
    setSimulation(sim);

    // Draw links
    const link = chart.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => {
        return linkWidth === 'value'
          ? linkWidthScale(d.value || 1)
          : (typeof linkWidth === 'number' ? linkWidth : 1);
      })
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#666')
          .attr('stroke-opacity', 1);

        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d.source.name || d.source.id} → ${d.target.name || d.target.id}</strong><br>
            ${formatValue(d.value, valueFormatter)}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#999')
          .attr('stroke-opacity', 0.6);
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onLinkClick) onLinkClick({
          source: d.source.id,
          target: d.target.id,
          value: d.value
        });
      });

    // Create node groups
    const node = chart.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(graphData.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
      );

    // Add circle for each node
    node.append('circle')
      .attr('r', d => {
        return nodeSize === 'value'
          ? nodeSizeScale(d.value || 1)
          : (typeof nodeSize === 'number' ? nodeSize : 10);
      })
      .attr('fill', d => color(d.group || d.category || d.type || d.id))
      .attr('stroke', d => d3.rgb(color(d.group || d.category || d.type || d.id)).darker(0.5))
      .attr('stroke-width', 1.5)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('stroke-width', 2);
        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d.name || d.id}</strong><br>
            ${d.value ? formatValue(d.value, valueFormatter) : ''}
            ${d.group || d.category || d.type ? `<br>${d.group || d.category || d.type}` : ''}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function() {
        d3.select(this).attr('stroke-width', 1.5);
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onNodeClick) onNodeClick(d);
      });

    // Add labels if enabled
    if (nodeLabels) {
      node.append('text')
        .attr('dx', d => {
          const radius = nodeSize === 'value'
            ? nodeSizeScale(d.value || 1)
            : (typeof nodeSize === 'number' ? nodeSize : 10);
          return radius + 5;
        })
        .attr('dy', '.35em')
        .attr('class', 'text-xs')
        .text(d => d.name || d.id)
        .style('pointer-events', 'none'); // Make labels non-interactive
    }

    // Set up the tick function for the simulation
    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node
        .attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag functions
    function dragstarted(event, d) {
      if (!event.active) sim.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) sim.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 5])
      .on('zoom', (event) => {
        chart.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Cleanup function
    return () => {
      if (simulation) {
        simulation.stop();
      }
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, simulation, onNodeClick, onLinkClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default NetworkGraph;