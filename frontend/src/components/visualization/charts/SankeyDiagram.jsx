import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { sankey, sankeyLinkHorizontal } from 'd3-sankey';
import { formatValue } from '../../../utils/formatting';

const SankeyDiagram = ({ 
  data, 
  config = {}, 
  height = 500,
  onNodeClick,
  onLinkClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const {
    title = '',
    nodeWidth = 15,
    nodePadding = 10,
    colorScheme = 'schemeCategory10',
    valueFormatter,
    margin = { top: 30, right: 30, bottom: 30, left: 30 }
  } = config;

  useEffect(() => {
    if (!data || !svgRef.current) return;
    if (!data.nodes || !data.links) {
      console.error('Invalid data format for Sankey diagram. Expected {nodes: [], links: []}');
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

    // Make a copy of the data to avoid modifying original
    const sankeyData = {
      nodes: [...data.nodes],
      links: [...data.links].map(d => ({...d}))
    };

    // Color scale for nodes
    const color = d3.scaleOrdinal(d3[colorScheme] || d3.schemeCategory10);

    // Create the sankey generator
    const sankeyGenerator = sankey()
      .nodeWidth(nodeWidth)
      .nodePadding(nodePadding)
      .extent([[0, 0], [chartWidth, chartHeight]]);

    // Generate the sankey data
    const sankeyLayout = sankeyGenerator(sankeyData);
    const { nodes, links } = sankeyLayout;

    // Draw the links
    const link = chart.append('g')
      .attr('class', 'links')
      .attr('fill', 'none')
      .attr('stroke-opacity', 0.4)
      .selectAll('path')
      .data(links)
      .enter()
      .append('path')
      .attr('d', sankeyLinkHorizontal())
      .attr('stroke', d => color(d.source.name))
      .attr('stroke-width', d => Math.max(1, d.width))
      .style('opacity', 0)
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke-opacity', 0.8)
          .attr('stroke-width', d => Math.max(1, d.width + 2));

        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d.source.name} → ${d.target.name}</strong><br>
            ${formatValue(d.value, valueFormatter)}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke-opacity', 0.4)
          .attr('stroke-width', d => Math.max(1, d.width));
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onLinkClick) onLinkClick({
          source: d.source.name,
          target: d.target.name,
          value: d.value
        });
      });

    // Animate links
    link.transition()
      .duration(800)
      .style('opacity', 1);

    // Draw the nodes
    const node = chart.append('g')
      .attr('class', 'nodes')
      .selectAll('rect')
      .data(nodes)
      .enter()
      .append('rect')
      .attr('x', d => d.x0)
      .attr('y', d => d.y0)
      .attr('height', d => d.y1 - d.y0)
      .attr('width', d => d.x1 - d.x0)
      .attr('fill', d => color(d.name))
      .attr('stroke', d => d3.rgb(color(d.name)).darker(0.5))
      .style('opacity', 0)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('fill', d3.rgb(color(d.name)).brighter(0.2));
        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d.name}</strong><br>
            ${formatValue(d.value, valueFormatter)}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function(event, d) {
        d3.select(this).attr('fill', color(d.name));
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onNodeClick) onNodeClick({
          name: d.name,
          value: d.value
        });
      });

    // Animate nodes
    node.transition()
      .duration(800)
      .style('opacity', 0.8);

    // Add node labels
    chart.append('g')
      .attr('class', 'node-labels')
      .selectAll('text')
      .data(nodes)
      .enter()
      .append('text')
      .attr('x', d => d.x0 < chartWidth / 2 ? d.x1 + 6 : d.x0 - 6)
      .attr('y', d => (d.y1 + d.y0) / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', d => d.x0 < chartWidth / 2 ? 'start' : 'end')
      .attr('class', 'text-xs font-semibold')
      .text(d => d.name)
      .style('opacity', 0)
      .transition()
      .duration(800)
      .delay(400)
      .style('opacity', 1);

    // Cleanup function
    return () => {
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, onNodeClick, onLinkClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default SankeyDiagram;