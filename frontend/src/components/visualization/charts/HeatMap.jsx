import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { formatValue } from '../../../utils/formatting';

const HeatMap = ({ 
  data, 
  config = {}, 
  height = 400,
  onCellClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const {
    xKey = 'x',
    yKey = 'y',
    valueKey = 'value',
    title = '',
    colorScheme = 'interpolateViridis',
    showValues = true,
    valueFormatter,
    legendTitle = 'Value',
    margin = { top: 50, right: 80, bottom: 50, left: 80 }
  } = config;

  useEffect(() => {
    if (!data || !data.length || !svgRef.current) return;

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

    // Extract unique x and y values
    const xValues = Array.from(new Set(data.map(d => d[xKey])));
    const yValues = Array.from(new Set(data.map(d => d[yKey])));

    // Define scales
    const xScale = d3.scaleBand()
      .domain(xValues)
      .range([0, chartWidth])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(yValues)
      .range([0, chartHeight])
      .padding(0.05);

    // Get the values for the color scale
    const valueExtent = d3.extent(data, d => +d[valueKey]);
    const colorScale = d3.scaleSequential(d3[colorScheme] || d3.interpolateViridis)
      .domain(valueExtent);

    // Add X axis
    chart.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('class', 'text-xs')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    // Add Y axis
    chart.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .attr('class', 'text-xs');

    // Add the heatmap cells
    const cells = chart.selectAll('.cell')
      .data(data)
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(d[xKey]))
      .attr('y', d => yScale(d[yKey]))
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d[valueKey]))
      .attr('stroke', 'white')
      .attr('stroke-width', 1)
      .style('opacity', 0)
      .on('mouseover', function(event, d) {
        d3.select(this).style('stroke', '#000').style('stroke-width', 2);
        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d[xKey]} / ${d[yKey]}</strong>: ${formatValue(d[valueKey], valueFormatter)}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function() {
        d3.select(this).style('stroke', 'white').style('stroke-width', 1);
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onCellClick) onCellClick(d);
      });

    // Animate cells
    cells.transition()
      .duration(500)
      .delay((d, i) => i * 5)
      .style('opacity', 1);

    // Add values inside cells if showValues is true
    if (showValues) {
      chart.selectAll('.cell-value')
        .data(data)
        .enter()
        .append('text')
        .attr('class', 'cell-value text-xs')
        .attr('x', d => xScale(d[xKey]) + xScale.bandwidth() / 2)
        .attr('y', d => yScale(d[yKey]) + yScale.bandwidth() / 2)
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .text(d => formatValue(d[valueKey], valueFormatter))
        .style('fill', d => {
          // Determine text color based on background color
          const rgb = d3.color(colorScale(d[valueKey]));
          const brightness = (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000;
          return brightness > 125 ? '#000000' : '#ffffff';
        })
        .style('font-weight', 'bold')
        .style('opacity', 0)
        .transition()
        .duration(500)
        .delay((d, i) => i * 5 + 300)
        .style('opacity', 1);
    }

    // Add color legend
    const legendWidth = 20;
    const legendHeight = chartHeight / 2;
    
    const legend = svg.append('g')
      .attr('transform', `translate(${width - margin.right + 20}, ${margin.top + chartHeight / 4})`);
    
    // Create gradient for the legend
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('x1', '0%')
      .attr('y1', '100%')
      .attr('x2', '0%')
      .attr('y2', '0%');
    
    // Add color stops
    const colorStops = d3.range(0, 1.01, 0.1);
    colorStops.forEach(stop => {
      gradient.append('stop')
        .attr('offset', `${stop * 100}%`)
        .attr('stop-color', colorScale(valueExtent[0] + stop * (valueExtent[1] - valueExtent[0])));
    });
    
    // Add gradient rectangle
    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#heatmap-gradient)');
    
    // Add legend scale
    const legendScale = d3.scaleLinear()
      .domain(valueExtent)
      .range([legendHeight, 0]);
    
    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(d3.axisRight(legendScale).ticks(5).tickFormat(d => formatValue(d, valueFormatter)))
      .selectAll('text')
      .attr('class', 'text-xs');
    
    // Add legend title
    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-xs font-semibold')
      .text(legendTitle);

    // Cleanup function
    return () => {
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, onCellClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default HeatMap;