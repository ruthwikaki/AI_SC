import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { formatValue } from '../../../utils/formatting';

const BarChart = ({ 
  data, 
  config = {}, 
  height = 400,
  onBarClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const {
    xKey = 'name',
    yKey = 'value',
    title = '',
    color = '#4f46e5',
    horizontal = false,
    showValues = true,
    valueFormatter,
    margin = { top: 30, right: 30, bottom: 50, left: 60 }
  } = config;

  useEffect(() => {
    if (!data || !data.length || !svgRef.current) return;

    // Clear existing chart if any
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
      chart.append('text')
        .attr('x', chartWidth / 2)
        .attr('y', -margin.top / 2)
        .attr('text-anchor', 'middle')
        .attr('class', 'text-sm font-semibold')
        .text(title);
    }

    // For horizontal bar chart, we swap x and y
    if (horizontal) {
      // X scale (values)
      const xScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => +d[yKey])])
        .nice()
        .range([0, chartWidth]);

      // Y scale (categories)
      const yScale = d3.scaleBand()
        .domain(data.map(d => d[xKey]))
        .range([0, chartHeight])
        .padding(0.2);

      // Add X axis
      chart.append('g')
        .attr('transform', `translate(0,${chartHeight})`)
        .call(d3.axisBottom(xScale))
        .selectAll('text')
        .attr('class', 'text-xs');

      // Add Y axis
      chart.append('g')
        .call(d3.axisLeft(yScale))
        .selectAll('text')
        .attr('class', 'text-xs');

      // Add bars
      const bars = chart.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('y', d => yScale(d[xKey]))
        .attr('height', yScale.bandwidth())
        .attr('x', 0)
        .attr('fill', color)
        .attr('width', 0) // Start at 0 for animation
        .on('mouseover', function(event, d) {
          d3.select(this).attr('fill', d3.color(color).darker(0.2));
          tooltipRef.current
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`<strong>${d[xKey]}</strong>: ${formatValue(d[yKey], valueFormatter)}`);
          d3.select(tooltipRef.current).classed('hidden', false);
        })
        .on('mouseout', function() {
          d3.select(this).attr('fill', color);
          d3.select(tooltipRef.current).classed('hidden', true);
        })
        .on('click', function(event, d) {
          if (onBarClick) onBarClick(d);
        });

      // Animate bars
      bars.transition()
        .duration(800)
        .attr('width', d => xScale(+d[yKey]))
        .delay((d, i) => i * 50);

      // Add bar labels if showValues is true
      if (showValues) {
        chart.selectAll('.bar-label')
          .data(data)
          .enter()
          .append('text')
          .attr('class', 'bar-label text-xs')
          .attr('y', d => yScale(d[xKey]) + yScale.bandwidth() / 2)
          .attr('x', d => xScale(+d[yKey]) + 5)
          .attr('dy', '.35em')
          .attr('opacity', 0)
          .text(d => formatValue(d[yKey], valueFormatter))
          .transition()
          .duration(800)
          .delay((d, i) => i * 50 + 400)
          .attr('opacity', 1);
      }
    } else {
      // X scale (categories)
      const xScale = d3.scaleBand()
        .domain(data.map(d => d[xKey]))
        .range([0, chartWidth])
        .padding(0.2);

      // Y scale (values)
      const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => +d[yKey])])
        .nice()
        .range([chartHeight, 0]);

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

      // Add bars
      const bars = chart.selectAll('.bar')
        .data(data)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => xScale(d[xKey]))
        .attr('width', xScale.bandwidth())
        .attr('y', chartHeight) // Start at bottom for animation
        .attr('height', 0) // Start with 0 height for animation
        .attr('fill', color)
        .on('mouseover', function(event, d) {
          d3.select(this).attr('fill', d3.color(color).darker(0.2));
          tooltipRef.current
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`<strong>${d[xKey]}</strong>: ${formatValue(d[yKey], valueFormatter)}`);
          d3.select(tooltipRef.current).classed('hidden', false);
        })
        .on('mouseout', function() {
          d3.select(this).attr('fill', color);
          d3.select(tooltipRef.current).classed('hidden', true);
        })
        .on('click', function(event, d) {
          if (onBarClick) onBarClick(d);
        });

      // Animate bars
      bars.transition()
        .duration(800)
        .attr('y', d => yScale(+d[yKey]))
        .attr('height', d => chartHeight - yScale(+d[yKey]))
        .delay((d, i) => i * 50);

      // Add bar labels if showValues is true
      if (showValues) {
        chart.selectAll('.bar-label')
          .data(data)
          .enter()
          .append('text')
          .attr('class', 'bar-label text-xs')
          .attr('text-anchor', 'middle')
          .attr('x', d => xScale(d[xKey]) + xScale.bandwidth() / 2)
          .attr('y', d => yScale(+d[yKey]) - 5)
          .attr('opacity', 0)
          .text(d => formatValue(d[yKey], valueFormatter))
          .transition()
          .duration(800)
          .delay((d, i) => i * 50 + 400)
          .attr('opacity', 1);
      }
    }

    // Cleanup function
    return () => {
      // Only remove the tooltip when component unmounts
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, onBarClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default BarChart;