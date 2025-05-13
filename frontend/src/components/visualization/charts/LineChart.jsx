import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { formatValue } from '../../../utils/formatting';

const LineChart = ({ 
  data, 
  config = {}, 
  height = 400,
  onPointClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const {
    xKey = 'date',
    yKey = 'value',
    title = '',
    color = '#4f46e5',
    curve = 'curveLinear',
    showArea = false,
    showPoints = true,
    areaOpacity = 0.1,
    valueFormatter,
    xAxisFormatter,
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

    // Define scales
    let xScale;
    // Check if data is time-based
    const isTimeData = data.some(d => d[xKey] instanceof Date) || 
      data.some(d => !isNaN(Date.parse(d[xKey])));
    
    if (isTimeData) {
      // Convert string dates to Date objects if necessary
      const dateData = data.map(d => ({
        ...d,
        [xKey]: d[xKey] instanceof Date ? d[xKey] : new Date(d[xKey])
      }));
      
      xScale = d3.scaleTime()
        .domain(d3.extent(dateData, d => d[xKey]))
        .range([0, chartWidth]);
        
      // Update data reference to use the date objects
      data = dateData;
    } else {
      // Numeric or categorical scale
      if (typeof data[0][xKey] === 'number') {
        xScale = d3.scaleLinear()
          .domain(d3.extent(data, d => +d[xKey]))
          .range([0, chartWidth]);
      } else {
        xScale = d3.scaleBand()
          .domain(data.map(d => d[xKey]))
          .range([0, chartWidth])
          .padding(0.1);
      }
    }

    // Y scale
    const yScale = d3.scaleLinear()
      .domain([0, d3.max(data, d => +d[yKey]) * 1.1]) // Add 10% padding at top
      .nice()
      .range([chartHeight, 0]);

    // Add X axis
    const xAxis = chart.append('g')
      .attr('transform', `translate(0,${chartHeight})`)
      .call(
        isTimeData 
          ? d3.axisBottom(xScale).tickFormat(xAxisFormatter ? d => xAxisFormatter(d) : d3.timeFormat('%b %d, %Y'))
          : d3.axisBottom(xScale).tickFormat(xAxisFormatter ? d => xAxisFormatter(d) : d => d)
      );
    
    xAxis.selectAll('text')
      .attr('class', 'text-xs')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    // Add Y axis
    chart.append('g')
      .call(d3.axisLeft(yScale).tickFormat(valueFormatter ? d => valueFormatter(d) : d => d))
      .selectAll('text')
      .attr('class', 'text-xs');

    // Create line generator
    const lineGenerator = d3.line()
      .x(d => xScale(d[xKey]) + (xScale.bandwidth ? xScale.bandwidth() / 2 : 0))
      .y(d => yScale(+d[yKey]))
      .curve(d3[curve] || d3.curveLinear);

    // Draw area if showArea is true
    if (showArea) {
      const areaGenerator = d3.area()
        .x(d => xScale(d[xKey]) + (xScale.bandwidth ? xScale.bandwidth() / 2 : 0))
        .y0(chartHeight)
        .y1(d => yScale(+d[yKey]))
        .curve(d3[curve] || d3.curveLinear);

      // Add the area path
      chart.append('path')
        .datum(data)
        .attr('class', 'area')
        .attr('fill', color)
        .attr('fill-opacity', 0)
        .attr('d', areaGenerator)
        .transition()
        .duration(1000)
        .attr('fill-opacity', areaOpacity);
    }

    // Add the line path
    const path = chart.append('path')
      .datum(data)
      .attr('class', 'line')
      .attr('fill', 'none')
      .attr('stroke', color)
      .attr('stroke-width', 2)
      .attr('d', lineGenerator);

    // Animate the line
    const pathLength = path.node().getTotalLength();
    path
      .attr('stroke-dasharray', pathLength + ' ' + pathLength)
      .attr('stroke-dashoffset', pathLength)
      .transition()
      .duration(1000)
      .attr('stroke-dashoffset', 0);

    // Add points if showPoints is true
    if (showPoints) {
      chart.selectAll('.point')
        .data(data)
        .enter()
        .append('circle')
        .attr('class', 'point')
        .attr('cx', d => xScale(d[xKey]) + (xScale.bandwidth ? xScale.bandwidth() / 2 : 0))
        .attr('cy', d => yScale(+d[yKey]))
        .attr('r', 0)
        .attr('fill', '#ffffff')
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .on('mouseover', function(event, d) {
          d3.select(this).attr('r', 6);
          tooltipRef.current
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
              <strong>${isTimeData ? d3.timeFormat('%b %d, %Y')(d[xKey]) : d[xKey]}</strong>: 
              ${formatValue(d[yKey], valueFormatter)}
            `);
          d3.select(tooltipRef.current).classed('hidden', false);
        })
        .on('mouseout', function() {
          d3.select(this).attr('r', 4);
          d3.select(tooltipRef.current).classed('hidden', true);
        })
        .on('click', function(event, d) {
          if (onPointClick) onPointClick(d);
        })
        .transition()
        .duration(800)
        .delay((d, i) => i * 50 + 300)
        .attr('r', 4);
    }

    // Add hover line for better interaction
    const hoverLine = chart.append('line')
      .attr('class', 'hover-line')
      .attr('x1', 0)
      .attr('x2', 0)
      .attr('y1', 0)
      .attr('y2', chartHeight)
      .style('stroke', '#ccc')
      .style('stroke-width', '1px')
      .style('stroke-dasharray', '3,3')
      .style('opacity', 0);

    // Add a transparent overlay for mouse interaction if not showing points
    chart.append('rect')
      .attr('class', 'overlay')
      .attr('width', chartWidth)
      .attr('height', chartHeight)
      .style('fill', 'none')
      .style('pointer-events', 'all')
      .on('mousemove', function(event) {
        const mouseX = d3.pointer(event)[0];
        
        // Find the closest data point
        let closestPoint;
        let minDistance = Infinity;
        
        data.forEach(d => {
          const x = xScale(d[xKey]) + (xScale.bandwidth ? xScale.bandwidth() / 2 : 0);
          const distance = Math.abs(x - mouseX);
          if (distance < minDistance) {
            minDistance = distance;
            closestPoint = d;
          }
        });
        
        if (closestPoint) {
          const x = xScale(closestPoint[xKey]) + (xScale.bandwidth ? xScale.bandwidth() / 2 : 0);
          
          // Update hover line
          hoverLine
            .attr('x1', x)
            .attr('x2', x)
            .style('opacity', 1);
          
          // Update tooltip
          tooltipRef.current
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px')
            .html(`
              <strong>${isTimeData ? d3.timeFormat('%b %d, %Y')(closestPoint[xKey]) : closestPoint[xKey]}</strong>: 
              ${formatValue(closestPoint[yKey], valueFormatter)}
            `);
          d3.select(tooltipRef.current).classed('hidden', false);
        }
      })
      .on('mouseout', function() {
        hoverLine.style('opacity', 0);
        d3.select(tooltipRef.current).classed('hidden', true);
      });

    // Cleanup function
    return () => {
      // Only remove the tooltip when component unmounts
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, onPointClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default LineChart;