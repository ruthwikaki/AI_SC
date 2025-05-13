import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { formatValue } from '../../../utils/formatting';

const PieChart = ({ 
  data, 
  config = {}, 
  height = 400,
  onSliceClick
}) => {
  const svgRef = useRef(null);
  const tooltipRef = useRef(null);

  const {
    nameKey = 'name',
    valueKey = 'value',
    title = '',
    colorScheme = 'schemeSet2',
    innerRadius = 0, // 0 for pie, >0 for donut
    padAngle = 0.02,
    cornerRadius = 4,
    showLabels = true,
    showValues = true,
    valueFormatter,
    showPercentages = true,
    margin = { top: 30, right: 30, bottom: 30, left: 30 }
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
    const radius = Math.min(chartWidth, chartHeight) / 2;

    // Create chart group
    const chart = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    // Add title if provided
    if (title) {
      svg.append('text')
        .attr('x', width / 2)
        .attr('y', margin.top / 2)
        .attr('text-anchor', 'middle')
        .attr('class', 'text-sm font-semibold')
        .text(title);
    }

    // Calculate total for percentages
    const total = d3.sum(data, d => +d[valueKey]);
    
    // Prepare the pie layout
    const pie = d3.pie()
      .value(d => +d[valueKey])
      .sort(null);
    
    // Color scheme
    const colorScale = d3.scaleOrdinal(d3[colorScheme] || d3.schemeSet2);
    
    // Create the arc generator
    const arc = d3.arc()
      .innerRadius(innerRadius * radius)
      .outerRadius(radius)
      .padAngle(padAngle)
      .cornerRadius(cornerRadius);
    
    // Larger arc for labels
    const outerArc = d3.arc()
      .innerRadius(radius * 0.9)
      .outerRadius(radius * 0.9);
    
    // Add the slices
    const slices = chart.selectAll('.slice')
      .data(pie(data))
      .enter()
      .append('g')
      .attr('class', 'slice');
    
    // Add the paths for the slices
    const paths = slices.append('path')
      .attr('d', arc)
      .attr('fill', (d, i) => colorScale(i))
      .attr('stroke', 'white')
      .style('stroke-width', '2px')
      .style('opacity', 0.8)
      .on('mouseover', function(event, d) {
        d3.select(this).style('opacity', 1);
        const percentage = ((d.data[valueKey] / total) * 100).toFixed(1);
        tooltipRef.current
          .style('opacity', 1)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
          .html(`
            <strong>${d.data[nameKey]}</strong>: ${formatValue(d.data[valueKey], valueFormatter)}
            ${showPercentages ? `<br>(${percentage}%)` : ''}
          `);
        d3.select(tooltipRef.current).classed('hidden', false);
      })
      .on('mouseout', function() {
        d3.select(this).style('opacity', 0.8);
        d3.select(tooltipRef.current).classed('hidden', true);
      })
      .on('click', function(event, d) {
        if (onSliceClick) onSliceClick(d.data);
      });
    
    // Animate slices
    paths
      .transition()
      .duration(1000)
      .attrTween('d', function(d) {
        const interpolate = d3.interpolate({ startAngle: 0, endAngle: 0 }, d);
        return function(t) {
          return arc(interpolate(t));
        };
      });

    // Add labels if showLabels is true
    if (showLabels) {
      // For outer labels with lines
      const polyline = slices.append('polyline')
        .attr('points', function(d) {
          const pos = outerArc.centroid(d);
          pos[0] = radius * 0.95 * (midAngle(d) < Math.PI ? 1 : -1);
          return [arc.centroid(d), outerArc.centroid(d), pos];
        })
        .attr('stroke', 'gray')
        .attr('fill', 'none')
        .attr('stroke-width', 1)
        .style('opacity', 0)
        .transition()
        .delay(1000)
        .duration(500)
        .style('opacity', 0.5);

      const labels = slices.append('text')
        .attr('transform', function(d) {
          const pos = outerArc.centroid(d);
          pos[0] = radius * (midAngle(d) < Math.PI ? 1.05 : -1.05);
          return `translate(${pos})`;
        })
        .attr('dy', '.35em')
        .attr('text-anchor', function(d) {
          return midAngle(d) < Math.PI ? 'start' : 'end';
        })
        .attr('class', 'text-xs')
        .text(function(d) {
          return d.data[nameKey];
        })
        .style('opacity', 0)
        .transition()
        .delay(1000)
        .duration(500)
        .style('opacity', 1);

      // Helper function for midpoint angle
      function midAngle(d) {
        return d.startAngle + (d.endAngle - d.startAngle) / 2;
      }
    }

    // Add value labels in the center for each slice if showValues and it's a donut chart
    if (showValues && innerRadius > 0) {
      slices.append('text')
        .attr('transform', function(d) {
          const c = arc.centroid(d);
          return `translate(${c})`;
        })
        .attr('dy', '.35em')
        .attr('text-anchor', 'middle')
        .attr('class', 'text-xs font-semibold')
        .text(function(d) {
          if (showPercentages) {
            const percentage = ((d.data[valueKey] / total) * 100).toFixed(1);
            return `${percentage}%`;
          } else {
            return formatValue(d.data[valueKey], valueFormatter);
          }
        })
        .style('opacity', 0)
        .style('fill', 'white')
        .transition()
        .delay(1200)
        .duration(300)
        .style('opacity', 1);
    }

    // Add center text if it's a donut chart
    if (innerRadius > 0) {
      chart.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', '.35em')
        .attr('class', 'text-sm font-semibold')
        .text(formatValue(total, valueFormatter))
        .style('opacity', 0)
        .transition()
        .delay(1200)
        .duration(300)
        .style('opacity', 1);
    }

    // Cleanup function
    return () => {
      if (tooltipRef.current) {
        d3.select(tooltipRef.current).remove();
        tooltipRef.current = null;
      }
    };
  }, [data, config, height, onSliceClick]);

  return (
    <div className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  );
};

export default PieChart;