// frontend/src/components/forecasting/widgets/ProductPyramid.jsx
import React, { useState, useEffect } from 'react';

const ProductPyramid = ({ data, onCategorySelect, selectedCategory }) => {
  const [pyramidData, setPyramidData] = useState([]);
  const [hoveredCategory, setHoveredCategory] = useState(null);

  useEffect(() => {
    if (data && data.results) {
      // Process data to create pyramid structure
      const categories = processCategoriesForPyramid(data.results);
      setPyramidData(categories);
    }
  }, [data]);

  function processCategoriesForPyramid(results) {
    // Extract categories and their metrics
    const categoryMap = new Map();
    
    const products = results.products || [];
    products.forEach(product => {
      const category = product.category || 'Uncategorized';
      if (!categoryMap.has(category)) {
        categoryMap.set(category, {
          name: category,
          productCount: 0,
          totalValue: 0,
          forecastValue: 0,
          abcDistribution: { A: 0, B: 0, C: 0 },
        });
      }
      
      const cat = categoryMap.get(category);
      cat.productCount++;
      cat.totalValue += product.stock_value || 0;
      cat.forecastValue += product.forecast_total || 0;
      cat.abcDistribution[product.abc_class || 'C']++;
    });
    
    // Convert to array and sort by value
    const categories = Array.from(categoryMap.values())
      .sort((a, b) => b.totalValue - a.totalValue)
      .slice(0, 5); // Top 5 categories
    
    // Calculate percentages for pyramid levels
    const totalValue = categories.reduce((sum, cat) => sum + cat.totalValue, 0);
    
    return categories.map((cat, index) => ({
      ...cat,
      level: index,
      percentage: (cat.totalValue / totalValue * 100).toFixed(1),
      color: getColorForLevel(index),
    }));
  }

  function getColorForLevel(level) {
    const colors = [
      '#DC2626', // red-600
      '#F59E0B', // amber-500
      '#10B981', // emerald-500
      '#3B82F6', // blue-500
      '#8B5CF6', // violet-500
    ];
    return colors[level] || '#6B7280';
  }

  const handleCategoryClick = (category) => {
    onCategorySelect(category.name);
  };

  return (
    <div className="space-y-6">
      {/* Pyramid Visualization */}
      <div className="relative">
        <svg width="100%" height="300" viewBox="0 0 300 250">
          {pyramidData.map((category, index) => {
            const width = 280 - (index * 50);
            const x = (300 - width) / 2;
            const y = index * 45 + 10;
            const isSelected = selectedCategory === category.name;
            const isHovered = hoveredCategory === category.name;
            
            return (
              <g key={category.name}>
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height="40"
                  fill={category.color}
                  opacity={isSelected ? 1 : (isHovered ? 0.8 : 0.7)}
                  stroke={isSelected ? '#1F2937' : 'none'}
                  strokeWidth={isSelected ? 2 : 0}
                  className="cursor-pointer transition-all duration-200"
                  onMouseEnter={() => setHoveredCategory(category.name)}
                  onMouseLeave={() => setHoveredCategory(null)}
                  onClick={() => handleCategoryClick(category)}
                />
                <text
                  x="150"
                  y={y + 25}
                  textAnchor="middle"
                  className="fill-white text-sm font-medium pointer-events-none"
                >
                  {category.name} ({category.percentage}%)
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Category Details */}
      <div className="space-y-3">
        {pyramidData.map((category) => {
          const isSelected = selectedCategory === category.name;
          
          return (
            <div
              key={category.name}
              className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                isSelected 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
              }`}
              onClick={() => handleCategoryClick(category)}
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-medium text-gray-900">{category.name}</h4>
                <div
                  className="w-4 h-4 rounded"
                  style={{ backgroundColor: category.color }}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-500">Products:</span>
                  <span className="ml-1 font-medium">{category.productCount}</span>
                </div>
                <div>
                  <span className="text-gray-500">Value:</span>
                  <span className="ml-1 font-medium">
                    ${(category.totalValue / 1000).toFixed(0)}K
                  </span>
                </div>
              </div>
              
              {/* ABC Distribution */}
              <div className="mt-2 flex items-center space-x-2">
                <span className="text-xs text-gray-500">ABC:</span>
                <div className="flex space-x-1">
                  {['A', 'B', 'C'].map(cls => (
                    <span
                      key={cls}
                      className={`text-xs px-2 py-1 rounded ${
                        cls === 'A' ? 'bg-red-100 text-red-700' :
                        cls === 'B' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-green-100 text-green-700'
                      }`}
                    >
                      {cls}: {category.abcDistribution[cls]}
                    </span>
                  ))}
                </div>
              </div>
              
              {/* Forecast Trend */}
              <div className="mt-2 flex items-center justify-between text-sm">
                <span className="text-gray-500">Forecast:</span>
                <span className={`font-medium ${
                  category.forecastValue > category.totalValue 
                    ? 'text-green-600' 
                    : 'text-red-600'
                }`}>
                  {category.forecastValue > category.totalValue ? '↑' : '↓'}
                  {' '}
                  {Math.abs(((category.forecastValue - category.totalValue) / category.totalValue * 100)).toFixed(1)}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Legend */}
      <div className="border-t pt-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Legend</h4>
        <div className="space-y-1 text-xs text-gray-600">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-red-100 rounded mr-2" />
            <span>A-Class: High value, low quantity (80% value)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-100 rounded mr-2" />
            <span>B-Class: Medium value/quantity (15% value)</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-100 rounded mr-2" />
            <span>C-Class: Low value, high quantity (5% value)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProductPyramid;
