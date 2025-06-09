// frontend/src/components/forecasting/ProductPyramid.jsx
import React, { useState, useEffect, useMemo } from 'react';
import { getProductCategories } from '../../services/forecasting';
import Loading from '../common/Loading';

const ProductPyramid = ({ onCategorySelect, selectedCategory }) => {
  const [categoryData, setCategoryData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCategories();
  }, []);

  const fetchCategories = async () => {
    try {
      const data = await getProductCategories();
      
      // Sort by value and take top 5-7 categories
      const sortedData = data
        .sort((a, b) => b.value - a.value)
        .slice(0, Math.min(7, data.length));
      
      // Assign colors dynamically
      const colors = [
        'bg-blue-500',
        'bg-green-500',
        'bg-yellow-500',
        'bg-purple-500',
        'bg-red-500',
        'bg-indigo-500',
        'bg-pink-500'
      ];
      
      const enrichedData = sortedData.map((cat, index) => ({
        ...cat,
        color: colors[index % colors.length]
      }));
      
      setCategoryData(enrichedData);
    } catch (error) {
      console.error('Error fetching categories:', error);
    } finally {
      setLoading(false);
    }
  };

  const totalValue = useMemo(() => 
    categoryData.reduce((sum, cat) => sum + cat.value, 0), 
    [categoryData]
  );

  if (loading) return <Loading />;
  
  if (categoryData.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Product Category Pyramid</h3>
        <p className="text-gray-500">No category data available</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold mb-4">Product Category Pyramid</h3>
      
      <div className="mb-6">
        <div className="text-sm text-gray-600 mb-2">Total Inventory Value</div>
        <div className="text-2xl font-bold text-gray-900">
          ${totalValue > 1000000 
            ? `${(totalValue / 1000000).toFixed(2)}M` 
            : totalValue > 1000 
            ? `${(totalValue / 1000).toFixed(2)}K`
            : totalValue.toFixed(2)
          }
        </div>
      </div>

      <div className="space-y-3">
        {categoryData.map((category, index) => {
          const isSelected = selectedCategory === category.name;
          const widthPercentage = 100 - (index * 12); // Dynamic width based on rank
          
          return (
            <div
              key={category.name}
              onClick={() => onCategorySelect(isSelected ? null : category.name)}
              className={`
                relative cursor-pointer transition-all duration-200
                ${isSelected ? 'transform scale-105' : 'hover:transform hover:scale-102'}
              `}
            >
              <div
                className={`
                  ${category.color} text-white rounded-lg p-3
                  ${isSelected ? 'ring-2 ring-offset-2 ring-blue-500' : ''}
                `}
                style={{ width: `${widthPercentage}%`, margin: '0 auto' }}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <div className="font-semibold">{category.name}</div>
                    <div className="text-sm opacity-90">
                      {category.product_count} products
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{category.percentage.toFixed(1)}%</div>
                    <div className="text-sm opacity-90">
                      ${category.value > 1000000 
                        ? `${(category.value / 1000000).toFixed(1)}M` 
                        : category.value > 1000 
                        ? `${(category.value / 1000).toFixed(1)}K`
                        : category.value.toFixed(0)
                      }
                    </div>
                  </div>
                </div>
              </div>
              
              {/* ABC Distribution */}
              {isSelected && category.abc_distribution && (
                <div className="mt-2 bg-gray-50 rounded-lg p-3 mx-auto" 
                     style={{ width: `${widthPercentage}%` }}>
                  <div className="text-sm font-medium text-gray-700 mb-1">
                    ABC Distribution
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-green-600">
                      A: {category.abc_distribution.A || 0} items
                    </span>
                    <span className="text-yellow-600">
                      B: {category.abc_distribution.B || 0} items
                    </span>
                    <span className="text-red-600">
                      C: {category.abc_distribution.C || 0} items
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="mt-6 text-sm text-gray-600">
        <div className="flex items-center mb-2">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
          <span>Click category to filter grid</span>
        </div>
        {selectedCategory && (
          <div className="flex items-center text-blue-600">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
            <span>Showing: {selectedCategory}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ProductPyramid;
