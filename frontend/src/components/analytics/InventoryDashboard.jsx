// frontend/src/components/analytics/InventoryDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Download, AlertCircle, TrendingUp, Package } from 'lucide-react';
import { analytics } from '../../services/api';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const InventoryDashboard = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Analysis states - all from backend
  const [safetyStock, setSafetyStock] = useState(null);
  const [abcAnalysis, setAbcAnalysis] = useState(null);
  const [forecast, setForecast] = useState(null);
  
  // Form states
  const [selectedProduct, setSelectedProduct] = useState('');
  const [forecastPeriod, setForecastPeriod] = useState(30);
  const [serviceLevel, setServiceLevel] = useState(0.95);
  const [leadTimeDays, setLeadTimeDays] = useState(7);
  const [products, setProducts] = useState([]);

  // Chart colors
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28'];

  useEffect(() => {
    fetchProducts();
  }, []);

  const fetchProducts = async () => {
    try {
      const response = await analytics.inventory.getProducts();
      // Fix: Extract the products array from the response
      setProducts(response.data.products || []);
    } catch (err) {
      console.error('Failed to fetch products:', err);
      setError(err.response?.data?.detail || 'Failed to fetch products');
      setProducts([]); // Set empty array on error
    }
  };

  const calculateSafetyStock = async () => {
    if (!selectedProduct) {
      setError('Please select a product');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await analytics.inventory.calculateSafetyStock({
        product_id: selectedProduct,
        service_level: serviceLevel,
        lead_time_days: leadTimeDays
      });
      setSafetyStock(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to calculate safety stock');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const performABCAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await analytics.inventory.performABCAnalysis({
        value_threshold_a: 0.8,
        value_threshold_b: 0.15
      });
      setAbcAnalysis(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to perform ABC analysis');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const generateForecast = async () => {
    if (!selectedProduct) {
      setError('Please select a product');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await analytics.inventory.generateForecast({
        product_id: selectedProduct,
        periods: forecastPeriod,
        method: 'auto'
      });
      setForecast(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to generate forecast');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const exportData = async (type) => {
    try {
      const response = await analytics.inventory.export(type);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `inventory_${type}_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to export data');
    }
  };

  // Helper function to prepare chart data
  const prepareABCChartData = () => {
    if (!abcAnalysis) return [];
    return [
      { name: 'Category A', value: abcAnalysis.category_a?.count || 0, percentage: abcAnalysis.category_a?.percentage || 0 },
      { name: 'Category B', value: abcAnalysis.category_b?.count || 0, percentage: abcAnalysis.category_b?.percentage || 0 },
      { name: 'Category C', value: abcAnalysis.category_c?.count || 0, percentage: abcAnalysis.category_c?.percentage || 0 }
    ];
  };

  return (
    <div className="space-y-6">
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Controls Section */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Select Product</label>
              <Select value={selectedProduct} onValueChange={setSelectedProduct}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a product" />
                </SelectTrigger>
                <SelectContent>
                  {products && products.length > 0 ? (
                    products.map((product) => (
                      <SelectItem key={product.id} value={product.id.toString()}>
                        {product.name} (SKU: {product.sku})
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="none" disabled>
                      No products available
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-sm font-medium mb-2 block">Forecast Period (days)</label>
              <Input
                type="number"
                value={forecastPeriod}
                onChange={(e) => setForecastPeriod(parseInt(e.target.value) || 30)}
                min="1"
                max="365"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Service Level</label>
              <Input
                type="number"
                value={serviceLevel}
                onChange={(e) => setServiceLevel(parseFloat(e.target.value) || 0.95)}
                min="0.5"
                max="0.99"
                step="0.01"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Lead Time (days)</label>
              <Input
                type="number"
                value={leadTimeDays}
                onChange={(e) => setLeadTimeDays(parseInt(e.target.value) || 7)}
                min="1"
                max="90"
              />
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <Button onClick={calculateSafetyStock} disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Calculate Safety Stock
            </Button>
            <Button onClick={performABCAnalysis} variant="outline" disabled={loading}>
              ABC Analysis
            </Button>
            <Button onClick={generateForecast} variant="outline" disabled={loading}>
              Generate Forecast
            </Button>
            <Button onClick={() => exportData('all')} variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Export All Data
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Safety Stock Results */}
      {safetyStock && (
        <Card>
          <CardHeader>
            <CardTitle>Safety Stock Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded">
                <p className="text-sm text-gray-600">Recommended Safety Stock</p>
                <p className="text-2xl font-bold text-blue-600">
                  {Math.round(safetyStock.safety_stock_quantity)} units
                </p>
              </div>
              <div className="text-center p-4 bg-green-50 rounded">
                <p className="text-sm text-gray-600">Reorder Point</p>
                <p className="text-2xl font-bold text-green-600">
                  {Math.round(safetyStock.reorder_point)} units
                </p>
              </div>
              <div className="text-center p-4 bg-orange-50 rounded">
                <p className="text-sm text-gray-600">Service Level</p>
                <p className="text-2xl font-bold text-orange-600">
                  {(safetyStock.service_level * 100).toFixed(0)}%
                </p>
              </div>
            </div>
            {safetyStock.confidence_interval && (
              <div className="mt-4 p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">
                  Confidence Interval: {Math.round(safetyStock.confidence_interval.lower)} - {Math.round(safetyStock.confidence_interval.upper)} units
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ABC Analysis Results */}
      {abcAnalysis && (
        <Card>
          <CardHeader>
            <CardTitle>ABC Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium mb-2">Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={prepareABCChartData()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percentage }) => `${name}: ${percentage.toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {prepareABCChartData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h3 className="text-sm font-medium mb-2">Category Details</h3>
                <div className="space-y-3">
                  {['category_a', 'category_b', 'category_c'].map((category, idx) => {
                    const data = abcAnalysis[category];
                    if (!data) return null;
                    const categoryName = category.split('_')[1].toUpperCase();
                    const bgColors = ['bg-blue-50', 'bg-green-50', 'bg-yellow-50'];
                    
                    return (
                      <div key={category} className={`p-3 ${bgColors[idx]} rounded`}>
                        <p className="font-medium">Category {categoryName}</p>
                        <p className="text-sm text-gray-600">
                          {data.count} items ({data.percentage.toFixed(1)}%)
                        </p>
                        <p className="text-sm">
                          Value: ${data.total_value.toLocaleString()}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Forecast Results */}
      {forecast && (
        <Card>
          <CardHeader>
            <CardTitle>Demand Forecast</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={forecast.forecast_data || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="historical" 
                  stroke="#8884d8" 
                  name="Historical Demand"
                  strokeWidth={2}
                />
                <Line 
                  type="monotone" 
                  dataKey="forecast" 
                  stroke="#82ca9d" 
                  name="Forecasted Demand"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                />
                {forecast.confidence_interval && (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="upper_bound" 
                      stroke="#ff7300" 
                      name="Upper Bound"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="lower_bound" 
                      stroke="#ff7300" 
                      name="Lower Bound"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                    />
                  </>
                )}
              </LineChart>
            </ResponsiveContainer>
            
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Forecast Method</p>
                <p className="font-medium">{forecast.method || 'Auto-selected'}</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Accuracy (MAPE)</p>
                <p className="font-medium">{forecast.accuracy?.toFixed(2) || 'N/A'}%</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Confidence Level</p>
                <p className="font-medium">{forecast.confidence_level || 95}%</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default InventoryDashboard;