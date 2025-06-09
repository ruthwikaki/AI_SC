// frontend/src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Alert, AlertDescription } from '../components/ui/alert';
import { 
  Package, Truck, Users, TrendingUp, AlertCircle, 
  DollarSign, ShoppingCart, Clock, CheckCircle 
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState({
    overview: null,
    recentOrders: [],
    inventoryAlerts: { low_stock: [], overstock: [] },
    supplierMetrics: null,
    logistics: null
  });

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch all dashboard data in parallel
      const [overviewRes, ordersRes, alertsRes, supplierRes, logisticsRes] = await Promise.all([
        api.get('/api/dashboard/overview'),
        api.get('/api/dashboard/recent-orders'),
        api.get('/api/dashboard/inventory-alerts'),
        api.get('/api/dashboard/supplier-metrics'),
        api.get('/api/dashboard/logistics-summary')
      ]);

      // Extract the actual data from responses
      setDashboardData({
        overview: overviewRes.data.metrics || {},
        recentOrders: ordersRes.data.recentOrders || ordersRes.data.orders || [],
        inventoryAlerts: alertsRes.data || { low_stock: [], overstock: [] },
        supplierMetrics: supplierRes.data || null,
        logistics: logisticsRes.data || null
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load dashboard data');
      console.error('Dashboard error:', err);
      // Set default data on error
      setDashboardData({
        overview: {},
        recentOrders: [],
        inventoryAlerts: { low_stock: [], overstock: [] },
        supplierMetrics: null,
        logistics: null
      });
    } finally {
      setLoading(false);
    }
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  // Transform inventory alerts to expected format
  const transformedAlerts = [
    ...dashboardData.inventoryAlerts.low_stock?.map(item => ({
      type: 'critical',
      product: item.name,
      message: `Low stock: ${item.current_stock} units (Reorder point: ${item.reorder_point})`
    })) || [],
    ...dashboardData.inventoryAlerts.overstock?.map(item => ({
      type: 'warning',
      product: item.name,
      message: `Overstock: ${item.current_stock} units`
    })) || []
  ];

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Supply Chain Dashboard</h1>
        <p className="text-gray-600">
          Welcome back! Here's your supply chain overview.
        </p>
      </div>

      {error && (
        <Alert className="mb-4" variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Inventory Value</p>
                <p className="text-2xl font-bold">
                  ${dashboardData.overview?.total_inventory_value?.toLocaleString() || '0'}
                </p>
                <p className="text-xs text-green-600 mt-1">
                  From {dashboardData.overview?.active_suppliers || 0} suppliers
                </p>
              </div>
              <Package className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Active Orders</p>
                <p className="text-2xl font-bold">
                  {dashboardData.overview?.active_orders || 0}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  In progress
                </p>
              </div>
              <ShoppingCart className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Low Stock Items</p>
                <p className="text-2xl font-bold">
                  {dashboardData.overview?.low_stock_items || 0}
                </p>
                <p className="text-xs text-red-600 mt-1">
                  Need reordering
                </p>
              </div>
              <AlertCircle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Suppliers</p>
                <p className="text-2xl font-bold">
                  {dashboardData.overview?.active_suppliers || 0}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Active partners
                </p>
              </div>
              <Users className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Inventory Alerts */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Inventory Alerts</span>
              <Button size="sm" variant="outline" onClick={() => navigate('/analytics')}>
                View All
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {transformedAlerts.length > 0 ? (
                transformedAlerts.slice(0, 5).map((alert, idx) => (
                  <div key={idx} className={`p-3 rounded-lg border ${
                    alert.type === 'critical' ? 'border-red-200 bg-red-50' :
                    alert.type === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                    'border-blue-200 bg-blue-50'
                  }`}>
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="font-medium text-sm">{alert.product}</p>
                        <p className="text-xs text-gray-600 mt-1">{alert.message}</p>
                      </div>
                      <AlertCircle className={`h-4 w-4 ${
                        alert.type === 'critical' ? 'text-red-600' :
                        alert.type === 'warning' ? 'text-yellow-600' :
                        'text-blue-600'
                      }`} />
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-500" />
                  <p className="text-sm">No alerts at this time</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Order Status Distribution */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Delivery Performance</CardTitle>
          </CardHeader>
          <CardContent>
            {dashboardData.logistics?.delivery_performance ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-2xl font-bold text-green-600">
                      {dashboardData.logistics.delivery_performance.completed || 0}
                    </p>
                    <p className="text-sm text-gray-600">Completed</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-600">
                      {dashboardData.logistics.delivery_performance.in_transit || 0}
                    </p>
                    <p className="text-sm text-gray-600">In Transit</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-yellow-600">
                      {dashboardData.logistics.delivery_performance.pending || 0}
                    </p>
                    <p className="text-sm text-gray-600">Pending</p>
                  </div>
                </div>
                <div className="mt-4 p-4 bg-gray-50 rounded">
                  <p className="text-sm text-gray-600">
                    Total Deliveries (Last 30 days): {dashboardData.logistics.delivery_performance.total_deliveries || 0}
                  </p>
                </div>
              </div>
            ) : (
              <div className="h-[200px] flex items-center justify-center text-gray-500">
                No delivery data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Orders and Supplier Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {/* Recent Orders */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Recent Orders</span>
              <Button size="sm" variant="outline" onClick={() => navigate('/query')}>
                View All
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left pb-2">Order #</th>
                    <th className="text-left pb-2">Supplier</th>
                    <th className="text-left pb-2">Status</th>
                    <th className="text-right pb-2">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {dashboardData.recentOrders && dashboardData.recentOrders.length > 0 ? (
                    dashboardData.recentOrders.slice(0, 5).map((order) => (
                      <tr key={order.id} className="border-b">
                        <td className="py-2 font-medium">{order.order_number || order.id}</td>
                        <td className="py-2">{order.supplier_name || 'N/A'}</td>
                        <td className="py-2">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            order.status === 'delivered' ? 'bg-green-100 text-green-800' :
                            order.status === 'shipped' ? 'bg-blue-100 text-blue-800' :
                            order.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {order.status}
                          </span>
                        </td>
                        <td className="py-2 text-right">${order.total_amount?.toLocaleString() || '0'}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="py-4 text-center text-gray-500">
                        No recent orders
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Supplier Performance */}
        <Card>
          <CardHeader>
            <CardTitle>Top Suppliers by Order Volume</CardTitle>
          </CardHeader>
          <CardContent>
            {dashboardData.supplierMetrics?.suppliers && dashboardData.supplierMetrics.suppliers.length > 0 ? (
              <div className="space-y-3">
                {dashboardData.supplierMetrics.suppliers.slice(0, 5).map((supplier) => (
                  <div key={supplier.id} className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-sm">{supplier.name}</p>
                      <p className="text-xs text-gray-600">
                        {supplier.order_count} orders • Rating: {supplier.rating}/5
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-sm">${supplier.total_business.toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-[200px] flex items-center justify-center text-gray-500">
                No supplier data available
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <div className="mt-6">
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Button 
                variant="outline" 
                className="h-20 flex-col"
                onClick={() => navigate('/analytics')}
              >
                <TrendingUp className="h-6 w-6 mb-2" />
                <span className="text-xs">View Analytics</span>
              </Button>
              <Button 
                variant="outline" 
                className="h-20 flex-col"
                onClick={() => navigate('/query')}
              >
                <Package className="h-6 w-6 mb-2" />
                <span className="text-xs">Query Data</span>
              </Button>
              <Button 
                variant="outline" 
                className="h-20 flex-col"
                onClick={() => navigate('/multi-tier')}
              >
                <Users className="h-6 w-6 mb-2" />
                <span className="text-xs">Supplier Network</span>
              </Button>
              <Button 
                variant="outline" 
                className="h-20 flex-col"
                onClick={() => navigate('/reports')}
              >
                <Clock className="h-6 w-6 mb-2" />
                <span className="text-xs">Generate Reports</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;