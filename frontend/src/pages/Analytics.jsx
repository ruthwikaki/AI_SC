// frontend/src/pages/Analytics.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, TrendingUp, Package, Truck, Users } from 'lucide-react';
import { analytics } from '../services/api';

// Import analytics components
import InventoryDashboard from '../components/analytics/InventoryDashboard';
import LogisticsDashboard from '../components/analytics/LogisticsDashboard';
import SupplierDashboard from '../components/analytics/SupplierDashboard';

const Analytics = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('inventory');
  const [overviewData, setOverviewData] = useState({
    inventory: null,
    logistics: null,
    supplier: null
  });

  useEffect(() => {
    fetchOverviewData();
  }, []);

  const fetchOverviewData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [invResponse, logResponse, supResponse] = await Promise.all([
        analytics.inventory.getOverview(),
        analytics.logistics.getOverview(),
        analytics.supplier.getOverview()
      ]);

      setOverviewData({
        inventory: invResponse.data,
        logistics: logResponse.data,
        supplier: supResponse.data
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load analytics data');
      console.error('Analytics error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Supply Chain Analytics</h1>
        <p className="text-gray-600">
          Comprehensive analytics and insights for your supply chain operations
        </p>
      </div>

      {error && (
        <Alert className="mb-4" variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Quick Stats Overview - Data from backend */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Inventory Value</p>
                <p className="text-2xl font-bold">
                  ${overviewData.inventory?.total_value?.toLocaleString() || '0'}
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
                <p className="text-sm text-gray-600">Active Shipments</p>
                <p className="text-2xl font-bold">
                  {overviewData.logistics?.active_shipments || 0}
                </p>
              </div>
              <Truck className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Supplier Performance</p>
                <p className="text-2xl font-bold">
                  {overviewData.supplier?.average_score?.toFixed(1) || '0'}%
                </p>
              </div>
              <Users className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Cost Savings</p>
                <p className="text-2xl font-bold">
                  ${overviewData.inventory?.cost_savings?.toLocaleString() || '0'}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Analytics Tabs */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle>Detailed Analytics</CardTitle>
            <Button onClick={fetchOverviewData} variant="outline" size="sm">
              Refresh Data
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="inventory">Inventory Optimization</TabsTrigger>
              <TabsTrigger value="logistics">Logistics Analytics</TabsTrigger>
              <TabsTrigger value="supplier">Supplier Performance</TabsTrigger>
            </TabsList>

            <TabsContent value="inventory" className="mt-6">
              <InventoryDashboard />
            </TabsContent>

            <TabsContent value="logistics" className="mt-6">
              <LogisticsDashboard />
            </TabsContent>

            <TabsContent value="supplier" className="mt-6">
              <SupplierDashboard />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default Analytics;