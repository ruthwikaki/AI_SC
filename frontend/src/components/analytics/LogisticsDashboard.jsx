// frontend/src/components/analytics/LogisticsDashboard.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, MapPin, Truck, Clock, DollarSign, AlertCircle } from 'lucide-react';
import api from '../../services/api';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const LogisticsDashboard = ({ data }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Analysis states
  const [routeOptimization, setRouteOptimization] = useState(null);
  const [carrierPerformance, setCarrierPerformance] = useState(null);
  const [deliveryAnalytics, setDeliveryAnalytics] = useState(null);
  
  // Form states
  const [selectedCarrier, setSelectedCarrier] = useState('');
  const [selectedRoute, setSelectedRoute] = useState('');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [carriers, setCarriers] = useState([]);
  const [routes, setRoutes] = useState([]);

  useEffect(() => {
    fetchCarriersAndRoutes();
  }, []);

  const fetchCarriersAndRoutes = async () => {
    try {
      const [carriersRes, routesRes] = await Promise.all([
        api.get('/api/analytics/logistics/carriers'),
        api.get('/api/analytics/logistics/routes')
      ]);
      // Fix: Extract the arrays from the response
      setCarriers(carriersRes.data.carriers || []);
      setRoutes(routesRes.data.routes || []);
    } catch (err) {
      console.error('Failed to fetch carriers/routes:', err);
      setCarriers([]);
      setRoutes([]);
    }
  };

  const optimizeRoutes = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/logistics/optimize-routes', {
        max_vehicles: 10,
        vehicle_capacity: 1000,
        optimization_objective: 'minimize_cost'
      });
      setRouteOptimization(response.data);
    } catch (err) {
      setError('Failed to optimize routes');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeCarrierPerformance = async () => {
    if (!selectedCarrier) {
      setError('Please select a carrier');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/logistics/carrier-performance', {
        carrier_id: selectedCarrier,
        start_date: dateRange.start,
        end_date: dateRange.end,
        metrics: ['on_time_delivery', 'damage_rate', 'cost_per_mile']
      });
      setCarrierPerformance(response.data);
    } catch (err) {
      setError('Failed to analyze carrier performance');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const analyzeDeliveries = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.post('/api/analytics/logistics/delivery-analytics', {
        start_date: dateRange.start,
        end_date: dateRange.end,
        group_by: 'day'
      });
      setDeliveryAnalytics(response.data);
    } catch (err) {
      setError('Failed to analyze deliveries');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const exportRouteOptimization = async () => {
    if (!routeOptimization) return;
    
    try {
      const response = await api.post('/api/analytics/logistics/export/routes', {
        route_data: routeOptimization
      }, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `optimized_routes_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      setError('Failed to export routes');
    }
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
          <CardTitle>Logistics Analysis Controls</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Select Carrier</label>
              <Select value={selectedCarrier} onValueChange={setSelectedCarrier}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose a carrier" />
                </SelectTrigger>
                <SelectContent>
                  {carriers && carriers.length > 0 ? (
                    carriers.map((carrier) => (
                      <SelectItem key={carrier.name} value={carrier.name}>
                        {carrier.name}
                      </SelectItem>
                    ))
                  ) : (
                    <SelectItem value="none" disabled>
                      No carriers available
                    </SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <label className="text-sm font-medium mb-2 block">Start Date</label>
              <Input
                type="date"
                value={dateRange.start}
                onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">End Date</label>
              <Input
                type="date"
                value={dateRange.end}
                onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
              />
            </div>
          </div>

          <div className="mt-4 flex gap-2">
            <Button onClick={optimizeRoutes} disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Optimize Routes
            </Button>
            <Button onClick={analyzeCarrierPerformance} variant="outline" disabled={loading}>
              Analyze Carrier
            </Button>
            <Button onClick={analyzeDeliveries} variant="outline" disabled={loading}>
              Delivery Analytics
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Carrier Overview - Show available carriers */}
      {carriers && carriers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Available Carriers</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {carriers.map((carrier) => (
                <div key={carrier.name} className="p-4 border rounded-lg">
                  <h4 className="font-medium">{carrier.name}</h4>
                  <p className="text-sm text-gray-600 mt-1">
                    Shipments: {carrier.shipment_count}
                  </p>
                  <p className="text-sm text-gray-600">
                    Performance: {carrier.performance_score}%
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Route Optimization Results */}
      {routeOptimization && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <CardTitle>Optimized Routes</CardTitle>
              <Button onClick={exportRouteOptimization} size="sm" variant="outline">
                Export Routes
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-blue-50 rounded">
                <MapPin className="h-6 w-6 mx-auto mb-2 text-blue-600" />
                <p className="text-sm text-gray-600">Total Stops</p>
                <p className="text-2xl font-bold text-blue-600">
                  {routeOptimization.total_stops}
                </p>
              </div>
              <div className="text-center p-4 bg-green-50 rounded">
                <Truck className="h-6 w-6 mx-auto mb-2 text-green-600" />
                <p className="text-sm text-gray-600">Vehicles Required</p>
                <p className="text-2xl font-bold text-green-600">
                  {routeOptimization.vehicles_used}
                </p>
              </div>
              <div className="text-center p-4 bg-orange-50 rounded">
                <Clock className="h-6 w-6 mx-auto mb-2 text-orange-600" />
                <p className="text-sm text-gray-600">Total Duration</p>
                <p className="text-2xl font-bold text-orange-600">
                  {routeOptimization.total_duration} hrs
                </p>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded">
                <DollarSign className="h-6 w-6 mx-auto mb-2 text-purple-600" />
                <p className="text-sm text-gray-600">Estimated Cost</p>
                <p className="text-2xl font-bold text-purple-600">
                  ${routeOptimization.total_cost?.toLocaleString() || '0'}
                </p>
              </div>
            </div>

            {/* Route Map */}
            {routeOptimization.routes && routeOptimization.routes.length > 0 && (
              <div className="h-96 rounded overflow-hidden">
                <MapContainer center={[40.7128, -74.0060]} zoom={10} style={{ height: '100%' }}>
                  <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  />
                  {routeOptimization.routes.map((route, idx) => (
                    <React.Fragment key={idx}>
                      {route.stops.map((stop, stopIdx) => (
                        <Marker key={`${idx}-${stopIdx}`} position={[stop.lat, stop.lng]}>
                          <Popup>
                            {stop.name}<br />
                            Delivery: {stop.delivery_time}
                          </Popup>
                        </Marker>
                      ))}
                      <Polyline
                        positions={route.stops.map(stop => [stop.lat, stop.lng])}
                        color={['blue', 'green', 'red', 'purple'][idx % 4]}
                      />
                    </React.Fragment>
                  ))}
                </MapContainer>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Carrier Performance Results */}
      {carrierPerformance && (
        <Card>
          <CardHeader>
            <CardTitle>Carrier Performance Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="text-center p-4 bg-green-50 rounded">
                <p className="text-sm text-gray-600">On-Time Delivery Rate</p>
                <p className="text-3xl font-bold text-green-600">
                  {carrierPerformance.on_time_rate?.toFixed(1) || '0'}%
                </p>
              </div>
              <div className="text-center p-4 bg-yellow-50 rounded">
                <p className="text-sm text-gray-600">Damage Rate</p>
                <p className="text-3xl font-bold text-yellow-600">
                  {carrierPerformance.damage_rate?.toFixed(2) || '0'}%
                </p>
              </div>
              <div className="text-center p-4 bg-blue-50 rounded">
                <p className="text-sm text-gray-600">Cost per Mile</p>
                <p className="text-3xl font-bold text-blue-600">
                  ${carrierPerformance.cost_per_mile?.toFixed(2) || '0'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium mb-2">Performance Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={carrierPerformance.trend_data || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="on_time_rate" 
                      stroke="#10b981" 
                      name="On-Time Rate (%)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="performance_score" 
                      stroke="#3b82f6" 
                      name="Overall Score"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h3 className="text-sm font-medium mb-2">Delivery Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={carrierPerformance.delivery_distribution || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="status" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Delivery Analytics Results */}
      {deliveryAnalytics && (
        <Card>
          <CardHeader>
            <CardTitle>Delivery Performance Analytics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="text-sm font-medium mb-2">Daily Delivery Volume</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={deliveryAnalytics.daily_volume || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="deliveries" fill="#82ca9d" />
                    <Bar dataKey="failed" fill="#ff6b6b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div>
                <h3 className="text-sm font-medium mb-2">Delivery Time Analysis</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={deliveryAnalytics.time_analysis || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="distance" name="Distance (miles)" />
                    <YAxis dataKey="duration" name="Duration (hours)" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Deliveries" data={deliveryAnalytics.time_analysis || []} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-sm font-medium mb-2">Delivery Insights</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gray-50 rounded">
                  <p className="text-sm text-gray-600">Average Delivery Time</p>
                  <p className="text-xl font-bold">{deliveryAnalytics.avg_delivery_time || '0'} hours</p>
                </div>
                <div className="p-4 bg-gray-50 rounded">
                  <p className="text-sm text-gray-600">Peak Delivery Hour</p>
                  <p className="text-xl font-bold">{deliveryAnalytics.peak_hour || '12'}:00</p>
                </div>
                <div className="p-4 bg-gray-50 rounded">
                  <p className="text-sm text-gray-600">Success Rate</p>
                  <p className="text-xl font-bold">{deliveryAnalytics.success_rate?.toFixed(1) || '0'}%</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default LogisticsDashboard;