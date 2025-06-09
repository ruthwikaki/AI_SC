// frontend/src/components/visualization/DashboardBuilder.jsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LayoutGrid, Plus, Save, Download, Trash2, Edit2, 
  BarChart3, LineChart, PieChart, Activity, Network, Layers,
  Loader2, Move, Maximize2, X
} from 'lucide-react';
import api from '../../services/api';
import GridLayout from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

// Import chart components
import BarChart from './charts/BarChart';
import LineChart from './charts/LineChart';
import PieChart from './charts/PieChart';
import HeatMap from './charts/HeatMap';
import NetworkGraph from './charts/NetworkGraph';
import SankeyDiagram from './charts/SankeyDiagram';

const DashboardBuilder = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dashboards, setDashboards] = useState([]);
  const [selectedDashboard, setSelectedDashboard] = useState(null);
  const [widgets, setWidgets] = useState([]);
  const [layout, setLayout] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  const [showAddWidget, setShowAddWidget] = useState(false);
  
  // New widget form state
  const [newWidget, setNewWidget] = useState({
    title: '',
    type: 'bar',
    dataSource: '',
    query: '',
    config: {}
  });

  const chartTypes = [
    { id: 'bar', name: 'Bar Chart', icon: BarChart3, component: BarChart },
    { id: 'line', name: 'Line Chart', icon: LineChart, component: LineChart },
    { id: 'pie', name: 'Pie Chart', icon: PieChart, component: PieChart },
    { id: 'heatmap', name: 'Heat Map', icon: Activity, component: HeatMap },
    { id: 'network', name: 'Network Graph', icon: Network, component: NetworkGraph },
    { id: 'sankey', name: 'Sankey Diagram', icon: Layers, component: SankeyDiagram }
  ];

  useEffect(() => {
    fetchDashboards();
  }, []);

  const fetchDashboards = async () => {
    try {
      const response = await api.get('/api/visualizations/dashboards');
      setDashboards(response.data);
    } catch (err) {
      console.error('Failed to fetch dashboards:', err);
    }
  };

  const loadDashboard = async (dashboardId) => {
    setLoading(true);
    try {
      const response = await api.get(`/api/visualizations/dashboards/${dashboardId}`);
      setSelectedDashboard(response.data);
      setWidgets(response.data.widgets || []);
      setLayout(response.data.layout || []);
      setIsEditing(false);
    } catch (err) {
      setError('Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  };

  const saveDashboard = async () => {
    if (!selectedDashboard) return;
    
    setLoading(true);
    try {
      await api.put(`/api/visualizations/dashboards/${selectedDashboard.id}`, {
        ...selectedDashboard,
        widgets,
        layout
      });
      setError(null);
      setIsEditing(false);
      fetchDashboards();
    } catch (err) {
      setError('Failed to save dashboard');
    } finally {
      setLoading(false);
    }
  };

  const createDashboard = async (name) => {
    setLoading(true);
    try {
      const response = await api.post('/api/visualizations/dashboards', {
        name,
        widgets: [],
        layout: []
      });
      await loadDashboard(response.data.id);
      fetchDashboards();
    } catch (err) {
      setError('Failed to create dashboard');
    } finally {
      setLoading(false);
    }
  };

  const deleteDashboard = async (dashboardId) => {
    if (!confirm('Are you sure you want to delete this dashboard?')) return;
    
    try {
      await api.delete(`/api/visualizations/dashboards/${dashboardId}`);
      if (selectedDashboard?.id === dashboardId) {
        setSelectedDashboard(null);
        setWidgets([]);
        setLayout([]);
      }
      fetchDashboards();
    } catch (err) {
      setError('Failed to delete dashboard');
    }
  };

  const addWidget = async () => {
    if (!newWidget.title || !newWidget.query) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      // First, execute the query to get data
      const dataResponse = await api.post('/api/database/query', {
        query: newWidget.query
      });

      const widgetId = `widget-${Date.now()}`;
      const newWidgetData = {
        id: widgetId,
        ...newWidget,
        data: dataResponse.data.results
      };

      // Add to widgets array
      setWidgets([...widgets, newWidgetData]);
      
      // Add to layout
      const newLayoutItem = {
        i: widgetId,
        x: (widgets.length * 2) % 12,
        y: Math.floor(widgets.length / 6) * 4,
        w: 4,
        h: 3,
        minW: 2,
        minH: 2
      };
      setLayout([...layout, newLayoutItem]);

      // Reset form
      setNewWidget({
        title: '',
        type: 'bar',
        dataSource: '',
        query: '',
        config: {}
      });
      setShowAddWidget(false);
    } catch (err) {
      setError('Failed to add widget');
    } finally {
      setLoading(false);
    }
  };

  const removeWidget = (widgetId) => {
    setWidgets(widgets.filter(w => w.id !== widgetId));
    setLayout(layout.filter(l => l.i !== widgetId));
  };

  const updateWidgetData = async (widgetId) => {
    const widget = widgets.find(w => w.id === widgetId);
    if (!widget) return;

    try {
      const dataResponse = await api.post('/api/database/query', {
        query: widget.query
      });
      
      setWidgets(widgets.map(w => 
        w.id === widgetId 
          ? { ...w, data: dataResponse.data.results }
          : w
      ));
    } catch (err) {
      console.error('Failed to update widget data:', err);
    }
  };

  const exportDashboard = async () => {
    if (!selectedDashboard) return;
    
    try {
      const response = await api.get(`/api/visualizations/dashboards/${selectedDashboard.id}/export`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `dashboard_${selectedDashboard.name}_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
    } catch (err) {
      setError('Failed to export dashboard');
    }
  };

  const renderWidget = (widget) => {
    const ChartType = chartTypes.find(ct => ct.id === widget.type)?.component;
    if (!ChartType) return <div>Unknown chart type</div>;

    return (
      <div className="h-full">
        <div className="flex justify-between items-center p-2 bg-gray-50 border-b">
          <h3 className="text-sm font-medium">{widget.title}</h3>
          {isEditing && (
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => updateWidgetData(widget.id)}
              >
                <Edit2 className="h-3 w-3" />
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => removeWidget(widget.id)}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          )}
        </div>
        <div className="p-4 h-[calc(100%-40px)]">
          <ChartType data={widget.data} config={widget.config} />
        </div>
      </div>
    );
  };

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Dashboard Builder</h1>
        <p className="text-gray-600">
          Create and customize dashboards with interactive visualizations
        </p>
      </div>

      {error && (
        <Alert className="mb-4" variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Dashboard List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="text-lg">Dashboards</CardTitle>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="sm">
                      <Plus className="h-4 w-4" />
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Dashboard</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4 mt-4">
                      <Input
                        placeholder="Dashboard name"
                        id="new-dashboard-name"
                      />
                      <Button
                        onClick={() => {
                          const name = document.getElementById('new-dashboard-name').value;
                          if (name) createDashboard(name);
                        }}
                      >
                        Create Dashboard
                      </Button>
                    </div>
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {dashboards.map((dashboard) => (
                  <div
                    key={dashboard.id}
                    className={`p-3 rounded cursor-pointer transition-colors ${
                      selectedDashboard?.id === dashboard.id
                        ? 'bg-blue-50 border border-blue-200'
                        : 'hover:bg-gray-50'
                    }`}
                    onClick={() => loadDashboard(dashboard.id)}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">{dashboard.name}</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteDashboard(dashboard.id);
                        }}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {dashboard.widgets?.length || 0} widgets
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Dashboard Canvas */}
        <div className="lg:col-span-3">
          {selectedDashboard ? (
            <Card>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle>{selectedDashboard.name}</CardTitle>
                  <div className="flex gap-2">
                    {!isEditing ? (
                      <Button onClick={() => setIsEditing(true)}>
                        <Edit2 className="h-4 w-4 mr-2" />
                        Edit
                      </Button>
                    ) : (
                      <>
                        <Button onClick={() => setShowAddWidget(true)}>
                          <Plus className="h-4 w-4 mr-2" />
                          Add Widget
                        </Button>
                        <Button onClick={saveDashboard} disabled={loading}>
                          <Save className="h-4 w-4 mr-2" />
                          Save
                        </Button>
                        <Button 
                          variant="outline" 
                          onClick={() => {
                            setIsEditing(false);
                            loadDashboard(selectedDashboard.id);
                          }}
                        >
                          Cancel
                        </Button>
                      </>
                    )}
                    <Button variant="outline" onClick={exportDashboard}>
                      <Download className="h-4 w-4 mr-2" />
                      Export
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center items-center h-96">
                    <Loader2 className="h-8 w-8 animate-spin" />
                  </div>
                ) : widgets.length === 0 ? (
                  <div className="text-center py-12 bg-gray-50 rounded">
                    <LayoutGrid className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-600">No widgets yet</p>
                    {isEditing && (
                      <Button
                        className="mt-4"
                        onClick={() => setShowAddWidget(true)}
                      >
                        Add Your First Widget
                      </Button>
                    )}
                  </div>
                ) : (
                  <GridLayout
                    className="layout"
                    layout={layout}
                    cols={12}
                    rowHeight={100}
                    width={1200}
                    isDraggable={isEditing}
                    isResizable={isEditing}
                    onLayoutChange={(newLayout) => setLayout(newLayout)}
                  >
                    {widgets.map((widget) => (
                      <div key={widget.id} className="bg-white border rounded shadow-sm">
                        {renderWidget(widget)}
                      </div>
                    ))}
                  </GridLayout>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="text-center py-12">
                <LayoutGrid className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600">Select or create a dashboard to get started</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* Add Widget Dialog */}
      <Dialog open={showAddWidget} onOpenChange={setShowAddWidget}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add Widget</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 mt-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Widget Title</label>
              <Input
                value={newWidget.title}
                onChange={(e) => setNewWidget({ ...newWidget, title: e.target.value })}
                placeholder="e.g., Monthly Sales Trend"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Chart Type</label>
              <div className="grid grid-cols-3 gap-2">
                {chartTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.id}
                      className={`p-4 rounded border text-center transition-colors ${
                        newWidget.type === type.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => setNewWidget({ ...newWidget, type: type.id })}
                    >
                      <Icon className="h-6 w-6 mx-auto mb-2" />
                      <span className="text-sm">{type.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">SQL Query</label>
              <textarea
                className="w-full h-32 p-3 border rounded font-mono text-sm"
                value={newWidget.query}
                onChange={(e) => setNewWidget({ ...newWidget, query: e.target.value })}
                placeholder="SELECT category, SUM(sales) as total_sales FROM orders GROUP BY category"
              />
            </div>

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowAddWidget(false)}>
                Cancel
              </Button>
              <Button onClick={addWidget} disabled={loading}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Add Widget
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default DashboardBuilder;