import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { useQuery } from '../hooks/useQuery';
import Navbar from '../components/common/Navbar';
import Sidebar from '../components/common/Sidebar';
import Loading from '../components/common/Loading';
import QueryInput from '../components/query/QueryInput';
import QuerySuggestions from '../components/query/QuerySuggestions';
import ChartViewer from '../components/visualization/ChartViewer';

const Dashboard = () => {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const { getRecentQueries } = useQuery();
  const navigate = useNavigate();
  
  const [metrics, setMetrics] = useState({
    inventoryValue: 0,
    orderFillRate: 0,
    onTimeDelivery: 0,
    supplierPerformance: 0,
    activeAlerts: 0
  });
  
  const [recentQueries, setRecentQueries] = useState([]);
  const [recentCharts, setRecentCharts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
      return;
    }

    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Get recent queries
        const queries = await getRecentQueries();
        setRecentQueries(queries.slice(0, 5));
        
        // Fetch KPI metrics (this would be a real API call)
        // Mock data for now
        setMetrics({
          inventoryValue: 2457000,
          orderFillRate: 94.7,
          onTimeDelivery: 92.3,
          supplierPerformance: 87.2,
          activeAlerts: 3
        });
        
        // Fetch recent visualizations (would be from an API)
        setRecentCharts([
          {
            id: 'chart1',
            title: 'Monthly Inventory Trends', 
            type: 'line',
            data: {/* chart data would be here */}
          },
          {
            id: 'chart2',
            title: 'Top Suppliers by Volume',
            type: 'bar',
            data: {/* chart data would be here */}
          },
          {
            id: 'chart3',
            title: 'On-Time Delivery by Region',
            type: 'heatmap',
            data: {/* chart data would be here */}
          },
          {
            id: 'chart4',
            title: 'Order Status Distribution',
            type: 'pie',
            data: {/* chart data would be here */}
          }
        ]);
        
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setLoading(false);
      }
    };

    if (isAuthenticated) {
      fetchDashboardData();
    }
  }, [isAuthenticated, authLoading, navigate, getRecentQueries]);

  if (authLoading) {
    return <Loading type="overlay" message="Authenticating..." />;
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto p-5">
          {loading ? (
            <Loading type="card" message="Loading dashboard..." />
          ) : (
            <>
              {/* Welcome Section */}
              <div className="mb-6">
                <h1 className="text-2xl font-semibold text-gray-800">
                  Welcome back, {user?.name || 'User'}
                </h1>
                <p className="text-gray-600">
                  Here's what's happening across your supply chain today
                </p>
              </div>
              
              {/* KPI Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
                <MetricCard 
                  title="Inventory Value" 
                  value={`$${metrics.inventoryValue.toLocaleString()}`} 
                  trend={+2.4}
                  icon="cube"
                />
                <MetricCard 
                  title="Order Fill Rate" 
                  value={`${metrics.orderFillRate}%`} 
                  trend={-0.8}
                  icon="clipboard-check"
                />
                <MetricCard 
                  title="On-Time Delivery" 
                  value={`${metrics.onTimeDelivery}%`} 
                  trend={+1.2}
                  icon="truck"
                />
                <MetricCard 
                  title="Supplier Performance" 
                  value={`${metrics.supplierPerformance}%`} 
                  trend={+0.5}
                  icon="users"
                />
                <MetricCard 
                  title="Active Alerts" 
                  value={metrics.activeAlerts} 
                  trend={null}
                  icon="bell"
                  alert={metrics.activeAlerts > 0}
                />
              </div>
              
              {/* Quick Ask Section */}
              <div className="mb-6">
                <div className="bg-white rounded-lg shadow p-5">
                  <h2 className="text-lg font-semibold mb-4">Ask a question</h2>
                  <QueryInput placeholder="Ask about your supply chain..." />
                  <div className="mt-3">
                    <h3 className="text-sm font-medium text-gray-700 mb-2">Suggested questions:</h3>
                    <QuerySuggestions />
                  </div>
                </div>
              </div>
              
              {/* Recent Visualizations */}
              <div className="mb-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">Recent Visualizations</h2>
                  <button 
                    className="text-blue-600 hover:text-blue-800"
                    onClick={() => navigate('/visualizations')}
                  >
                    View all
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {recentCharts.map(chart => (
                    <div key={chart.id} className="bg-white rounded-lg shadow overflow-hidden">
                      <div className="p-4">
                        <h3 className="font-medium">{chart.title}</h3>
                      </div>
                      <div className="h-64 p-2">
                        <ChartViewer 
                          type={chart.type} 
                          data={chart.data} 
                          height={250}
                          showControls={false}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Recent Queries */}
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">Recent Queries</h2>
                  <button 
                    className="text-blue-600 hover:text-blue-800"
                    onClick={() => navigate('/queries')}
                  >
                    View all
                  </button>
                </div>
                <div className="bg-white rounded-lg shadow overflow-hidden">
                  <ul className="divide-y divide-gray-200">
                    {recentQueries.length > 0 ? (
                      recentQueries.map(query => (
                        <li key={query.id} className="p-4 hover:bg-gray-50 cursor-pointer">
                          <div className="flex items-start">
                            <div className="flex-1">
                              <p className="font-medium">{query.text}</p>
                              <p className="text-sm text-gray-500">{new Date(query.timestamp).toLocaleString()}</p>
                            </div>
                            <button 
                              className="ml-4 text-gray-400 hover:text-blue-600"
                              onClick={() => navigate(`/queries/${query.id}`)}
                            >
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                            </button>
                          </div>
                        </li>
                      ))
                    ) : (
                      <li className="p-4 text-center text-gray-500">
                        No recent queries found
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
};

// Helper component for KPI metrics
const MetricCard = ({ title, value, trend, icon, alert = false }) => {
  return (
    <div className={`bg-white rounded-lg shadow p-4 ${alert ? 'border-l-4 border-red-500' : ''}`}>
      <div className="flex items-center mb-2">
        <span className="mr-2">
          <i className={`fas fa-${icon} text-gray-500`}></i>
        </span>
        <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      </div>
      <p className="text-2xl font-semibold">{value}</p>
      {trend !== null && (
        <div className={`mt-2 flex items-center text-sm ${trend >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          <span>
            {trend >= 0 ? (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </span>
          <span>{Math.abs(trend)}% from last month</span>
        </div>
      )}
    </div>
  );
};

export default Dashboard;