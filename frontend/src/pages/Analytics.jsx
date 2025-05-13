import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import Navbar from '../components/common/Navbar';
import Sidebar from '../components/common/Sidebar';
import Loading from '../components/common/Loading';
import InventoryDashboard from '../components/analytics/InventoryDashboard';
import SupplierDashboard from '../components/analytics/SupplierDashboard';
import LogisticsDashboard from '../components/analytics/LogisticsDashboard';

const Analytics = () => {
  const { isAuthenticated, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState('inventory');
  const [loading, setLoading] = useState(false);
  const [timeFrame, setTimeFrame] = useState('month'); // day, week, month, quarter, year
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [isAuthenticated, authLoading, navigate]);

  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate data refresh delay
    setTimeout(() => {
      setRefreshing(false);
    }, 1500);
  };

  if (authLoading) {
    return <Loading type="overlay" message="Authenticating..." />;
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'inventory':
        return <InventoryDashboard timeFrame={timeFrame} />;
      case 'supplier':
        return <SupplierDashboard timeFrame={timeFrame} />;
      case 'logistics':
        return <LogisticsDashboard timeFrame={timeFrame} />;
      default:
        return <InventoryDashboard timeFrame={timeFrame} />;
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto">
          <div className="px-6 py-4 bg-white border-b">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between">
              <div className="mb-4 md:mb-0">
                <h1 className="text-2xl font-semibold text-gray-800">Analytics</h1>
                <p className="text-gray-600">
                  {activeTab === 'inventory' && 'Analyze inventory performance and optimize stock levels'}
                  {activeTab === 'supplier' && 'Evaluate supplier performance and identify risks'}
                  {activeTab === 'logistics' && 'Track logistics metrics and delivery performance'}
                </p>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-2">
                {/* Time Frame Selector */}
                <div className="relative">
                  <select
                    value={timeFrame}
                    onChange={(e) => setTimeFrame(e.target.value)}
                    className="block w-full bg-white border border-gray-300 hover:border-gray-400 px-4 py-2 pr-8 rounded leading-tight focus:outline-none focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value="day">Today</option>
                    <option value="week">This Week</option>
                    <option value="month">This Month</option>
                    <option value="quarter">This Quarter</option>
                    <option value="year">This Year</option>
                  </select>
                </div>
                
                {/* Refresh Button */}
                <button
                  onClick={handleRefresh}
                  disabled={refreshing}
                  className="flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  {refreshing ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-700" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Refreshing...
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Refresh
                    </>
                  )}
                </button>
              </div>
            </div>
            
            {/* Tab Navigation */}
            <div className="mt-4 border-b border-gray-200">
              <nav className="-mb-px flex">
                <button
                  onClick={() => setActiveTab('inventory')}
                  className={`py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'inventory'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Inventory
                </button>
                <button
                  onClick={() => setActiveTab('supplier')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'supplier'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Supplier
                </button>
                <button
                  onClick={() => setActiveTab('logistics')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'logistics'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Logistics
                </button>
              </nav>
            </div>
          </div>
          
          {/* Dashboard Content */}
          <div className="p-6">
            {loading ? (
              <Loading type="card" message="Loading analytics..." />
            ) : (
              renderTabContent()
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Analytics;