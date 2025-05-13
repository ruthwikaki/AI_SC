import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import Navbar from '../components/common/Navbar';
import Sidebar from '../components/common/Sidebar';
import Loading from '../components/common/Loading';

const Settings = () => {
  const { user, isAuthenticated, loading: authLoading, updateUserProfile } = useAuth();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState('profile');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);
  
  // Profile settings
  const [profileForm, setProfileForm] = useState({
    name: '',
    email: '',
    title: '',
    department: '',
    phone: ''
  });
  
  // Database connection settings
  const [dbConnections, setDbConnections] = useState([]);
  const [newConnection, setNewConnection] = useState({
    name: '',
    type: 'postgresql',
    host: '',
    port: '',
    database: '',
    username: '',
    password: '',
    ssl: false
  });
  
  // Notification settings
  const [notifications, setNotifications] = useState({
    email: true,
    browser: true,
    supply_chain_alerts: true,
    inventory_alerts: true,
    delivery_alerts: true,
    weekly_reports: true,
    daily_summary: false
  });
  
  // Display settings
  const [displaySettings, setDisplaySettings] = useState({
    theme: 'light',
    density: 'comfortable',
    defaultDashboard: 'overview',
    defaultQueryView: 'visualization',
    chartColorScheme: 'default'
  });

  useEffect(() => {
    // Redirect to login if not authenticated
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
      return;
    }

    // Load user profile data
    if (user) {
      setProfileForm({
        name: user.name || '',
        email: user.email || '',
        title: user.title || '',
        department: user.department || '',
        phone: user.phone || ''
      });
    }

    // Load DB connections
    const fetchConnections = async () => {
      try {
        setLoading(true);
        // This would be an API call in a real implementation
        // Mocking the API response
        await new Promise(resolve => setTimeout(resolve, 800));
        
        setDbConnections([
          {
            id: 1,
            name: 'Production Database',
            type: 'postgresql',
            host: 'production.db.example.com',
            port: '5432',
            database: 'supply_chain_prod',
            username: 'readonly_user',
            status: 'connected'
          },
          {
            id: 2,
            name: 'Inventory System',
            type: 'mysql',
            host: 'inventory.example.com',
            port: '3306',
            database: 'inventory_system',
            username: 'readonly_user',
            status: 'connected'
          }
        ]);
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching database connections:', err);
        setError('Failed to load database connections.');
        setLoading(false);
      }
    };

    if (isAuthenticated) {
      fetchConnections();
    }
  }, [user, isAuthenticated, authLoading, navigate]);

  const handleProfileSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // This would call an actual API in a real implementation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update user profile
      await updateUserProfile(profileForm);
      
      setSuccess('Profile updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error updating profile:', err);
      setError('Failed to update profile. Please try again.');
      setLoading(false);
    }
  };

  const handleConnectionTest = async (connectionId) => {
    setLoading(true);
    
    try {
      // This would call an actual API in a real implementation
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setDbConnections(prevConnections => 
        prevConnections.map(conn => 
          conn.id === connectionId 
            ? { ...conn, status: 'connected' } 
            : conn
        )
      );
      
      setSuccess('Connection tested successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error testing connection:', err);
      
      setDbConnections(prevConnections => 
        prevConnections.map(conn => 
          conn.id === connectionId 
            ? { ...conn, status: 'failed' } 
            : conn
        )
      );
      
      setError('Connection test failed. Please check your settings.');
      setLoading(false);
    }
  };

  const handleNewConnectionSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // This would call an actual API in a real implementation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Add new connection to the list with a generated ID
      const newConn = {
        ...newConnection,
        id: Date.now(),
        status: 'pending'
      };
      
      setDbConnections([...dbConnections, newConn]);
      
      // Reset form
      setNewConnection({
        name: '',
        type: 'postgresql',
        host: '',
        port: '',
        database: '',
        username: '',
        password: '',
        ssl: false
      });
      
      setSuccess('Connection added successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error adding connection:', err);
      setError('Failed to add connection. Please try again.');
      setLoading(false);
    }
  };

  const handleNotificationChange = (e) => {
    const { name, checked } = e.target;
    setNotifications(prev => ({
      ...prev,
      [name]: checked
    }));
  };

  const handleDisplaySettingChange = (e) => {
    const { name, value } = e.target;
    setDisplaySettings(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const saveSettings = async (type) => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // This would call an actual API in a real implementation
      await new Promise(resolve => setTimeout(resolve, 800));
      
      setSuccess(`${type} settings saved successfully.`);
      setLoading(false);
    } catch (err) {
      console.error(`Error saving ${type} settings:`, err);
      setError(`Failed to save ${type} settings. Please try again.`);
      setLoading(false);
    }
  };

  if (authLoading) {
    return <Loading type="overlay" message="Authenticating..." />;
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Navbar />
        <main className="flex-1 overflow-y-auto">
          <div className="px-6 py-4 bg-white border-b">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between">
              <div className="mb-4 md:mb-0">
                <h1 className="text-2xl font-semibold text-gray-800">Settings</h1>
                <p className="text-gray-600">
                  Manage your profile, connections, and preferences
                </p>
              </div>
            </div>
            
            {/* Settings Tabs */}
            <div className="mt-4 border-b border-gray-200">
              <nav className="-mb-px flex">
                <button
                  onClick={() => setActiveTab('profile')}
                  className={`py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'profile'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Profile
                </button>
                <button
                  onClick={() => setActiveTab('database')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'database'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Database Connections
                </button>
                <button
                  onClick={() => setActiveTab('notifications')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'notifications'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Notifications
                </button>
                <button
                  onClick={() => setActiveTab('display')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'display'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Display
                </button>
              </nav>
            </div>
          </div>
          
          {/* Status Messages */}
          {success && (
            <div className="mx-6 mt-4 p-4 rounded-md bg-green-50 border border-green-200">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-green-800">{success}</p>
                </div>
                <div className="ml-auto pl-3">
                  <div className="-mx-1.5 -my-1.5">
                    <button
                      onClick={() => setSuccess(null)}
                      className="inline-flex rounded-md p-1.5 text-green-500 hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                    >
                      <span className="sr-only">Dismiss</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {error && (
            <div className="mx-6 mt-4 p-4 rounded-md bg-red-50 border border-red-200">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-red-800">{error}</p>
                </div>
                <div className="ml-auto pl-3">
                  <div className="-mx-1.5 -my-1.5">
                    <button
                      onClick={() => setError(null)}
                      className="inline-flex rounded-md p-1.5 text-red-500 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                    >
                      <span className="sr-only">Dismiss</span>
                      <svg className="h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Tab Content */}
          <div className="p-6">
            {loading && <Loading type="overlay" message="Saving changes..." />}
            
            {/* Profile Settings */}
            {activeTab === 'profile' && (
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-800">Profile Information</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Update your personal information and preferences
                  </p>
                </div>
                <form onSubmit={handleProfileSubmit} className="p-6 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                        Full Name
                      </label>
                      <input
                        type="text"
                        id="name"
                        value={profileForm.name}
                        onChange={(e) => setProfileForm({ ...profileForm, name: e.target.value })}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                        Email
                      </label>
                      <input
                        type="email"
                        id="email"
                        value={profileForm.email}
                        onChange={(e) => setProfileForm({ ...profileForm, email: e.target.value })}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="title" className="block text-sm font-medium text-gray-700">
                        Job Title
                      </label>
                      <input
                        type="text"
                        id="title"
                        value={profileForm.title}
                        onChange={(e) => setProfileForm({ ...profileForm, title: e.target.value })}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="department" className="block text-sm font-medium text-gray-700">
                        Department
                      </label>
                      <input
                        type="text"
                        id="department"
                        value={profileForm.department}
                        onChange={(e) => setProfileForm({ ...profileForm, department: e.target.value })}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                    </div>
                    
                    <div>
                      <label htmlFor="phone" className="block text-sm font-medium text-gray-700">
                        Phone Number
                      </label>
                      <input
                        type="tel"
                        id="phone"
                        value={profileForm.phone}
                        onChange={(e) => setProfileForm({ ...profileForm, phone: e.target.value })}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                    </div>
                  </div>
                  
                  <div className="flex justify-end">
                    <button
                      type="submit"
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      disabled={loading}
                    >
                      Save Profile
                    </button>
                  </div>
                </form>
              </div>
            )}
            
            {/* Database Connections */}
            {activeTab === 'database' && (
              <div className="space-y-6">
                {/* Existing Connections */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">Database Connections</h2>
                    <p className="mt-1 text-sm text-gray-500">
                      Manage your database connections for data access
                    </p>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Type
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Host
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Database
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Status
                          </th>
                          <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {dbConnections.map((connection) => (
                          <tr key={connection.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {connection.name}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {connection.type}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {connection.host}:{connection.port}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {connection.database}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                connection.status === 'connected' 
                                  ? 'bg-green-100 text-green-800' 
                                  : connection.status === 'failed'
                                  ? 'bg-red-100 text-red-800'
                                  : 'bg-yellow-100 text-yellow-800'
                              }`}>
                                {connection.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => handleConnectionTest(connection.id)}
                                className="text-blue-600 hover:text-blue-900 mr-4"
                              >
                                Test
                              </button>
                              <button className="text-red-600 hover:text-red-900">
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                {/* Add New Connection */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">Add New Connection</h2>
                    <p className="mt-1 text-sm text-gray-500">
                      Connect to a new database for data access
                    </p>
                  </div>
                  
                  <form onSubmit={handleNewConnectionSubmit} className="p-6 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="conn-name" className="block text-sm font-medium text-gray-700">
                          Connection Name
                        </label>
                        <input
                          type="text"
                          id="conn-name"
                          value={newConnection.name}
                          onChange={(e) => setNewConnection({ ...newConnection, name: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="conn-type" className="block text-sm font-medium text-gray-700">
                          Database Type
                        </label>
                        <select
                          id="conn-type"
                          value={newConnection.type}
                          onChange={(e) => setNewConnection({ ...newConnection, type: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="postgresql">PostgreSQL</option>
                          <option value="mysql">MySQL</option>
                          <option value="sqlserver">SQL Server</option>
                          <option value="oracle">Oracle</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="conn-host" className="block text-sm font-medium text-gray-700">
                          Host
                        </label>
                        <input
                          type="text"
                          id="conn-host"
                          value={newConnection.host}
                          onChange={(e) => setNewConnection({ ...newConnection, host: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="conn-port" className="block text-sm font-medium text-gray-700">
                          Port
                        </label>
                        <input
                          type="text"
                          id="conn-port"
                          value={newConnection.port}
                          onChange={(e) => setNewConnection({ ...newConnection, port: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="conn-database" className="block text-sm font-medium text-gray-700">
                          Database Name
                        </label>
                        <input
                          type="text"
                          id="conn-database"
                          value={newConnection.database}
                          onChange={(e) => setNewConnection({ ...newConnection, database: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="conn-username" className="block text-sm font-medium text-gray-700">
                          Username
                        </label>
                        <input
                          type="text"
                          id="conn-username"
                          value={newConnection.username}
                          onChange={(e) => setNewConnection({ ...newConnection, username: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="conn-password" className="block text-sm font-medium text-gray-700">
                          Password
                        </label>
                        <input
                          type="password"
                          id="conn-password"
                          value={newConnection.password}
                          onChange={(e) => setNewConnection({ ...newConnection, password: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div className="flex items-center">
                        <input
                          id="conn-ssl"
                          type="checkbox"
                          checked={newConnection.ssl}
                          onChange={(e) => setNewConnection({ ...newConnection, ssl: e.target.checked })}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="conn-ssl" className="ml-2 block text-sm text-gray-700">
                          Enable SSL/TLS
                        </label>
                      </div>
                    </div>
                    
                    <div className="flex justify-end">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        disabled={loading}
                      >
                        Add Connection
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            
            {/* Notification Settings */}
            {activeTab === 'notifications' && (
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-800">Notification Preferences</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Configure how and when you receive notifications
                  </p>
                </div>
                
                <div className="p-6 space-y-6">
                  <div className="space-y-4">
                    <h3 className="text-md font-medium text-gray-700">Delivery Methods</h3>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-email"
                        name="email"
                        type="checkbox"
                        checked={notifications.email}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-email" className="ml-2 block text-sm text-gray-700">
                        Email Notifications
                      </label>
                    </div>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-browser"
                        name="browser"
                        type="checkbox"
                        checked={notifications.browser}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-browser" className="ml-2 block text-sm text-gray-700">
                        Browser Notifications
                      </label>
                    </div>
                  </div>
                  
                  <div className="pt-5 border-t border-gray-200 space-y-4">
                    <h3 className="text-md font-medium text-gray-700">Alert Types</h3>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-supply-chain"
                        name="supply_chain_alerts"
                        type="checkbox"
                        checked={notifications.supply_chain_alerts}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-supply-chain" className="ml-2 block text-sm text-gray-700">
                        Supply Chain Alerts (disruptions, bottlenecks)
                      </label>
                    </div>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-inventory"
                        name="inventory_alerts"
                        type="checkbox"
                        checked={notifications.inventory_alerts}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-inventory" className="ml-2 block text-sm text-gray-700">
                        Inventory Alerts (low stock, stockouts)
                      </label>
                    </div>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-delivery"
                        name="delivery_alerts"
                        type="checkbox"
                        checked={notifications.delivery_alerts}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-delivery" className="ml-2 block text-sm text-gray-700">
                        Delivery Alerts (delays, exceptions)
                      </label>
                    </div>
                  </div>
                  
                  <div className="pt-5 border-t border-gray-200 space-y-4">
                    <h3 className="text-md font-medium text-gray-700">Report Frequency</h3>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-weekly"
                        name="weekly_reports"
                        type="checkbox"
                        checked={notifications.weekly_reports}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-weekly" className="ml-2 block text-sm text-gray-700">
                        Weekly Performance Reports
                      </label>
                    </div>
                    
                    <div className="flex items-center">
                      <input
                        id="notify-daily"
                        name="daily_summary"
                        type="checkbox"
                        checked={notifications.daily_summary}
                        onChange={handleNotificationChange}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <label htmlFor="notify-daily" className="ml-2 block text-sm text-gray-700">
                        Daily Activity Summary
                      </label>
                    </div>
                  </div>
                  
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => saveSettings('notification')}
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      disabled={loading}
                    >
                      Save Preferences
                    </button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Display Settings */}
            {activeTab === 'display' && (
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-200">
                  <h2 className="text-lg font-medium text-gray-800">Display Settings</h2>
                  <p className="mt-1 text-sm text-gray-500">
                    Customize your visual preferences
                  </p>
                </div>
                
                <div className="p-6 space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label htmlFor="theme" className="block text-sm font-medium text-gray-700">
                        Theme
                      </label>
                      <select
                        id="theme"
                        name="theme"
                        value={displaySettings.theme}
                        onChange={handleDisplaySettingChange}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="light">Light</option>
                        <option value="dark">Dark</option>
                        <option value="system">System Default</option>
                      </select>
                    </div>
                    
                    <div>
                      <label htmlFor="density" className="block text-sm font-medium text-gray-700">
                        UI Density
                      </label>
                      <select
                        id="density"
                        name="density"
                        value={displaySettings.density}
                        onChange={handleDisplaySettingChange}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="comfortable">Comfortable</option>
                        <option value="compact">Compact</option>
                      </select>
                    </div>
                    
                    <div>
                      <label htmlFor="defaultDashboard" className="block text-sm font-medium text-gray-700">
                        Default Dashboard
                      </label>
                      <select
                        id="defaultDashboard"
                        name="defaultDashboard"
                        value={displaySettings.defaultDashboard}
                        onChange={handleDisplaySettingChange}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="overview">Overview</option>
                        <option value="inventory">Inventory</option>
                        <option value="supplier">Supplier</option>
                        <option value="logistics">Logistics</option>
                      </select>
                    </div>
                    
                    <div>
                      <label htmlFor="defaultQueryView" className="block text-sm font-medium text-gray-700">
                        Default Query View
                      </label>
                      <select
                        id="defaultQueryView"
                        name="defaultQueryView"
                        value={displaySettings.defaultQueryView}
                        onChange={handleDisplaySettingChange}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="visualization">Visualization</option>
                        <option value="data">Data Table</option>
                        <option value="sql">SQL Query</option>
                      </select>
                    </div>
                    
                    <div>
                      <label htmlFor="chartColorScheme" className="block text-sm font-medium text-gray-700">
                        Chart Color Scheme
                      </label>
                      <select
                        id="chartColorScheme"
                        name="chartColorScheme"
                        value={displaySettings.chartColorScheme}
                        onChange={handleDisplaySettingChange}
                        className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="default">Default</option>
                        <option value="monochrome">Monochrome</option>
                        <option value="vibrant">Vibrant</option>
                        <option value="pastel">Pastel</option>
                      </select>
                    </div>
                  </div>
                  
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => saveSettings('display')}
                      className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      disabled={loading}
                    >
                      Save Settings
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default Settings;