import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import Navbar from '../components/common/Navbar';
import Sidebar from '../components/common/Sidebar';
import Loading from '../components/common/Loading';

const Admin = () => {
  const { user, isAuthenticated, loading: authLoading } = useAuth();
  const navigate = useNavigate();
  
  const [activeTab, setActiveTab] = useState('users');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  
  // Users state
  const [users, setUsers] = useState([]);
  const [newUser, setNewUser] = useState({
    name: '',
    email: '',
    role: 'user',
    title: '',
    department: ''
  });
  const [editingUser, setEditingUser] = useState(null);
  
  // Roles state
  const [roles, setRoles] = useState([]);
  const [newRole, setNewRole] = useState({
    name: '',
    description: '',
    permissions: []
  });
  const [editingRole, setEditingRole] = useState(null);
  
  // Clients state
  const [clients, setClients] = useState([]);
  const [newClient, setNewClient] = useState({
    name: '',
    industry: '',
    tier: 'standard',
    status: 'active'
  });
  const [editingClient, setEditingClient] = useState(null);
  
  // System Settings state
  const [systemSettings, setSystemSettings] = useState({
    activeModel: 'llama3',
    maxTokens: 4096,
    temperature: 0.7,
    logLevel: 'info',
    cacheEnabled: true,
    schemaRefreshInterval: 86400
  });
  
  // Audit Logs state
  const [auditLogs, setAuditLogs] = useState([]);
  const [auditLogFilters, setAuditLogFilters] = useState({
    user: '',
    action: '',
    dateFrom: '',
    dateTo: ''
  });

  useEffect(() => {
    // Redirect if not authenticated or not admin
    if (!authLoading && (!isAuthenticated || (user && user.role !== 'admin'))) {
      navigate('/');
      return;
    }

    // Load initial data
    const loadAdminData = async () => {
      try {
        setLoading(true);
        // This would be real API calls in production
        await Promise.all([
          fetchUsers(),
          fetchRoles(),
          fetchClients(),
          fetchSystemSettings(),
          fetchAuditLogs()
        ]);
        setLoading(false);
      } catch (err) {
        console.error('Error loading admin data:', err);
        setError('Failed to load admin data. Please refresh the page.');
        setLoading(false);
      }
    };

    if (isAuthenticated && user && user.role === 'admin') {
      loadAdminData();
    }
  }, [isAuthenticated, authLoading, user, navigate]);

  // Mock API functions
  const fetchUsers = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock user data
    const mockUsers = [
      { id: 1, name: 'Admin User', email: 'admin@example.com', role: 'admin', title: 'System Administrator', department: 'IT', status: 'active', created: '2023-01-15' },
      { id: 2, name: 'John Smith', email: 'john@example.com', role: 'manager', title: 'Supply Chain Manager', department: 'Operations', status: 'active', created: '2023-02-20' },
      { id: 3, name: 'Emily Johnson', email: 'emily@example.com', role: 'user', title: 'Inventory Analyst', department: 'Operations', status: 'active', created: '2023-03-10' },
      { id: 4, name: 'Michael Brown', email: 'michael@example.com', role: 'user', title: 'Logistics Coordinator', department: 'Logistics', status: 'inactive', created: '2023-04-05' }
    ];
    
    setUsers(mockUsers);
  };

  const fetchRoles = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 600));
    
    // Mock role data
    const mockRoles = [
      { id: 1, name: 'admin', description: 'System Administrator with full access', permissions: ['read:all', 'write:all', 'admin:all'], systemRole: true },
      { id: 2, name: 'manager', description: 'Manager with department-level access', permissions: ['read:all', 'write:department', 'admin:none'], systemRole: false },
      { id: 3, name: 'user', description: 'Standard user with basic access', permissions: ['read:assigned', 'write:none', 'admin:none'], systemRole: true },
      { id: 4, name: 'analyst', description: 'Analyst with read-only access', permissions: ['read:all', 'write:none', 'admin:none'], systemRole: false }
    ];
    
    setRoles(mockRoles);
  };

  const fetchClients = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 700));
    
    // Mock client data
    const mockClients = [
      { id: 1, name: 'Acme Corporation', industry: 'Manufacturing', tier: 'enterprise', status: 'active', created: '2023-01-10', users: 25 },
      { id: 2, name: 'TechGlobal', industry: 'Technology', tier: 'premium', status: 'active', created: '2023-02-15', users: 12 },
      { id: 3, name: 'Logistics Plus', industry: 'Logistics', tier: 'standard', status: 'active', created: '2023-03-20', users: 5 },
      { id: 4, name: 'RetailOne', industry: 'Retail', tier: 'standard', status: 'inactive', created: '2023-04-25', users: 0 }
    ];
    
    setClients(mockClients);
  };

  const fetchSystemSettings = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 400));
    
    // Settings would be fetched from API
    // Using the default settings from state
  };

  const fetchAuditLogs = async () => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // Mock audit log data
    const mockAuditLogs = [
      { id: 1, timestamp: '2023-10-25T14:30:00Z', user: 'admin@example.com', action: 'USER_CREATE', details: 'Created user john@example.com', ip: '192.168.1.1' },
      { id: 2, timestamp: '2023-10-25T15:45:00Z', user: 'admin@example.com', action: 'ROLE_UPDATE', details: 'Modified role: analyst', ip: '192.168.1.1' },
      { id: 3, timestamp: '2023-10-26T09:15:00Z', user: 'john@example.com', action: 'LOGIN', details: 'User login', ip: '192.168.1.2' },
      { id: 4, timestamp: '2023-10-26T11:30:00Z', user: 'system', action: 'SYSTEM_SETTING', details: 'Updated model configuration', ip: '127.0.0.1' },
      { id: 5, timestamp: '2023-10-27T10:20:00Z', user: 'emily@example.com', action: 'DATA_EXPORT', details: 'Exported inventory report', ip: '192.168.1.3' }
    ];
    
    setAuditLogs(mockAuditLogs);
  };

  // Handler functions for user management
  const handleAddUser = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create new user with generated ID
      const newUserWithId = {
        ...newUser,
        id: Date.now(),
        status: 'active',
        created: new Date().toISOString().split('T')[0]
      };
      
      setUsers([...users, newUserWithId]);
      
      // Reset form
      setNewUser({
        name: '',
        email: '',
        role: 'user',
        title: '',
        department: ''
      });
      
      setSuccess('User added successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error adding user:', err);
      setError('Failed to add user. Please try again.');
      setLoading(false);
    }
  };

  const handleUpdateUser = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update user in list
      setUsers(users.map(user => 
        user.id === editingUser.id ? editingUser : user
      ));
      
      setEditingUser(null);
      setSuccess('User updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error updating user:', err);
      setError('Failed to update user. Please try again.');
      setLoading(false);
    }
  };

  const handleToggleUserStatus = async (userId) => {
    setLoading(true);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Toggle user status
      setUsers(users.map(user => 
        user.id === userId
          ? { ...user, status: user.status === 'active' ? 'inactive' : 'active' }
          : user
      ));
      
      setSuccess('User status updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error toggling user status:', err);
      setError('Failed to update user status. Please try again.');
      setLoading(false);
    }
  };

  // Handler functions for role management
  const handleAddRole = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create new role with generated ID
      const newRoleWithId = {
        ...newRole,
        id: Date.now(),
        systemRole: false
      };
      
      setRoles([...roles, newRoleWithId]);
      
      // Reset form
      setNewRole({
        name: '',
        description: '',
        permissions: []
      });
      
      setSuccess('Role added successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error adding role:', err);
      setError('Failed to add role. Please try again.');
      setLoading(false);
    }
  };

  const handleUpdateRole = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update role in list
      setRoles(roles.map(role => 
        role.id === editingRole.id ? editingRole : role
      ));
      
      setEditingRole(null);
      setSuccess('Role updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error updating role:', err);
      setError('Failed to update role. Please try again.');
      setLoading(false);
    }
  };

  // Handler functions for client management
  const handleAddClient = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Create new client with generated ID
      const newClientWithId = {
        ...newClient,
        id: Date.now(),
        created: new Date().toISOString().split('T')[0],
        users: 0
      };
      
      setClients([...clients, newClientWithId]);
      
      // Reset form
      setNewClient({
        name: '',
        industry: '',
        tier: 'standard',
        status: 'active'
      });
      
      setSuccess('Client added successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error adding client:', err);
      setError('Failed to add client. Please try again.');
      setLoading(false);
    }
  };

  const handleUpdateClient = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update client in list
      setClients(clients.map(client => 
        client.id === editingClient.id ? editingClient : client
      ));
      
      setEditingClient(null);
      setSuccess('Client updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error updating client:', err);
      setError('Failed to update client. Please try again.');
      setLoading(false);
    }
  };

  const handleToggleClientStatus = async (clientId) => {
    setLoading(true);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Toggle client status
      setClients(clients.map(client => 
        client.id === clientId
          ? { ...client, status: client.status === 'active' ? 'inactive' : 'active' }
          : client
      ));
      
      setSuccess('Client status updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error toggling client status:', err);
      setError('Failed to update client status. Please try again.');
      setLoading(false);
    }
  };

  // Handler functions for system settings
  const handleUpdateSystemSettings = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    try {
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      // In a real app, this would update settings on the server
      
      setSuccess('System settings updated successfully.');
      setLoading(false);
    } catch (err) {
      console.error('Error updating system settings:', err);
      setError('Failed to update system settings. Please try again.');
      setLoading(false);
    }
  };

  // Handler functions for audit log filtering
  const handleAuditLogFilterChange = (e) => {
    const { name, value } = e.target;
    setAuditLogFilters({
      ...auditLogFilters,
      [name]: value
    });
  };

  const applyAuditLogFilters = () => {
    // In a real app, this would make an API call with filters
    setLoading(true);
    
    // Simulate API delay
    setTimeout(() => {
      // For demo, just use the mock data
      setLoading(false);
    }, 800);
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
                <h1 className="text-2xl font-semibold text-gray-800">Admin Dashboard</h1>
                <p className="text-gray-600">
                  Manage users, roles, clients, and system settings
                </p>
              </div>
            </div>
            
            {/* Admin Tabs */}
            <div className="mt-4 border-b border-gray-200">
              <nav className="-mb-px flex flex-wrap">
                <button
                  onClick={() => setActiveTab('users')}
                  className={`py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'users'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Users
                </button>
                <button
                  onClick={() => setActiveTab('roles')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'roles'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Roles & Permissions
                </button>
                <button
                  onClick={() => setActiveTab('clients')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'clients'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Clients
                </button>
                <button
                  onClick={() => setActiveTab('system')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'system'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  System Settings
                </button>
                <button
                  onClick={() => setActiveTab('audit')}
                  className={`ml-8 py-2 px-4 font-medium text-sm border-b-2 ${
                    activeTab === 'audit'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Audit Logs
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
            {loading && !authLoading && <Loading type="overlay" message="Processing..." />}
            
            {/* Users Tab */}
            {activeTab === 'users' && (
              <div className="space-y-6">
                {/* User List */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                    <div>
                      <h2 className="text-lg font-medium text-gray-800">Users</h2>
                      <p className="mt-1 text-sm text-gray-500">
                        Manage system users and their access
                      </p>
                    </div>
                    <button
                      onClick={() => setEditingUser(null)}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Add User
                    </button>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Email
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Role
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Title / Department
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
                        {users.map((user) => (
                          <tr key={user.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {user.name}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {user.email}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                {user.role}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {user.title}
                              {user.title && user.department && ' / '}
                              {user.department}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                user.status === 'active' 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-red-100 text-red-800'
                              }`}>
                                {user.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => setEditingUser(user)}
                                className="text-blue-600 hover:text-blue-900 mr-4"
                              >
                                Edit
                              </button>
                              <button 
                                onClick={() => handleToggleUserStatus(user.id)}
                                className={`${
                                  user.status === 'active' 
                                    ? 'text-red-600 hover:text-red-900' 
                                    : 'text-green-600 hover:text-green-900'
                                }`}
                              >
                                {user.status === 'active' ? 'Deactivate' : 'Activate'}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                {/* User Form */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">
                      {editingUser ? 'Edit User' : 'Add New User'}
                    </h2>
                  </div>
                  
                  <form onSubmit={editingUser ? handleUpdateUser : handleAddUser} className="p-6 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                          Full Name
                        </label>
                        <input
                          type="text"
                          id="name"
                          value={editingUser ? editingUser.name : newUser.name}
                          onChange={(e) => editingUser 
                            ? setEditingUser({ ...editingUser, name: e.target.value })
                            : setNewUser({ ...newUser, name: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                          Email
                        </label>
                        <input
                          type="email"
                          id="email"
                          value={editingUser ? editingUser.email : newUser.email}
                          onChange={(e) => editingUser 
                            ? setEditingUser({ ...editingUser, email: e.target.value })
                            : setNewUser({ ...newUser, email: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="role" className="block text-sm font-medium text-gray-700">
                          Role
                        </label>
                        <select
                          id="role"
                          value={editingUser ? editingUser.role : newUser.role}
                          onChange={(e) => editingUser 
                            ? setEditingUser({ ...editingUser, role: e.target.value })
                            : setNewUser({ ...newUser, role: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          {roles.map(role => (
                            <option key={role.id} value={role.name}>{role.name}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="title" className="block text-sm font-medium text-gray-700">
                          Job Title
                        </label>
                        <input
                          type="text"
                          id="title"
                          value={editingUser ? editingUser.title : newUser.title}
                          onChange={(e) => editingUser 
                            ? setEditingUser({ ...editingUser, title: e.target.value })
                            : setNewUser({ ...newUser, title: e.target.value })
                          }
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
                          value={editingUser ? editingUser.department : newUser.department}
                          onChange={(e) => editingUser 
                            ? setEditingUser({ ...editingUser, department: e.target.value })
                            : setNewUser({ ...newUser, department: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                      
                      {!editingUser && (
                        <div>
                          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                            Temporary Password
                          </label>
                          <input
                            type="password"
                            id="password"
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                            required
                          />
                          <p className="mt-1 text-xs text-gray-500">
                            User will be prompted to change on first login
                          </p>
                        </div>
                      )}
                    </div>
                    
                    <div className="flex justify-end">
                      {editingUser && (
                        <button
                          type="button"
                          onClick={() => setEditingUser(null)}
                          className="mr-3 inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Cancel
                        </button>
                      )}
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        {editingUser ? 'Save Changes' : 'Add User'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            
            {/* Roles Tab */}
            {activeTab === 'roles' && (
              <div className="space-y-6">
                {/* Role List */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                    <div>
                      <h2 className="text-lg font-medium text-gray-800">Roles & Permissions</h2>
                      <p className="mt-1 text-sm text-gray-500">
                        Manage user roles and their permissions
                      </p>
                    </div>
                    <button
                      onClick={() => setEditingRole(null)}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Add Role
                    </button>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Description
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Permissions
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Type
                          </th>
                          <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {roles.map((role) => (
                          <tr key={role.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {role.name}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {role.description}
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-500">
                              <div className="flex flex-wrap gap-1">
                                {role.permissions.map((permission, index) => (
                                  <span key={index} className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">
                                    {permission}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {role.systemRole ? (
                                <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-purple-100 text-purple-800">
                                  System
                                </span>
                              ) : (
                                <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                  Custom
                                </span>
                              )}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => setEditingRole(role)}
                                className="text-blue-600 hover:text-blue-900 mr-4"
                                disabled={role.systemRole}
                              >
                                Edit
                              </button>
                              <button 
                                className="text-red-600 hover:text-red-900"
                                disabled={role.systemRole}
                              >
                                Delete
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                {/* Role Form */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">
                      {editingRole ? 'Edit Role' : 'Add New Role'}
                    </h2>
                  </div>
                  
                  <form onSubmit={editingRole ? handleUpdateRole : handleAddRole} className="p-6 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="role-name" className="block text-sm font-medium text-gray-700">
                          Role Name
                        </label>
                        <input
                          type="text"
                          id="role-name"
                          value={editingRole ? editingRole.name : newRole.name}
                          onChange={(e) => editingRole 
                            ? setEditingRole({ ...editingRole, name: e.target.value })
                            : setNewRole({ ...newRole, name: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="role-description" className="block text-sm font-medium text-gray-700">
                          Description
                        </label>
                        <input
                          type="text"
                          id="role-description"
                          value={editingRole ? editingRole.description : newRole.description}
                          onChange={(e) => editingRole 
                            ? setEditingRole({ ...editingRole, description: e.target.value })
                            : setNewRole({ ...newRole, description: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Permissions
                      </label>
                      <div className="bg-gray-50 rounded-md p-4 space-y-4">
                        <div>
                          <h3 className="text-sm font-medium text-gray-700 mb-2">Read Access</h3>
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                            <div className="flex items-center">
                              <input
                                id="read-none"
                                name="read"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('read:none')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('read:'))
                                    .concat(['read:none']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="read-none" className="ml-2 block text-sm text-gray-700">
                                No Access
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="read-assigned"
                                name="read"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('read:assigned')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('read:'))
                                    .concat(['read:assigned']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="read-assigned" className="ml-2 block text-sm text-gray-700">
                                Assigned Only
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="read-all"
                                name="read"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('read:all')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('read:'))
                                    .concat(['read:all']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="read-all" className="ml-2 block text-sm text-gray-700">
                                All Data
                              </label>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-sm font-medium text-gray-700 mb-2">Write Access</h3>
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                            <div className="flex items-center">
                              <input
                                id="write-none"
                                name="write"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('write:none')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('write:'))
                                    .concat(['write:none']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="write-none" className="ml-2 block text-sm text-gray-700">
                                No Access
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="write-department"
                                name="write"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('write:department')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('write:'))
                                    .concat(['write:department']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="write-department" className="ml-2 block text-sm text-gray-700">
                                Department Only
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="write-all"
                                name="write"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('write:all')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('write:'))
                                    .concat(['write:all']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="write-all" className="ml-2 block text-sm text-gray-700">
                                All Data
                              </label>
                            </div>
                          </div>
                        </div>
                        
                        <div>
                          <h3 className="text-sm font-medium text-gray-700 mb-2">Admin Access</h3>
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                            <div className="flex items-center">
                              <input
                                id="admin-none"
                                name="admin"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('admin:none')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('admin:'))
                                    .concat(['admin:none']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="admin-none" className="ml-2 block text-sm text-gray-700">
                                No Access
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="admin-users"
                                name="admin"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('admin:users')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('admin:'))
                                    .concat(['admin:users']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="admin-users" className="ml-2 block text-sm text-gray-700">
                                User Management
                              </label>
                            </div>
                            <div className="flex items-center">
                              <input
                                id="admin-all"
                                name="admin"
                                type="radio"
                                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                                checked={(editingRole ? editingRole.permissions : newRole.permissions).includes('admin:all')}
                                onChange={() => {
                                  const newPermissions = (editingRole ? editingRole.permissions : newRole.permissions)
                                    .filter(p => !p.startsWith('admin:'))
                                    .concat(['admin:all']);
                                  
                                  if (editingRole) {
                                    setEditingRole({ ...editingRole, permissions: newPermissions });
                                  } else {
                                    setNewRole({ ...newRole, permissions: newPermissions });
                                  }
                                }}
                              />
                              <label htmlFor="admin-all" className="ml-2 block text-sm text-gray-700">
                                Full Admin
                              </label>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-end">
                      {editingRole && (
                        <button
                          type="button"
                          onClick={() => setEditingRole(null)}
                          className="mr-3 inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Cancel
                        </button>
                      )}
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        {editingRole ? 'Save Changes' : 'Add Role'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            
            {/* Clients Tab */}
            {activeTab === 'clients' && (
              <div className="space-y-6">
                {/* Client List */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                    <div>
                      <h2 className="text-lg font-medium text-gray-800">Clients</h2>
                      <p className="mt-1 text-sm text-gray-500">
                        Manage organization clients and their settings
                      </p>
                    </div>
                    <button
                      onClick={() => setEditingClient(null)}
                      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                    >
                      Add Client
                    </button>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Name
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Industry
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Subscription Tier
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Users
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
                        {clients.map((client) => (
                          <tr key={client.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {client.name}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {client.industry}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                client.tier === 'enterprise' 
                                  ? 'bg-purple-100 text-purple-800' 
                                  : client.tier === 'premium'
                                  ? 'bg-blue-100 text-blue-800'
                                  : 'bg-green-100 text-green-800'
                              }`}>
                                {client.tier}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {client.users}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                client.status === 'active' 
                                  ? 'bg-green-100 text-green-800' 
                                  : 'bg-red-100 text-red-800'
                              }`}>
                                {client.status}
                              </span>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                              <button 
                                onClick={() => setEditingClient(client)}
                                className="text-blue-600 hover:text-blue-900 mr-4"
                              >
                                Edit
                              </button>
                              <button 
                                onClick={() => handleToggleClientStatus(client.id)}
                                className={`${
                                  client.status === 'active' 
                                    ? 'text-red-600 hover:text-red-900' 
                                    : 'text-green-600 hover:text-green-900'
                                }`}
                              >
                                {client.status === 'active' ? 'Deactivate' : 'Activate'}
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                {/* Client Form */}
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">
                      {editingClient ? 'Edit Client' : 'Add New Client'}
                    </h2>
                  </div>
                  
                  <form onSubmit={editingClient ? handleUpdateClient : handleAddClient} className="p-6 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="client-name" className="block text-sm font-medium text-gray-700">
                          Client Name
                        </label>
                        <input
                          type="text"
                          id="client-name"
                          value={editingClient ? editingClient.name : newClient.name}
                          onChange={(e) => editingClient 
                            ? setEditingClient({ ...editingClient, name: e.target.value })
                            : setNewClient({ ...newClient, name: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="client-industry" className="block text-sm font-medium text-gray-700">
                          Industry
                        </label>
                        <input
                          type="text"
                          id="client-industry"
                          value={editingClient ? editingClient.industry : newClient.industry}
                          onChange={(e) => editingClient 
                            ? setEditingClient({ ...editingClient, industry: e.target.value })
                            : setNewClient({ ...newClient, industry: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          required
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="client-tier" className="block text-sm font-medium text-gray-700">
                          Subscription Tier
                        </label>
                        <select
                          id="client-tier"
                          value={editingClient ? editingClient.tier : newClient.tier}
                          onChange={(e) => editingClient 
                            ? setEditingClient({ ...editingClient, tier: e.target.value })
                            : setNewClient({ ...newClient, tier: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="standard">Standard</option>
                          <option value="premium">Premium</option>
                          <option value="enterprise">Enterprise</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="client-status" className="block text-sm font-medium text-gray-700">
                          Status
                        </label>
                        <select
                          id="client-status"
                          value={editingClient ? editingClient.status : newClient.status}
                          onChange={(e) => editingClient 
                            ? setEditingClient({ ...editingClient, status: e.target.value })
                            : setNewClient({ ...newClient, status: e.target.value })
                          }
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="active">Active</option>
                          <option value="inactive">Inactive</option>
                        </select>
                      </div>
                    </div>
                    
                    <div className="flex justify-end">
                      {editingClient && (
                        <button
                          type="button"
                          onClick={() => setEditingClient(null)}
                          className="mr-3 inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Cancel
                        </button>
                      )}
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        {editingClient ? 'Save Changes' : 'Add Client'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            
            {/* System Settings Tab */}
            {activeTab === 'system' && (
              <div className="space-y-6">
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">System Settings</h2>
                    <p className="mt-1 text-sm text-gray-500">
                      Configure global system parameters
                    </p>
                  </div>
                  
                  <form onSubmit={handleUpdateSystemSettings} className="p-6 space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label htmlFor="active-model" className="block text-sm font-medium text-gray-700">
                          Active LLM Model
                        </label>
                        <select
                          id="active-model"
                          value={systemSettings.activeModel}
                          onChange={(e) => setSystemSettings({ ...systemSettings, activeModel: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="llama3">Llama 3</option>
                          <option value="mistral">Mistral</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="max-tokens" className="block text-sm font-medium text-gray-700">
                          Max Tokens
                        </label>
                        <input
                          type="number"
                          id="max-tokens"
                          value={systemSettings.maxTokens}
                          onChange={(e) => setSystemSettings({ ...systemSettings, maxTokens: parseInt(e.target.value) })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="temperature" className="block text-sm font-medium text-gray-700">
                          Temperature
                        </label>
                        <input
                          type="number"
                          step="0.1"
                          min="0"
                          max="1"
                          id="temperature"
                          value={systemSettings.temperature}
                          onChange={(e) => setSystemSettings({ ...systemSettings, temperature: parseFloat(e.target.value) })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="log-level" className="block text-sm font-medium text-gray-700">
                          Log Level
                        </label>
                        <select
                          id="log-level"
                          value={systemSettings.logLevel}
                          onChange={(e) => setSystemSettings({ ...systemSettings, logLevel: e.target.value })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="debug">Debug</option>
                          <option value="info">Info</option>
                          <option value="warn">Warning</option>
                          <option value="error">Error</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="schema-refresh" className="block text-sm font-medium text-gray-700">
                          Schema Refresh Interval (seconds)
                        </label>
                        <input
                          type="number"
                          id="schema-refresh"
                          value={systemSettings.schemaRefreshInterval}
                          onChange={(e) => setSystemSettings({ ...systemSettings, schemaRefreshInterval: parseInt(e.target.value) })}
                          className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                      
                      <div className="flex items-center h-full">
                        <input
                          id="cache-enabled"
                          type="checkbox"
                          checked={systemSettings.cacheEnabled}
                          onChange={(e) => setSystemSettings({ ...systemSettings, cacheEnabled: e.target.checked })}
                          className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                        />
                        <label htmlFor="cache-enabled" className="ml-2 block text-sm text-gray-700">
                          Enable Query Cache
                        </label>
                      </div>
                    </div>
                    
                    <div className="flex justify-end">
                      <button
                        type="submit"
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        Save Settings
                      </button>
                    </div>
                  </form>
                </div>
                
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">System Maintenance</h2>
                    <p className="mt-1 text-sm text-gray-500">
                      Perform system maintenance operations
                    </p>
                  </div>
                  
                  <div className="p-6 space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-md font-medium text-gray-800 mb-2">Cache Management</h3>
                        <p className="text-sm text-gray-600 mb-3">
                          Clear system caches to ensure fresh data
                        </p>
                        <button
                          type="button"
                          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Clear Cache
                        </button>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-md font-medium text-gray-800 mb-2">Schema Management</h3>
                        <p className="text-sm text-gray-600 mb-3">
                          Refresh database schema information
                        </p>
                        <button
                          type="button"
                          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Refresh Schema
                        </button>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-md font-medium text-gray-800 mb-2">Model Health</h3>
                        <p className="text-sm text-gray-600 mb-3">
                          Check LLM model health status
                        </p>
                        <button
                          type="button"
                          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Run Health Check
                        </button>
                      </div>
                      
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h3 className="text-md font-medium text-gray-800 mb-2">System Backup</h3>
                        <p className="text-sm text-gray-600 mb-3">
                          Create a backup of system configuration
                        </p>
                        <button
                          type="button"
                          className="inline-flex justify-center py-2 px-4 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                        >
                          Create Backup
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {/* Audit Logs Tab */}
            {activeTab === 'audit' && (
              <div className="space-y-6">
                <div className="bg-white shadow rounded-lg overflow-hidden">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-800">Audit Logs</h2>
                    <p className="mt-1 text-sm text-gray-500">
                      View system audit logs and activity history
                    </p>
                  </div>
                  
                  <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      <div>
                        <label htmlFor="audit-user" className="block text-sm font-medium text-gray-700 mb-1">
                          User
                        </label>
                        <input
                          type="text"
                          id="audit-user"
                          name="user"
                          value={auditLogFilters.user}
                          onChange={handleAuditLogFilterChange}
                          className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                          placeholder="Filter by user"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="audit-action" className="block text-sm font-medium text-gray-700 mb-1">
                          Action
                        </label>
                        <select
                          id="audit-action"
                          name="action"
                          value={auditLogFilters.action}
                          onChange={handleAuditLogFilterChange}
                          className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        >
                          <option value="">All Actions</option>
                          <option value="LOGIN">Login</option>
                          <option value="USER_CREATE">User Creation</option>
                          <option value="USER_UPDATE">User Update</option>
                          <option value="ROLE_UPDATE">Role Update</option>
                          <option value="SYSTEM_SETTING">System Setting</option>
                          <option value="DATA_EXPORT">Data Export</option>
                        </select>
                      </div>
                      
                      <div>
                        <label htmlFor="audit-date-from" className="block text-sm font-medium text-gray-700 mb-1">
                          From Date
                        </label>
                        <input
                          type="date"
                          id="audit-date-from"
                          name="dateFrom"
                          value={auditLogFilters.dateFrom}
                          onChange={handleAuditLogFilterChange}
                          className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                      
                      <div>
                        <label htmlFor="audit-date-to" className="block text-sm font-medium text-gray-700 mb-1">
                          To Date
                        </label>
                        <input
                          type="date"
                          id="audit-date-to"
                          name="dateTo"
                          value={auditLogFilters.dateTo}
                          onChange={handleAuditLogFilterChange}
                          className="block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        />
                      </div>
                    </div>
                    
                    <div className="mt-4 flex justify-end">
                      <button
                        type="button"
                        onClick={applyAuditLogFilters}
                        className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                      >
                        Apply Filters
                      </button>
                    </div>
                  </div>
                  
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Timestamp
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            User
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Action
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Details
                          </th>
                          <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            IP Address
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {auditLogs.map((log) => (
                          <tr key={log.id}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {new Date(log.timestamp).toLocaleString()}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {log.user}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                {log.action}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-sm text-gray-500">
                              {log.details}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                              {log.ip}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
                    <div className="flex justify-between items-center">
                      <div className="text-sm text-gray-700">
                        Showing <span className="font-medium">1</span> to <span className="font-medium">{auditLogs.length}</span> of <span className="font-medium">{auditLogs.length}</span> logs
                      </div>
                      <div className="flex-1 flex justify-end">
                        <button
                          type="button"
                          className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={true}
                        >
                          Previous
                        </button>
                        <button
                          type="button"
                          className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                          disabled={true}
                        >
                          Next
                        </button>
                      </div>
                    </div>
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

export default Admin;