import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaBars, FaServer, FaCog, FaChartLine, FaSearch, FaDatabase } from 'react-icons/fa';

const Navbar = ({ toggleSidebar }) => {
  const [serverStatus, setServerStatus] = useState('unknown');
  const [modelCount, setModelCount] = useState(0);
  const location = useLocation();

  useEffect(() => {
    // Fetch server status on component mount
    const checkServerStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          const data = await response.json();
          setServerStatus(data.status);
          setModelCount(data.models.filter(model => model.loaded).length);
        } else {
          setServerStatus('offline');
        }
      } catch (error) {
        console.error('Error checking server status:', error);
        setServerStatus('offline');
      }
    };

    checkServerStatus();
    
    // Poll server status every 30 seconds
    const interval = setInterval(checkServerStatus, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <nav className="bg-gray-800 text-white p-4 flex items-center justify-between shadow-md">
      <div className="flex items-center">
        <button 
          onClick={toggleSidebar}
          className="text-white p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-600 mr-2"
        >
          <FaBars className="h-6 w-6" />
        </button>
        
        <Link to="/" className="flex items-center">
          <span className="font-bold text-xl ml-2">Supply Chain LLM</span>
        </Link>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="hidden md:flex items-center">
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            serverStatus === 'healthy' ? 'bg-green-500' : 
            serverStatus === 'offline' ? 'bg-red-500' : 'bg-yellow-500'
          }`}>
            {serverStatus === 'healthy' ? 'Server Online' : 
             serverStatus === 'offline' ? 'Server Offline' : 'Server Initializing'}
          </span>
          {serverStatus === 'healthy' && (
            <span className="ml-2 text-xs text-gray-300">
              {modelCount} {modelCount === 1 ? 'Model' : 'Models'} Loaded
            </span>
          )}
        </div>

        <div className="hidden md:flex space-x-2">
          <Link 
            to="/query" 
            className={`px-3 py-2 rounded hover:bg-gray-700 flex items-center ${
              location.pathname === '/query' ? 'bg-gray-700' : ''
            }`}
          >
            <FaSearch className="mr-1" /> Query
          </Link>
          <Link 
            to="/data" 
            className={`px-3 py-2 rounded hover:bg-gray-700 flex items-center ${
              location.pathname === '/data' ? 'bg-gray-700' : ''
            }`}
          >
            <FaDatabase className="mr-1" /> Data
          </Link>
          <Link 
            to="/analytics" 
            className={`px-3 py-2 rounded hover:bg-gray-700 flex items-center ${
              location.pathname === '/analytics' ? 'bg-gray-700' : ''
            }`}
          >
            <FaChartLine className="mr-1" /> Analytics
          </Link>
          <Link 
            to="/server" 
            className={`px-3 py-2 rounded hover:bg-gray-700 flex items-center ${
              location.pathname === '/server' ? 'bg-gray-700' : ''
            }`}
          >
            <FaServer className="mr-1" /> Server
          </Link>
          <Link 
            to="/settings" 
            className={`px-3 py-2 rounded hover:bg-gray-700 flex items-center ${
              location.pathname === '/settings' ? 'bg-gray-700' : ''
            }`}
          >
            <FaCog className="mr-1" /> Settings
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;