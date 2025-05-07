import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  FaSearch, 
  FaDatabase, 
  FaChartLine, 
  FaServer, 
  FaCog,
  FaHome,
  FaTable,
  FaTruck,
  FaWarehouse,
  FaBoxes,
  FaTools,
  FaUserCog,
  FaFileUpload
} from 'react-icons/fa';

const Sidebar = ({ isOpen }) => {
  const location = useLocation();

  const mainMenuItems = [
    { path: '/', icon: <FaHome />, label: 'Dashboard' },
    { path: '/query', icon: <FaSearch />, label: 'Query LLM' },
    { path: '/data', icon: <FaDatabase />, label: 'Data Management' },
    { path: '/analytics', icon: <FaChartLine />, label: 'Analytics' },
  ];

  const dataMenuItems = [
    { path: '/data/upload', icon: <FaFileUpload />, label: 'Upload Data' },
    { path: '/data/inventory', icon: <FaBoxes />, label: 'Inventory' },
    { path: '/data/suppliers', icon: <FaTruck />, label: 'Suppliers' },
    { path: '/data/warehouses', icon: <FaWarehouse />, label: 'Warehouses' },
    { path: '/data/tables', icon: <FaTable />, label: 'Data Tables' },
  ];

  const systemMenuItems = [
    { path: '/server', icon: <FaServer />, label: 'Server Status' },
    { path: '/settings', icon: <FaCog />, label: 'Settings' },
    { path: '/system/models', icon: <FaTools />, label: 'Model Management' },
    { path: '/system/users', icon: <FaUserCog />, label: 'User Management' },
  ];

  const renderMenuItems = (items) => {
    return items.map((item) => (
      <Link 
        key={item.path}
        to={item.path} 
        className={`
          flex items-center py-3 px-4 text-gray-300 hover:bg-gray-700 rounded-md mb-1
          ${location.pathname === item.path ? 'bg-gray-700 text-white' : ''}
        `}
      >
        <span className="mr-3">{item.icon}</span>
        <span>{item.label}</span>
      </Link>
    ));
  };

  return (
    <div 
      className={`
        fixed left-0 top-16 h-full bg-gray-800 text-white transition-all duration-300 overflow-y-auto
        ${isOpen ? 'w-64' : 'w-0'}
      `}
      style={{ zIndex: 40 }}
    >
      <div className={`p-4 ${isOpen ? 'block' : 'hidden'}`}>
        <div className="mb-8">
          <h3 className="text-xs uppercase font-bold tracking-wider text-gray-500 mb-2">
            Main Menu
          </h3>
          <div className="space-y-1">
            {renderMenuItems(mainMenuItems)}
          </div>
        </div>

        <div className="mb-8">
          <h3 className="text-xs uppercase font-bold tracking-wider text-gray-500 mb-2">
            Data Management
          </h3>
          <div className="space-y-1">
            {renderMenuItems(dataMenuItems)}
          </div>
        </div>

        <div className="mb-8">
          <h3 className="text-xs uppercase font-bold tracking-wider text-gray-500 mb-2">
            System
          </h3>
          <div className="space-y-1">
            {renderMenuItems(systemMenuItems)}
          </div>
        </div>
        
        <div className="pt-4 mt-6 border-t border-gray-700">
          <div className="px-4 py-2">
            <div className="text-xs text-gray-500">Server Status</div>
            <div className="flex items-center mt-1">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
              <span className="text-sm">System Online</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;