import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { Bars3Icon, BellIcon, UserCircleIcon, XMarkIcon } from '@heroicons/react/24/outline';

const Navbar = ({ onMenuClick, user }) => {
  const navigate = useNavigate();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Use the user prop instead of useAuth
  const mockUser = user || { name: 'Test User', email: 'test@example.com' };

  const handleLogout = () => {
    console.log('Logout clicked');
    // Clear auth data
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user');
    sessionStorage.removeItem('auth_token');
    sessionStorage.removeItem('user');
    navigate('/login');
  };

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={onMenuClick}
              className="px-4 text-gray-500 focus:outline-none focus:text-gray-600 lg:hidden"
              aria-label="Toggle sidebar"
            >
              {isMobileMenuOpen ? (
                <XMarkIcon className="h-6 w-6" />
              ) : (
                <Bars3Icon className="h-6 w-6" />
              )}
            </button>
            
            <div className="flex-shrink-0 flex items-center ml-4">
              <h1 className="text-xl font-semibold text-gray-800">Supply Chain LLM</h1>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button 
              className="text-gray-500 hover:text-gray-700"
              aria-label="Notifications"
            >
              <BellIcon className="h-6 w-6" />
            </button>

            <div className="relative group">
              <button 
                className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                aria-label="User menu"
              >
                <UserCircleIcon className="h-8 w-8 text-gray-400" />
                <span className="ml-2 text-gray-700">{mockUser.name}</span>
              </button>
              
              {/* Dropdown menu - could be implemented later */}
              <div className="hidden group-hover:block absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
                <NavLink 
                  to="/settings" 
                  className={({ isActive }) => 
                    `block px-4 py-2 text-sm ${isActive ? 'bg-gray-100 text-gray-900' : 'text-gray-700 hover:bg-gray-100'}`
                  }
                >
                  Settings
                </NavLink>
                <button
                  onClick={handleLogout}
                  className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                >
                  Logout
                </button>
              </div>
            </div>

            <button
              onClick={handleLogout}
              className="text-gray-500 hover:text-gray-700 text-sm hidden sm:block"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;