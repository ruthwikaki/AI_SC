import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Bars3Icon, BellIcon, UserCircleIcon } from '@heroicons/react/24/outline';

const Navbar = ({ onMenuClick, user }) => {
  const navigate = useNavigate();
  
  // Use the user prop instead of useAuth
  const mockUser = user || { name: 'Test User', email: 'test@example.com' };

  const handleLogout = () => {
    console.log('Logout clicked');
    navigate('/login');
  };

  return (
    <nav className="bg-white shadow-sm border-b">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <button
              onClick={onMenuClick}
              className="px-4 text-gray-500 focus:outline-none focus:text-gray-600"
            >
              <Bars3Icon className="h-6 w-6" />
            </button>
            
            <div className="flex-shrink-0 flex items-center ml-4">
              <h1 className="text-xl font-semibold text-gray-800">Supply Chain LLM</h1>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <button className="text-gray-500 hover:text-gray-700">
              <BellIcon className="h-6 w-6" />
            </button>

            <div className="relative">
              <button className="flex items-center text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                <UserCircleIcon className="h-8 w-8 text-gray-400" />
                <span className="ml-2 text-gray-700">{mockUser.name}</span>
              </button>
            </div>

            <button
              onClick={handleLogout}
              className="text-gray-500 hover:text-gray-700 text-sm"
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
