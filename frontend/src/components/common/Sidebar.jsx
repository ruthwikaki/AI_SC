import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';

// Icons
import {
  HomeIcon,
  QuestionMarkCircleIcon,
  ChartBarIcon,
  CubeTransparentIcon,
  Cog6ToothIcon,
  ShieldCheckIcon,
  DocumentTextIcon,
  CircleStackIcon,
  XMarkIcon,
  ArrowTrendingUpIcon  // Add this for Forecasting
} from '@heroicons/react/24/outline';

const Sidebar = ({ isOpen, onClose }) => {
  const location = useLocation();
  const { user } = useAuth();

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Query', href: '/query', icon: QuestionMarkCircleIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Forecasting', href: '/forecasting', icon: ArrowTrendingUpIcon },  // Add this line
    { name: 'Multi-Tier View', href: '/multi-tier', icon: CubeTransparentIcon },
    //{ name: 'Database Explorer', href: '/database', icon: CircleStackIcon },
    //{ name: 'Reports', href: '/reports', icon: DocumentTextIcon },
    { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
  ];

  // Add admin link if user is admin
  if (user && user.role === 'admin') {
    navigation.push({ name: 'Admin', href: '/admin', icon: ShieldCheckIcon });
  }

  return (
    <>
      {/* Mobile sidebar overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-gray-600 bg-opacity-75 z-30 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-40 w-64 bg-white border-r border-gray-200 
        transform transition-transform duration-300 ease-in-out
        lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 flex flex-col pt-5 pb-4 overflow-y-auto">
            <div className="flex items-center justify-between flex-shrink-0 px-4">
              <span className="text-lg font-semibold text-indigo-600">Supply Chain LLM</span>
              <button
                onClick={onClose}
                className="lg:hidden text-gray-500 hover:text-gray-700"
                aria-label="Close sidebar"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>
            <div className="mt-5 flex-1 px-2 space-y-1">
              {navigation.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={({ isActive }) => `
                    group flex items-center px-2 py-2 text-sm font-medium rounded-md
                    ${isActive
                      ? 'bg-indigo-50 text-indigo-600'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'}
                  `}
                  aria-current={({ isActive }) => isActive ? 'page' : undefined}
                  onClick={() => {
                    // Close mobile sidebar after navigation
                    if (window.innerWidth < 1024) {
                      onClose();
                    }
                  }}
                >
                  {({ isActive }) => (
                    <>
                      <item.icon
                        className={`
                          mr-3 flex-shrink-0 h-6 w-6
                          ${isActive ? 'text-indigo-600' : 'text-gray-400 group-hover:text-gray-500'}
                        `}
                        aria-hidden="true"
                      />
                      {item.name}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
          <div className="flex-shrink-0 flex border-t border-gray-200 p-4">
            <div className="flex-shrink-0 w-full group block">
              <div className="flex items-center">
                <div>
                  <div className="h-9 w-9 rounded-full bg-indigo-600 flex items-center justify-center text-white">
                    {user ? user.username?.charAt(0).toUpperCase() : 'U'}
                  </div>
                </div>
                <div className="ml-3">
                  <p className="text-sm font-medium text-gray-700">{user ? user.username : 'Guest'}</p>
                  <p className="text-xs font-medium text-gray-500 group-hover:text-gray-700">
                    {user ? user.email : 'Not logged in'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
