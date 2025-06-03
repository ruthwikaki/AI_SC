import React, { useState } from 'react';
import { MagnifyingGlassIcon, TableCellsIcon, CircleStackIcon } from '@heroicons/react/24/outline';

const Database = () => {
  const [selectedTable, setSelectedTable] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Mock data for demonstration
  const tables = [
    { name: 'suppliers', count: 1234, lastUpdated: '2025-06-01' },
    { name: 'products', count: 5678, lastUpdated: '2025-06-01' },
    { name: 'inventory', count: 3456, lastUpdated: '2025-06-01' },
    { name: 'orders', count: 9012, lastUpdated: '2025-05-31' },
    { name: 'shipments', count: 4567, lastUpdated: '2025-05-31' },
    { name: 'warehouses', count: 89, lastUpdated: '2025-05-30' },
  ];

  const filteredTables = tables.filter(table => 
    table.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900">Database Explorer</h1>
        <p className="mt-2 text-gray-600">
          Browse and query your supply chain database tables
        </p>
      </div>

      {/* Search Bar */}
      <div className="mb-6">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
            placeholder="Search tables..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Tables Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filteredTables.map((table) => (
          <div
            key={table.name}
            onClick={() => setSelectedTable(table.name)}
            className={`relative rounded-lg border p-6 cursor-pointer transition-all hover:shadow-md ${
              selectedTable === table.name
                ? 'border-indigo-600 bg-indigo-50'
                : 'border-gray-300 bg-white hover:border-gray-400'
            }`}
          >
            <div className="flex items-center">
              <CircleStackIcon className={`h-8 w-8 ${
                selectedTable === table.name ? 'text-indigo-600' : 'text-gray-400'
              }`} />
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-900">{table.name}</h3>
                <p className="text-sm text-gray-500">{table.count.toLocaleString()} records</p>
              </div>
            </div>
            <div className="mt-4">
              <p className="text-xs text-gray-500">
                Last updated: {new Date(table.lastUpdated).toLocaleDateString()}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Selected Table Preview */}
      {selectedTable && (
        <div className="mt-8 bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Table: {selectedTable}
              </h3>
              <button className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                <TableCellsIcon className="-ml-0.5 mr-2 h-4 w-4" />
                View Full Table
              </button>
            </div>
            
            <div className="border-t border-gray-200 pt-4">
              <p className="text-sm text-gray-500">
                Sample data preview will be displayed here. Integration with actual database required.
              </p>
              
              {/* Mock sample data */}
              <div className="mt-4 overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Column 1
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Column 2
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Column 3
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        Sample Data
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        Sample Value
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        Sample Info
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Coming Soon Notice */}
      <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-md p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">
              Feature in Development
            </h3>
            <div className="mt-2 text-sm text-yellow-700">
              <p>
                Full database exploration capabilities are coming soon. This includes:
              </p>
              <ul className="list-disc list-inside mt-1">
                <li>Real-time data browsing</li>
                <li>Custom SQL query execution</li>
                <li>Data export functionality</li>
                <li>Schema visualization</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Database;