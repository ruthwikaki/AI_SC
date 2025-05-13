import React from 'react';

// Three different loading components for different contexts
const LoadingSpinner = () => (
  <div className="flex justify-center items-center">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
  </div>
);

const LoadingOverlay = ({ message = 'Loading...' }) => (
  <div className="fixed inset-0 bg-gray-900 bg-opacity-50 flex justify-center items-center z-50">
    <div className="bg-white p-5 rounded-lg shadow-xl flex flex-col items-center space-y-3">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
      <p className="text-gray-700">{message}</p>
    </div>
  </div>
);

const LoadingCard = ({ height = 'h-32', message = 'Loading data...' }) => (
  <div className={`bg-white shadow rounded-lg ${height} flex flex-col justify-center items-center p-4`}>
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mb-3"></div>
    <p className="text-gray-500 text-sm">{message}</p>
  </div>
);

const LoadingTable = ({ rows = 5 }) => (
  <div className="bg-white shadow overflow-hidden sm:rounded-md">
    <div className="animate-pulse">
      <div className="bg-gray-200 h-12 flex items-center px-4"></div>
      {[...Array(rows)].map((_, index) => (
        <div key={index} className="border-t border-gray-200">
          <div className="px-4 py-4 sm:px-6">
            <div className="flex items-center">
              <div className="bg-gray-300 h-4 w-2/3 rounded"></div>
            </div>
            <div className="mt-2">
              <div className="bg-gray-300 h-3 w-1/2 rounded"></div>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
);

// Main component that decides which loading indicator to show
const Loading = ({ type = 'spinner', message, height, rows }) => {
  switch (type) {
    case 'overlay':
      return <LoadingOverlay message={message} />;
    case 'card':
      return <LoadingCard height={height} message={message} />;
    case 'table':
      return <LoadingTable rows={rows} />;
    case 'spinner':
    default:
      return <LoadingSpinner />;
  }
};

export default Loading;