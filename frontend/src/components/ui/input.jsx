import React from 'react';

export const Input = ({ className = '', type = 'text', ...props }) => {
  return (
    <input
      type={type}
      className={`w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed ${className}`}
      {...props}
    />
  );
};
