import React from 'react';

const Loading = ({ size = 'medium', fullScreen = false, message = '' }) => {
  const sizeClasses = {
    small: 'w-4 h-4',
    medium: 'w-8 h-8',
    large: 'w-12 h-12'
  };

  // Fallback to medium size if invalid size is provided
  const spinnerSize = sizeClasses[size] || sizeClasses.medium;

  const Spinner = () => (
    <div className={`${spinnerSize} relative`}>
      <div className="absolute inset-0 border-4 border-gray-200 rounded-full"></div>
      <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 bg-white bg-opacity-90 flex items-center justify-center z-50">
        <div className="text-center">
          <Spinner />
          {message && <p className="mt-4 text-gray-600">{message}</p>}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <Spinner />
      {message && <p className="mt-4 text-gray-600">{message}</p>}
    </div>
  );
};

export default Loading;