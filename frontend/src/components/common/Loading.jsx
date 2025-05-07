import React from 'react';

const Loading = ({ size = 'md', text = 'Loading...', fullScreen = false }) => {
  // Determine spinner size based on the size prop
  const spinnerSizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
    xl: 'w-16 h-16'
  };
  
  const spinnerSize = spinnerSizes[size] || spinnerSizes.md;
  
  // Full screen loading overlay
  if (fullScreen) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-gray-900 bg-opacity-60 z-50">
        <div className="text-center">
          <div className="flex justify-center">
            <div className={`${spinnerSize} border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin`}></div>
          </div>
          {text && <p className="mt-4 text-white font-semibold">{text}</p>}
        </div>
      </div>
    );
  }
  
  // Inline loading spinner
  return (
    <div className="flex items-center justify-center py-4">
      <div className={`${spinnerSize} border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin`}></div>
      {text && <p className="ml-3 text-gray-700">{text}</p>}
    </div>
  );
};

// Specific variants for different loading states
export const QueryLoading = () => (
  <div className="my-6 p-6 bg-gray-50 rounded-lg border border-gray-200">
    <Loading 
      size="lg" 
      text="Processing your query. The LLM is analyzing supply chain data..." 
    />
    <div className="mt-4 text-sm text-gray-500">
      <p className="text-center">Complex supply chain analyses may take a moment.</p>
      <div className="mt-3 h-2 bg-gray-200 rounded overflow-hidden">
        <div 
          className="h-full bg-blue-500 rounded animate-pulse"
          style={{ width: '60%' }}
        ></div>
      </div>
    </div>
  </div>
);

export const DataLoading = () => (
  <div className="flex flex-col items-center justify-center py-8">
    <Loading size="lg" text="Loading data..." />
    <p className="mt-2 text-sm text-gray-500">Retrieving supply chain data records...</p>
  </div>
);

export const ModelLoading = () => (
  <div className="flex flex-col items-center justify-center h-40 bg-gray-50 rounded-lg border border-gray-200">
    <Loading size="lg" text="Loading model..." />
    <p className="mt-2 text-sm text-gray-500">This may take a few moments</p>
  </div>
);

export default Loading;