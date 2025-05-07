import React, { Component } from 'react';
import { FaExclamationTriangle, FaRedo, FaHome } from 'react-icons/fa';
import { Link } from 'react-router-dom';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Catch errors in any components below and re-render with error message
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error info to error tracking service or console
    console.error("Error caught by ErrorBoundary:", error, errorInfo);
  }
  
  handleReset = () => {
    this.setState({ 
      hasError: false,
      error: null,
      errorInfo: null
    });
  }

  render() {
    if (this.state.hasError) {
      // Render custom fallback UI
      return (
        <div className="min-h-full flex flex-col items-center justify-center p-6 bg-gray-50 rounded-lg border border-gray-200 text-center">
          <div className="rounded-full bg-red-100 p-4 mb-4">
            <FaExclamationTriangle className="h-8 w-8 text-red-600" />
          </div>
          
          <h2 className="text-xl font-bold text-gray-800 mb-2">Something went wrong</h2>
          
          <p className="text-gray-600 mb-6">
            {this.props.fallbackMessage || "We encountered an error while rendering this component."}
          </p>
          
          {this.props.showErrorDetails && (
            <div className="mb-6 p-4 bg-gray-100 rounded text-left overflow-auto w-full max-w-md">
              <p className="font-mono text-sm text-red-600 whitespace-pre-wrap">
                {this.state.error && this.state.error.toString()}
              </p>
              <p className="font-mono text-xs text-gray-700 mt-2 whitespace-pre-wrap">
                {this.state.errorInfo && this.state.errorInfo.componentStack}
              </p>
            </div>
          )}
          
          <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-2">
            <button
              onClick={this.handleReset}
              className="flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <FaRedo className="mr-2" /> Try Again
            </button>
            
            <Link
              to="/"
              className="flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <FaHome className="mr-2" /> Go to Dashboard
            </Link>
          </div>
        </div>
      );
    }

    // If no error occurred, render children normally
    return this.props.children;
  }
}

// API Error UI component for fetch/API errors
export const ApiError = ({ 
  error, 
  retry = null,
  message = "We couldn't retrieve the data you requested."
}) => {
  return (
    <div className="p-6 bg-red-50 rounded-lg border border-red-200 text-center">
      <div className="rounded-full bg-red-100 p-3 mx-auto w-fit mb-4">
        <FaExclamationTriangle className="h-6 w-6 text-red-600" />
      </div>
      
      <h3 className="text-lg font-medium text-red-800 mb-2">Error Loading Data</h3>
      
      <p className="text-red-700 mb-4">{message}</p>
      
      {error && (
        <p className="text-sm text-red-600 mb-4 p-2 bg-red-100 rounded">
          {error.toString()}
        </p>
      )}
      
      {retry && (
        <button
          onClick={retry}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
        >
          <FaRedo className="mr-2" /> Retry
        </button>
      )}
    </div>
  );
};

export default ErrorBoundary;