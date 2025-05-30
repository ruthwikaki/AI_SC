import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null,
      errorInfo: null,
      errorCount: 0
    };
  }

  static getDerivedStateFromError(error) {
    // Update state to show fallback UI and store the error
    return { 
      hasError: true,
      error: error
    };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details for debugging
    console.error('Error caught by boundary:', error, errorInfo);
    
    // Update state with full error information
    this.setState({
      errorInfo: errorInfo,
      errorCount: this.state.errorCount + 1
    });
  }

  handleReset = () => {
    // Reset error state instead of reloading the page
    this.setState({ 
      hasError: false, 
      error: null,
      errorInfo: null 
    });
    
    // If props include an onReset callback, call it
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="max-w-xl w-full bg-white shadow-lg rounded-lg p-8">
            <div className="text-center">
              <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-100">
                <svg className="h-6 w-6 text-red-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              
              <h1 className="mt-4 text-2xl font-bold text-gray-900">Something went wrong</h1>
              
              <p className="mt-2 text-gray-600">
                We apologize for the inconvenience. An unexpected error has occurred.
              </p>
              
              {/* Show error message if available */}
              {this.state.error && (
                <div className="mt-4 p-4 bg-red-50 rounded-md">
                  <p className="text-sm text-red-800 font-mono">
                    {this.state.error.toString()}
                  </p>
                </div>
              )}
              
              {/* Show additional debug info in development */}
              {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
                <details className="mt-4 text-left">
                  <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
                    Error details (Development only)
                  </summary>
                  <pre className="mt-2 p-4 bg-gray-100 rounded-md text-xs overflow-auto">
                    {this.state.errorInfo.componentStack}
                  </pre>
                </details>
              )}
              
              <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center">
                <button
                  onClick={this.handleReset}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                >
                  Try Again
                </button>
                
                <button
                  onClick={() => window.location.href = '/'}
                  className="px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
                >
                  Go to Home
                </button>
              </div>
              
              {this.state.errorCount > 2 && (
                <p className="mt-4 text-sm text-gray-500">
                  Multiple errors detected. If this persists, please contact support.
                </p>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;