import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { AuthProvider } from './hooks/useAuth';
import Dashboard from './pages/Dashboard';
import QueryPage from './pages/QueryPage';
import Analytics from './pages/Analytics';
import MultiTier from './pages/MultiTier';
import Settings from './pages/Settings';
import Admin from './pages/Admin';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import ErrorBoundary from './components/common/ErrorBoundary';

// Protected Route Component
const ProtectedRoute = ({ element }) => {
  // Check if user is authenticated (simplified)
  const isAuthenticated = localStorage.getItem('auth_token');
  
  if (!isAuthenticated) {
    // Redirect to login if not authenticated
    return <Navigate to="/login" replace />;
  }
  
  return element;
};

// Admin Route Component
const AdminRoute = ({ element }) => {
  // Check if user is authenticated and is admin (simplified)
  const isAuthenticated = localStorage.getItem('auth_token');
  
  // In a real app, you'd also check if the user has admin role
  // For now, we'll just use the auth token check
  if (!isAuthenticated) {
    // Redirect to login if not authenticated
    return <Navigate to="/login" replace />;
  }
  
  return element;
};

function App() {
  return (
    <ErrorBoundary>
      <AuthProvider>
        <Router>
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            
            {/* Protected Routes */}
            <Route 
              path="/" 
              element={<ProtectedRoute element={<Dashboard />} />} 
            />
            <Route 
              path="/queries" 
              element={<ProtectedRoute element={<QueryPage />} />} 
            />
            <Route 
              path="/queries/:queryId" 
              element={<ProtectedRoute element={<QueryPage />} />} 
            />
            <Route 
              path="/analytics" 
              element={<ProtectedRoute element={<Analytics />} />} 
            />
            <Route 
              path="/multitier" 
              element={<ProtectedRoute element={<MultiTier />} />} 
            />
            <Route 
              path="/settings" 
              element={<ProtectedRoute element={<Settings />} />} 
            />
            
            {/* Admin Routes */}
            <Route 
              path="/admin" 
              element={<AdminRoute element={<Admin />} />} 
            />
            
            {/* Fallback Route */}
            <Route 
              path="*" 
              element={<Navigate to="/" replace />} 
            />
          </Routes>
        </Router>
      </AuthProvider>
    </ErrorBoundary>
  );
}

export default App;