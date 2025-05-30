import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link } from 'react-router-dom';
import { AuthProvider, useAuth } from './hooks/useAuth';

// Pages
import Dashboard from './pages/Dashboard';
import QueryPage from './pages/QueryPage';
import Analytics from './pages/Analytics';
import MultiTier from './pages/MultiTier';
import Settings from './pages/Settings';
import Admin from './pages/Admin';

// Auth Components
import Login from './components/auth/Login';
import Register from './components/auth/Register';

// Common Components
import Navbar from './components/common/Navbar';
import Sidebar from './components/common/Sidebar';
import ErrorBoundary from './components/common/ErrorBoundary';
import Loading from './components/common/Loading';

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return <Loading fullScreen={true} message="Loading..." />;
  }
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
};

// Admin Route Component - Now properly checks role
const AdminRoute = ({ children }) => {
  const { user, loading } = useAuth();
  
  if (loading) {
    return <Loading fullScreen={true} message="Loading..." />;
  }
  
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  
  // Check if user has admin role
  if (user.role !== 'admin' && user.role !== 'administrator') {
    return <Navigate to="/dashboard" replace />;
  }
  
  return children;
};

// Layout Component for authenticated pages
const AuthenticatedLayout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = React.useState(true);
  const { user } = useAuth();
  
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} user={user} />
      <div className="flex">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className={`flex-1 transition-all duration-300 ${sidebarOpen ? 'lg:ml-64' : 'ml-0'}`}>
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};

function App() {
  return (
    <ErrorBoundary>
      <Router>
        <AuthProvider>
          <Routes>
            {/* Public Routes */}
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            
            {/* Protected Routes */}
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <Dashboard />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            {/* Query routes with optional queryId parameter */}
            <Route path="/query" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <QueryPage />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            <Route path="/query/:queryId" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <QueryPage />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            <Route path="/analytics" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <Analytics />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            <Route path="/multi-tier" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <MultiTier />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            <Route path="/settings" element={
              <ProtectedRoute>
                <AuthenticatedLayout>
                  <Settings />
                </AuthenticatedLayout>
              </ProtectedRoute>
            } />
            
            {/* Admin Routes - Now properly protected by role */}
            <Route path="/admin" element={
              <AdminRoute>
                <AuthenticatedLayout>
                  <Admin />
                </AuthenticatedLayout>
              </AdminRoute>
            } />
            
            {/* Default Route */}
            <Route path="/" element={<Navigate to="/login" replace />} />
            
            {/* 404 Route */}
            <Route path="*" element={
              <div className="min-h-screen flex items-center justify-center">
                <div className="text-center">
                  <h1 className="text-6xl font-bold text-gray-300">404</h1>
                  <p className="text-xl text-gray-600 mt-4">Page not found</p>
                  <Link to="/dashboard" className="mt-6 inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                    Go to Dashboard
                  </Link>
                </div>
              </div>
            } />
          </Routes>
        </AuthProvider>
      </Router>
    </ErrorBoundary>
  );
}

export default App;