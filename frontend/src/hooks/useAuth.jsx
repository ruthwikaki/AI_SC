import { useState, useEffect, useContext, createContext } from 'react';
import authService from '../services/auth';

// Create auth context
const AuthContext = createContext();

// Auth Provider component
export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Check if user is already logged in on mount
  useEffect(() => {
    const checkAuthStatus = async () => {
      try {
        setLoading(true);
        const currentUser = await authService.getCurrentUser();
        
        if (currentUser) {
          setUser(currentUser);
        }
      } catch (err) {
        console.error('Authentication check failed:', err);
        // Do not set error here, as this is just a check
      } finally {
        setLoading(false);
      }
    };
    
    checkAuthStatus();
  }, []);
  
  // Login function
  const login = async (email, password, rememberMe = false) => {
    try {
      setLoading(true);
      setError(null);
      
      const userData = await authService.login(email, password, rememberMe);
      setUser(userData);
      
      return { success: true, user: userData };
    } catch (err) {
      setError(err.message || 'Login failed');
      return { success: false, error: err.message || 'Login failed' };
    } finally {
      setLoading(false);
    }
  };
  
  // Register function
  const register = async (userData) => {
    try {
      setLoading(true);
      setError(null);
      
      const newUser = await authService.register(userData);
      
      // Note: typically we don't log in automatically after registration
      // as email verification might be required
      
      return { success: true, user: newUser };
    } catch (err) {
      setError(err.message || 'Registration failed');
      return { success: false, error: err.message || 'Registration failed' };
    } finally {
      setLoading(false);
    }
  };
  
  // Logout function
  const logout = async () => {
    try {
      setLoading(true);
      await authService.logout();
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Update user profile
  const updateUserProfile = async (profileData) => {
    try {
      setLoading(true);
      setError(null);
      
      const updatedUser = await authService.updateProfile(profileData);
      setUser({ ...user, ...updatedUser });
      
      return { success: true, user: updatedUser };
    } catch (err) {
      setError(err.message || 'Profile update failed');
      return { success: false, error: err.message || 'Profile update failed' };
    } finally {
      setLoading(false);
    }
  };
  
  // Change password
  const changePassword = async (currentPassword, newPassword) => {
    try {
      setLoading(true);
      setError(null);
      
      await authService.changePassword(currentPassword, newPassword);
      
      return { success: true };
    } catch (err) {
      setError(err.message || 'Password change failed');
      return { success: false, error: err.message || 'Password change failed' };
    } finally {
      setLoading(false);
    }
  };
  
  // Reset password (forgot password)
  const resetPassword = async (email) => {
    try {
      setLoading(true);
      setError(null);
      
      await authService.resetPassword(email);
      
      return { success: true };
    } catch (err) {
      setError(err.message || 'Password reset failed');
      return { success: false, error: err.message || 'Password reset failed' };
    } finally {
      setLoading(false);
    }
  };
  
  // Computed value for isAuthenticated
  const isAuthenticated = !!user;
  
  // Check if user has a specific role
  const hasRole = (role) => {
    if (!user) return false;
    return user.role === role;
  };
  
  // Check if user has a specific permission
  const hasPermission = (permission) => {
    if (!user || !user.permissions) return false;
    return user.permissions.includes(permission);
  };
  
  // Auth context value
  const value = {
    user,
    loading,
    error,
    isAuthenticated,
    login,
    register,
    logout,
    updateUserProfile,
    changePassword,
    resetPassword,
    hasRole,
    hasPermission
  };
  
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

// Custom hook for using auth context
export function useAuth() {
  const context = useContext(AuthContext);
  
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  
  return context;
}

export default useAuth;