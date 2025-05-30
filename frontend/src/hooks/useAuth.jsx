// frontend/src/hooks/useAuth.jsx
import { createContext, useContext, useState, useEffect } from 'react';
import authService from '../services/auth';
import api from '../services/api';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on mount
    const token = localStorage.getItem('authToken');
    const refreshToken = localStorage.getItem('refreshToken');
    const savedUser = authService.getCurrentUser();
    
    if (token && savedUser) {
      setUser(savedUser);
      // Set the authorization header
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }
    
    setIsLoading(false);
  }, []);

  const login = async (email, password, rememberMe = true) => {
    try {
      const response = await authService.login(email, password, rememberMe);
      setUser(response.user);
      return response;
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  };

  const register = async (firstName, lastName, email, password, company, jobTitle) => {
    try {
      const response = await authService.register({
        firstName,
        lastName,
        email,
        password,
        company,
        jobTitle
      });
      
      // Don't auto-login after registration, let user login manually
      return response;
    } catch (error) {
      console.error('Registration failed:', error);
      throw error;
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
  };

  const refreshAccessToken = async () => {
    try {
      const response = await authService.refreshToken();
      if (response.user) {
        setUser(response.user);
      }
      return response.token;
    } catch (error) {
      // If refresh fails, logout the user
      logout();
      throw error;
    }
  };

  // Add isAuthenticated computed value
  const isAuthenticated = !!user;

  return (
    <AuthContext.Provider value={{
      user,
      login,
      register,
      logout,
      refreshAccessToken,
      isLoading,
      loading: isLoading, // Provide both for compatibility
      isAuthenticated
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};