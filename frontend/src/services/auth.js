console.log('AUTH.JS LOADED - FIXED VERSION');
import api from './api';

const authService = {
  // Register new user
  async register(userData) {
    try {
      const registrationData = {
        username: userData.email.split('@')[0],
        email: userData.email,
        password: userData.password,
      };
      
      const response = await api.post('/api/auth/register', registrationData);
      return response.data;
    } catch (error) {
      console.error('Registration error:', error.response?.data);
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error('Registration failed. Please try again.');
    }
  },

  // Login user
  async login(email, password, rememberMe = true) {
    try {
      console.log('=== AUTH SERVICE LOGIN ===');
      console.log('Email received:', email);
      console.log('Password received:', password);
      console.log('Email type:', typeof email);
      
      // Extract username from email
      const username = email && typeof email === 'string' && email.includes('@') 
        ? email.split('@')[0] 
        : email;
      
      console.log('Extracted username:', username);
      
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      console.log('Sending to API:', formData.toString());
      
      const response = await api.post('/api/auth/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      console.log('Login successful!', response.data);
      
      if (response.data.access_token) {
        const storage = rememberMe ? localStorage : sessionStorage;
        storage.setItem('authToken', response.data.access_token);
        
        if (response.data.refresh_token) {
          storage.setItem('refreshToken', response.data.refresh_token);
        }
        
        api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
        
        // Get user details
        const userResponse = await api.get('/api/auth/me');
        const user = userResponse.data;
        storage.setItem('user', JSON.stringify(user));
        
        return {
          user,
          token: response.data.access_token,
          refreshToken: response.data.refresh_token
        };
      }
      
      throw new Error('Login failed - no access token received');
    } catch (error) {
      console.error('Auth service - login error:', error);
      console.error('Error response:', error.response?.data);
      
      if (error.response?.status === 401) {
        throw new Error(error.response.data?.detail || 'Invalid credentials');
      }
      
      throw error;
    }
  },

  // Logout user
  logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
    sessionStorage.removeItem('authToken');
    sessionStorage.removeItem('refreshToken');
    sessionStorage.removeItem('user');
    delete api.defaults.headers.common['Authorization'];
  },

  // Get current user
  getCurrentUser() {
    const userStr = localStorage.getItem('user') || sessionStorage.getItem('user');
    if (userStr) {
      try {
        return JSON.parse(userStr);
      } catch {
        return null;
      }
    }
    return null;
  },

  // Check if authenticated
  isAuthenticated() {
    return !!(localStorage.getItem('authToken') || sessionStorage.getItem('authToken'));
  }
};

export default authService;
