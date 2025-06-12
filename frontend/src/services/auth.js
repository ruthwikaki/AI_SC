import api from './api';

const authService = {
  async login(credentials) {
    try {
      console.log('Auth service - login attempt with:', credentials);
      
      // OAuth2PasswordRequestForm expects form data with 'username' field
      const formData = new URLSearchParams();
      formData.append('username', credentials.email);
      formData.append('password', credentials.password);
      
      console.log('Sending login request to /api/auth/token');
      
      const response = await api.post('/api/auth/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      console.log('Login successful:', response.data);
      
      const { access_token, token_type, expires_at, user_id, role, permissions } = response.data;
      
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('token_type', token_type);
      localStorage.setItem('expires_at', expires_at);
      localStorage.setItem('user_id', user_id);
      localStorage.setItem('user_role', role);
      localStorage.setItem('user_permissions', JSON.stringify(permissions));
      
      api.defaults.headers.common['Authorization'] = `${token_type} ${access_token}`;
      
      return response.data;
    } catch (error) {
      console.error('Auth service - login error:', error);
      if (error.response) {
        console.error('Error response:', error.response.data);
      }
      throw error;
    }
  },
  
  async register(userData) {
    try {
      const response = await api.post('/api/auth/register', userData);
      return response.data;
    } catch (error) {
      console.error('Auth service - register error:', error);
      throw error;
    }
  },
  
  async logout() {
    try {
      await api.post('/api/auth/logout');
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
      localStorage.removeItem('expires_at');
      localStorage.removeItem('user_id');
      localStorage.removeItem('user_role');
      localStorage.removeItem('user_permissions');
      delete api.defaults.headers.common['Authorization'];
      return true;
    } catch (error) {
      console.error('Auth service - logout error:', error);
      throw error;
    }
  },
  
  async getCurrentUser() {
    try {
      const response = await api.get('/api/auth/me');
      return response.data;
    } catch (error) {
      console.error('Auth service - get current user error:', error);
      throw error;
    }
  },
  
  getStoredToken() {
    return localStorage.getItem('access_token');
  },
  
  isAuthenticated() {
    const token = this.getStoredToken();
    const expiresAt = localStorage.getItem('expires_at');
    
    if (!token || !expiresAt) {
      return false;
    }
    
    const expiryDate = new Date(expiresAt);
    return expiryDate > new Date();
  },
};

export default authService;
