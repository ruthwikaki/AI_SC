// src/services/auth.js
import api from './api';

const AUTH_TOKEN_KEY = 'authToken';
const REFRESH_TOKEN_KEY = 'refreshToken';
const USER_KEY = 'user';

class AuthService {
  async login(email, password) {
    try {
      console.log('Auth service - login attempt:', { email });
      
      // OAuth2PasswordRequestForm expects username and password
      // We'll use email as username since that's what users enter
      const formData = new URLSearchParams();
      formData.append('username', email);  // Send email as username
      formData.append('password', password);
      
      const response = await api.post('/api/auth/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      console.log('Login successful:', response.data);

      const { access_token, token_type, expires_at, user_id, role, permissions } = response.data;

      // Store tokens
      localStorage.setItem(AUTH_TOKEN_KEY, access_token);
      
      // Store user info
      const user = {
        id: user_id,
        email: email,
        role: role,
        permissions: permissions || []
      };
      localStorage.setItem(USER_KEY, JSON.stringify(user));

      return {
        token: access_token,
        user
      };
    } catch (error) {
      console.error('Auth service - login error:', error);
      console.error('Error response:', error.response?.data);
      
      if (error.response?.status === 401) {
        throw new Error(error.response.data.detail || 'Invalid credentials');
      }
      
      throw new Error('Login failed: ' + (error.response?.data?.detail || error.message));
    }
  }

  async register(userData) {
    try {
      const response = await api.post('/api/auth/register', userData);
      return response.data;
    } catch (error) {
      console.error('Registration error:', error);
      throw new Error(error.response?.data?.detail || 'Registration failed');
    }
  }

  async logout() {
    try {
      await api.post('/api/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear local storage regardless
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(REFRESH_TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      sessionStorage.clear();
    }
  }

  async getCurrentUser() {
    try {
      const response = await api.get('/api/auth/me');
      return response.data;
    } catch (error) {
      console.error('Get current user error:', error);
      throw error;
    }
  }

  getStoredUser() {
    const userStr = localStorage.getItem(USER_KEY);
    return userStr ? JSON.parse(userStr) : null;
  }

  getToken() {
    return localStorage.getItem(AUTH_TOKEN_KEY);
  }

  isAuthenticated() {
    return !!this.getToken();
  }

  hasPermission(permission) {
    const user = this.getStoredUser();
    return user?.permissions?.includes(permission) || false;
  }

  hasRole(role) {
    const user = this.getStoredUser();
    return user?.role === role;
  }

  isAdmin() {
    return this.hasRole('admin');
  }
}

export default new AuthService();
