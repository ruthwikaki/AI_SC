import api from './api';

const authService = {
  // Register new user
  async register(userData) {
    try {
      // Backend expects username, email, and password
      const registrationData = {
        username: userData.email.split('@')[0], // Use email prefix as username
        email: userData.email,
        password: userData.password,
        first_name: userData.firstName,
        last_name: userData.lastName,
        company: userData.company,
        job_title: userData.jobTitle
      };
      
      const response = await api.post('/auth/register', registrationData);
      return response.data;
    } catch (error) {
      if (error.response?.data?.detail) {
        throw new Error(error.response.data.detail);
      }
      throw new Error('Registration failed. Please try again.');
    }
  },

  // Login user
  async login(email, password, rememberMe = true) {
    try {
      // OAuth2 compatible login
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);
      
      const response = await api.post('/auth/token', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      if (response.data.access_token) {
        // Use appropriate storage based on rememberMe
        const storage = rememberMe ? localStorage : sessionStorage;
        
        storage.setItem('authToken', response.data.access_token);
        
        // Store refresh token if provided
        if (response.data.refresh_token) {
          storage.setItem('refreshToken', response.data.refresh_token);
        }
        
        api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
        
        // Get user details
        const userResponse = await api.get('/auth/me');
        const user = userResponse.data;
        storage.setItem('user', JSON.stringify(user));
        
        return { 
          user, 
          token: response.data.access_token,
          refreshToken: response.data.refresh_token 
        };
      }
      
      throw new Error('Login failed');
    } catch (error) {
      throw error;
    }
  },

  // Refresh access token
  async refreshToken() {
    try {
      const refreshToken = localStorage.getItem('refreshToken') || sessionStorage.getItem('refreshToken');
      
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }
      
      const formData = new URLSearchParams();
      formData.append('refresh_token', refreshToken);
      formData.append('grant_type', 'refresh_token');
      
      const response = await api.post('/auth/token/refresh', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      if (response.data.access_token) {
        // Determine which storage was used
        const storage = localStorage.getItem('authToken') ? localStorage : sessionStorage;
        
        storage.setItem('authToken', response.data.access_token);
        api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`;
        
        // Update refresh token if a new one is provided
        if (response.data.refresh_token) {
          storage.setItem('refreshToken', response.data.refresh_token);
        }
        
        // Get updated user details
        const userResponse = await api.get('/auth/me');
        const user = userResponse.data;
        storage.setItem('user', JSON.stringify(user));
        
        return { 
          user, 
          token: response.data.access_token,
          refreshToken: response.data.refresh_token 
        };
      }
      
      throw new Error('Token refresh failed');
    } catch (error) {
      throw error;
    }
  },

  // Logout user
  logout() {
    // Clear from both storages to be sure
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