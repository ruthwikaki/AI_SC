import api from './api';

const authService = {
  login: async (credentials) => {
    try {
      console.log('Auth service - login attempt with:', credentials);
      
      // Ensure we have email/username and password
      if (!credentials.email || !credentials.password) {
        throw new Error('Email and password are required');
      }
      
      // Convert to form data for OAuth2 compatibility
      const formData = new URLSearchParams();
      formData.append('username', credentials.email); // OAuth2 expects 'username' field
      formData.append('password', credentials.password);
      
      console.log('Sending login request to /api/auth/token');
      
      const response = await api.post('/api/auth/token', formData.toString(), {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });
      
      console.log('Login response:', response.data);
      const data = response.data;
      
      // Store token and user info
      if (data.access_token) {
        localStorage.setItem('authToken', data.access_token);
        
        // Create user object from token response
        const user = {
          id: data.user_id,
          email: credentials.email,
          role: data.role,
          permissions: data.permissions || []
        };
        
        localStorage.setItem('user', JSON.stringify(user));
        
        // Set default auth header
        api.defaults.headers.common['Authorization'] = `Bearer ${data.access_token}`;
        
        // Try to fetch full user details
        try {
          const userResponse = await api.get('/api/auth/me');
          if (userResponse.data) {
            localStorage.setItem('user', JSON.stringify(userResponse.data));
            return { ...data, user: userResponse.data };
          }
        } catch (error) {
          console.warn('Could not fetch user details, using token data');
        }
        
        return { ...data, user };
      }
      
      throw new Error('No access token received');
    } catch (error) {
      console.error('Auth service - login error:', error);
      console.error('Error response:', error.response?.data);
      throw error;
    }
  },

  register: async (userData) => {
    try {
      const response = await api.post('/api/auth/register', userData);
      return response.data;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  },

  logout: () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    sessionStorage.removeItem('authToken');
    sessionStorage.removeItem('user');
    delete api.defaults.headers.common['Authorization'];
  },

  getCurrentUser: () => {
    try {
      const userStr = localStorage.getItem('user');
      
      // Check for invalid values
      if (!userStr || userStr === 'undefined' || userStr === 'null') {
        return null;
      }
      
      // Try to parse the user data
      const user = JSON.parse(userStr);
      
      // Validate that it's a valid user object
      if (user && typeof user === 'object' && (user.id || user.email)) {
        return user;
      }
      
      // Invalid user data, clear it
      console.warn('Invalid user data found, clearing...');
      localStorage.removeItem('user');
      return null;
    } catch (error) {
      console.error('Error parsing user data:', error);
      console.error('Corrupted data:', localStorage.getItem('user'));
      // Clear corrupted data
      localStorage.removeItem('user');
      localStorage.removeItem('authToken');
      return null;
    }
  },

  getToken: () => {
    const token = localStorage.getItem('authToken');
    // Check for invalid token values
    if (token && token !== 'undefined' && token !== 'null') {
      return token;
    }
    return null;
  },

  isAuthenticated: () => {
    const token = authService.getToken();
    return !!token;
  },

  refreshToken: async () => {
    try {
      const response = await api.post('/api/auth/refresh');
      const { access_token } = response.data;
      
      if (access_token) {
        localStorage.setItem('authToken', access_token);
        api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
      }
      
      return response.data;
    } catch (error) {
      console.error('Token refresh error:', error);
      authService.logout();
      throw error;
    }
  },

  // Utility function to clear all auth data
  clearAuthData: () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    sessionStorage.removeItem('authToken');
    sessionStorage.removeItem('user');
    delete api.defaults.headers.common['Authorization'];
    console.log('All auth data cleared');
  }
};

// Clear any corrupted data on load
if (typeof window !== 'undefined') {
  const userStr = localStorage.getItem('user');
  if (userStr === 'undefined' || userStr === 'null' || userStr === '') {
    console.log('Found corrupted auth data, clearing...');
    authService.clearAuthData();
  }
}

export default authService;