import api from './api';

const authService = {
  /**
   * Login with email and password
   * @param {string} email - User email
   * @param {string} password - User password
   * @param {boolean} rememberMe - Whether to remember the login
   * @returns {Promise<Object>} User data
   */
  login: async (email, password, rememberMe = false) => {
    try {
      const response = await api.post('/auth/login', {
        email,
        password,
        remember_me: rememberMe,
      });
      
      // Store tokens in localStorage
      localStorage.setItem('auth_token', response.token);
      
      if (response.refresh_token) {
        localStorage.setItem('refresh_token', response.refresh_token);
      }
      
      return response.user;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Register a new user
   * @param {Object} userData - User registration data
   * @returns {Promise<Object>} New user data
   */
  register: async (userData) => {
    try {
      const response = await api.post('/auth/register', userData);
      return response.user;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Logout the current user
   * @returns {Promise<void>}
   */
  logout: async () => {
    try {
      // Call logout endpoint to invalidate token on server
      await api.post('/auth/logout');
    } catch (error) {
      console.error('Logout API error:', error);
      // Continue with logout process even if API call fails
    } finally {
      // Clear tokens from localStorage
      localStorage.removeItem('auth_token');
      localStorage.removeItem('refresh_token');
    }
  },
  
  /**
   * Get the current logged in user
   * @returns {Promise<Object|null>} User data if logged in, null otherwise
   */
  getCurrentUser: async () => {
    // Check if token exists
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      return null;
    }
    
    try {
      const response = await api.get('/auth/me');
      return response.user;
    } catch (error) {
      // If token is invalid, clear it
      if (error.status === 401) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
      }
      return null;
    }
  },
  
  /**
   * Update user profile
   * @param {Object} profileData - Profile data to update
   * @returns {Promise<Object>} Updated user data
   */
  updateProfile: async (profileData) => {
    try {
      const response = await api.put('/auth/profile', profileData);
      return response.user;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Change user password
   * @param {string} currentPassword - Current password
   * @param {string} newPassword - New password
   * @returns {Promise<Object>} Success response
   */
  changePassword: async (currentPassword, newPassword) => {
    try {
      const response = await api.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Request password reset for an email
   * @param {string} email - User email
   * @returns {Promise<Object>} Success response
   */
  resetPassword: async (email) => {
    try {
      const response = await api.post('/auth/reset-password', {
        email,
      });
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Confirm password reset with token
   * @param {string} token - Reset token
   * @param {string} newPassword - New password
   * @returns {Promise<Object>} Success response
   */
  confirmResetPassword: async (token, newPassword) => {
    try {
      const response = await api.post('/auth/reset-password/confirm', {
        token,
        new_password: newPassword,
      });
      return response;
    } catch (error) {
      throw error;
    }
  },
  
  /**
   * Check if user has a specific role
   * @param {Object} user - User object
   * @param {string} role - Role to check
   * @returns {boolean} True if user has the role
   */
  hasRole: (user, role) => {
    if (!user) return false;
    return user.role === role;
  },
  
  /**
   * Check if user has a specific permission
   * @param {Object} user - User object
   * @param {string} permission - Permission to check
   * @returns {boolean} True if user has the permission
   */
  hasPermission: (user, permission) => {
    if (!user || !user.permissions) return false;
    return user.permissions.includes(permission);
  },
};

export default authService;