/**
 * Utility functions for input validation
 */

/**
 * Check if a value is empty (null, undefined, empty string, empty array, empty object)
 * @param {any} value - Value to check
 * @returns {boolean} True if value is empty
 */
export const isEmpty = (value) => {
    if (value === null || value === undefined) {
      return true;
    }
    
    if (typeof value === 'string' && value.trim() === '') {
      return true;
    }
    
    if (Array.isArray(value) && value.length === 0) {
      return true;
    }
    
    if (typeof value === 'object' && Object.keys(value).length === 0) {
      return true;
    }
    
    return false;
  };
  
  /**
   * Validate email format
   * @param {string} email - Email to validate
   * @returns {boolean} True if email is valid
   */
  export const isValidEmail = (email) => {
    if (isEmpty(email)) {
      return false;
    }
    
    // RFC 5322 Official Standard Email Regex
    const emailRegex = /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return emailRegex.test(String(email).toLowerCase());
  };
  
  /**
   * Validate password strength
   * @param {string} password - Password to validate
   * @param {Object} options - Validation options
   * @returns {Object} Validation result with pass flag and message
   */
  export const validatePassword = (password, options = {}) => {
    const {
      minLength = 8,
      requireUppercase = true,
      requireLowercase = true,
      requireNumbers = true,
      requireSpecialChars = true,
    } = options;
    
    if (isEmpty(password)) {
      return { pass: false, message: 'Password cannot be empty' };
    }
    
    if (password.length < minLength) {
      return { pass: false, message: `Password must be at least ${minLength} characters long` };
    }
    
    if (requireUppercase && !/[A-Z]/.test(password)) {
      return { pass: false, message: 'Password must contain at least one uppercase letter' };
    }
    
    if (requireLowercase && !/[a-z]/.test(password)) {
      return { pass: false, message: 'Password must contain at least one lowercase letter' };
    }
    
    if (requireNumbers && !/\d/.test(password)) {
      return { pass: false, message: 'Password must contain at least one number' };
    }
    
    if (requireSpecialChars && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      return { pass: false, message: 'Password must contain at least one special character' };
    }
    
    return { pass: true, message: 'Password is valid' };
  };
  
  /**
   * Check if passwords match
   * @param {string} password - Password
   * @param {string} confirmPassword - Confirmation password
   * @returns {boolean} True if passwords match
   */
  export const passwordsMatch = (password, confirmPassword) => {
    return password === confirmPassword;
  };
  
  /**
   * Validate URL format
   * @param {string} url - URL to validate
   * @returns {boolean} True if URL is valid
   */
  export const isValidUrl = (url) => {
    if (isEmpty(url)) {
      return false;
    }
    
    try {
      new URL(url);
      return true;
    } catch (error) {
      return false;
    }
  };
  
  /**
   * Validate phone number format
   * @param {string} phone - Phone number to validate
   * @returns {boolean} True if phone number is valid
   */
  export const isValidPhone = (phone) => {
    if (isEmpty(phone)) {
      return false;
    }
    
    // Basic international phone number validation
    // Allows formats like: +1234567890, 1234567890, 123-456-7890, (123) 456-7890
    const phoneRegex = /^(\+?\d{1,3}[- ]?)?\(?(?:\d{3})\)?[- ]?\d{3}[- ]?\d{4}$/;
    return phoneRegex.test(phone);
  };
  
  /**
   * Validate date format
   * @param {string} date - Date string to validate
   * @param {string} format - Expected format (default: 'YYYY-MM-DD')
   * @returns {boolean} True if date is valid and in the correct format
   */
  export const isValidDate = (date, format = 'YYYY-MM-DD') => {
    if (isEmpty(date)) {
      return false;
    }
    
    if (format === 'YYYY-MM-DD') {
      // Simple validation for YYYY-MM-DD format
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      
      if (!dateRegex.test(date)) {
        return false;
      }
      
      const parsedDate = new Date(date);
      return !isNaN(parsedDate.getTime());
    }
    
    // Add support for other formats as needed
    return false;
  };
  
  /**
   * Validate if a date is in the past
   * @param {string|Date} date - Date to validate
   * @returns {boolean} True if date is in the past
   */
  export const isDateInPast = (date) => {
    if (isEmpty(date)) {
      return false;
    }
    
    const parsedDate = new Date(date);
    
    if (isNaN(parsedDate.getTime())) {
      return false;
    }
    
    const now = new Date();
    return parsedDate < now;
  };
  
  /**
   * Validate if a date is in the future
   * @param {string|Date} date - Date to validate
   * @returns {boolean} True if date is in the future
   */
  export const isDateInFuture = (date) => {
    if (isEmpty(date)) {
      return false;
    }
    
    const parsedDate = new Date(date);
    
    if (isNaN(parsedDate.getTime())) {
      return false;
    }
    
    const now = new Date();
    return parsedDate > now;
  };
  
  /**
   * Validate a credit card number using the Luhn algorithm
   * @param {string} cardNumber - Credit card number to validate
   * @returns {boolean} True if card number is valid
   */
  export const isValidCreditCard = (cardNumber) => {
    if (isEmpty(cardNumber)) {
      return false;
    }
    
    // Remove non-digit characters
    const digits = cardNumber.replace(/\D/g, '');
    
    if (digits.length < 13 || digits.length > 19) {
      return false;
    }
    
    // Luhn algorithm
    let sum = 0;
    let doubled = false;
    
    // Loop from right to left
    for (let i = digits.length - 1; i >= 0; i--) {
      let digit = parseInt(digits.charAt(i), 10);
      
      if (doubled) {
        digit *= 2;
        if (digit > 9) {
          digit -= 9;
        }
      }
      
      sum += digit;
      doubled = !doubled;
    }
    
    return sum % 10 === 0;
  };
  
  /**
   * Validate a form object
   * @param {Object} formData - Form data to validate
   * @param {Object} validationRules - Validation rules
   * @returns {Object} Validation result with errors and isValid flag
   */
  export const validateForm = (formData, validationRules) => {
    const errors = {};
    let isValid = true;
    
    Object.keys(validationRules).forEach((field) => {
      const value = formData[field];
      const rules = validationRules[field];
      
      // Check required
      if (rules.required && isEmpty(value)) {
        errors[field] = rules.errorMessages?.required || 'This field is required';
        isValid = false;
        return;
      }
      
      // Skip other validations if value is empty and not required
      if (isEmpty(value) && !rules.required) {
        return;
      }
      
      // Check email
      if (rules.email && !isValidEmail(value)) {
        errors[field] = rules.errorMessages?.email || 'Please enter a valid email address';
        isValid = false;
        return;
      }
      
      // Check min length
      if (rules.minLength && value.length < rules.minLength) {
        errors[field] = rules.errorMessages?.minLength || `Must be at least ${rules.minLength} characters`;
        isValid = false;
        return;
      }
      
      // Check max length
      if (rules.maxLength && value.length > rules.maxLength) {
        errors[field] = rules.errorMessages?.maxLength || `Must be no more than ${rules.maxLength} characters`;
        isValid = false;
        return;
      }
      
      // Check pattern
      if (rules.pattern && !new RegExp(rules.pattern).test(value)) {
        errors[field] = rules.errorMessages?.pattern || 'Invalid format';
        isValid = false;
        return;
      }
      
      // Check custom validator
      if (rules.validator && typeof rules.validator === 'function') {
        const validatorResult = rules.validator(value, formData);
        
        if (validatorResult !== true) {
          errors[field] = validatorResult || rules.errorMessages?.validator || 'Invalid value';
          isValid = false;
          return;
        }
      }
    });
    
    return { errors, isValid };
  };
  
  /**
   * Get error message for a field
   * @param {Object} errors - Form errors object
   * @param {string} field - Field name
   * @returns {string|null} Error message or null
   */
  export const getFieldError = (errors, field) => {
    return errors[field] || null;
  };
  
  /**
   * Check if a field has an error
   * @param {Object} errors - Form errors object
   * @param {string} field - Field name
   * @returns {boolean} True if field has an error
   */
  export const hasFieldError = (errors, field) => {
    return !!errors[field];
  };