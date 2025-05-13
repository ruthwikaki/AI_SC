/**
 * Utility functions for formatting data
 */

/**
 * Format a number with commas as thousands separators
 * @param {number} number - Number to format
 * @param {number} decimals - Number of decimal places (default: 0)
 * @returns {string} Formatted number
 */
export const formatNumber = (number, decimals = 0) => {
    if (number === null || number === undefined || isNaN(number)) {
      return '-';
    }
    
    return number.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };
  
  /**
   * Format a currency value
   * @param {number} value - Value to format
   * @param {string} currency - Currency code (default: 'USD')
   * @param {string} locale - Locale for formatting (default: 'en-US')
   * @returns {string} Formatted currency
   */
  export const formatCurrency = (value, currency = 'USD', locale = 'en-US') => {
    if (value === null || value === undefined || isNaN(value)) {
      return '-';
    }
    
    return new Intl.NumberFormat(locale, {
      style: 'currency',
      currency,
    }).format(value);
  };
  
  /**
   * Format a percentage value
   * @param {number} value - Value to format (e.g., 0.75 for 75%)
   * @param {number} decimals - Number of decimal places (default: 1)
   * @returns {string} Formatted percentage
   */
  export const formatPercentage = (value, decimals = 1) => {
    if (value === null || value === undefined || isNaN(value)) {
      return '-';
    }
    
    return `${(value * 100).toFixed(decimals)}%`;
  };
  
  /**
   * Format a date
   * @param {string|Date} date - Date to format
   * @param {string} format - Format string (default: 'MM/DD/YYYY')
   * @returns {string} Formatted date
   */
  export const formatDate = (date, format = 'MM/DD/YYYY') => {
    if (!date) {
      return '-';
    }
    
    const d = new Date(date);
    
    if (isNaN(d.getTime())) {
      return '-';
    }
    
    // Simple formatter based on format string
    const formatMap = {
      'MM': String(d.getMonth() + 1).padStart(2, '0'),
      'M': String(d.getMonth() + 1),
      'DD': String(d.getDate()).padStart(2, '0'),
      'D': String(d.getDate()),
      'YYYY': d.getFullYear(),
      'YY': String(d.getFullYear()).slice(-2),
      'HH': String(d.getHours()).padStart(2, '0'),
      'H': String(d.getHours()),
      'mm': String(d.getMinutes()).padStart(2, '0'),
      'm': String(d.getMinutes()),
      'ss': String(d.getSeconds()).padStart(2, '0'),
      's': String(d.getSeconds()),
    };
    
    let formattedDate = format;
    
    // Replace format tokens with actual values
    Object.keys(formatMap).forEach((key) => {
      formattedDate = formattedDate.replace(key, formatMap[key]);
    });
    
    return formattedDate;
  };
  
  /**
   * Format a datetime
   * @param {string|Date} date - Date to format
   * @param {boolean} includeSeconds - Whether to include seconds (default: false)
   * @returns {string} Formatted datetime
   */
  export const formatDateTime = (date, includeSeconds = false) => {
    if (!date) {
      return '-';
    }
    
    const format = includeSeconds
      ? 'MM/DD/YYYY HH:mm:ss'
      : 'MM/DD/YYYY HH:mm';
    
    return formatDate(date, format);
  };
  
  /**
   * Format a relative time (e.g., "2 hours ago")
   * @param {string|Date} date - Date to format
   * @returns {string} Relative time
   */
  export const formatRelativeTime = (date) => {
    if (!date) {
      return '-';
    }
    
    const d = new Date(date);
    
    if (isNaN(d.getTime())) {
      return '-';
    }
    
    const now = new Date();
    const diffMs = now - d;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    const diffWeeks = Math.floor(diffDays / 7);
    const diffMonths = Math.floor(diffDays / 30);
    const diffYears = Math.floor(diffDays / 365);
    
    if (diffSecs < 60) {
      return diffSecs === 1 ? '1 second ago' : `${diffSecs} seconds ago`;
    } else if (diffMins < 60) {
      return diffMins === 1 ? '1 minute ago' : `${diffMins} minutes ago`;
    } else if (diffHours < 24) {
      return diffHours === 1 ? '1 hour ago' : `${diffHours} hours ago`;
    } else if (diffDays < 7) {
      return diffDays === 1 ? '1 day ago' : `${diffDays} days ago`;
    } else if (diffWeeks < 4) {
      return diffWeeks === 1 ? '1 week ago' : `${diffWeeks} weeks ago`;
    } else if (diffMonths < 12) {
      return diffMonths === 1 ? '1 month ago' : `${diffMonths} months ago`;
    } else {
      return diffYears === 1 ? '1 year ago' : `${diffYears} years ago`;
    }
  };
  
  /**
   * Truncate text to a specified length and add ellipsis
   * @param {string} text - Text to truncate
   * @param {number} length - Maximum length (default: 30)
   * @returns {string} Truncated text
   */
  export const truncateText = (text, length = 30) => {
    if (!text) {
      return '';
    }
    
    if (text.length <= length) {
      return text;
    }
    
    return `${text.substring(0, length)}...`;
  };
  
  /**
   * Convert bytes to a human-readable format
   * @param {number} bytes - Number of bytes
   * @param {number} decimals - Number of decimal places (default: 2)
   * @returns {string} Formatted size
   */
  export const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
  };
  
  /**
   * Format a duration in seconds to a human-readable format
   * @param {number} seconds - Duration in seconds
   * @returns {string} Formatted duration
   */
  export const formatDuration = (seconds) => {
    if (seconds === null || seconds === undefined || isNaN(seconds)) {
      return '-';
    }
    
    if (seconds < 60) {
      return `${seconds}s`;
    }
    
    const minutes = Math.floor(seconds / 60);
    
    if (minutes < 60) {
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds}s`;
    }
    
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    
    if (hours < 24) {
      return `${hours}h ${remainingMinutes}m`;
    }
    
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    
    return `${days}d ${remainingHours}h`;
  };
  
  /**
   * Format a value based on its type
   * @param {any} value - Value to format
   * @param {string} type - Type of value (number, currency, percentage, date, datetime)
   * @param {Object} options - Additional formatting options
   * @returns {string} Formatted value
   */
  export const formatValue = (value, type, options = {}) => {
    switch (type) {
      case 'number':
        return formatNumber(value, options.decimals);
      case 'currency':
        return formatCurrency(value, options.currency, options.locale);
      case 'percentage':
        return formatPercentage(value, options.decimals);
      case 'date':
        return formatDate(value, options.format);
      case 'datetime':
        return formatDateTime(value, options.includeSeconds);
      case 'relative':
        return formatRelativeTime(value);
      case 'bytes':
        return formatBytes(value, options.decimals);
      case 'duration':
        return formatDuration(value);
      default:
        return value !== null && value !== undefined ? String(value) : '-';
    }
  };
  
  /**
   * Format a camelCase or snake_case string to title case
   * @param {string} str - String to format
   * @returns {string} Formatted string
   */
  export const formatTitleCase = (str) => {
    if (!str) {
      return '';
    }
    
    // Replace underscores and camelCase with spaces
    const spaced = str
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .trim();
    
    // Capitalize first letter of each word
    return spaced
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  };
  
  /**
   * Format a string to camelCase
   * @param {string} str - String to format
   * @returns {string} Camel case string
   */
  export const toCamelCase = (str) => {
    if (!str) {
      return '';
    }
    
    return str
      .replace(/(?:^\w|[A-Z]|\b\w)/g, (letter, index) => 
        index === 0 ? letter.toLowerCase() : letter.toUpperCase()
      )
      .replace(/\s+|_+|-+/g, '');
  };
  
  /**
   * Format a string to snake_case
   * @param {string} str - String to format
   * @returns {string} Snake case string
   */
  export const toSnakeCase = (str) => {
    if (!str) {
      return '';
    }
    
    return str
      .replace(/\s+/g, '_')
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '')
      .replace(/_+/g, '_');
  };