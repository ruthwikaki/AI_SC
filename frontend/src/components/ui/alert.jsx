import React from 'react';

export const Alert = ({ variant = 'default', className = '', children, ...props }) => {
  const variants = {
    default: 'bg-blue-50 text-blue-900 border-blue-200',
    destructive: 'bg-red-50 text-red-900 border-red-200',
    warning: 'bg-yellow-50 text-yellow-900 border-yellow-200',
    success: 'bg-green-50 text-green-900 border-green-200',
  };
  
  return (
    <div
      className={`px-4 py-3 border rounded-md ${variants[variant]} ${className}`}
      role="alert"
      {...props}
    >
      {children}
    </div>
  );
};

export const AlertDescription = ({ className = '', children, ...props }) => {
  return (
    <div className={`text-sm ${className}`} {...props}>
      {children}
    </div>
  );
};

export const AlertTitle = ({ className = '', children, ...props }) => {
  return (
    <h5 className={`font-medium mb-1 ${className}`} {...props}>
      {children}
    </h5>
  );
};
