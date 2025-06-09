import React, { useState } from 'react';

export const Dialog = ({ open, onOpenChange, children }) => {
  return (
    <>
      {React.Children.map(children, child => {
        if (child.type === DialogTrigger) {
          return React.cloneElement(child, { onOpenChange });
        }
        if (child.type === DialogContent) {
          return open ? React.cloneElement(child, { onOpenChange }) : null;
        }
        return child;
      })}
    </>
  );
};

export const DialogTrigger = ({ children, onOpenChange, asChild }) => {
  const handleClick = () => onOpenChange(true);
  
  if (asChild && React.isValidElement(children)) {
    return React.cloneElement(children, { onClick: handleClick });
  }
  
  return (
    <button onClick={handleClick}>
      {children}
    </button>
  );
};

export const DialogContent = ({ className = '', children, onOpenChange }) => {
  return (
    <>
      <div 
        className="fixed inset-0 bg-black/50 z-50" 
        onClick={() => onOpenChange(false)}
      />
      <div className={`fixed left-[50%] top-[50%] z-50 max-h-[85vh] w-[90vw] max-w-[450px] translate-x-[-50%] translate-y-[-50%] rounded-lg bg-white p-6 shadow-lg ${className}`}>
        {children}
        <button
          className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-white transition-opacity hover:opacity-100"
          onClick={() => onOpenChange(false)}
        >
          <span className="sr-only">Close</span>
          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </>
  );
};

export const DialogHeader = ({ className = '', ...props }) => (
  <div className={`flex flex-col space-y-1.5 text-center sm:text-left ${className}`} {...props} />
);

export const DialogTitle = ({ className = '', ...props }) => (
  <h3 className={`text-lg font-semibold ${className}`} {...props} />
);
