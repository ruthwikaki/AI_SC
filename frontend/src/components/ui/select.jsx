import React, { useState, useRef, useEffect } from 'react';

export const Select = ({ children, value, onValueChange }) => {
  const [isOpen, setIsOpen] = useState(false);
  const selectRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (selectRef.current && !selectRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={selectRef}>
      {React.Children.map(children, child => {
        if (child.type === SelectTrigger) {
          return React.cloneElement(child, { value, onValueChange, isOpen, setIsOpen });
        }
        if (child.type === SelectContent) {
          return React.cloneElement(child, { value, onValueChange, isOpen, setIsOpen });
        }
        return child;
      })}
    </div>
  );
};

export const SelectTrigger = ({ children, value, isOpen, setIsOpen, className = '' }) => {
  return (
    <button
      type="button"
      className={`w-full px-3 py-2 text-sm text-left border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${className}`}
      onClick={() => setIsOpen(!isOpen)}
    >
      {children}
    </button>
  );
};

export const SelectValue = ({ placeholder = 'Select...', value }) => {
  return <span>{value || placeholder}</span>;
};

export const SelectContent = ({ children, value, onValueChange, isOpen, setIsOpen }) => {
  if (!isOpen) return null;

  return (
    <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
      {React.Children.map(children, child => {
        if (child.type === SelectItem) {
          return React.cloneElement(child, { 
            onValueChange: (val) => {
              onValueChange(val);
              setIsOpen(false);
            },
            isSelected: value === child.props.value
          });
        }
        return child;
      })}
    </div>
  );
};

export const SelectItem = ({ children, value, onValueChange, isSelected }) => {
  return (
    <button
      type="button"
      className={`w-full px-3 py-2 text-sm text-left hover:bg-gray-100 focus:outline-none focus:bg-gray-100 ${isSelected ? 'bg-gray-100' : ''}`}
      onClick={() => onValueChange(value)}
    >
      {children}
    </button>
  );
};
