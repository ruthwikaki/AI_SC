import React, { createContext, useContext, useState } from 'react';

const TabsContext = createContext();

export const Tabs = ({ defaultValue, value, onValueChange, children, className = '' }) => {
  const [selectedTab, setSelectedTab] = useState(value || defaultValue);
  
  const handleTabChange = (newValue) => {
    setSelectedTab(newValue);
    if (onValueChange) {
      onValueChange(newValue);
    }
  };
  
  return (
    <TabsContext.Provider value={{ selectedTab: value || selectedTab, onTabChange: handleTabChange }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
};

export const TabsList = ({ children, className = '' }) => {
  return (
    <div className={`flex space-x-1 border-b border-gray-200 ${className}`}>
      {children}
    </div>
  );
};

export const TabsTrigger = ({ value, children, className = '' }) => {
  const { selectedTab, onTabChange } = useContext(TabsContext);
  const isSelected = selectedTab === value;
  
  return (
    <button
      type="button"
      className={`px-4 py-2 text-sm font-medium transition-colors focus:outline-none ${
        isSelected
          ? 'text-blue-600 border-b-2 border-blue-600'
          : 'text-gray-600 hover:text-gray-900'
      } ${className}`}
      onClick={() => onTabChange(value)}
    >
      {children}
    </button>
  );
};

export const TabsContent = ({ value, children, className = '' }) => {
  const { selectedTab } = useContext(TabsContext);
  
  if (selectedTab !== value) return null;
  
  return <div className={className}>{children}</div>;
};
