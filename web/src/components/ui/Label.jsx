import React from 'react';

export const Label = ({ children, className = '', ...props }) => {
  return (
    <label 
      className={`block text-xs font-medium text-gray-400 mb-1.5 ml-1 ${className}`}
      {...props}
    >
      {children}
    </label>
  );
};
