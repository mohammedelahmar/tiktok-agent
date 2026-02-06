import React from 'react';

const Input = ({ className, label, error, ...props }) => {
  return (
    <div className="w-full">
      {label && (
        <label className="block text-xs font-medium text-gray-400 mb-1.5 ml-1">
          {label}
        </label>
      )}
      <div className="relative group">
        <input
          className={`
            w-full bg-[#0A0A0A] text-gray-200 
            border border-white/10 rounded-xl px-4 py-3
            focus:outline-none focus:ring-2 focus:ring-rose-500/50 focus:border-rose-500/50
            placeholder:text-gray-600 transition-all duration-300
            hover:border-white/20
            ${error ? 'border-red-500/50 focus:ring-red-500/20' : ''}
            ${className || ''}
          `}
          {...props}
        />
        {/* Glow effect on focus (optional extra polish) */}
        <div className="absolute inset-0 rounded-xl bg-rose-500/5 opacity-0 group-focus-within:opacity-100 pointer-events-none transition-opacity duration-300" />
      </div>
      {error && (
        <p className="text-red-400 text-xs mt-1 ml-1">{error}</p>
      )}
    </div>
  );
};

export default Input;
