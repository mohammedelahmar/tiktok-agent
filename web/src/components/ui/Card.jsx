import React from 'react';
import { cn } from '../../lib/utils';

// Helper function shim if not already present
// We will create lib/utils.js later if needed, but for now assuming direct usage or I'll implement a simple one here
// Actually, let's just use clsx/tailwind-merge inline or create the util file first.
// I see clsx and tailwind-merge in package.json, so I should create the utility first.

const Card = ({ className, children, ...props }) => {
  return (
    <div 
      className={`
        relative backdrop-blur-xl bg-carbon-900/60 
        border border-white/5 rounded-2xl 
        shadow-xl shadow-black/20
        ${className || ''}
      `}
      {...props}
    >
      {/* Inner Glow Gradient */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
      
      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

export default Card;
