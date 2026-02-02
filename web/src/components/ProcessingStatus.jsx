import React, { useEffect, useState } from 'react';
import { Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

const ProcessingStatus = ({ job, onReset }) => {
  if (!job) return null;

  const getStatusIcon = () => {
    switch (job.status) {
      case 'completed': return <CheckCircle className="text-green-500" size={32} />;
      case 'failed': return <AlertCircle className="text-red-500" size={32} />;
      default: return <Loader2 className="animate-spin text-primary-500" size={32} />;
    }
  };

  const getProgressColor = () => {
     switch (job.status) {
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-primary-500';
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-900 rounded-xl p-8 border border-slate-800 shadow-xl text-center max-w-2xl mx-auto"
    >
      <div className="flex flex-col items-center gap-4 mb-6">
        <div className="p-4 bg-slate-800/50 rounded-full border border-slate-700">
           {getStatusIcon()}
        </div>
        <div>
          <h3 className="text-xl font-bold text-slate-100 capitalize">
            {job.status === 'pending' ? 'Queued' : job.status}
          </h3>
          <p className="text-slate-400 mt-1">{job.message || "Processing..."}</p>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-slate-800 rounded-full h-3 mb-2 overflow-hidden">
        <motion.div 
          className={`h-full ${getProgressColor()}`}
          initial={{ width: 0 }}
          animate={{ width: job.status === 'completed' ? '100%' : job.status === 'failed' ? '100%' : '60%' }}
          transition={{ duration: 0.5 }}
        />
      </div>
      
      {/* Time Check */}
      <div className="flex justify-between text-xs text-slate-500 font-mono mb-8">
         <span>Started: {new Date(job.created_at * 1000).toLocaleTimeString()}</span>
         {job.status === 'processing' && (
             <span className="flex items-center gap-1"><Clock size={12}/> Est. remaining: calculating...</span>
         )}
      </div>

       {job.status === 'failed' && (
           <button 
             onClick={onReset}
             className="px-6 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors border border-slate-600"
           >
             Try Again
           </button>
       )}

    </motion.div>
  );
};

export default ProcessingStatus;
