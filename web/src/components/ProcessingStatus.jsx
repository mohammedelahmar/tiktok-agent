import React from 'react';
import { Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';


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
    <div 
      className="bg-slate-900/80 backdrop-blur-lg rounded-xl p-8 border border-slate-700/50 shadow-2xl text-center max-w-2xl mx-auto relative overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500"
    >
      {/* Background Pulse */}
      <div className="absolute inset-0 bg-gradient-to-r from-primary-500/10 to-purple-500/10 animate-pulse pointer-events-none"></div>

      <div className="flex flex-col items-center gap-4 mb-8 relative z-10">
        <div className="relative">
           <div className="absolute inset-0 bg-primary-500 blur-xl opacity-20 rounded-full"></div>
           <div className="p-4 bg-slate-800 rounded-full border border-slate-600 relative">
              {getStatusIcon()}
           </div>
        </div>
        <div>
          <h3 className="text-2xl font-bold text-white capitalize tracking-tight">
            {job.status === 'pending' ? 'Queued' : job.status === 'processing' ? 'Processing Video' : job.status}
          </h3>
          <p className="text-slate-300 mt-2 text-lg">{job.message || "Initializing..."}</p>
        </div>
      </div>

      {/* Steps Visualization */}
      {job.status === 'processing' && (
        <div className="grid grid-cols-4 gap-2 mb-8 text-xs text-slate-500">
           {['Download', 'Analyze', 'Clip', 'Polish'].map((step, i) => (
             <div key={step} className="flex flex-col items-center gap-2">
                <div className={`w-full h-1 rounded-full ${i < 2 ? 'bg-primary-500' : 'bg-slate-800'} transition-colors duration-500`}></div>
                <span className={i < 2 ? 'text-primary-400 font-medium' : ''}>{step}</span>
             </div>
           ))}
        </div>
      )}

      {/* Progress Bar */}
      <div className="w-full bg-slate-800 rounded-full h-2 mb-4 overflow-hidden relative z-10">
        <div 
          className={`h-full ${getProgressColor()} shadow-[0_0_10px_rgba(59,130,246,0.5)] transition-all duration-500`}
          style={{ width: job.status === 'completed' ? '100%' : job.status === 'failed' ? '100%' : '60%' }}
        />
      </div>
      
      {/* Time Check */}
      <div className="flex justify-center text-sm text-slate-400 font-mono mb-6 gap-4">
         {job.created_at && (
            <span className="flex items-center gap-1.5 px-3 py-1 bg-slate-800/50 rounded-md">
                <Clock size={14} className="text-primary-400"/> 
                Started: {new Date(job.created_at * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
            </span>
         )}
      </div>

       {job.status === 'failed' && (
           <button 
             onClick={onReset}
             className="px-8 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl transition-all shadow-lg hover:shadow-red-500/25 font-semibold"
           >
             Try Again
           </button>
       )}

    </div>
  );
};

export default ProcessingStatus;
