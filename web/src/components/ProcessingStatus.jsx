import React from 'react';
import { Loader2, CheckCircle, AlertCircle, Clock } from 'lucide-react';
import Card from './ui/Card';
import Button from './ui/Button';

const ProcessingStatus = ({ job, onReset }) => {
  if (!job) return null;

  const getStatusIcon = () => {
    switch (job.status) {
      case 'completed': return <CheckCircle className="text-emerald-400" size={32} />;
      case 'failed': return <AlertCircle className="text-red-500" size={32} />;
      default: return <Loader2 className="animate-spin text-rose-500" size={32} />;
    }
  };

  const getProgressColor = () => {
     switch (job.status) {
      case 'completed': return 'bg-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]';
      case 'failed': return 'bg-red-500 shadow-[0_0_15px_rgba(239,68,68,0.5)]';
      default: return 'bg-gradient-to-r from-rose-500 to-orange-500 shadow-[0_0_15px_rgba(244,63,94,0.5)]';
    }
  };

  return (
    <Card 
      className="p-8 text-center max-w-3xl mx-auto overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-500 border-rose-500/10"
    >
      {/* Background Pulse */}
      <div className="absolute inset-0 bg-gradient-to-r from-rose-500/5 to-orange-500/5 animate-pulse-slow pointer-events-none" />

      <div className="flex flex-col items-center gap-6 mb-10 relative z-10">
        <div className="relative">
           <div className={`absolute inset-0 blur-2xl opacity-40 rounded-full ${job.status === 'failed' ? 'bg-red-500' : 'bg-rose-500'}`}></div>
           <div className="p-6 bg-black/80 rounded-full border border-white/10 relative shadow-2xl">
              {getStatusIcon()}
           </div>
        </div>
        <div>
          <h3 className="text-3xl font-bold text-white capitalize tracking-tight drop-shadow-lg">
            {job.status === 'pending' ? 'Queued' : job.status === 'processing' ? 'Processing Video' : job.status}
          </h3>
          <p className="text-gray-400 mt-2 text-lg">{job.message || "Initializing..."}</p>
        </div>
      </div>

      {/* Steps Visualization */}
      {job.status === 'processing' && (
        <div className="grid grid-cols-4 gap-4 mb-10 text-xs text-gray-500 relative z-10 font-mono">
           {['Download', 'Analyze', 'Clip', 'Polish'].map((step, i) => (
             <div key={step} className="flex flex-col items-center gap-3">
                <div className={`w-full h-1.5 rounded-full overflow-hidden bg-white/5`}>
                     <div className={`h-full w-full ${i < 2 ? 'bg-rose-500' : 'translate-x-[-100%]'} transition-transform duration-1000 ease-in-out`} />
                </div>
                <span className={i < 2 ? 'text-rose-400 font-bold uppercase tracking-wider' : 'uppercase tracking-wider'}>{step}</span>
             </div>
           ))}
        </div>
      )}

      {/* Progress Bar */}
      <div className="w-full bg-black/50 rounded-full h-3 mb-6 overflow-hidden relative z-10 border border-white/5">
        <div 
          className={`h-full ${getProgressColor()} transition-all duration-1000 ease-[cubic-bezier(0.4,0,0.2,1)]`}
          style={{ width: job.status === 'completed' ? '100%' : job.status === 'failed' ? '100%' : '60%' }}
        >
             <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]" />
        </div>
      </div>
      
      {/* Time Check */}
      <div className="flex justify-center text-sm text-gray-500 font-mono mb-8 gap-4 relative z-10">
         {job.created_at && (
            <span className="flex items-center gap-2 px-4 py-1.5 bg-black/40 rounded-lg border border-white/5">
                <Clock size={14} className="text-rose-400"/> 
                Started: <span className="text-gray-300">{new Date(job.created_at * 1000).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
            </span>
         )}
      </div>

       {job.status === 'failed' && (
           <Button 
             variant="danger"
             onClick={onReset}
             size="lg"
             className="relative z-10"
           >
             Try Again
           </Button>
       )}

    </Card>
  );
};

export default ProcessingStatus;
