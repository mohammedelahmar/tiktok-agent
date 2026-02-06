import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Clock, CheckCircle2, AlertCircle, Loader2, Play, FileVideo } from 'lucide-react';

const API_BASE = '/api';

export default function HistorySection({ onLoadJob }) {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
    // Poll for updates every 5s while looking at history
    const interval = setInterval(fetchHistory, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await axios.get(`${API_BASE}/history`);
      setJobs(res.data);
    } catch (err) {
      console.error("Failed to fetch history", err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle2 className="text-green-400" size={18} />;
      case 'analyzed': return <CheckCircle2 className="text-cyan-400" size={18} />;
      case 'processing': return <Loader2 className="text-yellow-400 animate-spin" size={18} />;
      case 'failed': return <AlertCircle className="text-red-400" size={18} />;
      default: return <Clock className="text-gray-400" size={18} />;
    }
  };

  const formatDate = (ts) => {
    return new Date(ts * 1000).toLocaleString();
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-3 mb-6">
        <Clock className="text-cyan-400" size={24} />
        <h2 className="text-2xl font-bold text-white">History</h2>
      </div>

      <div className="grid gap-4">
        {loading && jobs.length === 0 ? (
          <div className="text-center py-12 text-gray-500">Loading history...</div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 text-gray-500 bg-white/5 rounded-xl border border-white/5">
            No history found. Start processing videos!
          </div>
        ) : (
          jobs.map((job) => (
            <div 
              key={job.job_id}
              className="group relative overflow-hidden bg-black/40 border border-white/5 rounded-xl p-4 transition-all hover:bg-white/5 hover:border-white/10"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div className={`p-3 rounded-lg ${
                    job.status === 'completed' ? 'bg-green-500/10' :
                    job.status === 'failed' ? 'bg-red-500/10' :
                    'bg-cyan-500/10'
                  }`}>
                    {getStatusIcon(job.status)}
                  </div>
                  
                  <div>
                    <h3 className="font-medium text-white flex items-center gap-2">
                      {job.config?.source_type === 'youtube' ? 'YouTube Video' : 'Local File'}
                      <span className="text-xs text-gray-500 px-2 py-0.5 bg-white/5 rounded-full">
                        {job.status.toUpperCase()}
                      </span>
                    </h3>
                    <p className="text-sm text-gray-400 mt-1 line-clamp-1 max-w-md">
                      {job.config?.source_path}
                    </p>
                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                      <span>{formatDate(job.created_at)}</span>
                      <span>•</span>
                      <span>{job.job_id.slice(0, 8)}...</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {(job.status === 'completed' || job.status === 'analyzed') && (
                     <button 
                        onClick={() => onLoadJob(job)}
                        className="px-4 py-2 text-sm font-medium text-cyan-400 bg-cyan-950/30 rounded-lg hover:bg-cyan-900/50 transition-colors flex items-center gap-2"
                     >
                        <Play size={14} />
                        View
                     </button>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
