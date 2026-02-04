import React, { useState } from 'react';
import { Play, Clock, Star, Edit3, ArrowRight, Video } from 'lucide-react';
import VideoTrimmer from './VideoTrimmer';
import axios from 'axios';

const API_BASE = '/api';

const ReviewStage = ({ job, onRenderStart }) => {
  // job is now the result object directly: { candidates: [], video_filename: ... }
  const [clips, setClips] = useState(job.candidates || []);
  const [editingClip, setEditingClip] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Sync clips when job changes
  React.useEffect(() => {
    if (job.candidates) {
      setClips(job.candidates);
    }
  }, [job]);

  // Debug removed

  const videoFilename = job.video_filename;
  // Use absolute URL to avoid proxy issues if frontend wasn't restarted or proxy is flaky
  const videoUrl = `http://localhost:8000/inputs/${videoFilename}`;

  const handleEdit = (clip) => {
    setEditingClip(clip);
  };

  const handleSaveTrim = (start, end) => {
    setClips(prev => prev.map(c => 
      c.id === editingClip.id ? { ...c, start, end } : c
    ));
    setEditingClip(null);
  };

  const formatTime = (seconds) => {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
  };

  const handleRenderAll = async () => {
    try {
      setIsSubmitting(true);
      // Call render endpoint
      // The job object we receive is the "result" part of the status.
      // It doesn't contain the job_id, so we need to pass it as a separate prop
      const payload = {
        job_id: job.job_id_ref, // We need to update App.jsx to pass this
        clips: clips,
        format_method: job.config.format_method, // Inherit from original config
        watermark_enabled: job.config.watermark_enabled,
        watermark_text: job.config.watermark_text,
        generate_thumbnail: job.config.generate_thumbnail,
        captions_enabled: job.config.captions_enabled
      };

      const response = await axios.post(`${API_BASE}/render`, payload);
      onRenderStart(response.data.job_id); // This might be same ID or new ID

    } catch (err) {
      console.error("Render error", err);
      alert("Failed to start render: " + err.message);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-2 tracking-tight">Review Candidates</h2>
        <p className="text-slate-400">AI found {clips.length} viral moments. Adjust them below before rendering.</p>
      </div>

      {/* Clip List */}
      <div className="grid gap-4 max-w-3xl mx-auto">
        {clips.map((clip) => (
          <div key={clip.id} className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center justify-between hover:border-slate-700 transition-colors group">
            
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-slate-800 flex items-center justify-center text-slate-500Group-hover:text-primary-400">
                 <Video size={24} className="text-slate-500 group-hover:text-primary-400 transition-colors" />
              </div>
              
              <div>
                <div className="flex items-center gap-3 mb-1">
                   <h4 className="font-semibold text-white">Clip #{clip.id + 1}</h4>
                   <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full flex items-center gap-1">
                      <Star size={10} /> {clip.score.toFixed(2)}
                   </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-slate-400">
                   <span className="flex items-center gap-1"><Clock size={14} /> {formatTime(clip.start)} - {formatTime(clip.end)}</span>
                   <span>Duration: {(clip.end - clip.start).toFixed(1)}s</span>
                </div>
              </div>
            </div>

            <button 
              onClick={() => handleEdit(clip)}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white transition-all flex items-center gap-2 border border-transparent hover:border-slate-600"
            >
              <Edit3 size={16} />
              Adjust
            </button>
          </div>
        ))}
      </div>

      {/* Action Footer */}
      <div className="flex justify-center pt-8 border-t border-slate-900/50">
          <button
            onClick={handleRenderAll}
            disabled={isSubmitting}
            className="group relative inline-flex items-center justify-center px-8 py-4 font-semibold text-white transition-all duration-200 bg-emerald-600 rounded-full hover:bg-emerald-500 hover:scale-105 active:scale-95 disabled:opacity-50 disabled:pointer-events-none shadow-lg shadow-emerald-900/20"
          >
            <span className="relative flex items-center gap-2 text-lg">
              {isSubmitting ? 'Starting Render...' : 'Render All Clips'}
              <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
            </span>
          </button>
      </div>

      {/* Trimmer Modal */}
      {editingClip && (
        <VideoTrimmer
          key={editingClip.id}
          videoUrl={videoUrl}
          initialStart={editingClip.start}
          initialEnd={editingClip.end}
          onSave={handleSaveTrim}
          onCancel={() => setEditingClip(null)}
        />
      )}

    </div>
  );
};

export default ReviewStage;
