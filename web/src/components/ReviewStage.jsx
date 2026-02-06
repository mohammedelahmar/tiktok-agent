import React, { useState } from 'react';
import { Play, Clock, Star, Edit3, ArrowRight, Video } from 'lucide-react';
import VideoTrimmer from './VideoTrimmer';
import axios from 'axios';
import Card from './ui/Card';

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
        captions_enabled: job.config.captions_enabled,
        generate_metadata: job.config.generate_metadata
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
        <h2 className="text-3xl font-bold mb-2 tracking-tight text-white">Review Candidates</h2>
        <p className="text-gray-400">AI found <span className="text-rose-400 font-bold">{clips.length}</span> viral moments. Adjust them below before rendering.</p>
      </div>

      {/* Clip List */}
      <div className="grid gap-4 max-w-3xl mx-auto">
        {clips.map((clip) => (
          <Card key={clip.id} className="p-4 flex items-center justify-between group hover:border-rose-500/30 transition-all duration-300">
            
            <div className="flex items-center gap-5">
              <div className="w-14 h-14 rounded-xl bg-black/50 flex items-center justify-center border border-white/5 group-hover:border-rose-500/20 group-hover:bg-rose-500/10 transition-all">
                 <Video size={24} className="text-gray-500 group-hover:text-rose-400 transition-colors" />
              </div>
              
              <div>
                <div className="flex items-center gap-3 mb-1">
                   <h4 className="font-bold text-white text-lg">Clip #{clip.id + 1}</h4>
                   <span className="text-xs bg-cyan-950/30 text-cyan-400 px-2 py-0.5 rounded-full flex items-center gap-1 border border-cyan-500/20 shadow-lg shadow-cyan-500/10">
                      <Star size={10} className="fill-cyan-400" /> {clip.score.toFixed(2)}
                   </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-400 font-mono">
                   <span className="flex items-center gap-1.5"><Clock size={14} className="text-gray-500" /> {formatTime(clip.start)} - {formatTime(clip.end)}</span>
                   <span className="text-gray-600">|</span>
                   <span>Duration: <span className="text-gray-300">{(clip.end - clip.start).toFixed(1)}s</span></span>
                </div>
              </div>
            </div>

            <button 
              onClick={() => handleEdit(clip)}
              className="px-5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white transition-all flex items-center gap-2 border border-white/5 hover:border-white/20 shadow-lg hover:shadow-xl active:scale-95"
            >
              <Edit3 size={16} />
              Adjust
            </button>
          </Card>
        ))}
      </div>

      {/* Action Footer */}
      <div className="flex justify-center pt-8 border-t border-white/5">
          <button
            onClick={handleRenderAll}
            disabled={isSubmitting}
            className="group relative inline-flex items-center justify-center px-10 py-5 font-bold text-white transition-all duration-300 bg-gradient-to-r from-emerald-600 to-teal-600 rounded-full hover:bg-emerald-500 hover:scale-105 active:scale-95 disabled:opacity-50 disabled:pointer-events-none shadow-xl shadow-emerald-500/20"
          >
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]"></span>
            <span className="relative flex items-center gap-3 text-lg">
              {isSubmitting ? 'Starting Render...' : 'Render All Clips'}
              <ArrowRight size={22} className="group-hover:translate-x-1 transition-transform" />
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
