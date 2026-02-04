import React, { useState, useRef, useEffect } from 'react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { Play, Pause, Scissors, Check, X, RotateCcw } from 'lucide-react';

const VideoTrimmer = ({ videoUrl, initialStart = 0, initialEnd = 10, onSave, onCancel }) => {
  const [playing, setPlaying] = useState(false);
  const [range, setRange] = useState([initialStart, initialEnd]);
  const videoRef = useRef(null);
  const [duration, setDuration] = useState(0);

  // Time update handler to enforce loop
  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    const currentTime = videoRef.current.currentTime;
    
    // If we go past the end, loop back to start
    if (currentTime >= range[1]) {
      videoRef.current.currentTime = range[0];
      if (playing) {
          videoRef.current.play();
      }
    }
  };

  const handleLoadedMetadata = () => {
     if(videoRef.current) {
         setDuration(videoRef.current.duration);
         // Initial seek
         videoRef.current.currentTime = range[0];
     }
  };

  const handleSliderChange = (newRange) => {
    setRange(newRange);
    if (!videoRef.current) return;

    // If start changed, seek to start
    if (newRange[0] !== range[0]) {
        videoRef.current.currentTime = newRange[0];
    }
    // If end changed, preview end cut (3s before end)
    if (newRange[1] !== range[1]) {
        videoRef.current.currentTime = Math.max(newRange[0], newRange[1] - 3);
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    
    if (playing) {
        videoRef.current.pause();
    } else {
        videoRef.current.play();
    }
    setPlaying(!playing);
  };
  
  // Sync state with video events
  const onPlay = () => setPlaying(true);
  const onPause = () => setPlaying(false);

  const previewClip = () => {
      if (!videoRef.current) return;
      videoRef.current.currentTime = range[0];
      videoRef.current.play();
      setPlaying(true);
  };

  const formatTime = (seconds) => {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-in fade-in duration-300">
      <div className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-4xl shadow-2xl flex flex-col max-h-[90vh] overflow-y-auto">
        
        {/* Header */}
        <div className="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-950/50">
          <h3 className="text-xl font-bold flex items-center gap-2 text-white">
            <Scissors size={20} className="text-primary-400" />
            Trim Clip
          </h3>
          <button onClick={onCancel} className="text-slate-400 hover:text-white transition-colors">
            <X size={24} />
          </button>
        </div>

        {/* Player Area */}
        <div className="relative aspect-video bg-black group flex justify-center items-center">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-full object-contain"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={onPlay}
              onPause={onPause}
              onClick={togglePlay}
            />
            
            {/* Overlay Play Button */}
            {!playing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/20 group-hover:bg-black/40 transition-colors cursor-pointer" onClick={togglePlay}>
                    <div className="bg-white/20 hover:bg-white/30 backdrop-blur-sm p-4 rounded-full transition-all hover:scale-110">
                        <Play size={48} className="text-white fill-white" />
                    </div>
                </div>
            )}
        </div>

        {/* Controls Area */}
        <div className="p-6 space-y-6 bg-slate-900">
            
            {/* Time Display */}
            <div className="flex justify-between text-sm font-medium text-slate-400">
                <span>Start: <span className="text-primary-400">{formatTime(range[0])}</span></span>
                <span>Duration: <span className="text-white">{formatTime(range[1] - range[0])}</span></span>
                <span>End: <span className="text-primary-400">{formatTime(range[1])}</span></span>
            </div>

            {/* Slider */}
            <div className="px-2">
                <Slider
                    range
                    min={0}
                    // Ensure max is at least the end of the clip, or defaults to 100 (or video duration)
                    max={Math.max(duration, range[1], 100)} 
                    value={range}
                    onChange={handleSliderChange}
                    step={0.1}
                    trackStyle={[{ backgroundColor: '#6366f1' }]}
                    handleStyle={[
                        { borderColor: '#818cf8', backgroundColor: '#fff', opacity: 1 },
                        { borderColor: '#818cf8', backgroundColor: '#fff', opacity: 1 }
                    ]}
                    railStyle={{ backgroundColor: '#334155' }}
                />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-between items-center pt-2">
                <div className="flex gap-2">
                     <button
                        onClick={togglePlay}
                        className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white transition-colors"
                        title={playing ? "Pause" : "Play"}
                     >
                        {playing ? <Pause size={20} /> : <Play size={20} />}
                     </button>
                     <button
                        onClick={previewClip}
                        className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-white transition-colors"
                        title="Preview Selection Loop"
                     >
                        <RotateCcw size={20} />
                     </button>
                </div>

                <div className="flex gap-3">
                    <button
                        onClick={onCancel}
                        className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-colors"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={() => onSave(range[0], range[1])}
                        className="px-6 py-2 rounded-lg bg-primary-600 hover:bg-primary-700 text-white font-semibold shadow-lg hover:shadow-primary-500/20 transition-all flex items-center gap-2"
                    >
                        <Check size={18} />
                        Confirm Trim
                    </button>
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default VideoTrimmer;
