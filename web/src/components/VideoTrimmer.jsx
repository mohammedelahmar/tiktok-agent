import React, { useState, useRef, useEffect } from 'react';
import Slider from 'rc-slider';
import 'rc-slider/assets/index.css';
import { Play, Pause, Scissors, Check, X, RotateCcw } from 'lucide-react';
import Button from './ui/Button';

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
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/90 backdrop-blur-xl p-4 animate-in fade-in duration-300">
      <div className="bg-[#0A0A0A] border border-white/10 rounded-2xl w-full max-w-5xl shadow-2xl flex flex-col max-h-[95vh] overflow-hidden">
        
        {/* Header */}
        <div className="p-5 border-b border-white/5 flex justify-between items-center bg-black/40">
          <h3 className="text-xl font-bold flex items-center gap-3 text-white">
            <span className="p-1.5 bg-rose-500/20 rounded-lg">
                <Scissors size={20} className="text-rose-400" />
            </span>
            Trim Clip
          </h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-white transition-colors bg-white/5 hover:bg-white/10 p-2 rounded-full">
            <X size={20} />
          </button>
        </div>

        {/* Player Area - Darkened for focus */}
        <div className="relative flex-1 bg-black flex justify-center items-center overflow-hidden">
            <video
              ref={videoRef}
              src={videoUrl}
              className="w-full h-full max-h-[60vh] object-contain"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={onPlay}
              onPause={onPause}
              onClick={togglePlay}
            />
            
            {/* Overlay Play Button */}
            {!playing && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 hover:bg-black/50 transition-colors cursor-pointer" onClick={togglePlay}>
                    <div className="bg-white/10 hover:bg-rose-500/80 backdrop-blur-md p-6 rounded-full transition-all hover:scale-110 group shadow-xl">
                        <Play size={48} className="text-white fill-white ml-1 group-hover:scale-105 transition-transform" />
                    </div>
                </div>
            )}
        </div>

        {/* Controls Area */}
        <div className="p-6 space-y-6 bg-[#0F0F0F] border-t border-white/5">
            
            {/* Time Display */}
            <div className="flex justify-between text-sm font-medium text-gray-400 font-mono">
                <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider">Start</span>
                    <span className="text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">{formatTime(range[0])}</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider">Duration</span>
                    <span className="text-white">{formatTime(range[1] - range[0])}</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs uppercase tracking-wider">End</span>
                    <span className="text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20">{formatTime(range[1])}</span>
                </div>
            </div>

            {/* Slider */}
            <div className="px-2 py-2">
                <Slider
                    range
                    min={0}
                    max={Math.max(duration, range[1], 100)} 
                    value={range}
                    onChange={handleSliderChange}
                    step={0.1}
                    trackStyle={[{ backgroundColor: '#f43f5e', height: 6 }]} // Rose-500
                    railStyle={{ backgroundColor: '#333', height: 6 }}
                    handleStyle={[
                        { borderColor: '#f43f5e', backgroundColor: '#fff', opacity: 1, boxShadow: '0 0 10px rgba(244, 63, 94, 0.5)', width: 20, height: 20, marginTop: -7 },
                        { borderColor: '#f43f5e', backgroundColor: '#fff', opacity: 1, boxShadow: '0 0 10px rgba(244, 63, 94, 0.5)', width: 20, height: 20, marginTop: -7 }
                    ]}
                />
            </div>

            {/* Action Buttons */}
            <div className="flex justify-between items-center pt-2">
                <div className="flex gap-2">
                     <button
                        onClick={togglePlay}
                        className="p-3 rounded-xl bg-white/5 hover:bg-white/10 text-white transition-colors border border-white/5"
                        title={playing ? "Pause" : "Play"}
                     >
                        {playing ? <Pause size={20} /> : <Play size={20} />}
                     </button>
                     <button
                        onClick={previewClip}
                        className="p-3 rounded-xl bg-white/5 hover:bg-white/10 text-white transition-colors border border-white/5"
                        title="Preview Selection Loop"
                     >
                        <RotateCcw size={20} />
                     </button>
                </div>

                <div className="flex gap-3">
                    <Button
                        variant="ghost"
                        onClick={onCancel}
                    >
                        Cancel
                    </Button>
                    <Button
                        variant="primary"
                        onClick={() => onSave(range[0], range[1])}
                        className="shadow-lg shadow-rose-500/20"
                    >
                        <Check size={18} className="mr-2" />
                        Confirm Trim
                    </Button>
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default VideoTrimmer;
