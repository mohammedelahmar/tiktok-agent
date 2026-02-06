import React from 'react';
import { Sliders, Video, Scissors, Timer, Image as ImageIcon } from 'lucide-react';
import Card from './ui/Card';
import { Label } from './ui/Label';
import Input from './ui/Input';

const SettingsSection = ({ params, onChange }) => {
  const handleChange = (key, value) => {
    onChange({ ...params, [key]: value });
  };

  const Toggle = ({ checked, onChange, label, subtext }) => (
    <div className="space-y-2">
       <div className="flex items-center justify-between">
          <label className="text-sm font-medium text-gray-300">{label}</label>
          <div 
            onClick={() => onChange(!checked)}
            className={`w-12 h-6 rounded-full relative cursor-pointer transition-all duration-300 shadow-inner ${
                checked ? 'bg-rose-600 shadow-rose-900/20' : 'bg-slate-800 shadow-black/20'
            }`}
          >
             <div className={`absolute top-1 w-4 h-4 rounded-full bg-white shadow-md transition-all duration-300 ${
                 checked ? 'left-7 shadow-rose-500/50' : 'left-1'
             }`} />
          </div>
       </div>
       {subtext && <p className="text-xs text-gray-500">{subtext}</p>}
    </div>
  );

  const RangeSlider = ({ value, min, max, step, onChange, label, displayValue }) => (
    <div className="space-y-4">
      <div className="flex justify-between text-sm">
        <label className="text-gray-300 font-medium">{label}</label>
        <span className="text-rose-400 font-mono bg-rose-500/10 px-2 py-0.5 rounded text-xs border border-rose-500/20">
          {displayValue}
        </span>
      </div>
      <div className="relative w-full h-1.5 bg-white/10 rounded-full">
         <div 
            className="absolute top-0 left-0 h-full bg-gradient-to-r from-rose-500 to-orange-500 rounded-full"
            style={{ width: `${((value - min) / (max - min)) * 100}%` }}
         />
         <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <div 
            className="absolute top-1/2 -mt-2.5 w-5 h-5 bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.5)] pointer-events-none transition-all"
            style={{ left: `calc(${((value - min) / (max - min)) * 100}% - 10px)` }}
        />
      </div>
    </div>
  );

  return (
    <Card className="p-6">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
        <span className="bg-gradient-to-br from-rose-500 to-orange-500 p-2 rounded-lg shadow-lg shadow-rose-500/20">
            <Sliders size={18} className="text-white" />
        </span>
        Processing Settings
      </h2>

      <div className="space-y-8">
        
        {/* Clip Settings */}
        <div className="space-y-6">
           <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
            <Scissors size={14} /> Clips
          </h3>

          <RangeSlider 
             label="Detailed Duration (seconds)"
             value={params.duration}
             min={5}
             max={60}
             step={1}
             onChange={(val) => handleChange('duration', val)}
             displayValue={`${params.duration}s`}
          />

          <RangeSlider 
             label="Number of Clips"
             value={params.num_clips}
             min={1}
             max={10}
             step={1}
             onChange={(val) => handleChange('num_clips', val)}
             displayValue={params.num_clips}
          />
        </div>

        <div className="h-px bg-white/5" />

        {/* Video Format Settings */}
         <div className="space-y-6">
           <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
            <Video size={14} /> Style
          </h3>

           <div className="space-y-3">
            <Label>Format Method</Label>
            <div className="grid grid-cols-3 gap-2 bg-black/40 p-1 rounded-xl border border-white/5">
              {['crop', 'blur', 'bars'].map((method) => (
                <button
                  key={method}
                  onClick={() => handleChange('format_method', method)}
                  className={`py-2 px-3 rounded-lg text-xs font-medium capitalize transition-all duration-300 ${
                    params.format_method === method
                      ? 'bg-gradient-to-r from-rose-600 to-orange-600 text-white shadow-lg shadow-rose-900/20'
                      : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {method}
                </button>
              ))}
            </div>
          </div>

          <Toggle 
            label="Watermark"
            checked={params.watermark_enabled}
            onChange={(val) => handleChange('watermark_enabled', val)}
          />

           {params.watermark_enabled && (
             <Input
                placeholder="@username"
                value={params.watermark_text}
                onChange={(e) => handleChange('watermark_text', e.target.value)}
                className="mt-2"
            />
           )}
        </div>

        <div className="h-px bg-white/5" />

        {/* AI Features */}
        <div className="space-y-6">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                <Timer size={14} /> AI Features
            </h3>
            
            <Toggle 
                label="AI Captions"
                subtext="Auto-generate subtitles (Karaoke style)"
                checked={params.captions_enabled}
                onChange={(val) => handleChange('captions_enabled', val)}
            />

             <Toggle 
                label="AI Metadata (SEO)"
                subtext="Viral Title, Description & Hashtags"
                checked={params.generate_metadata}
                onChange={(val) => handleChange('generate_metadata', val)}
            />

            <Toggle 
                label="Generate Thumbnails"
                subtext="Create JPG cover for each clip"
                checked={params.generate_thumbnail}
                onChange={(val) => handleChange('generate_thumbnail', val)}
            />
        </div>

        <div className="h-px bg-white/5" />

        {/* Review Mode */}
        <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-blue-500/10 border border-purple-500/20 backdrop-blur-md">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                    <Scissors size={16} className="text-purple-400" />
                    <span className="text-sm font-medium text-purple-200">Review Mode</span>
                </div>
                <div 
                    onClick={() => handleChange('mode', params.mode === 'analyze' ? 'process' : 'analyze')}
                    className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${
                        params.mode === 'analyze' ? 'bg-purple-600' : 'bg-slate-700'
                    }`}
                >
                    <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${
                        params.mode === 'analyze' ? 'left-6' : 'left-1'
                    }`} />
                </div>
            </div>
            <p className="text-xs text-purple-400/60">
                {params.mode === 'analyze' 
                    ? "Identify clips first, then trim manually." 
                    : "Auto-process everything immediately."}
            </p>
        </div>

      </div>
    </Card>
  );
};

export default SettingsSection;
