import React from 'react';
import { Sliders, Video, Scissors, Timer, Image as ImageIcon } from 'lucide-react';

const SettingsSection = ({ params, onChange }) => {
  const handleChange = (key, value) => {
    onChange({ ...params, [key]: value });
  };

  return (
    <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 shadow-xl mb-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <span className="bg-primary-600 p-1.5 rounded-lg"><Sliders size={18} /></span>
        Processing Settings
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Clip Settings */}
        <div className="space-y-6">
           <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider flex items-center gap-2">
            <Scissors size={14} /> Clips
          </h3>

          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <label className="text-slate-300">Detailed Duration (seconds)</label>
              <span className="text-primary-400 font-mono bg-primary-950/50 px-2 rounded">
                {params.duration}s
              </span>
            </div>
            <input
              type="range"
              min="5"
              max="60"
              step="1"
              value={params.duration}
              onChange={(e) => handleChange('duration', parseFloat(e.target.value))}
              className="w-full accent-primary-500 h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          <div className="space-y-3">
             <div className="flex justify-between text-sm">
              <label className="text-slate-300">Number of Clips</label>
              <span className="text-primary-400 font-mono bg-primary-950/50 px-2 rounded">
                {params.num_clips}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={params.num_clips}
              onChange={(e) => handleChange('num_clips', parseInt(e.target.value))}
              className="w-full accent-primary-500 h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>

        {/* Video Format Settings */}
         <div className="space-y-6">
           <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider flex items-center gap-2">
            <Video size={14} /> Style
          </h3>

           <div className="space-y-2">
            <label className="text-sm text-slate-300 block mb-1">Format Method</label>
            <div className="grid grid-cols-3 gap-2">
              {['crop', 'blur', 'bars'].map((method) => (
                <button
                  key={method}
                  onClick={() => handleChange('format_method', method)}
                  className={`py-2 px-3 rounded-lg border text-sm capitalize transition-all ${
                    params.format_method === method
                      ? 'bg-primary-950 border-primary-500 text-primary-400'
                      : 'bg-slate-950 border-slate-700 text-slate-400 hover:border-slate-600'
                  }`}
                >
                  {method}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
             <div className="flex items-center justify-between">
                <label className="text-sm text-slate-300">Watermark</label>
                <div 
                  onClick={() => handleChange('watermark_enabled', !params.watermark_enabled)}
                  className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${
                      params.watermark_enabled ? 'bg-primary-600' : 'bg-slate-700'
                  }`}
                >
                   <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${
                       params.watermark_enabled ? 'left-6' : 'left-1'
                   }`} />
                </div>
             </div>
             {params.watermark_enabled && (
                <input
                    type="text"
                    value={params.watermark_text}
                    onChange={(e) => handleChange('watermark_text', e.target.value)}
                    placeholder="@username"
                    className="w-full bg-slate-950 border border-slate-700 rounded-md py-2 px-3 text-sm focus:ring-1 focus:ring-primary-500 outline-none"
                />
             )}
          </div>
           
           <div className="flex items-center gap-3 p-3 bg-slate-950/50 rounded-lg border border-slate-800">
               <ImageIcon size={16} className="text-slate-400" />
               <div className="flex-1">
                   <p className="text-sm text-slate-300">Generate Thumbnails</p>
                   <p className="text-xs text-slate-500">Create JPG cover for each clip</p>
               </div>
               <div 
                  onClick={() => handleChange('generate_thumbnail', !params.generate_thumbnail)}
                  className={`w-10 h-5 rounded-full relative cursor-pointer transition-colors ${
                      params.generate_thumbnail ? 'bg-primary-600' : 'bg-slate-700'
                  }`}
                >
                   <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${
                       params.generate_thumbnail ? 'left-6' : 'left-1'
                   }`} />
                </div>
           </div>

        </div>
      </div>
    </div>
  );
};

export default SettingsSection;
