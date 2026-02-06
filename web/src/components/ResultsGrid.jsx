import React from 'react';
import { Download, Play, Share2 } from 'lucide-react';
import Card from './ui/Card';
import Button from './ui/Button';

const ResultsGrid = ({ results, onReset }) => {
  if (!results || !results.clips || results.clips.length === 0) return null;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
       <div className="flex items-center justify-between mb-6">
         <h2 className="text-2xl font-bold text-white flex items-center gap-2">
           Generated Clips 
           <span className="text-gray-500 text-lg font-normal">({results.clips.length})</span>
         </h2>
         <Button 
           variant="ghost"
           size="sm"
           onClick={onReset}
         >
           Start New Project
         </Button>
       </div>

       <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
         {results.clips.map((clip, index) => (
           <Card 
             key={index}
             className="overflow-hidden group hover:border-rose-500/30 transition-all duration-300"
           >
             {/* Neon Border Glow on Hover */}
             <div className="absolute inset-0 rounded-2xl ring-1 ring-inset ring-transparent group-hover:ring-rose-500/30 transition-all pointer-events-none" />

             <div className="aspect-[9/16] bg-black relative group-hover:ring-1 ring-white/10 transition-all">
               {/* Video/Thumbnail Placeholder */}
               <video 
                  controls 
                  className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
                  poster={clip.thumbnail ? `http://localhost:8000/files/${clip.thumbnail}` : null}
                  src={`http://localhost:8000/files/${clip.filename}`}
               />
               
               {/* Score Badge */}
               <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-lg text-xs font-bold font-mono text-cyan-400 border border-cyan-500/30 flex items-center gap-1 shadow-lg">
                 <span className="text-cyan-400 drop-shadow-[0_0_5px_rgba(34,211,238,0.8)]">★</span> {clip.score.toFixed(2)}
               </div>
             </div>

             <div className="p-5 space-y-4 bg-gradient-to-b from-transparent to-black/40">
               <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-bold text-white text-lg group-hover:text-rose-400 transition-colors shadow-black drop-shadow-md">Clip #{index + 1}</h4>
                    <p className="text-xs text-gray-500 font-mono flex items-center gap-2 mt-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>
                      {clip.start.toFixed(1)}s - {clip.end.toFixed(1)}s
                    </p>
                  </div>
               </div>

               <div className="grid grid-cols-2 gap-3 pt-2">
                 <a 
                   href={`http://localhost:8000/files/${clip.filename}`} 
                   download
                   className="flex items-center justify-center gap-2 py-2.5 px-3 bg-white text-black hover:bg-gray-200 rounded-xl text-sm font-bold transition-all shadow-lg shadow-white/10 hover:scale-[1.02] active:scale-[0.98]"
                 >
                   <Download size={16} /> Download
                 </a>
                 <button className="flex items-center justify-center gap-2 py-2.5 px-3 bg-white/5 hover:bg-white/10 text-gray-300 rounded-xl text-sm font-medium transition-all hover:text-white border border-white/10 hover:border-white/20">
                   <Share2 size={16} /> Share
                 </button>
               </div>
               
               {clip.metadata && (
                 <div className={`mt-4 p-4 rounded-xl border text-sm backdrop-blur-md ${
                    clip.metadata.error 
                      ? 'bg-red-950/20 border-red-500/20 text-red-200' 
                      : 'bg-black/40 border-white/5'
                 }`}>
                    {clip.metadata.error ? (
                        <div className="flex items-start gap-2">
                             <span className="text-xl">⚠️</span>
                             <div>
                                 <h5 className="font-semibold mb-1">Metadata Failed</h5>
                                 <p className="text-xs opacity-70">{clip.metadata.error}</p>
                             </div>
                        </div>
                    ) : (
                        <div className="space-y-2">
                            <h5 className="font-semibold text-rose-400 leading-tight">{clip.metadata.title}</h5>
                            <p className="text-gray-400 text-xs line-clamp-2">{clip.metadata.description}</p>
                            <div className="flex flex-wrap gap-1.5 pt-1">
                            {clip.metadata.hashtags && clip.metadata.hashtags.map((tag, i) => (
                                <span key={i} className="text-cyan-400 text-[10px] bg-cyan-950/30 px-2 py-0.5 rounded border border-cyan-500/20">
                                {tag}
                                </span>
                            ))}
                            </div>
                        </div>
                    )}
                 </div>
               )}
             </div>
           </Card>
         ))}
       </div>
    </div>
  );
};


export default ResultsGrid;
