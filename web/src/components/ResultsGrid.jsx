import React from 'react';
import { Download, Play, Share2 } from 'lucide-react';
import { motion } from 'framer-motion';

const ResultsGrid = ({ results, onReset }) => {
  if (!results || !results.clips || results.clips.length === 0) return null;

  return (
    <div className="space-y-6">
       <div className="flex items-center justify-between mb-6">
         <h2 className="text-2xl font-bold text-slate-100">
           Generated Clips <span className="text-slate-500 text-base font-normal">({results.clips.length})</span>
         </h2>
         <button 
           onClick={onReset}
           className="text-primary-400 hover:text-primary-300 text-sm font-medium"
         >
           Start New Project
         </button>
       </div>

       <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
         {results.clips.map((clip, index) => (
           <motion.div 
             key={index}
             initial={{ opacity: 0, scale: 0.9 }}
             animate={{ opacity: 1, scale: 1 }}
             transition={{ delay: index * 0.1 }}
             className="bg-slate-900 rounded-xl overflow-hidden border border-slate-800 group hover:border-primary-500/50 transition-colors"
           >
             <div className="aspect-[9/16] bg-black relative">
               {/* Video/Thumbnail Placeholder */}
               <video 
                  controls 
                  className="w-full h-full object-cover"
                  poster={clip.thumbnail ? `http://localhost:8000/files/${clip.thumbnail}` : null}
                  src={`http://localhost:8000/files/${clip.filename}`}
               />
               
               {/* Score Badge */}
               <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md px-2 py-1 rounded text-xs font-mono text-green-400 border border-green-500/30">
                 Score: {clip.score.toFixed(2)}
               </div>
             </div>

             <div className="p-4 space-y-3">
               <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium text-slate-200">Clip #{index + 1}</h4>
                    <p className="text-xs text-slate-500 font-mono">
                      {clip.start.toFixed(1)}s - {clip.end.toFixed(1)}s
                    </p>
                  </div>
               </div>

               <div className="grid grid-cols-2 gap-2 pt-2">
                 <a 
                   href={`http://localhost:8000/files/${clip.filename}`} 
                   download
                   className="flex items-center justify-center gap-2 py-2 px-3 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 transition-colors"
                 >
                   <Download size={16} /> Download
                 </a>
                 <button className="flex items-center justify-center gap-2 py-2 px-3 bg-primary-600/10 hover:bg-primary-600/20 text-primary-400 rounded-lg text-sm transition-colors border border-primary-500/20">
                   <Share2 size={16} /> Share
                 </button>
               </div>
             </div>
           </motion.div>
         ))}
       </div>
    </div>
  );
};

export default ResultsGrid;
