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
             initial={{ opacity: 0, scale: 0.9, y: 20 }}
             animate={{ opacity: 1, scale: 1, y: 0 }}
             whileHover={{ y: -5, transition: { duration: 0.2 } }}
             transition={{ delay: index * 0.1 }}
             className="bg-slate-900/50 backdrop-blur-sm rounded-xl overflow-hidden border border-slate-800 group hover:border-primary-500/50 hover:shadow-xl hover:shadow-primary-500/10 transition-all duration-300"
           >
             <div className="aspect-[9/16] bg-black relative group-hover:ring-1 ring-white/10 transition-all">
               {/* Video/Thumbnail Placeholder */}
               <video 
                  controls 
                  className="w-full h-full object-cover"
                  poster={clip.thumbnail ? `http://localhost:8000/files/${clip.thumbnail}` : null}
                  src={`http://localhost:8000/files/${clip.filename}`}
               />
               
               {/* Score Badge */}
               <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md px-2 py-1 rounded-md text-xs font-bold font-mono text-green-400 border border-green-500/30 flex items-center gap-1 shadow-lg">
                 <span>★</span> {clip.score.toFixed(2)}
               </div>
             </div>

             <div className="p-5 space-y-4">
               <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-bold text-slate-100 text-lg group-hover:text-primary-400 transition-colors">Clip #{index + 1}</h4>
                    <p className="text-xs text-slate-500 font-mono flex items-center gap-2 mt-1">
                      <span className="w-2 h-2 rounded-full bg-primary-500/50"></span>
                      {clip.start.toFixed(1)}s - {clip.end.toFixed(1)}s
                    </p>
                  </div>
               </div>

               <div className="grid grid-cols-2 gap-3 pt-2">
                 <a 
                   href={`http://localhost:8000/files/${clip.filename}`} 
                   download
                   className="flex items-center justify-center gap-2 py-2.5 px-3 bg-white text-slate-900 hover:bg-slate-200 rounded-lg text-sm font-semibold transition-all shadow-lg shadow-white/5 hover:scale-[1.02] active:scale-[0.98]"
                 >
                   <Download size={16} /> Download
                 </a>
                 <button className="flex items-center justify-center gap-2 py-2.5 px-3 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-sm font-medium transition-all hover:text-white border border-slate-700 hover:border-slate-600">
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
