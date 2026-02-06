import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sparkles, Zap, Github } from 'lucide-react';
import InputSection from './components/InputSection';
import SettingsSection from './components/SettingsSection';
import ProcessingStatus from './components/ProcessingStatus';
import ResultsGrid from './components/ResultsGrid';
import ReviewStage from './components/ReviewStage';

const API_BASE = '/api';

function App() {
  // Application State
  const [stage, setStage] = useState('input'); // input, processing, results
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // Processing Parameters
  const [params, setParams] = useState({
    source_type: 'youtube', // or 'file'
    source_path: '',
    num_clips: 1,
    duration: 15,
    min_gap: 1.0,
    format_method: 'crop',
    watermark_enabled: false,
    watermark_text: '',
    generate_thumbnail: true,
    captions_enabled: false,
    generate_metadata: false,

    face_detection: 'mediapipe',
    mode: 'analyze' // 'process' or 'analyze'
  });

  // Poll for job status
  useEffect(() => {
    let interval;
    if (stage === 'processing' && jobId) {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE}/status/${jobId}`);
          setJobStatus(response.data);

          if (response.data.status === 'completed') {
            setResults(response.data.result);
            setStage('results');
            clearInterval(interval);
          } else if (response.data.status === 'analyzed') {
            // New state for manual review
            // Fix: Extract the inner result payload, not the whole job status
            // Explicitly construct object to avoid any spread issues
            const resultData = response.data.result || {};
            setResults({ 
              candidates: resultData.candidates || [],
              video_filename: resultData.video_filename,
              mode: resultData.mode,
              success: resultData.success,
              job_id_ref: jobId,
              config: response.data.config 
            }); 
            setStage('review');
            clearInterval(interval);
          } else if (response.data.status === 'failed') {
            setError(response.data.message);
            // Don't clear interval immediately, maybe wait or show error state
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Polling error", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [stage, jobId]);

  const handleStartProcessing = async () => {
    if (!params.source_path) {
      alert("Please provide a video Source!");
      return;
    }

    try {
      setJobStatus({ status: 'preparing', message: 'Connecting to server...' });
      setStage('processing');
      setError(null);
      
      let endpoint = params.source_type === 'youtube' ? '/process/youtube' : '/process/file';
      let payload = {};

      if (params.source_type === 'youtube') {
         // Query params for YouTube
         endpoint += `?url=${encodeURIComponent(params.source_path)}`;
         endpoint += `&num_clips=${params.num_clips}`;
         endpoint += `&duration=${params.duration}`;
         endpoint += `&format_method=${params.format_method}`;
         if (params.watermark_enabled) endpoint += `&watermark=${encodeURIComponent(params.watermark_text)}`;
         if (params.captions_enabled) endpoint += `&captions=true`;
         if (params.generate_metadata) endpoint += `&generate_metadata=true`;
         if (params.mode === 'analyze') endpoint += `&mode=analyze`;
      } else {
         // Query params for File
         endpoint += `?filename=${encodeURIComponent(params.source_path)}`;
         endpoint += `&num_clips=${params.num_clips}`;
         endpoint += `&duration=${params.duration}`;
         endpoint += `&format_method=${params.format_method}`;
         if (params.watermark_enabled) endpoint += `&watermark=${encodeURIComponent(params.watermark_text)}`;
         if (params.captions_enabled) endpoint += `&captions=true`;
         if (params.generate_metadata) endpoint += `&generate_metadata=true`;
         if (params.mode === 'analyze') endpoint += `&mode=analyze`;
      }

      const response = await axios.post(`${API_BASE}${endpoint}`);
      setJobId(response.data.job_id);
      setJobStatus({ status: 'pending', message: 'Job started...' });

    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || err.message);
      setStage('input');
    }
  };

  const resetApp = () => {
    setStage('input');
    setJobId(null);
    setJobStatus(null);
    setResults(null);
    setError(null);
    setParams(prev => ({ ...prev, source_path: '' })); // Keep settings, clear input
  };
  
  const handleRenderStart = (newJobId) => {
      setJobId(newJobId);
      setStage('processing');
      setJobStatus({ status: 'pending', message: 'Starting render...' });
      // Polling will restart due to stage change + jobId presence
  };

  return (
    <div className="relative min-h-screen font-sans selection:bg-rose-500/30 overflow-x-hidden">
      
      {/* Ambient Atmosphere (Orbs) */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
         {/* Top-Left Rose Orb */}
         <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-rose-600/20 rounded-full blur-[120px] animate-pulse-slow" />
         {/* Bottom-Right Cyan Orb */}
         <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-cyan-600/20 rounded-full blur-[120px] animate-pulse-slow animation-delay-2000" />
         {/* Center/Random Purple Orb for extra depth */}
         <div className="absolute top-[30%] left-[60%] w-[300px] h-[300px] bg-purple-600/10 rounded-full blur-[100px] animate-pulse-slow animation-delay-4000" />
      </div>

      {/* Header */}
      <header className="relative z-50 border-b border-white/5 bg-black/20 backdrop-blur-md sticky top-0">
        <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-rose-500 to-orange-500 p-2 rounded-xl shadow-lg shadow-rose-500/20">
              <Sparkles size={20} className="text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              TikTok Agent <span className="text-xs font-mono text-cyan-400 bg-cyan-900/30 px-1.5 py-0.5 rounded ml-2">BETA</span>
            </h1>
          </div>
          <div className="flex items-center gap-4">
             <a href="#" className="p-2 hover:bg-white/5 rounded-full transition-colors text-gray-400 hover:text-white">
               <Github size={20} />
             </a>
          </div>
        </div>
      </header>

      <main className="relative z-10 max-w-[1440px] mx-auto px-6 py-8">
        
        {/* Error Banner */}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-red-200 flex items-center gap-3 backdrop-blur-md">
             <Zap size={20} />
             <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-12 gap-8">
          
          {/* Main Stage (8 Cols) */}
          <div className="col-span-12 lg:col-span-8 space-y-8">
            {stage === 'input' && (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 space-y-8">
                <div className="mb-6">
                  <h2 className="text-4xl font-bold mb-2 tracking-tight text-white">Create Viral Clips</h2>
                  <p className="text-gray-400 text-lg">AI-powered video extraction and formatting for TikTok & Shorts</p>
                </div>

                <InputSection 
                  onInputParamsChange={(changes) => setParams(prev => ({ ...prev, ...changes }))} 
                />
                
                {/* Mobile Settings (only show below main on mobile, traditionally sidebar on desktop) */}
                <div className="lg:hidden">
                   <SettingsSection params={params} onChange={setParams} />
                </div>

                <div className="flex justify-start pt-4">
                  <button
                    onClick={handleStartProcessing}
                    className="group relative inline-flex items-center justify-center px-10 py-5 font-bold text-white transition-all duration-300 bg-gradient-to-r from-rose-600 to-orange-600 rounded-2xl shadow-xl shadow-rose-500/25 hover:shadow-rose-500/40 hover:-translate-y-1 active:scale-95"
                  >
                    <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]"></span>
                    <span className="relative flex items-center gap-3 text-lg">
                      <Zap size={22} className="text-white group-hover:text-yellow-200 transition-colors" />
                      Start Processing
                    </span>
                  </button>
                </div>
              </div>
            )}

            {/* Processing Stage */}
            {stage === 'processing' && (
               <ProcessingStatus job={jobStatus} onReset={resetApp} />
            )}

            {/* Results Stage */}
            {stage === 'results' && (
               <ResultsGrid results={results} onReset={resetApp} />
            )}
            
            {/* Review Stage */}
            {stage === 'review' && (
               <ReviewStage job={results} onRenderStart={handleRenderStart} />
            )}
          </div>

          {/* Sidebar (4 Cols) */}
          <div className="hidden lg:block col-span-4 space-y-6">
            {/* Stats / Info Card could go here */}
            {stage === 'input' && (
              <SettingsSection 
                params={params} 
                onChange={setParams} 
              />
            )}
            
            {/* Example Stats Card */}
            <div className="p-6 rounded-2xl bg-black/40 border border-white/5 backdrop-blur-xl">
               <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Quick Stats</h3>
               <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Credits Remaining</span>
                    <span className="text-cyan-400 font-mono">∞</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Model</span>
                    <span className="text-rose-400 font-mono">Gemini 2.0</span>
                  </div>
               </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}

export default App;
