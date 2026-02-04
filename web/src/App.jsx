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
         if (params.mode === 'analyze') endpoint += `&mode=analyze`;
      } else {
         // Query params for File
         endpoint += `?filename=${encodeURIComponent(params.source_path)}`;
         endpoint += `&num_clips=${params.num_clips}`;
         endpoint += `&duration=${params.duration}`;
         endpoint += `&format_method=${params.format_method}`;
         if (params.watermark_enabled) endpoint += `&watermark=${encodeURIComponent(params.watermark_text)}`;
         if (params.captions_enabled) endpoint += `&captions=true`;
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
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-primary-500/30">
      
      {/* Header */}
      <header className="border-b border-slate-900 bg-slate-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-gradient-to-br from-primary-500 to-primary-700 p-2 rounded-lg">
              <Sparkles size={20} className="text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              TikTok Agent
            </h1>
          </div>
          <div className="flex items-center gap-4">
             <a href="#" className="p-2 hover:bg-slate-900 rounded-full transition-colors text-slate-400 hover:text-white">
               <Github size={20} />
             </a>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        
        {/* Error Banner */}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-200 flex items-center gap-3">
             <Zap size={20} />
             <p>{error}</p>
          </div>
        )}

        {/* Input Stage */}
        {stage === 'input' && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="text-center mb-10">
              <h2 className="text-4xl font-bold mb-3 tracking-tight">Create Viral Clips</h2>
              <p className="text-slate-400 text-lg">AI-powered video extraction and formatting for TikTok & Shorts</p>
            </div>

            <InputSection 
              onInputParamsChange={(changes) => setParams(prev => ({ ...prev, ...changes }))} 
            />
            
            <SettingsSection 
              params={params} 
              onChange={setParams} 
            />

            <div className="flex justify-center mt-8">
              <button
                onClick={handleStartProcessing}
                className="group relative inline-flex items-center justify-center px-8 py-4 font-semibold text-white transition-all duration-200 bg-primary-600 rounded-full hover:bg-primary-700 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-600 focus:ring-offset-slate-900"
              >
                <span className="absolute inset-0 w-full h-full -mt-1 rounded-lg opacity-30 bg-gradient-to-b from-transparent via-transparent to-black"></span>
                <span className="relative flex items-center gap-2 text-lg">
                  <Zap size={20} className="group-hover:text-yellow-300 transition-colors" />
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
      </main>
    </div>
  );
}

export default App;
