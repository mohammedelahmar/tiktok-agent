import React, { useState } from 'react';
import { Upload, Youtube, FileVideo, Link } from 'lucide-react';
import axios from 'axios';

const InputSection = ({ onInputParamsChange }) => {
  const [activeTab, setActiveTab] = useState('youtube');
  const [url, setUrl] = useState('');
  const [filename, setFilename] = useState('');
  const [isUploading, setIsUploading] = useState(false);

  const handleUrlChange = (e) => {
    setUrl(e.target.value);
    onInputParamsChange({ source_type: 'youtube', source_path: e.target.value });
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setFilename(response.data.original_name);
      onInputParamsChange({ source_type: 'file', source_path: response.data.filename });
    } catch (error) {
      console.error('Upload failed', error);
      alert('Upload failed: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 shadow-xl mb-6">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <span className="bg-primary-600 p-1.5 rounded-lg"><Link size={18} /></span>
        Input Source
      </h2>
      
      <div className="flex gap-4 mb-6 bg-slate-800/50 p-1 rounded-lg w-fit">
        <button
          onClick={() => setActiveTab('youtube')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
            activeTab === 'youtube' 
              ? 'bg-primary-600 text-white shadow-lg shadow-primary-900/20' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <Youtube size={18} />
          YouTube URL
        </button>
        <button
          onClick={() => setActiveTab('file')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${
            activeTab === 'file' 
              ? 'bg-primary-600 text-white shadow-lg shadow-primary-900/20' 
              : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
          }`}
        >
          <FileVideo size={18} />
          Upload File
        </button>
      </div>

      <div className="relative">
        {activeTab === 'youtube' ? (
          <div className="space-y-2">
            <label className="text-sm text-slate-400 font-medium">YouTube Video URL</label>
            <div className="relative">
              <input
                type="text"
                value={url}
                onChange={handleUrlChange}
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full bg-slate-950 border border-slate-700 rounded-lg py-3 px-4 focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all placeholder:text-slate-600"
              />
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-sm text-slate-400 font-medium">Local Video File</label>
            <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-primary-500 transition-colors bg-slate-950/30">
              <input
                type="file"
                onChange={handleFileUpload}
                accept="video/*"
                className="hidden"
                id="video-upload"
              />
              <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center gap-3">
                <div className={`p-4 rounded-full bg-slate-800 ${isUploading ? 'animate-pulse' : ''}`}>
                  <Upload size={24} className="text-primary-400" />
                </div>
                <div>
                  <p className="font-medium text-slate-200">
                    {filename || (isUploading ? "Uploading..." : "Click to browse or drag file")}
                  </p>
                  <p className="text-sm text-slate-500 mt-1">MP4, MKV, MOV supported</p>
                </div>
              </label>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InputSection;
