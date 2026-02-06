import React, { useState } from 'react';
import { Upload, Youtube, FileVideo, Link } from 'lucide-react';
import axios from 'axios';
import Card from './ui/Card';
import Input from './ui/Input';
import Button from './ui/Button';

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
    <Card className="p-8">
      <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
        <span className="bg-gradient-to-br from-rose-500 to-orange-500 p-2 rounded-lg shadow-lg shadow-rose-500/20">
          <Link size={18} className="text-white" />
        </span>
        Input Source
      </h2>
      
      <div className="flex gap-4 mb-8 p-1 bg-black/40 rounded-xl w-fit border border-white/5 backdrop-blur-md">
        <button
          onClick={() => setActiveTab('youtube')}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-lg transition-all font-medium ${
            activeTab === 'youtube' 
              ? 'bg-gradient-to-r from-rose-600 to-orange-600 text-white shadow-lg shadow-rose-500/20' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <Youtube size={18} />
          YouTube URL
        </button>
        <button
          onClick={() => setActiveTab('file')}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-lg transition-all font-medium ${
            activeTab === 'file' 
              ? 'bg-gradient-to-r from-rose-600 to-orange-600 text-white shadow-lg shadow-rose-500/20' 
              : 'text-gray-400 hover:text-white hover:bg-white/5'
          }`}
        >
          <FileVideo size={18} />
          Upload File
        </button>
      </div>

      <div className="relative animate-in fade-in duration-300">
        {activeTab === 'youtube' ? (
          <div className="space-y-3">
            <Input
              label="YouTube Video URL"
              value={url}
              onChange={handleUrlChange}
              placeholder="https://www.youtube.com/watch?v=..."
            />
          </div>
        ) : (
          <div className="space-y-3">
            <label className="block text-xs font-medium text-gray-400 mb-1.5 ml-1">Local Video File</label>
            <div className="border-2 border-dashed border-white/10 rounded-xl p-10 text-center hover:border-rose-500/50 hover:bg-white/5 transition-all cursor-pointer group bg-black/20">
              <input
                type="file"
                onChange={handleFileUpload}
                accept="video/*"
                className="hidden"
                id="video-upload"
              />
              <label htmlFor="video-upload" className="cursor-pointer flex flex-col items-center gap-4 w-full h-full">
                <div className={`p-4 rounded-full bg-white/5 group-hover:bg-rose-500/20 transition-colors ${isUploading ? 'animate-pulse' : ''}`}>
                  <Upload size={24} className="text-gray-400 group-hover:text-rose-400" />
                </div>
                <div>
                  <p className="font-medium text-gray-200 group-hover:text-white transition-colors">
                    {filename || (isUploading ? "Uploading..." : "Click to browse or drag file")}
                  </p>
                  <p className="text-sm text-gray-500 mt-2">MP4, MKV, MOV supported</p>
                </div>
              </label>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default InputSection;
