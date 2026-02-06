# TikTok Agent Features

This document provides a comprehensive overview of all features available in the TikTok Agent application.

## Core Processing Features

### 🤖 AI Viral Detection
The core engine analyzes video content to identify segments with high viral potential.
- **Motion Analysis**: Calculates motion energy to find action-packed scenes.
- **Scene Detection**: Identifies scene boundaries to ensure clips have logical start and end points.
- **Scoring**: Assigns a "virality score" to each candidate clip.

### 🎯 Smart Hook Optimization
- Automatically detects the most engaging moments to serve as "hooks" (the first 3 seconds associated with high retention).
- Optimizes clip start times to capture these hooks.

### 📱 Intelligent reformatting (Landscape to 9:16)
Converts standard 16:9 YouTube videos into 9:16 vertical shorts.
- **Crop Mode**: Smartly crops the center of the video (ideal for talking heads).
- **Blur Mode**: Adds a blurred background fill for preserving full context (ideal for gaming/screencasts).

### 📝 AI Metadata Generation
Integrates with Google Gemini LLM to generate SEO-optimized metadata.
- **Viral Titles**: Generates click-worthy titles.
- **Descriptions**: Writes engaging video descriptions.
- **Hashtags**: Generates relevant, high-traffic hashtags.

## User Interface (Web App)

### 💎 Neon Glass UI
A premium, "Cyberpunk Lab" aesthetic featuring:
- **Glassmorphism**: Translucent panels and blur effects.
- **Ambient Backgrounds**: Animated glowing orbs for a dynamic feel.
- **Responsive Design**: Works on Desktop and Mobile.

### ✂️ Precision Trimmer
A dedicated review stage for AI-selected candidates.
- **Frame-Perfect Control**: Fine-tune the start and end times of any clip.
- **Preview**: Watch the clip before rendering.
- **Score Visualization**: See the AI's confidence score for each candidate.

### 📜 History Dashboard
A persistent record of your work.
- **Job Tracking**: View all past jobs (Completed, Failed, Processing).
- **Resume Capability**: Re-open any past job to view results or re-download clips.
- **Status Monitoring**: Real-time progress updates.

## Technical Features

### 💾 Job Persistence
- **State Preservation**: All jobs are saved to a local JSON database (`jobs.json`).
- **Crash Recovery**: If the server stops, your job history remains when you restart.

### 🌊 Watermarking
- **Text Watermarks**: Overlay custom text (e.g., channel name) on videos.
- **Image Watermarks**: Support for logo overlays (CLI/Code level).

### 📥 Multi-Source Support
- **YouTube Downloader**: Paste a URL to automatically download and process.
- **File Upload**: Drag and drop local video files for processing.

### 🛠️ CLI & API
- **Full REST API**: Powered by FastAPI, allowing for external integration.
- **CLI Mode**: Run `main.py` for headless automation scripts.
