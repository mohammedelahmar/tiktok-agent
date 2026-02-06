# 🌟 TikTok Agent

**TikTok Agent** is an advanced AI-powered tool that automatically extracts viral-worthy clips from long-form videos and formats them for TikTok, YouTube Shorts, and Instagram Reels.

It features a **modern Web UI** where you can review AI-selected candidates, manually trim clips with frame precision, and batch render your final videos.

---

## ✨ Features

*   **🤖 AI Viral Detection**: Automatically identifies the most engaging segments using motion scoring and scene analysis.
*   **🖥️ Interactive Web UI**: A beautiful, dark-mode interface for managing your video projects.
*   **✂️ Precision Trimmer**: Watch AI-selected clips and fine-tune start/end times with a frame-perfect scrubber before rendering.
*   **📱 9:16 Auto-Formatting**: Smart cropping and background blur modes to convert landscape video to vertical format.
*   **🎯 Smart Hook Optimization**: Automatically detects and preserves the most engaging "hook" moments.
*   **📝 AI Metadata Generator**: Generates viral titles, descriptions, and hashtags using Google Gemini LLM.
*   **🌊 Watermarking**: Add custom text or image watermarks.
*   **📥 Multi-Source**: Support for YouTube URLs and local file uploads.

---

## 🚀 Quick Start (Web App)

### Prerequisites
*   Python 3.8+
*   Node.js & npm
*   FFmpeg (must be in your system PATH)
*   **Google Gemini API Key** (optional, for AI metadata)

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tiktok_agent.git
cd tiktok_agent

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Setup Environment Variables
# Create a .env file and add your API key:
# GEMINI_API_KEY=your_api_key_here
```

### 2. Frontend Setup

```bash
cd web
npm install
```

### 3. Run the App

You need to run the backend and frontend in separate terminals.

**Terminal 1 (Backend):**
```bash
# From root directory
python web_api.py
```

**Terminal 2 (Frontend):**
```bash
# From web directory
npm run dev
```

Open your browser at `http://localhost:5173` to start creating!

---

## 🛠️ Usage Guide

1.  **Input**: Paste a YouTube URL or upload a video file.
2.  **Configure**: Set the number of clips, duration, and format style (Crop/Blur).
    *   *Optional*: Enable **"AI Metadata (SEO)"** to generate titles/tags (requires API key).
3.  **Analyze**: The AI scans the video (take a coffee break ☕).
4.  **Review**:
    *   You'll see a list of "Viral Candidates" found by the AI.
    *   Click **Adjust** on any clip to open the Trimmer.
    *   Use the sliders to fix the timing.
    *   Click **Confirm**.
5.  **Render**: Click "Render All Clips" to generate the final MP4s.

---

## 💻 CLI Usage (Advanced)

You can also use the tool purely from the command line for automation.

```bash
# Process a YouTube video
python main.py --youtube https://youtu.be/VIDEO_ID --num-clips 3

# Process a local file
python main.py --file video.mp4

# Advanced options
python main.py --file video.mp4 --format blur --watermark --watermark-text "@MyChannel"
```

### Key Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--youtube` | YouTube URL to download | |
| `--file` | Local file path | |
| `--num-clips` | Number of clips to extract | 1 |
| `--duration` | Target duration in seconds | 15.0 |
| `--format` | `crop` (center), `blur` (blurred bg), `bars` | crop |
| `--use-gpu` | Enable GPU acceleration | False |

---

## 🏗️ Architecture

*   **Backend**: Python, FastAPI, OpenCV, PyTorch, MoviePy
*   **Frontend**: React, Vite, TailwindCSS, Lucide Icons
*   **AI Models**:
    *   **Face Detection**: MediaPipe / OpenCV DNN
    *   **Visual Interest**: Motion energy & scene complexity analysis

---

## 📄 License

MIT License. Free to use and modify!
