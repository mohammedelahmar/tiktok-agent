# 🌟 TikTok Agent

TikTok Agent is an advanced tool for automatically extracting viral-worthy clips from videos and formatting them for TikTok. It uses engagement modeling to identify the most promising segments of your content and transforms them into TikTok-ready videos.

---

## ✨ Features

* **Viral Clip Extraction**: Automatically identifies the most engaging segments of videos
* **Multiple Clips**: Extract several viral clips from a single video
* **Video Formatting**: Converts videos to TikTok's 9:16 aspect ratio using:

  * Crop (center focus)
  * Blur (background blur)
  * Bars (letterbox)
* **Watermarking**: Add text or image watermarks to your videos
* **Multi-source Input**: Process local video files or YouTube videos
* **Cloud Storage**: Automatically upload processed clips to Google Drive or AWS S3
* **Face Detection**: Uses advanced face detection to improve framing (OpenCV or MediaPipe)
* **Performance Optimization**: Parallel processing and optional GPU acceleration

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tiktok_agent.git
cd tiktok_agent

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Interactive Mode

For an easy guided experience:

```bash
python main.py
```

### Command Line Interface

**Basic usage:**

```bash
# Process a local video file
python main.py --file /path/to/video.mp4

# Process a YouTube video
python main.py --youtube https://www.youtube.com/watch?v=VIDEO_ID
```

**Extract multiple clips:**

```bash
python main.py --file /path/to/video.mp4 --num-clips 3 --duration 15
```

**Add watermarks:**

```bash
python main.py --youtube https://www.youtube.com/watch?v=VIDEO_ID --watermark --watermark-text "@YourTikTokHandle"
```

**Upload to cloud:**

```bash
# Google Drive
python main.py --file /path/to/video.mp4 --upload-cloud --cloud-provider gdrive

# AWS S3
python main.py --file /path/to/video.mp4 --upload-cloud --cloud-provider s3
```

---

## 📁 Configuration

Configure the tool using:

* Command line arguments
* Environment variables (prefixed with `TIKTOK_`)
* The `config.py` file

### Key Options

| Option                  | Description                           | Default        |
| ----------------------- | ------------------------------------- | -------------- |
| `OUTPUT_FORMAT`         | Video output format                   | `mp4`          |
| `DEFAULT_CLIP_DURATION` | Length of extracted clips             | `15.0` seconds |
| `WATERMARK_ENABLED`     | Enable watermark                      | `False`        |
| `USE_ENGAGEMENT_MODEL`  | Use ML model for viral clip detection | `True`         |
| `FACE_DETECTOR`         | Face detection method                 | `mediapipe`    |
| `CLOUD_STORAGE_ENABLED` | Enable cloud uploads                  | `False`        |

---

## ⚖️ Advanced Features

### Face Detection

```bash
python main.py --file video.mp4 --face-detector mediapipe
```

### Performance Optimization

```bash
python main.py --youtube https://youtu.be/video_id --workers 8 --use-gpu
```

### Format Options

```bash
python main.py --file video.mp4 --format blur --watermark
```

### Cloud Storage Setup

#### Google Drive

1. Create credentials in the Google Cloud Console
2. Save the credentials file as `credentials.json`
3. Run with:

```bash
python main.py --upload-cloud --cloud-provider gdrive
```

#### AWS S3

Configure with:

```bash
python main.py --upload-cloud --cloud-provider s3 --s3-bucket your-bucket --s3-region your-region
```

---

## 📄 License

**MIT License**

---

## 💪 Contributing

Contributions welcome! Please feel free to submit a Pull Request.
