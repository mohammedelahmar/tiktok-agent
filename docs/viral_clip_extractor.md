# Viral Clip Extractor Documentation

## 📄 Overview

The `viral_clip_extractor.py` file defines the `ViralClipExtractor` class, a powerful tool for analyzing videos and automatically identifying the most engaging segments. It simplifies the process of turning long-form content into TikTok-ready viral clips using intelligent scoring and automated trimming.

---

## 🧠 How It Works

The extractor uses a machine learning engagement model to:

1. **Segment** the video into 1-second intervals (or custom duration)
2. **Score** each segment for engagement potential
3. **Locate** the highest-scoring continuous window(s) of specified duration
4. **Extract** and save those clips as individual videos

---

## ✨ Features

* **🎯 Smart Scoring**: Uses an AI model to find the most engaging video parts
* **🎬 Single or Multiple Clips**: Extract one or several high-scoring clips
* **⏱️ Custom Durations**: Define the length of each clip
* **📄 Metadata Embedding**: Embed clip info (score, timestamps) in the output file
* **⚡ Fast Performance**: Supports parallel processing for multiple clips

---

## 🔧 Core Methods

### `extract_best_clip()`

Extracts the single most engaging clip from a video.

**Parameters:**

* `video_path`: Path to the source video
* `clip_duration`: Duration of the extracted clip in seconds
* `output_path`: Where to save the output clip (optional)
* `segment_duration`: Duration of analysis segments (default: 1.0s)

**Returns:**

```python
(clip_path, start_time, end_time, score)
```

---

### `extract_multiple_clips()`

Extracts multiple non-overlapping viral clips from a video.

**Parameters:**

* `video_path`: Path to the source video
* `num_clips`: Number of clips to extract
* `clip_duration`: Duration of each clip
* `min_gap`: Minimum spacing between clips (in seconds)
* `output_prefix`: Prefix for naming output clips
* `parallel_processing`: Enable multi-threaded processing (optional)

**Returns:**

```python
[(clip_path, start_time, end_time, score), ...]
```

---

## 🔁 Workflow

1. **Initialization**: Loads the engagement scoring model
2. **Segment Scoring**: Breaks video into small pieces and evaluates them
3. **Window Selection**: Finds best-scoring continuous segment(s)
4. **Clip Cutting**: Extracts clips from original video
5. **Metadata Saving**: Embeds metadata in the output filename

---

## 🚀 Example Usage

```python
# Initialize extractor
extractor = ViralClipExtractor()

# Extract a single 15-second viral clip
clip_path, start, end, score = extractor.extract_best_clip(
    video_path="my_video.mp4",
    clip_duration=15.0
)

# Extract three 10-second viral clips
clips = extractor.extract_multiple_clips(
    video_path="my_video.mp4",
    num_clips=3,
    clip_duration=10.0
)
```

---

## 🔍 Behind the Scenes

The extractor uses a **sliding window** technique to compute average scores over time windows and select the most engaging sections. For multiple clips, it iteratively finds the top-scoring window, excludes it (with a buffer), and continues searching.

---

## 📂 Output Format

Each output file includes:

* Clip index (for multiple clips)
* Engagement score
* Start & end times

**Example filename:**

```
original_video_1_score0.87_10s-25s.mp4
```

---

## 🎥 Why Use This?

TikTok creators, editors, and content marketers can instantly:

* Save time scanning long videos
* Automatically identify viral-worthy highlights
* Produce short-form content ready for sharing

This tool takes your raw video and turns it into TikTok gold — intelligently and effortlessly.
