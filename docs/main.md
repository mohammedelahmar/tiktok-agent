# Documentation: `main.py`

## Overview

`main.py` is the main entry point for the TikTok Agent application. It provides a command-line interface (CLI) and an interactive mode for extracting viral clips from local or YouTube videos, formatting them to TikTok's 9:16 ratio, and optionally uploading them to cloud storage.

---

## Main Functionalities

* Accepts input video via file path or YouTube URL
* Allows clip customization (duration, number of clips, formatting method, watermark)
* Provides performance tuning (GPU usage, parallel processing)
* Supports watermarking (text or image)
* Offers cloud storage uploads (Google Drive, AWS S3)
* Includes an interactive mode for ease of use

---

## Key Functions

### `main()`

Handles argument parsing, program execution, video processing, formatting, and optional cloud uploading.

#### Steps:

1. Parse CLI arguments
2. Determine if interactive mode is needed
3. Download or load video
4. Extract viral clip(s)
5. Format the clip(s) for TikTok
6. Optionally, upload result(s) to cloud storage

---

### `interactive_mode(args)`

Prompts the user for input options via console and file dialogs, allowing:

* Selecting input source (file or YouTube)
* Setting number/duration of clips
* Choosing formatting method and watermark options
* Selecting face detection, GPU, workers, and cloud options

Returns the updated `args` object.

---

### `get_video(args)`

Downloads a YouTube video or loads a local file based on the provided arguments.

* Returns the local file path of the video

---

## CLI Arguments Summary

| Argument               | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `--youtube` / `--file` | Input source (YouTube URL or local file path)        |
| `--duration`           | Duration of clip in seconds                          |
| `--num-clips`          | Number of viral clips to extract                     |
| `--format`             | Output formatting style: `crop`, `blur`, or `bars`   |
| `--watermark`          | Enable watermarking                                  |
| `--watermark-type`     | Watermark type: `text` or `image`                    |
| `--output`             | Output path or prefix for formatted clips            |
| `--workers`            | Number of parallel workers                           |
| `--face-detector`      | Face detection method: `opencv`, `mediapipe`, `none` |
| `--use-gpu`            | Enable GPU acceleration                              |
| `--upload-cloud`       | Enable cloud upload                                  |
| `--cloud-provider`     | Choose `gdrive` or `s3`                              |

#### Additional Google Drive & AWS S3 options are available via grouped arguments.

---

## Logging

* Configurable log levels via `--log-level`
* Uses `utils.logger` to handle info, error, and debug messages

---

## Dependencies

* `torch`
* `tkinter`
* `argparse`, `os`, `sys`, `logging`
* Project modules: `config`, `utils`, `core`

---

## Execution

Run the script via:

```bash
python main.py [options]
```

Or let the tool guide you interactively:

```bash
python main.py
```

---

## Notes

* Handles both single and multiple clip extraction
* Smart formatting and watermarking included
* Designed for scalability and user customization
