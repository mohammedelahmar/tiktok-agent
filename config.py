import os
from pathlib import Path
import torch  # Add import

# Base paths
PROJECT_ROOT = Path(__file__).parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# GPU settings
USE_GPU = os.environ.get("TIKTOK_USE_GPU", "1").lower() in ("1", "true", "yes", "y")
GPU_DEVICE = os.environ.get("TIKTOK_GPU_DEVICE", "0")  # For multi-GPU systems
# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available() if USE_GPU else False

# Video settings - allow override from environment variables
DEFAULT_CLIP_DURATION = float(os.environ.get("TIKTOK_DEFAULT_CLIP_DURATION", 15))  # seconds
OUTPUT_RESOLUTION = (
    int(os.environ.get("TIKTOK_OUTPUT_WIDTH", 1080)), 
    int(os.environ.get("TIKTOK_OUTPUT_HEIGHT", 1920))
)  # 9:16 portrait (width, height)
OUTPUT_QUALITY = os.environ.get("TIKTOK_OUTPUT_QUALITY", "high")  # high, medium, low
OUTPUT_FORMAT = os.environ.get("TIKTOK_OUTPUT_FORMAT", "mp4")

# Watermark settings
WATERMARK_ENABLED = os.environ.get("TIKTOK_WATERMARK_ENABLED", "0").lower() in ("1", "true", "yes", "y")
WATERMARK_TYPE = os.environ.get("TIKTOK_WATERMARK_TYPE", "text")  # text or image
WATERMARK_TEXT = os.environ.get("TIKTOK_WATERMARK_TEXT", "@YourUsername")
WATERMARK_IMAGE = os.environ.get("TIKTOK_WATERMARK_IMAGE", "")  # Path to watermark image
WATERMARK_POSITION = os.environ.get("TIKTOK_WATERMARK_POSITION", "bottom-right")  # top-left, top-right, bottom-left, bottom-right, center
WATERMARK_OPACITY = float(os.environ.get("TIKTOK_WATERMARK_OPACITY", "0.7"))  # 0.0 to 1.0
WATERMARK_PADDING = int(os.environ.get("TIKTOK_WATERMARK_PADDING", "20"))  # Padding from edges in pixels
WATERMARK_TEXT_SIZE = int(os.environ.get("TIKTOK_WATERMARK_TEXT_SIZE", "40"))
WATERMARK_TEXT_COLOR = os.environ.get("TIKTOK_WATERMARK_TEXT_COLOR", "white")
WATERMARK_FONT = os.environ.get("TIKTOK_WATERMARK_FONT", "")  # Path to font file or empty for default

# Model settings
USE_ENGAGEMENT_MODEL = os.environ.get("TIKTOK_USE_ENGAGEMENT_MODEL", "1").lower() in ("1", "true", "yes", "y")
MODEL_WEIGHTS_PATH = MODELS_DIR / "engagement_weights.pth"

# Face detection settings
FACE_DETECTOR = os.environ.get("TIKTOK_FACE_DETECTOR", "mediapipe")  # opencv, mediapipe, or none

# YouTube download settings
YT_DOWNLOAD_RESOLUTION = os.environ.get("TIKTOK_YT_DOWNLOAD_RESOLUTION", "720p")

# Parallelism settings
MAX_WORKERS = int(os.environ.get("TIKTOK_MAX_WORKERS", os.cpu_count() or 4))

# Face detection model URLs
FACE_DETECTION_PROTOTXT_URL = os.environ.get(
    "TIKTOK_FACE_DETECTION_PROTOTXT_URL", 
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
FACE_DETECTION_MODEL_URL = os.environ.get(
    "TIKTOK_FACE_DETECTION_MODEL_URL", 
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)