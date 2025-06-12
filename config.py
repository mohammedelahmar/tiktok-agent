import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Video settings
DEFAULT_CLIP_DURATION = 15  # seconds
OUTPUT_RESOLUTION = (1080, 1920)  # 9:16 portrait (width, height)
OUTPUT_QUALITY = "high"  # high, medium, low
OUTPUT_FORMAT = "mp4"

# Model settings
USE_ENGAGEMENT_MODEL = True
MODEL_WEIGHTS_PATH = MODELS_DIR / "engagement_weights.pth"

# YouTube download settings
YT_DOWNLOAD_RESOLUTION = "720p"