import os
import urllib.request
from pathlib import Path
import config
from utils.logger import logger

def download_face_detection_model():
    """Download the face detection model files if they don't exist"""
    models_dir = Path(config.PROJECT_ROOT) / "models" / "face_detection"
    os.makedirs(models_dir, exist_ok=True)
    
    prototxt_path = models_dir / "deploy.prototxt"
    caffemodel_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    # URLs for the face detection model files - now from config with fallbacks
    prototxt_url = getattr(config, 'FACE_DETECTION_PROTOTXT_URL', 
                          "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt")
    caffemodel_url = getattr(config, 'FACE_DETECTION_MODEL_URL', 
                            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    
    try:
        # Download prototxt if it doesn't exist
        if not prototxt_path.exists():
            logger.info(f"Downloading face detection model prototxt from {prototxt_url}...")
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        
        # Download caffemodel if it doesn't exist
        if not caffemodel_path.exists():
            logger.info(f"Downloading face detection model weights from {caffemodel_url} (this may take a while)...")
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
            
        logger.info(f"Face detection model files downloaded successfully to {models_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading face detection model: {str(e)}")
        return False

if __name__ == "__main__":
    download_face_detection_model()