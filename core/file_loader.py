import os
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip

from utils.logger import logger

class VideoFileLoader:
    def __init__(self):
        """Initialize the video file loader"""
        pass
    
    def load(self, file_path):
        """Load and validate a local video file
        
        Args:
            file_path: Path to the video file
            
        Returns:
            str: Absolute path to the video file if valid, None otherwise
        """
        file_path = Path(file_path).absolute()
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return None
            
        # Check if it's a video file
        if not self._is_valid_video(str(file_path)):
            logger.error(f"Not a valid video file: {file_path}")
            return None
            
        logger.info(f"Successfully loaded video: {file_path}")
        return str(file_path)
    
    def _is_valid_video(self, file_path):
        """Check if a file is a valid video
        
        Args:
            file_path: Path to the file
            
        Returns:
            bool: True if valid video, False otherwise
        """
        try:
            # Try to open with OpenCV
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False
                
            # Read one frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False
                
            # Try to get video info with MoviePy
            with VideoFileClip(file_path) as clip:
                duration = clip.duration
                if duration <= 0:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating video file: {str(e)}")
            return False